import os
import time

import torch

from torchtitan import utils
from torchtitan.config_manager import TORCH_DTYPE_MAP, JobConfig
from torchtitan.datasets import build_tokenizer
from torchtitan.logging import init_logger, logger
from torchtitan.metrics import build_gpu_memory_monitor
from torchtitan.models import (model_name_to_cls, model_name_to_tokenizer,
                               models_config)
from torchtitan.parallelisms import ParallelDims, models_parallelize_fns


def prepare(job_config: JobConfig):
    init_logger()
    logger.info(f"Starting job: {job_config.job.description}")

    # used for colorful printing
    color = utils.Color  # if job_config.metrics.enable_color_printing else utils.NoColor

    # take control of garbage collection to avoid stragglers
    gc_handler = utils.GarbageCollection(gc_freq=job_config.training.gc_freq)

    # init distributed
    world_size = int(os.environ["WORLD_SIZE"])
    parallel_dims = ParallelDims(
        dp_shard=job_config.training.data_parallel_shard_degree,
        dp_replicate=job_config.training.data_parallel_replicate_degree,
        tp=job_config.training.tensor_parallel_degree,
        pp=job_config.experimental.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=job_config.training.enable_loss_parallel,
    )
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.cuda.set_device(device)
    utils.init_distributed(job_config)

    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type="cuda")
    assert not parallel_dims.pp_enabled
    model_name = job_config.model.name

    # build tokenizer
    tokenizer_type = model_name_to_tokenizer[model_name]
    tokenizer = build_tokenizer(tokenizer_type, job_config.model.tokenizer_path)

    # build model (using meta init)
    model_cls = model_name_to_cls[model_name]
    model_config = models_config[model_name][job_config.model.flavor]
    # set the model configs from training inputs:
    # 1. norm type to decide which norm layer to use
    # 2. vocab size from tokenizer
    # 3. max_seq_len base on inputs
    model_config.norm_type = job_config.model.norm_type
    model_config.vocab_size = tokenizer.n_words
    model_config.max_seq_len = job_config.training.seq_len

    with torch.device("meta"):
        model = model_cls.from_model_args(model_config)

    # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
    models_parallelize_fns[model_name](model, world_mesh, parallel_dims, job_config)

    # move sharded model to CPU/GPU and initialize weights via DTensor
    init_device = "cpu" if job_config.checkpoint.create_seed_checkpoint else "cuda"
    model.to_empty(device=init_device)
    model.init_weights()
    model.eval()

    return model_config, model, color, parallel_dims


def main(job_config: JobConfig):
    model_config, model, color, parallel_dims = prepare(job_config)
    bs = job_config.training.batch_size
    seq_len = job_config.training.seq_len
    vocab_size = model_config.vocab_size

    gpu_memory_monitor = build_gpu_memory_monitor()

    logger.info(
        f"{color.red}dp enabled? {parallel_dims.dp_enabled}, "
        f"tp enabled? {parallel_dims.tp_enabled}{color.reset}")
    logger.info(f"warming up on model: {job_config.model.name} {job_config.model.flavor}")
    if parallel_dims.tp_enabled:
        generator = torch.Generator(device='cpu').manual_seed(1105)
    else:
        generator = None

    if (not parallel_dims.dp_enabled) and parallel_dims.tp_enabled:
        dtype = TORCH_DTYPE_MAP[job_config.training.mixed_precision_param]
        model = model.to(dtype=dtype)
    with torch.no_grad():
        input_ids = torch.randint(0, vocab_size, (bs, seq_len), generator=generator).to(device="cuda")
        model(input_ids)
    logger.info("warmup done")

    time_last_log = time.perf_counter()
    ntokens_since_last_log = 0
    for i in range(100):
        gpu_mem_stats = gpu_memory_monitor.get_peak_stats()
        with torch.no_grad():
            input_ids = torch.randint(0, vocab_size, (bs, seq_len), generator=generator).to(device="cuda")
            model(input_ids)
            ntokens_since_last_log += bs * seq_len
        if (i + 1) % 10 == 0:
            time_delta = time.perf_counter() - time_last_log
            logger.info(
                f"iteration {i+1} done "
                f"{color.cyan}memory: {gpu_mem_stats.max_reserved_gib:5.2f}GiB"
                f"({gpu_mem_stats.max_reserved_pct:.2f}%)  "
                f"{color.yellow}bs={bs}, seq_len={seq_len} "
                f"{color.green}latency per inference: {time_delta / 10:.2f} sec "
                f"{color.red}throughput: {ntokens_since_last_log / (time_delta*parallel_dims.tp):.2f} tokens/device/sec "
                f"{color.reset}"
            )
            time_last_log = time.perf_counter()
            ntokens_since_last_log = 0
        gpu_memory_monitor.reset_peak_stats()


if __name__ == "__main__":
    config = JobConfig()
    config.parse_args()
    main(config)
    torch.distributed.destroy_process_group()

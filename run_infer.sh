#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

if [ -f venv/bin/activate ]; then
    source venv/bin/activate
elif [ -d /apps/dat/cv/felixyu/venv_torchtitan ]; then
    cp -r /apps/dat/cv/felixyu/venv_torchtitan ./venv
    source venv/bin/activate
else
    echo "No venv found"
    exit -1
fi
pip install wandb -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# use envs as local overrides for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_llama_train.sh
NGPU=${NGPU:-"8"}
LOG_RANK=${LOG_RANK:-0}
CONFIG_FILE=${CONFIG_FILE:-"./train_configs/debug_model.toml"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    inference.py --job.config_file ${CONFIG_FILE} $overrides

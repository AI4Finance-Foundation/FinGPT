#!/usr/bin/env bash

# export HF_HOME=/path/to/huggingface/

python train.py --actor-model facebook/opt-sent-1.3b --deployment-type single_gpu --output-dir checkpoints
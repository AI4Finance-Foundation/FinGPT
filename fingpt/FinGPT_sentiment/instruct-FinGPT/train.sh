#!/usr/bin/env bash

# export HF_HOME=/path/to/huggingface/

python train.py --actor-model decapoda-research/sent-llama-7b-hf --deployment-type single_node --output-dir checkpoints
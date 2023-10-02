#!/usr/bin/env bash

# export HF_HOME=/mnt/boyu.muzhe/model/huggingface/

# facebook/opt-1.3b
# python chat.py --phase=infer --path=/mnt/boyu.muzhe/code/DeepSpeedExamples/applications/DeepSpeed-Chat/output_quake/actor-models/sent-llama-7b

# meta-llama/Llama-2-7b
# /xfs/home/tensor_zy/chatds/llama/llama2-7b-hf
# deepspeed --num_gpus 8 ./inference/batchbot.py --path facebook/opt-1.3b --max_new_tokens 16 --debug
# deepspeed --num_gpus 8 ./inference/batchbot.py --path /xfs/home/tensor_zy/chatds/llama/llama2-7b-hf --max_new_tokens 16 --debug true
# deepspeed --num_gpus 8 ./inference/batchbot.py --path checkpoints/actor-models/sent-llama-7b --max_new_tokens 16 --debug true

python ./inference/batchbot_torch.py --path /xfs/home/tensor_zy/chatds/llama/llama2-7b-hf --max_new_tokens 16 --debug

python ./inference/batchbot_torch.py --path checkpoints/actor-models/sent-llama2-7b --max_new_tokens 16 --debug
nohup python ./inference/batchbot_torch.py --path checkpoints/actor-models/sent-llama2-7b --max_new_tokens 16 > llama2-7b-in 2>&1 &
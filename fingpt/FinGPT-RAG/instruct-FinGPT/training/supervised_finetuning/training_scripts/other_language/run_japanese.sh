#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
MODEL=sberbank-ai/mGPT
OUTPUT_PATH=./output_japanese_mGPT_1.3b_4data
mkdir -p $OUTPUT_PATH
# The Japanese data we found mostly only contain one response without another
# "rejected" response. Thus we only test the step 1 finetuning and use
# a data_split of 10,0,0 (keep all data for step 1).
deepspeed main.py \
   --data_path mkqa-Japanese Cohere/miracl-ja-queries-22-12 lmqg/qg_jaquad lmqg/qag_jaquad \
   --data_split 10,0,0 \
   --model_name_or_path $MODEL \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 2 \
   --learning_rate 9.65e-6 \
   --num_train_epochs 16  \
   --deepspeed_config ./ds_config.json \
   --deepspeed --seed 1234 --num_warmup_steps 0 \
   --lr_scheduler_type cosine \
   --output_dir $OUTPUT_PATH \
   &> $OUTPUT_PATH/training.log

#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
MODEL=bigscience/bloom-1b1
OUTPUT_PATH=./output_chinese_bloom_1.1b_4data
mkdir -p $OUTPUT_PATH
# The Chinese data we found mostly only contain one response without another
# "rejected" response. Thus we only test the step 1 finetuning and use
# a data_split of 10,0,0 (keep all data for step 1).
deepspeed main.py \
   --data_path wangrui6/Zhihu-KOL Cohere/miracl-zh-queries-22-12 Hello-SimpleAI/HC3-Chinese mkqa-Chinese \
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

export NCCL_IGNORE_DISABLED_P2P=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=0


deepspeed \
--include localhost:2,3 \
train_lora.py \
--run_name dow30v3-llama2-5e-5lr-qkvogud \
--base_model llama2 \
--dataset dow30-20230601-20230930-llama,dow30nobasics-20230601-20230930-llama,dow30v3-20221231-20230531-llama*2 \
--max_length 4096 \
--batch_size 1 \
--gradient_accumulation_steps 16 \
--learning_rate 5e-5 \
--num_epochs 5 \
--log_interval 10 \
--warmup_ratio 0.03 \
--scheduler constant \
--evaluation_strategy steps \
--ds_config config.json
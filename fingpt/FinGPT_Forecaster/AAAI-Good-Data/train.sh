export NCCL_IGNORE_DISABLED_P2P=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=0

deepspeed \
train_lora.py \
--run_name llama3-8b-a100-5e-5lr \
--base_model llama3 \
--dataset "/content/drive/MyDrive/Colab Notebooks/AI4Finance/FinForecaster/Benchmark with Llama3 8b Data/fingpt-forecaster-1105/train/" \
--test_dataset "/content/drive/MyDrive/Colab Notebooks/AI4Finance/FinForecaster/Benchmark with Llama3 8b Data/fingpt-forecaster-1105/test/" \
--max_length 8000 \
--batch_size 2 \
--gradient_accumulation_steps 8 \
--learning_rate 5e-5 \
--num_epochs 5 \
--log_interval 10 \
--warmup_ratio 0.03 \
--scheduler constant \
--evaluation_strategy steps \
--ds_config config.json \


export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_IGNORE_DISABLED_P2P=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=0



#---- Generalization ----

# deepspeed train_lora.py \
# --run_name GRCLS-sentiment-chatglm2-linear-1e-4lr \
# --base_model chatglm2 \
# --dataset headline-cls-instruct,finred-cls-instruct*2,ner-cls-instruct*7 \
# --test_dataset sentiment-cls-instruct \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 1 \
# --log_interval 10 \
# --warmup_ratio 0.03 \
# --scheduler linear \
# --evaluation_strategy steps \
# --ds_config config_hf.json

# deepspeed train_lora.py \
# --run_name GRCLS-sentiment-llama2-linear-small \
# --base_model llama2 \
# --test_dataset sentiment-cls-instruct \
# --dataset headline-cls-instruct,finred-cls-instruct*2,ner-cls-instruct*7 \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 1 \
# --log_interval 10 \
# --warmup_ratio 0 \
# --scheduler linear \
# --evaluation_strategy steps \
# --eval_steps 100 \
# --ds_config config_hf.json

# deepspeed train_lora.py \
# --run_name GRCLS-sentiment-falcon-linear-small \
# --base_model falcon \
# --test_dataset sentiment-cls-instruct \
# --dataset headline-cls-instruct,finred-cls-instruct*2,ner-cls-instruct*7 \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 1 \
# --log_interval 10 \
# --warmup_ratio 0 \
# --scheduler linear \
# --evaluation_strategy steps \
# --eval_steps 100 \
# --ds_config config_hf.json

# deepspeed train_lora.py \
# --run_name GRCLS-sentiment-qwen-linear-small \
# --base_model qwen \
# --test_dataset sentiment-cls-instruct \
# --dataset headline-cls-instruct,finred-cls-instruct*2,ner-cls-instruct*7 \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 1 \
# --log_interval 10 \
# --warmup_ratio 0 \
# --scheduler linear \
# --evaluation_strategy steps \
# --eval_steps 100 \
# --ds_config config_hf.json

# deepspeed train_lora.py \
# --run_name GRCLS-sentiment-bloom-linear-small \
# --base_model bloom \
# --test_dataset sentiment-cls-instruct \
# --dataset headline-cls-instruct,finred-cls-instruct*2,ner-cls-instruct*7 \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 1 \
# --log_interval 10 \
# --warmup_ratio 0 \
# --scheduler linear \
# --evaluation_strategy steps \
# --eval_steps 100 \
# --ds_config config_hf.json

# deepspeed train_lora.py \
# --run_name GRCLS-sentiment-mpt-linear-small \
# --base_model mpt \
# --dataset headline-cls-instruct,finred-cls-instruct*2,ner-cls-instruct*7 \
# --test_dataset sentiment-cls-instruct \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 1 \
# --log_interval 10 \
# --warmup_ratio 0.03 \
# --scheduler linear \
# --evaluation_strategy steps \
# --eval_steps 100 \
# --ds_config config_hf.json


#---- Multi-Task ----

# deepspeed train_lora.py \
# --run_name MT-chatglm2-linear \
# --base_model chatglm2 \
# --dataset sentiment-train,headline,finred*3,ner*15 \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 4

# deepspeed train_lora.py \
# --run_name MT-falcon-linear \
# --base_model falcon \
# --dataset sentiment-train,headline,finred*3,ner*15 \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 4

# deepspeed train_lora.py \
# --run_name MT-qwen-linear \
# --base_model qwen \
# --dataset sentiment-train,headline,finred*3,ner*15 \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 4

# deepspeed train_lora.py \
# --run_name MT-mpt-linear \
# --base_model mpt \
# --dataset sentiment-train,headline,finred*3,ner*15 \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 4

# deepspeed train_lora.py \
# --run_name MT-bloom-linear \
# --base_model bloom \
# --dataset sentiment-train,headline,finred*3,ner*15 \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 4

# deepspeed train_lora.py \
# --run_name MT-llama2-linear \
# --base_model llama2 \
# --dataset sentiment-train,headline,finred*3,ner*15 \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 4 \
# --log_interval 10


#---- FinEval ----

# deepspeed train_lora.py \
# --run_name fineval-internlm-linear \
# --base_model internlm \
# --dataset data/fingpt-fineval \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 50 \
# --log_interval 10

# deepspeed train_lora.py \
# --run_name fineval-llama2-linear \
# --base_model llama2 \
# --dataset data/fingpt-fineval \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 50 \
# --log_interval 10

# deepspeed train_lora.py \
# --run_name fineval-chatglm2-linear \
# --base_model chatglm2 \
# --dataset data/fingpt-fineval \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 50 \
# --log_interval 10

# deepspeed train_lora.py \
# --run_name fineval-falcon-linear \
# --base_model falcon \
# --dataset data/fingpt-fineval \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 50 \
# --log_interval 10

# deepspeed train_lora.py \
# --run_name fineval-qwen-linear \
# --base_model qwen \
# --dataset data/fingpt-fineval \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 50 \
# --log_interval 10

# deepspeed train_lora.py \
# --run_name fineval-mpt-linear \
# --base_model mpt \
# --dataset data/fingpt-fineval \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 50 \
# --log_interval 10

# deepspeed train_lora.py \
# --run_name fineval-bloom-linear \
# --base_model bloom \
# --dataset data/fingpt-fineval \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 50 \
# --log_interval 10


#---- ConvFinQA ----

# deepspeed train_lora.py \
# --run_name convfinqa-llama2-linear \
# --base_model llama2 \
# --ds_config config_hf.json \
# --dataset data/fingpt-convfinqa \
# --max_length 2048 \
# --batch_size 1 \
# --learning_rate 1e-4 \
# --num_epochs 4

# deepspeed train_lora.py \
# --run_name convfinqa-chatglm2-linear \
# --base_model chatglm2 \
# --dataset data/fingpt-convfinqa \
# --max_length 2048 \
# --batch_size 1 \
# --learning_rate 1e-4 \
# --num_epochs 4

# deepspeed train_lora.py \
# --run_name convfinqa-falcon-linear \
# --base_model falcon \
# --dataset data/fingpt-convfinqa \
# --max_length 2048 \
# --batch_size 1 \
# --learning_rate 1e-4 \
# --num_epochs 4

# deepspeed train_lora.py \
# --run_name convfinqa-qwen-linear \
# --base_model qwen \
# --dataset data/fingpt-convfinqa \
# --max_length 2048 \
# --batch_size 1 \
# --learning_rate 1e-4 \
# --num_epochs 4

# deepspeed train_lora.py \
# --run_name convfinqa-mpt-linear \
# --base_model mpt \
# --dataset data/fingpt-convfinqa \
# --max_length 2048 \
# --batch_size 1 \
# --learning_rate 1e-4 \
# --num_epochs 4

# deepspeed train_lora.py \
# --run_name convfinqa-bloom-linear \
# --base_model bloom \
# --dataset data/fingpt-convfinqa \
# --max_length 2048 \
# --batch_size 1 \
# --learning_rate 1e-4 \
# --num_epochs 4


#---- NER ----

# deepspeed train_lora.py \
# --run_name ner-llama2-linear \
# --base_model llama2 \
# --dataset data/fingpt-ner \
# --ds_config config_hf.json \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 100 \
# --log_interval 10

# deepspeed train_lora.py \
# --run_name ner-chatglm2-linear \
# --base_model chatglm2 \
# --dataset data/fingpt-ner \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 100 \
# --log_interval 10

# deepspeed train_lora.py \
# --run_name ner-falcon-linear \
# --base_model falcon \
# --dataset data/fingpt-ner \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 100 \
# --log_interval 10

# deepspeed train_lora.py \
# --run_name ner-qwen-linear \
# --base_model qwen \
# --dataset data/fingpt-ner \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 100 \
# --log_interval 10

# deepspeed train_lora.py \
# --run_name ner-mpt-linear \
# --base_model mpt \
# --dataset data/fingpt-ner \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 100 \
# --log_interval 10

# deepspeed train_lora.py \
# --run_name ner-bloom-linear \
# --base_model bloom \
# --dataset data/fingpt-ner \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 100 \
# --log_interval 10


#---- Headline (IE) ----

# deepspeed train_lora.py \
# --run_name headline-internlm-linear \
# --base_model internlm \
# --dataset data/fingpt-headline \
# --ds_config config_hf.json \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 8

# deepspeed train_lora.py \
# --run_name headline-llama2-linear \
# --base_model llama2 \
# --dataset data/fingpt-headline \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 8

# deepspeed train_lora.py \
# --run_name headline-chatglm2-linear \
# --base_model chatglm2 \
# --dataset data/fingpt-headline \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 8

# deepspeed train_lora.py \
# --run_name headline-falcon-linear \
# --base_model falcon \
# --dataset data/fingpt-headline \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 8

# deepspeed train_lora.py \
# --run_name headline-qwen-linear \
# --base_model qwen \
# --dataset data/fingpt-headline \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 8

# deepspeed train_lora.py \
# --run_name headline-mpt-linear \
# --base_model mpt \
# --dataset data/fingpt-headline \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 8

# deepspeed train_lora.py \
# --run_name headline-bloom-linear \
# --base_model bloom \
# --dataset data/fingpt-headline \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 8

#---- Sentiment Analysis ----

# deepspeed train_lora.py \
# --run_name sentiment-internlm-linear \
# --base_model internlm \
# --dataset data/fingpt-sentiment-train \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 8

# deepspeed train_lora.py \
# --run_name sentiment-llama2-linear \
# --base_model llama2 \
# --dataset data/fingpt-sentiment-train \
# --ds_config config_hf.json \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 8

# deepspeed train_lora.py \
# --run_name sentiment-chatglm2-linear \
# --base_model chatglm2 \
# --dataset data/fingpt-sentiment-train \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 8

# deepspeed train_lora.py \
# --run_name sentiment-falcon-linear \
# --base_model falcon \
# --dataset data/fingpt-sentiment-train \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 8

# deepspeed train_lora.py \
# --run_name sentiment-qwen-linear \
# --base_model qwen \
# --dataset data/fingpt-sentiment-train \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 8

# deepspeed train_lora.py \
# --run_name sentiment-mpt-linear \
# --base_model mpt \
# --dataset data/fingpt-sentiment-train \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 8

# deepspeed train_lora.py \
# --run_name sentiment-bloom-linear \
# --base_model bloom \
# --dataset data/fingpt-sentiment-train \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 8


#---- Relation Extraction ----

# deepspeed train_lora.py \
# --run_name finred-llama2-linear \
# --base_model llama2 \
# --dataset data/fingpt-finred-re \
# --ds_config config_hf.json \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 8

# deepspeed train_lora.py \
# --run_name finred-chatglm2-linear \
# --base_model chatglm2 \
# --dataset data/fingpt-finred-re \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 8

# deepspeed train_lora.py \
# --run_name finred-falcon-linear \
# --base_model falcon \
# --dataset data/fingpt-finred-re \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 8

# deepspeed train_lora.py \
# --run_name finred-qwen-linear \
# --base_model qwen \
# --dataset data/fingpt-finred-re \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 8

# deepspeed train_lora.py \
# --run_name finred-mpt-linear \
# --base_model mpt \
# --dataset data/fingpt-finred-re \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 8

# deepspeed train_lora.py \
# --run_name finred-bloom-linear \
# --base_model bloom \
# --dataset data/fingpt-finred-re \
# --max_length 512 \
# --batch_size 4 \
# --learning_rate 1e-4 \
# --num_epochs 8
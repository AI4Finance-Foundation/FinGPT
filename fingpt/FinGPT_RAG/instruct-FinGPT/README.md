
A fast, affordable, scalable and open system framework for enabling end-to-end Instruction Tuning experience to generate high-quality Instruct-FinGPT models at all scales.

## Training
Use the following command to instruction finetune the llama7b model on the financial sentiment analysis datasets.
```
python train.py --actor-model /path/to/llama7b --deployment-type single_node --output-dir checkpoints
```

Choose the expected deployment-type，(ranging from single_gpu, single_node to multi_node)。These deployment type corresponds to different training scripts in the "training" folder. Modify the parameters of these scripts according to needs. Specially, if you want to finetune with LoRA, you can modify the script in training/supervised_finetuning/single_node/run_sent-llama-7b.sh as:
```
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT

python3 main.py \
   --data_path zeroshot/twitter-financial-news-sentiment chiapudding/kaggle-financial-sentiment \
   --data_split 2,4,4 \
   --model_name_or_path decapoda-research/llama-7b-hf \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 512 \
   --learning_rate 1e-5 \
   --weight_decay 0.1 \
   --num_train_epochs 2  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   --lora_dim 128 \
   --lora_module_name model.layers. \
   &> $OUTPUT/training.log
   ```

## Testing
Use the following command to implement inference.
```
python ./inference/batchbot.py --path checkpoints/actor-models/sent-llama-7b --max_new_tokens 16 --local_rank 0
```


## Acknowledgement
This code is developed based on [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-chat).

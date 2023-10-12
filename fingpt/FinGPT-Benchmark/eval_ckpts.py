from transformers.integrations import TensorBoardCallback
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from transformers.trainer import TRAINING_ARGS_NAME
from torch.utils.tensorboard import SummaryWriter
import datasets
import torch
import os
import sys
import wandb
import argparse
from datetime import datetime
from functools import partial
from utils import *
from glob import glob

# LoRA
from peft import (
    TaskType,
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,   
)

os.environ['WANDB_API_KEY'] = 'ecf1e5e4f47441d46822d38a3249d62e8fc94db4'
os.environ['WANDB_PROJECT'] = 'fingpt-omni'


def main(args):
        
    model_name = parse_model_name(args.base_model)
    
    # load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # load_in_8bit=True,
        # device_map="auto",
        trust_remote_code=True
    )
    if args.local_rank == 0:
        print(model)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if args.base_model != 'mpt':
        tokenizer.padding_side = "left"
    if args.base_model == 'qwen':
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids('<|endoftext|>')
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<|extra_0|>')
    if not tokenizer.pad_token or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    
    dataset_list = load_dataset(args.test_dataset)        
    dataset_test = datasets.concatenate_datasets([d['test'] for d in dataset_list])
    print(dataset_test[0])
        
    dataset_test = dataset_test.map(partial(tokenize, args, tokenizer))
    print('original dataset length: ', len(dataset_test))
    dataset_test = dataset_test.filter(lambda x: not x['exceed_max_length'])
    print('filtered dataset length: ', len(dataset_test))
    dataset_test = dataset_test.remove_columns(['instruction', 'input', 'output', 'exceed_max_length'])
    
    print(dataset_test[0])

    # config
    # deepspeed_config = './config_newnew.json'
    # deepspeed_config = './config_new.json'
    # deepspeed_config = './config.json'
    
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y%m%d%H%M')
    
    training_args = TrainingArguments(
        output_dir=f'finetuned_models/{args.run_name}_{formatted_time}', # 保存位置
        logging_steps=args.log_interval,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        dataloader_num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.scheduler,
        save_steps=0.1,
        eval_steps=0.1,
        fp16=True,
        # fp16_full_eval=True,
        # deepspeed=args.ds_config,
        evaluation_strategy=args.evaluation_strategy,
        remove_unused_columns=False,
        report_to='wandb',
        run_name=args.run_name
    )
    if not args.base_model == 'mpt':
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    
    ckpts = glob(f'finetuned_models/{args.peft_model}/checkpoint-*')
    for ckpt in ckpts:
        model_ckpt = PeftModel.from_pretrained(model, ckpt)

        trainer = Trainer(
            model=model_ckpt, 
            args=training_args, 
            train_dataset=dataset_test,
            eval_dataset=dataset_test, 
            data_collator=DataCollatorForSeq2Seq(
                tokenizer, padding=True,
                return_tensors="pt"
            ),
        )

        # if torch.__version__ >= "2" and sys.platform != "win32":
        #     model = torch.compile(model)

        torch.cuda.empty_cache()
        
        metrics = trainer.evaluate()
        if args.local_rank == 0:
            print(ckpt, metrics)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--run_name", default='local-test', type=str)
    parser.add_argument("--test_dataset", required=True, type=str)
    parser.add_argument("--peft_model", required=True, type=str)    
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--base_model", required=True, type=str, choices=['chatglm2', 'llama2', 'falcon', 'internlm', 'qwen', 'mpt', 'bloom'])
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--batch_size", default=4, type=int, help="The train batch size per device")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="The learning rate")
    parser.add_argument("--num_epochs", default=8, type=float, help="The training epochs")
    parser.add_argument("--num_workers", default=8, type=int, help="dataloader workers")
    parser.add_argument("--log_interval", default=100, type=int)
    parser.add_argument("--warmup_ratio", default=0.05, type=float)
    parser.add_argument("--ds_config", default='./config_new.json', type=str)
    parser.add_argument("--scheduler", default='linear', type=str)
    parser.add_argument("--instruct_template", default='default')
    parser.add_argument("--evaluation_strategy", default='steps', type=str)    
    args = parser.parse_args()
    
    wandb.login()
    main(args)
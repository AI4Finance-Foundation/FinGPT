from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, get_peft_model, LoraConfig, TaskType  # 0.4.0
import torch
import argparse


from fpb import test_fpb, test_fpb_mlt
from fiqa import test_fiqa, test_fiqa_mlt 
from tfns import test_tfns
from nwgi import test_nwgi
from headline import test_headline
from ner import test_ner
from convfinqa import test_convfinqa
from fineval import test_fineval
from finred import test_re


import sys
sys.path.append('../')
from utils import *


def main(args):
    if args.from_remote:
        model_name = parse_model_name(args.base_model, args.from_remote)
    else:
        model_name = '../' + parse_model_name(args.base_model)
        

    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, 
        # load_in_8bit=True
        device_map="auto",
        # fp16=True
    )
    model.model_parallel = True

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"
    if args.base_model == 'qwen':
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids('<|endoftext|>')
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<|extra_0|>')
    if not tokenizer.pad_token or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    
    print(f'pad: {tokenizer.pad_token_id}, eos: {tokenizer.eos_token_id}')
    
    # peft_config = LoraConfig(
    #     task_type=TaskType.CAUSAL_LM,
    #     inference_mode=False,
    #     r=8,
    #     lora_alpha=32,
    #     lora_dropout=0.1,
    #     target_modules=lora_module_dict[args.base_model],
    #     bias='none',
    # )
    # model = get_peft_model(model, peft_config)
    # model.load_state_dict(torch.load(args.peft_model + '/pytorch_model.bin'))

    model = PeftModel.from_pretrained(model, args.peft_model)
    model = model.eval()
    
    with torch.no_grad():
        for data in args.dataset.split(','):
            if data == 'fpb':
                test_fpb(args, model, tokenizer)
            elif data == 'fpb_mlt':
                test_fpb_mlt(args, model, tokenizer)
            elif data == 'fiqa':
                test_fiqa(args, model, tokenizer)
            elif data == 'fiqa_mlt':
                test_fiqa_mlt(args, model, tokenizer)
            elif data == 'tfns':
                test_tfns(args, model, tokenizer)
            elif data == 'nwgi':
                test_nwgi(args, model, tokenizer)
            elif data == 'headline':
                test_headline(args, model, tokenizer)
            elif data == 'ner':
                test_ner(args, model, tokenizer)
            elif data == 'convfinqa':
                test_convfinqa(args, model, tokenizer)
            elif data == 'fineval':
                test_fineval(args, model, tokenizer)
            elif data == 're':
                test_re(args, model, tokenizer)
            else:
                raise ValueError('undefined dataset.')
    
    print('Evaluation Ends.')
        


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--base_model", required=True, type=str, choices=['chatglm2', 'llama2', 'llama2-13b', 'llama2-13b-nr', 'baichuan', 'falcon', 'internlm', 'qwen', 'mpt', 'bloom'])
    parser.add_argument("--peft_model", required=True, type=str)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--batch_size", default=4, type=int, help="The train batch size per device")
    parser.add_argument("--instruct_template", default='default')
    parser.add_argument("--from_remote", default=False, type=bool)    

    args = parser.parse_args()
    
    print(args.base_model)
    print(args.peft_model)
    
    main(args)

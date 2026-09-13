from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, get_peft_model, LoraConfig, TaskType  # 0.4.0
import torch
import argparse
import os


from fpb import test_fpb, test_fpb_mlt
from fiqa import test_fiqa, test_fiqa_mlt
from tfns import test_tfns
from nwgi import test_nwgi
from headline import test_headline
from ner import test_ner
from convfinqa import test_convfinqa
from fineval import test_fineval
from finred import test_re


# Increase HuggingFace Hub timeout to handle slow network connections or large file downloads
os.environ.setdefault("HF_HUB_TIMEOUT", "120")

import sys
sys.path.append('../')
from utils import *


def main(args):
    if args.from_remote:
        model_name = parse_model_name(args.base_model, args.from_remote)
    else:
        model_name = '../' + parse_model_name(args.base_model)
        

    # Create offload folder if it doesn't exist
    offload_folder = "./offload"
    os.makedirs(offload_folder, exist_ok=True)
    
    # Try loading with different memory configurations
    try:
        # First try with 8-bit quantization for memory efficiency
        print("Attempting to load model with 8-bit quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, 
            load_in_8bit=True,
            device_map="auto",
            offload_folder=offload_folder,
        )
    except Exception as e:
        print(f"8-bit loading failed: {e}")
        try:
            # Fallback to 4-bit quantization
            print("Attempting to load model with 4-bit quantization...")
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True, 
                quantization_config=bnb_config,
                device_map="auto",
                offload_folder=offload_folder,
            )
        except Exception as e2:
            print(f"4-bit loading failed: {e2}")
            # Final fallback to full precision with memory optimization
            print("Loading model in full precision with memory optimization...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True, 
                device_map="auto",
                offload_folder=offload_folder,
                low_cpu_mem_usage=True,
            )
    
    # Clear CUDA cache and set up memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU Memory after model load: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
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
            print(f"Starting evaluation for dataset: {data}")
            try:
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
                
                # Clear memory after each dataset evaluation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"GPU Memory after {data}: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            except Exception as e:
                print(f"Error during evaluation of {data}: {e}")
                # Continue with next dataset instead of failing completely
                continue
    
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

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import argparse
import re
import torch
import logging
# import transformers  # noqa: F401
from transformers import pipeline, set_seed
from transformers import AutoConfig, OPTForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
import deepspeed
import os
import pandas as pd
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",
                        type=str,
                        help="Directory containing trained actor model")
    parser.add_argument("--debug",
                        type=bool,
                        default=False,
                        help="whether debug mode is on")
    parser.add_argument("--local_rank",
                        type=int,
                        help="local rank")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate per response",
    )
    args = parser.parse_args()
    return args


def get_generator(path, local_rank):
    
    if 'llama' in path:
        tokenizer = LlamaTokenizer.from_pretrained(path, fast_tokenizer=True)
        tokenizer.pad_token = tokenizer.eos_token
        model_config = AutoConfig.from_pretrained(path)
        model = LlamaForCausalLM.from_pretrained(path,
                                                 from_tf=bool(".ckpt" in path),
                                                 config=model_config, 
                                                #  device_map='auto'
                                                #  , load_in_8bit=True
                                                 )
        model.config.end_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
        model.resize_token_embeddings(len(tokenizer))
        model.eval()
        generator = pipeline("text-generation",
                            model=model,
                            tokenizer=tokenizer,
                            device=local_rank)
    else:
        tokenizer = AutoTokenizer.from_pretrained(path, fast_tokenizer=True)
        tokenizer.pad_token = tokenizer.eos_token
        model_config = AutoConfig.from_pretrained(path)
        model = OPTForCausalLM.from_pretrained(path,
                                           from_tf=bool(".ckpt" in path),
                                           config=model_config, 
                                        #    device_map='auto'
                                        #    , load_in_8bit=True
                                           )

        model.config.end_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
        model.resize_token_embeddings(len(tokenizer))
        model.eval()
        generator = pipeline("text-generation",
                            model=model,
                            tokenizer=tokenizer,
                            device=local_rank)
    return generator


def process_response(response):
    output = str(response[0]["generated_text"])
    output = output.split("Assistant:")[1].strip()
    # output = output.replace("<|endoftext|></s>", "")
    output = output.replace("<|endoftext|>", "")
    output = output.replace("</s>", "")

    return output


def main(args):
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    # world_size = torch.distributed.get_world_size()
    # local_rank = int(os.getenv('LOCAL_RANK', args.local_rank))
    # args.local_rank=int(os.getenv("LOCAL_RANK", -1))
    # local_rank = args.local_rank
    generator = get_generator(args.path, args.local_rank)
    # generator.model = deepspeed.init_inference(generator.model,
    #                                       mp_size=world_size,
    #                                       dtype=torch.half,
    #                                       replace_with_kernel_inject=True)
    
    set_seed(42)

    # load huggingface finacial phrasebank dataset
    # https://huggingface.co/datasets/financial_phrasebank
    dataset_fpb = load_dataset("financial_phrasebank", "sentences_50agree")
    label_mapping_fpb = {"negative": 0, "neutral": 1, "positive": 2}
    text_inputs_fpb = dataset_fpb['train']['sentence']
    labels_fpb = dataset_fpb['train']['label']
    recons_fpb = {"name": "fpb", "sentence": text_inputs_fpb, "label": labels_fpb, "label_mapping": label_mapping_fpb}

    dataset_fiqa = load_dataset("pauri32/fiqa-2018")
    label_mapping_fiqa = {"negative": 2, "neutral": 1, "positive": 0}
    text_inputs_fiqa = dataset_fiqa['test']['sentence']
    labels_fiqa = dataset_fiqa['test']['label']
    recons_fiqa = {"name": "fiqa", "sentence": text_inputs_fiqa, "label": labels_fiqa, "label_mapping": label_mapping_fiqa}

    # dataset_fpb_num = pd.read_csv("data/FPB_filtered_number.csv")
    # label_mapping_fpb_num = {"negative": 0, "neutral": 1, "positive": 2}
    # text_inputs_fpb_num = dataset_fpb_num['sentence']
    # labels_fpb_num = dataset_fpb_num['sentiment'].apply(lambda x: label_mapping_fpb_num[x])
    # recons_fpb_num = {"name": "fpb_num", "sentence": text_inputs_fpb_num, "label": labels_fpb_num, "label_mapping": label_mapping_fpb_num}

    dataset_twitter = load_dataset("zeroshot/twitter-financial-news-sentiment")
    label_mapping_twitter = {"negative": 0, "neutral": 2, "positive": 1}
    text_inputs_twitter = dataset_twitter['validation']['text']
    labels_twitter = dataset_twitter['validation']['label']
    recons_twitter = {"name": "twitter", "sentence": text_inputs_twitter, "label": labels_twitter, "label_mapping": label_mapping_twitter}


    # dataset_list = [recons_fpb, recons_fiqa, recons_twitter]
    dataset_list = [recons_twitter]
    # dataset_list = [recons_fpb_num]

    for dataset in dataset_list:
        labels = []
        preds = []

        text_inputs = dataset['sentence']
        process_inputs = [f"Human: Determine the sentiment of the financial news as negative, neutral or positive: {text_inputs[i]} Assistant: " for i in range(len(text_inputs))]
        labels = dataset['label']
        label_mapping = dataset['label_mapping']

        if args.debug:
            process_inputs = process_inputs[:100]
            labels = labels[:100]

        # response = generator(process_inputs, max_new_tokens=args.max_new_tokens, do_sample=True)
        start = time.time()
        response = generator(process_inputs, max_new_tokens=args.max_new_tokens)
        end = time.time()
        # print(response)
        outputs = [process_response(response[i]) for i in range(len(response))]

        for i in range(len(outputs)):
            if "negative" in outputs[i]:
                preds.append(label_mapping["negative"])
            elif "neutral" in outputs[i]:
                preds.append(label_mapping["neutral"])
            elif "positive" in outputs[i]:
                preds.append(label_mapping["positive"])
            else:
                preds.append(label_mapping["neutral"])
        
        print(f"Dataset: {dataset['name']}")
        print(f"Process time: {end-start}")
        print(f"Accuracy: {accuracy_score(labels, preds)}")
        print(f"F1: {f1_score(labels, preds, average='macro')}")
        print(f"Confusion Matrix: {confusion_matrix(labels, preds)}")


if __name__ == "__main__":
    # Silence warnings about `max_new_tokens` and `max_length` being set
    logging.getLogger("transformers").setLevel(logging.ERROR)

    args = parse_args()
    print(args)
    main(args)

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import argparse
import re
import logging
import transformers  # noqa: F401
from transformers import pipeline, set_seed
from transformers import AutoConfig, OPTForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",
                        type=str,
                        help="Directory containing trained actor model")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate per response",
    )
    parser.add_argument("--debug",
                        action='store_true',
                        help="whether debug")
    args = parser.parse_args()
    return args


def get_generator(path):
    
    if 'llama' in path:
        tokenizer = LlamaTokenizer.from_pretrained(path, fast_tokenizer=True)
        tokenizer.pad_token = tokenizer.eos_token
        model_config = AutoConfig.from_pretrained(path)
        model = LlamaForCausalLM.from_pretrained(path,
                                                 from_tf=bool(".ckpt" in path),
                                                 config=model_config, device_map='auto', load_in_8bit=True)
        model.config.end_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
        model.resize_token_embeddings(len(tokenizer))
        model.eval()
        generator = pipeline("text-generation",
                            model=model,
                            tokenizer=tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(path, fast_tokenizer=True)
        tokenizer.pad_token = tokenizer.eos_token
        model_config = AutoConfig.from_pretrained(path)
        model = OPTForCausalLM.from_pretrained(path,
                                           from_tf=bool(".ckpt" in path),
                                           config=model_config, device_map='auto', load_in_8bit=True)

        model.config.end_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
        model.resize_token_embeddings(len(tokenizer))
        model.eval()
        generator = pipeline("text-generation",
                            model=model,
                            tokenizer=tokenizer)
    return generator


def get_user_input(user_input):
    tmp = input("Enter input (type 'quit' to exit, 'clear' to clean memory): ")
    new_inputs = f"Human: {tmp}\n Assistant: "
    user_input += f" {new_inputs}"
    return user_input, tmp == "quit", tmp == "clear"


def get_model_response(generator, user_input, max_new_tokens):
    response = generator(user_input, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
    return response


def process_response(response):
    output = str(response[0]["generated_text"])
    output = output.split("Assistant:")[1].strip()
    # output = output.replace("<|endoftext|></s>", "")
    output = output.replace("<|endoftext|>", "")
    output = output.replace("</s>", "")
 
    return output


def main(args):
    generator = get_generator(args.path)
    set_seed(42)

    # load huggingface finacial phrasebank dataset
    # https://huggingface.co/datasets/financial_phrasebank
    # dataset = load_dataset("financial_phrasebank", "sentences_50agree")
    # # eval_dataloader = DataLoader(dataset, batch_size=8)

    # text_inputs = dataset['train']['sentence']
    # labels = dataset['train']['label']

    # read the contextualized texts
    dataset2 = pd.read_csv("/xfs/home/tensor_zy/chatds/inference/sent_valid_classified_scraped_partial.csv")
    text_inputs = dataset2['contextualized_sentence']
    labels = dataset2['label']
    label_mapping_twitter = {"negative": 0, "neutral": 2, "positive": 1}

    process_inputs = [f"Human: Determine the sentiment of the financial news as negative, neutral or positive: {text_inputs[i]} Assistant: " for i in range(len(text_inputs))]
    
    preds = []
    args.debug = False

    if args.debug:
        process_inputs = process_inputs[:32]
        labels = labels[:32]

    response = generator(process_inputs, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=0.7)
    if args.debug:
        for r in response:
            print(r)

    outputs = [process_response(response[i]) for i in range(len(response))]

    for i in range(len(outputs)):
        if "negative" in outputs[i]:
            preds.append(0)
        elif "neutral" in outputs[i]:
            preds.append(1)
        elif "positive" in outputs[i]:
            preds.append(2)
        else:
            preds.append(1)
    
    print(f"Accuracy: {accuracy_score(labels, preds)}")
    print(f"F1: {f1_score(labels, preds, average='macro')}")
    print(f"confusion matrix: {confusion_matrix(labels, preds)}")



if __name__ == "__main__":
    # Silence warnings about `max_new_tokens` and `max_length` being set
    logging.getLogger("transformers").setLevel(logging.ERROR)

    args = parse_args()
    main(args)



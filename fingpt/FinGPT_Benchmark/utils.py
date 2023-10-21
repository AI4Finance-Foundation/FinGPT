import os
import datasets


template_dict = {
    'default': 'Instruction: {instruction}\nInput: {input}\nAnswer: '
}

lora_module_dict = {
    'chatglm2': ['query_key_value'],
    'falcon': ['query_key_value'],
    'bloom': ['query_key_value'],
    'internlm': ['q_proj', 'k_proj', 'v_proj'],
    'llama2': ['q_proj', 'k_proj', 'v_proj'],
    'qwen': ["c_attn"],
    'mpt': ['Wqkv'],
}


def get_prompt(template, instruction, input):
    
    if instruction:
        return template_dict[template].format(instruction=instruction, input=input)
    else:
        return input


def test_mapping(args, feature):
    
    prompt = get_prompt(
        args.instruct_template,
        feature['instruction'],
        feature['input']
    )    
    return {
        "prompt": prompt,
    }


def tokenize(args, tokenizer, feature):
    
    prompt = get_prompt(
        args.instruct_template,
        feature['instruction'],
        feature['input']
    )
    prompt_ids = tokenizer(
        prompt, padding=False,
        max_length=args.max_length, truncation=True
    )['input_ids']
    target_ids = tokenizer(
        feature['output'].strip(), padding=False,
        max_length=args.max_length, truncation=True,
        add_special_tokens=False
    )['input_ids']
    
    input_ids = prompt_ids + target_ids
    exceed_max_length = len(input_ids) >= args.max_length
    
    # Add EOS Token
    if input_ids[-1] != tokenizer.eos_token_id and not exceed_max_length:
        input_ids.append(tokenizer.eos_token_id)
    
    label_ids = [tokenizer.pad_token_id] * len(prompt_ids) + input_ids[len(prompt_ids):]
    
    return {
        "input_ids": input_ids,
        "labels": label_ids,
        "exceed_max_length": exceed_max_length
    }


def parse_model_name(name, from_remote=False):
    
    if name == 'chatglm2':
        return 'THUDM/chatglm2-6b' if from_remote else 'base_models/chatglm2-6b'
    elif name == 'llama2':
        return 'meta-llama/Llama-2-7b-hf' if from_remote else 'base_models/Llama-2-7b-hf'
        # return 'NousResearch/Llama-2-7b-hf' if from_remote else 'base_models/Llama-2-7b-hf-nous'
    elif name == 'falcon':
        return 'tiiuae/falcon-7b' if from_remote else 'base_models/falcon-7b'
    elif name == 'internlm':
        return 'internlm/internlm-7b' if from_remote else 'base_models/internlm-7b'
    elif name == 'qwen':
        return 'Qwen/Qwen-7B' if from_remote else 'base_models/Qwen-7B'
    elif name == 'mpt':
        return 'cekal/mpt-7b-peft-compatible' if from_remote else 'base_models/mpt-7b-peft-compatible'
        # return 'mosaicml/mpt-7b' if from_remote else 'base_models/mpt-7b'
    elif name == 'bloom':
        return 'bigscience/bloom-7b1' if from_remote else 'base_models/bloom-7b1'
    else:
        raise ValueError(f"Undefined base model {name}")
        
    
def load_dataset(names, from_remote=False):
    dataset_names = [d for d in names.split(',')]
    dataset_list = []
    for name in dataset_names:
        rep = 1
        if not os.path.exists(name):
            rep = int(name.split('*')[1]) if '*' in name else 1
            name = ('FinGPT/fingpt-' if from_remote else 'data/fingpt-') + name.split('*')[0]
        tmp_dataset = datasets.load_dataset(name) if from_remote else datasets.load_from_disk(name)
        if 'test' not in tmp_dataset:
            if 'train' in tmp_dataset:
                tmp_dataset = tmp_dataset['train']
            tmp_dataset = tmp_dataset.train_test_split(0.2, shuffle=True, seed=42)
            
        dataset_list.extend([tmp_dataset] * rep)
    return dataset_list
        
        

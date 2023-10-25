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
    """
    Parse the model name and return the appropriate path based on whether
    the model is to be fetched from a remote source or from a local source.

    Args:
    - name (str): Name of the model.
    - from_remote (bool): If True, return the remote path, else return the local path.

    Returns:
    - str: The appropriate path for the given model name.
    """
    model_paths = {
        'chatglm2': ('THUDM/chatglm2-6b', 'base_models/chatglm2-6b'),
        'llama2': ('meta-llama/Llama-2-7b-hf', 'base_models/Llama-2-7b-hf'),
        'llama2-13b': ('meta-llama/Llama-2-13b-hf', 'base_models/Llama-2-13b-hf'),
        'falcon': ('tiiuae/falcon-7b', 'base_models/falcon-7b'),
        'internlm': ('internlm/internlm-7b', 'base_models/internlm-7b'),
        'qwen': ('Qwen/Qwen-7B', 'base_models/Qwen-7B'),
        'mpt': ('cekal/mpt-7b-peft-compatible', 'base_models/mpt-7b-peft-compatible'),
        'bloom': ('bigscience/bloom-7b1', 'base_models/bloom-7b1')
    }

    if name in model_paths:
        return model_paths[name][0] if from_remote else model_paths[name][1]
    else:
        valid_model_names = ', '.join(model_paths.keys())
        raise ValueError(f"Undefined base model '{name}'. Valid model names are: {valid_model_names}")
    
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
        
        

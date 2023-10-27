import os
import datasets

# A dictionary to store various prompt templates.
template_dict = {
    'default': 'Instruction: {instruction}\nInput: {input}\nAnswer: '
}

# A dictionary to store the LoRA module mapping for different models.
lora_module_dict = {
    'chatglm2': ['query_key_value'],
    'falcon': ['query_key_value'],
    'bloom': ['query_key_value'],
    'internlm': ['q_proj', 'k_proj', 'v_proj'],
    'llama2': ['q_proj', 'k_proj', 'v_proj'],
    'qwen': ["c_attn"],
    'mpt': ['Wqkv'],
}


# Function to generate prompts based on the instruction, input, and chosen template.
def get_prompt(template, instruction, input):
    # If there's an instruction, format the prompt accordingly.
    # Otherwise, just return the input as is.
    if instruction:
        return template_dict[template].format(instruction=instruction, input=input)
    else:
        return input

# Function to map the dataset features to prompt for testing.
def test_mapping(args, feature):
    # Generate the prompt based on the instruction and input from the feature.
    prompt = get_prompt(
        args.instruct_template,
        feature['instruction'],
        feature['input']
    )    
    return {
        "prompt": prompt,
    }

# Function to tokenize the prompts and targets for training/testing.
def tokenize(args, tokenizer, feature):
    """
    Tokenizes the input prompt and target/output for model training or evaluation.

    Args:
    args (Namespace): A namespace object containing various settings and configurations.
    tokenizer (Tokenizer): A tokenizer object used to convert text into tokens.
    feature (dict): A dictionary containing 'input', 'instruction', and 'output' fields.

    Returns:
    dict: A dictionary containing tokenized 'input_ids', 'labels', and a flag 'exceed_max_length'.
    """
    # Generate the prompt.
    prompt = get_prompt(
        args.instruct_template,
        feature['instruction'],
        feature['input']
    )
    # Tokenize the prompt.
    prompt_ids = tokenizer(
        prompt,
        padding=False,
        max_length=args.max_length,
        truncation=True
    )['input_ids']

    # Tokenize the target/output.
    target_ids = tokenizer(
        feature['output'].strip(),
        padding=False,
        max_length=args.max_length,
        truncation=True,
        add_special_tokens=False
    )['input_ids']

    # Combine tokenized prompt and target output.
    input_ids = prompt_ids + target_ids

    # Check if the combined length exceeds the maximum allowed length.
    exceed_max_length = len(input_ids) >= args.max_length

    # Add an end-of-sequence (EOS) token if it's not already present
    # and if the sequence length is within the limit.
    if input_ids[-1] != tokenizer.eos_token_id and not exceed_max_length:
        input_ids.append(tokenizer.eos_token_id)

    # Create label IDs for training.
    # The labels should start from where the prompt ends, and be padded for the prompt portion.
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
    """
    Load one or multiple datasets based on the provided names and source location.

    Args:
    names (str): A comma-separated list of dataset names. Each name can be followed by '*n' to indicate replication.
    from_remote (bool): If True, load the dataset from Hugging Face's model hub. Otherwise, load it from a local disk.

    Returns:
    List[Dataset]: A list of loaded datasets. Each dataset is possibly replicated based on the input names.
    """
    # Split the dataset names by commas for handling multiple datasets
    dataset_names = names.split(',')
    dataset_list = []

    for name in dataset_names:
        # Initialize replication factor to 1
        replication_factor = 1
        dataset_name = name

        # Check if the dataset name includes a replication factor
        if '*' in name:
            dataset_name, replication_factor = name.split('*')
            replication_factor = int(replication_factor)
            if replication_factor < 1:
                raise ValueError("Replication factor must be a positive integer.")

        # Construct the correct dataset path or name based on the source location
        dataset_path_or_name = ('FinGPT/fingpt-' if from_remote else 'data/fingpt-') + dataset_name
        if not os.path.exists(dataset_path_or_name) and not from_remote:
            raise FileNotFoundError(f"The dataset path {dataset_path_or_name} does not exist.")

        # Load the dataset
        try:
            tmp_dataset = datasets.load_dataset(dataset_path_or_name) if from_remote else datasets.load_from_disk(
                dataset_path_or_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load the dataset: {str(e)}")

        # Check for 'test' split and create it from 'train' if necessary
        if 'test' not in tmp_dataset:
            if 'train' in tmp_dataset:
                tmp_dataset = tmp_dataset['train']
                tmp_dataset = tmp_dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
            else:
                raise ValueError("The dataset must contain a 'train' or 'test' split.")

        # Append the possibly replicated dataset to the list
        dataset_list.extend([tmp_dataset] * replication_factor)

    return dataset_list

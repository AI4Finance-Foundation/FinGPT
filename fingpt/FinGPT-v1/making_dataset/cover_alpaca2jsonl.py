import argparse
import json
import logging
from tqdm import tqdm

def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}

def load_data(data_path: str) -> list:
    try:
        with open(data_path, encoding='utf-8') as f:
            examples = json.load(f)
            return examples
    except FileNotFoundError:
        logging.error(f"Data file not found at {data_path}")
        return []
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON from {data_path}")
        return []

def save_data(save_path: str, examples: list) -> None:
    with open(save_path, 'w') as f:
        for example in tqdm(examples, desc="Formatting.."):
            formatted_example = format_example(example)
            f.write(json.dumps(formatted_example) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/alpaca_data.json")
    parser.add_argument("--save_path", type=str, default="data/alpaca_data.jsonl")

    args = parser.parse_args()

    examples = load_data(args.data_path)
    if examples:
        save_data(args.save_path, examples)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

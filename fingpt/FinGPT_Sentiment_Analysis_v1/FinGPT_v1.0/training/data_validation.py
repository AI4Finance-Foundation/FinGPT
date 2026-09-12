"""Validation helpers for processed FinGPT-v1 training datasets."""


REQUIRED_COLUMNS = {"input_ids", "seq_len"}


def validate_training_dataset(dataset):
    """Validate a processed dataset before model and GPU initialization."""
    dataset_length = len(dataset)
    if dataset_length < 2:
        raise ValueError(
            "dataset must contain at least 2 examples to create a train/validation split"
        )

    missing_columns = REQUIRED_COLUMNS - set(dataset.column_names)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"processed dataset is missing required column(s): {missing}; "
            "run making_dataset/tokenize_dataset_rows.py first"
        )

    for index, example in enumerate(dataset.select(range(min(dataset_length, 100)))):
        if not example["input_ids"]:
            raise ValueError(f"dataset example {index} has empty input_ids")
        if example["seq_len"] < 1 or example["seq_len"] > len(example["input_ids"]):
            raise ValueError(
                f"dataset example {index} has invalid seq_len={example['seq_len']} "
                f"for {len(example['input_ids'])} input_ids"
            )

    return dataset_length
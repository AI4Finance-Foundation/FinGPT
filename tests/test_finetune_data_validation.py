import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "fingpt"
    / "FinGPT_Sentiment_Analysis_v1"
    / "FinGPT_v1.0"
    / "training"
    / "data_validation.py"
)
SPEC = importlib.util.spec_from_file_location("data_validation", MODULE_PATH)
data_validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(data_validation)


class FakeDataset:
    column_names = ["input_ids", "seq_len"]

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def select(self, indexes):
        return [self.rows[index] for index in indexes]


def test_valid_dataset_passes():
    dataset = FakeDataset([
        {"input_ids": [1, 2, 3], "seq_len": 2},
        {"input_ids": [4, 5], "seq_len": 1},
    ])

    assert data_validation.validate_training_dataset(dataset) == 2


@pytest.mark.parametrize(
    "dataset, message",
    [
        (FakeDataset([]), "at least 2 examples"),
        (FakeDataset([{"input_ids": [1]}]), "at least 2 examples"),
    ],
)
def test_dataset_must_be_large_enough_for_split(dataset, message):
    with pytest.raises(ValueError, match=message):
        data_validation.validate_training_dataset(dataset)


def test_missing_columns_are_reported():
    dataset = FakeDataset([
        {"input_ids": [1]},
        {"input_ids": [2]},
    ])
    dataset.column_names = ["input_ids"]

    with pytest.raises(ValueError, match="seq_len"):
        data_validation.validate_training_dataset(dataset)


def test_invalid_sequence_length_is_reported():
    dataset = FakeDataset([
        {"input_ids": [1, 2], "seq_len": 3},
        {"input_ids": [3], "seq_len": 1},
    ])

    with pytest.raises(ValueError, match="invalid seq_len"):
        data_validation.validate_training_dataset(dataset)
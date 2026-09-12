import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "fingpt"
    / "FinGPT_Others"
    / "FinGPT_Trading"
    / "transaction_data.py"
)
SPEC = importlib.util.spec_from_file_location("transaction_data", MODULE_PATH)
transaction_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transaction_data)


TRANSACTIONS = [
    {
        "timestamp": "2026-09-11T10:00:00Z",
        "symbol": "reliance.ns",
        "price": 2500,
        "quantity": 10,
        "side": "BUY",
    },
    {
        "timestamp": "2026-09-11T10:01:00Z",
        "symbol": "RELIANCE.NS",
        "price": 2498,
        "quantity": 4,
        "side": "sell",
    },
]


def test_transaction_is_normalized():
    normalized = transaction_data.validate_transaction(TRANSACTIONS[0])

    assert normalized["symbol"] == "RELIANCE.NS"
    assert normalized["side"] == "buy"
    assert normalized["timestamp"].tzinfo is not None


def test_transactions_are_aggregated_into_features():
    features = transaction_data.aggregate_transactions(TRANSACTIONS)

    assert features["event_count"] == 2
    assert features["total_quantity"] == 14
    assert features["total_notional"] == pytest.approx(34992)
    assert features["vwap"] == pytest.approx(2499.4285714285716)
    assert features["buy_sell_imbalance"] == pytest.approx(6 / 14)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("price", 0, "positive number"),
        ("quantity", -1, "positive number"),
        ("side", "hold", "buy or sell"),
        ("timestamp", "tomorrow", "ISO-8601"),
    ],
)
def test_invalid_transaction_fields_are_rejected(field, value, message):
    transaction = dict(TRANSACTIONS[0], **{field: value})

    with pytest.raises(ValueError, match=message):
        transaction_data.validate_transaction(transaction)


def test_mixed_symbols_cannot_be_aggregated():
    transaction = dict(TRANSACTIONS[1], symbol="TCS.NS")

    with pytest.raises(ValueError, match="one symbol"):
        transaction_data.aggregate_transactions([TRANSACTIONS[0], transaction])
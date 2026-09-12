"""Validation and aggregation helpers for transaction-level market data."""

from datetime import datetime


REQUIRED_FIELDS = {"timestamp", "symbol", "price", "quantity", "side"}
VALID_SIDES = {"buy", "sell"}


def validate_transaction(transaction):
    """Validate one transaction event and return a normalized copy."""
    if not isinstance(transaction, dict):
        raise ValueError("transaction must be an object")

    missing_fields = REQUIRED_FIELDS - transaction.keys()
    if missing_fields:
        raise ValueError("missing transaction field(s): " + ", ".join(sorted(missing_fields)))

    timestamp = transaction["timestamp"]
    if not isinstance(timestamp, str):
        raise ValueError("timestamp must be an ISO-8601 string")
    try:
        parsed_timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("timestamp must be an ISO-8601 string") from error

    symbol = transaction["symbol"]
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")

    price = transaction["price"]
    quantity = transaction["quantity"]
    if isinstance(price, bool) or not isinstance(price, (int, float)) or price <= 0:
        raise ValueError("price must be a positive number")
    if isinstance(quantity, bool) or not isinstance(quantity, (int, float)) or quantity <= 0:
        raise ValueError("quantity must be a positive number")

    side = transaction["side"].lower() if isinstance(transaction["side"], str) else ""
    if side not in VALID_SIDES:
        raise ValueError("side must be buy or sell")

    normalized = dict(transaction)
    normalized.update({
        "timestamp": parsed_timestamp,
        "symbol": symbol.strip().upper(),
        "price": float(price),
        "quantity": float(quantity),
        "side": side,
    })
    return normalized


def aggregate_transactions(transactions):
    """Aggregate validated transactions into model-ready market features."""
    normalized_transactions = [validate_transaction(transaction) for transaction in transactions]
    if not normalized_transactions:
        raise ValueError("transactions must contain at least one event")

    symbols = {transaction["symbol"] for transaction in normalized_transactions}
    if len(symbols) != 1:
        raise ValueError("transactions must contain one symbol per aggregation window")

    buy_quantity = sum(
        transaction["quantity"]
        for transaction in normalized_transactions
        if transaction["side"] == "buy"
    )
    sell_quantity = sum(
        transaction["quantity"]
        for transaction in normalized_transactions
        if transaction["side"] == "sell"
    )
    total_quantity = buy_quantity + sell_quantity
    total_notional = sum(
        transaction["price"] * transaction["quantity"]
        for transaction in normalized_transactions
    )

    return {
        "symbol": normalized_transactions[0]["symbol"],
        "event_count": len(normalized_transactions),
        "total_quantity": total_quantity,
        "total_notional": total_notional,
        "vwap": total_notional / total_quantity,
        "buy_sell_imbalance": (buy_quantity - sell_quantity) / total_quantity,
    }
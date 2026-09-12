# Transaction Data Features

`transaction_data.py` provides a small, provider-neutral starting point for
transaction-level market data. Each event must contain:

```json
{"timestamp":"2026-09-11T10:00:00Z","symbol":"RELIANCE.NS","price":2500.0,"quantity":10,"side":"buy"}
```

The helper validates the event and aggregates one symbol per time window into:

- event count
- total quantity and notional value
- volume-weighted average price (VWAP)
- buy/sell quantity imbalance

Buy/sell imbalance is a market-activity feature, not a ground-truth sentiment
label. It should be aligned to the observation timestamp and evaluated against
future returns using chronological splits. Do not use future trades, revised
records, or post-event labels when constructing the input window.

Example:

```python
from transaction_data import aggregate_transactions

features = aggregate_transactions([
    {"timestamp": "2026-09-11T10:00:00Z", "symbol": "RELIANCE.NS", "price": 2500, "quantity": 10, "side": "buy"},
    {"timestamp": "2026-09-11T10:01:00Z", "symbol": "RELIANCE.NS", "price": 2498, "quantity": 4, "side": "sell"},
])
```

This is a data-preparation primitive, not a trading strategy or profitability
claim. A training pipeline still needs a documented target, transaction costs,
slippage, and out-of-sample evaluation.
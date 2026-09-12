"""Market-specific ticker symbols used by the Forecaster data providers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MarketSymbol:
    display: str
    price: str
    provider: str


def resolve_symbol(symbol, market="US"):
    """Resolve one user ticker into price and news-provider symbols."""
    value = symbol.strip().upper()
    if not value:
        raise ValueError("ticker symbol cannot be empty")

    market = market.upper()
    if market == "US":
        return MarketSymbol(display=value, price=value, provider=value)

    if market not in {"INDIA_NSE", "INDIA_BSE"}:
        raise ValueError("market must be US, INDIA_NSE, or INDIA_BSE")

    exchange, suffix = ("NSE", ".NS") if market == "INDIA_NSE" else ("BSE", ".BO")
    if value.startswith(("NSE:", "BSE:")):
        value = value.split(":", 1)[1]
    if value.endswith((".NS", ".BO")):
        value = value.rsplit(".", 1)[0]

    return MarketSymbol(
        display=f"{exchange}:{value}",
        price=f"{value}{suffix}",
        provider=f"{exchange}:{value}",
    )
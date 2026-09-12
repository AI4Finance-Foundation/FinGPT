import sys
from pathlib import Path


FORECASTER_DIR = Path(__file__).resolve().parents[1] / "fingpt" / "FinGPT_Forecaster"
sys.path.insert(0, str(FORECASTER_DIR))

from market_symbols import resolve_symbol


def test_us_symbol_is_unchanged():
    assert resolve_symbol("aapl") == resolve_symbol("AAPL", "US")


def test_india_nse_symbol_is_resolved_for_each_provider():
    symbol = resolve_symbol("RELIANCE", "INDIA_NSE")

    assert symbol.display == "NSE:RELIANCE"
    assert symbol.price == "RELIANCE.NS"
    assert symbol.provider == "NSE:RELIANCE"


def test_india_bse_accepts_provider_and_yfinance_formats():
    assert resolve_symbol("BSE:TCS", "INDIA_BSE").price == "TCS.BO"
    assert resolve_symbol("TCS.BO", "INDIA_BSE").provider == "BSE:TCS"


def test_invalid_market_and_empty_symbol_are_rejected():
    for symbol, market in (("", "US"), ("AAPL", "EUROPE")):
        try:
            resolve_symbol(symbol, market)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid symbol input should raise ValueError")
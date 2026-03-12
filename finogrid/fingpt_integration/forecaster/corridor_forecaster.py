"""
FinGPT Forecaster Adapter for Finogrid.

Adapts FinGPT's stock forecasting pipeline for corridor risk signals:
- Stablecoin depeg risk monitoring (USDT, USDC)
- Corridor macro risk scoring (news + fundamentals → risk signal)
- Partner SLA drift alerts

Adapted from FinGPT/fingpt/FinGPT_Forecaster/data.py and prompt.py.
Runs in the Process Improvement and Treasury Strategy agents — NOT hot path.
"""
from __future__ import annotations

import structlog
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional

log = structlog.get_logger()

# Stablecoin tickers available on yfinance
STABLECOIN_TICKERS = {
    "USDT": "USDT-USD",
    "USDC": "USDC-USD",
}

# Macro proxies for each corridor (local currency vs USD)
CORRIDOR_FX_TICKERS = {
    "BR": "BRL=X",
    "AR": "ARS=X",
    "VN": "VND=X",
    "IN": "INR=X",
    "AE": "AED=X",
    "ID": "IDR=X",
    "PH": "PHP=X",
    "NG": "NGN=X",
}


class CorridorForecaster:
    """
    Uses FinGPT's forecasting approach to generate corridor risk signals.

    For each corridor: fetches FX data, runs FinGPT to generate a
    structured risk assessment (positive developments, concerns, signal).
    """

    def __init__(self, openai_client=None, fingpt_model=None):
        self.openai = openai_client      # For GPT-4 labeling (FinGPT approach)
        self.fingpt = fingpt_model       # FinGPT forecaster (optional)

    def get_fx_returns(self, corridor_code: str, weeks: int = 4) -> dict:
        """Fetch recent FX returns for a corridor's local currency."""
        ticker = CORRIDOR_FX_TICKERS.get(corridor_code)
        if not ticker:
            return {}

        end = datetime.now()
        start = end - timedelta(weeks=weeks)
        try:
            df = yf.download(ticker, start=start, end=end, interval="1wk", progress=False)
            if df.empty:
                return {"corridor": corridor_code, "error": "No data"}
            closes = df["Close"].dropna()
            weekly_return = float((closes.iloc[-1] / closes.iloc[0] - 1) * 100)
            return {
                "corridor": corridor_code,
                "ticker": ticker,
                "weeks": weeks,
                "weekly_return_pct": round(weekly_return, 4),
                "latest_rate": float(closes.iloc[-1]),
                "direction": "strengthening" if weekly_return > 0 else "weakening",
            }
        except Exception as e:
            log.error("fx_fetch_failed", corridor=corridor_code, error=str(e))
            return {"corridor": corridor_code, "error": str(e)}

    def get_stablecoin_depeg_risk(self, asset: str = "USDT") -> dict:
        """Monitor stablecoin peg stability."""
        ticker = STABLECOIN_TICKERS.get(asset.upper())
        if not ticker:
            return {}

        try:
            df = yf.download(ticker, period="7d", interval="1h", progress=False)
            if df.empty:
                return {"asset": asset, "error": "No data"}
            prices = df["Close"].dropna()
            current = float(prices.iloc[-1])
            low_7d = float(prices.min())
            high_7d = float(prices.max())
            depeg_bps = abs(current - 1.0) * 10_000

            return {
                "asset": asset,
                "current_price": round(current, 6),
                "low_7d": round(low_7d, 6),
                "high_7d": round(high_7d, 6),
                "depeg_bps": round(depeg_bps, 1),
                "risk_level": "high" if depeg_bps > 50 else "medium" if depeg_bps > 10 else "low",
            }
        except Exception as e:
            log.error("depeg_check_failed", asset=asset, error=str(e))
            return {"asset": asset, "error": str(e)}

    def generate_corridor_risk_signal(
        self,
        corridor_code: str,
        news_sentiment_scores: list[dict],
        fx_data: Optional[dict] = None,
    ) -> dict:
        """
        Combine news sentiment + FX data → corridor risk signal.
        Adapted from FinGPT Forecaster's positive/negative/prediction structure.
        """
        fx_data = fx_data or self.get_fx_returns(corridor_code)
        avg_sentiment = (
            sum(n.get("sentiment_score", 0) for n in news_sentiment_scores) / len(news_sentiment_scores)
            if news_sentiment_scores else 0
        )

        # Heuristic risk signal (in production: run FinGPT forecaster model)
        fx_direction = fx_data.get("direction", "unknown")
        fx_return = fx_data.get("weekly_return_pct", 0.0)

        if avg_sentiment > 0.3 and fx_direction == "strengthening":
            risk_level = "low"
            signal = "favorable"
        elif avg_sentiment < -0.3 or (fx_direction == "weakening" and abs(fx_return) > 3):
            risk_level = "high"
            signal = "caution"
        else:
            risk_level = "medium"
            signal = "neutral"

        return {
            "corridor": corridor_code,
            "risk_level": risk_level,
            "signal": signal,
            "avg_news_sentiment": round(avg_sentiment, 3),
            "fx_return_pct": fx_data.get("weekly_return_pct"),
            "fx_direction": fx_direction,
            "basis": "fingpt_sentiment + fx_data",
        }

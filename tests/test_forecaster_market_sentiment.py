import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd


FORECASTER_DIR = Path(__file__).resolve().parents[1] / "fingpt" / "FinGPT_Forecaster"
if str(FORECASTER_DIR) not in sys.path:
    sys.path.insert(0, str(FORECASTER_DIR))

from market_sentiment import (
    dataset_csv_path,
    enrich_recent_market_sentiment,
    format_market_sentiment_prompt,
)


def _build_payload(entries):
    return {"ticker": "AAPL", "found": True, "daily_trend": entries}


def test_enrich_recent_market_sentiment_aggregates_recent_windows():
    data = pd.DataFrame(
        {
            "Start Date": [pd.Timestamp("2026-04-01"), pd.Timestamp("2026-04-08")],
            "End Date": [pd.Timestamp("2026-04-07"), pd.Timestamp("2026-04-10")],
        }
    )

    source_payloads = {
        "reddit": _build_payload(
            [
                {"date": "2026-04-03", "mentions": 12, "sentiment_score": 0.40},
                {"date": "2026-04-05", "mentions": 18, "sentiment_score": 0.20},
                {"date": "2026-04-09", "mentions": 8, "sentiment_score": 0.10},
            ]
        ),
        "x": _build_payload(
            [
                {"date": "2026-04-02", "mentions": 10, "sentiment_score": 0.10},
                {"date": "2026-04-06", "mentions": 5, "sentiment_score": 0.00},
                {"date": "2026-04-10", "mentions": 6, "sentiment_score": -0.10},
            ]
        ),
        "news": _build_payload(
            [
                {"date": "2026-04-04", "mentions": 7, "sentiment_score": 0.05},
                {"date": "2026-04-08", "mentions": 9, "sentiment_score": 0.15},
            ]
        ),
        "polymarket": _build_payload([]),
    }

    def fetcher(url, headers):
        assert headers["X-API-Key"] == "demo-key"
        for source, payload in source_payloads.items():
            if "/{}/".format(source) in url:
                return payload
        raise AssertionError("unexpected url {}".format(url))

    enriched = enrich_recent_market_sentiment(
        data,
        "AAPL",
        api_key="demo-key",
        fetcher=fetcher,
        today=date(2026, 4, 10),
    )

    first_window = json.loads(enriched.loc[0, "MarketSentiment"])
    second_window = json.loads(enriched.loc[1, "MarketSentiment"])

    assert first_window["available"] is True
    assert first_window["coverage"] == 3
    assert first_window["total_mentions"] == 52
    assert first_window["average_sentiment_label"] == "mixed"
    assert first_window["source_alignment"] == "mixed"
    assert first_window["sources"]["reddit"]["total_mentions"] == 30

    assert second_window["available"] is True
    assert second_window["coverage"] == 3
    assert second_window["total_mentions"] == 23
    assert second_window["sources"]["x"]["average_sentiment_score"] == -0.1


def test_enrich_recent_market_sentiment_is_noop_without_key():
    data = pd.DataFrame(
        {
            "Start Date": [pd.Timestamp("2026-04-01")],
            "End Date": [pd.Timestamp("2026-04-07")],
        }
    )

    enriched = enrich_recent_market_sentiment(data, "AAPL", api_key="", today=date(2026, 4, 10))

    assert json.loads(enriched.loc[0, "MarketSentiment"]) == {}


def test_enrich_recent_market_sentiment_skips_windows_outside_supported_history():
    data = pd.DataFrame(
        {
            "Start Date": [pd.Timestamp("2025-10-01")],
            "End Date": [pd.Timestamp("2025-10-07")],
        }
    )

    enriched = enrich_recent_market_sentiment(data, "AAPL", api_key="demo-key", today=date(2026, 4, 10))

    assert json.loads(enriched.loc[0, "MarketSentiment"]) == {}


def test_format_market_sentiment_prompt_renders_structured_block():
    payload = {
        "available": True,
        "average_sentiment_score": 0.183,
        "average_sentiment_label": "mixed",
        "coverage": 3,
        "source_alignment": "aligned",
        "total_mentions": 41,
        "sources": {
            "reddit": {"average_sentiment_score": 0.25, "total_mentions": 20},
            "x": {"average_sentiment_score": 0.15, "total_mentions": 11},
            "news": {"average_sentiment_score": 0.15, "total_mentions": 10},
        },
    }

    block = format_market_sentiment_prompt(payload)

    assert "[Market Sentiment Signals]:" in block
    assert "Average sentiment score: 0.183 (mixed)" in block
    assert "Source coverage: 3/4" in block
    assert "- Reddit: sentiment 0.250, mentions 20" in block


def test_format_market_sentiment_prompt_is_empty_without_available_data():
    assert format_market_sentiment_prompt({}) == ""
    assert format_market_sentiment_prompt('{"available": false}') == ""


def test_dataset_csv_path_tracks_optional_sentiment_suffix():
    with_basics = dataset_csv_path("AAPL", "/tmp/data", "2026-01-01", "2026-02-01")
    no_basics = dataset_csv_path(
        "AAPL",
        "/tmp/data",
        "2026-01-01",
        "2026-02-01",
        with_basics=False,
    )
    with_sentiment = dataset_csv_path(
        "AAPL",
        "/tmp/data",
        "2026-01-01",
        "2026-02-01",
        with_market_sentiment=True,
    )

    assert with_basics.endswith("/AAPL_2026-01-01_2026-02-01.csv")
    assert no_basics.endswith("/AAPL_2026-01-01_2026-02-01_nobasics.csv")
    assert with_sentiment.endswith("/AAPL_2026-01-01_2026-02-01_sentiment.csv")

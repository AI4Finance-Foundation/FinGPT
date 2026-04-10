import json
import os
from datetime import date, datetime, timedelta
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

SUPPORTED_SOURCES = ("reddit", "x", "news", "polymarket")
DEFAULT_BASE_URL = "https://api.adanos.org"
MAX_LOOKBACK_DAYS = 90


def dataset_csv_path(
    symbol,
    data_dir,
    start_date,
    end_date,
    with_basics=True,
    with_market_sentiment=False,
):
    suffix = ""
    if not with_basics:
        suffix += "_nobasics"
    if with_market_sentiment:
        suffix += "_sentiment"
    return f"{data_dir}/{symbol}_{start_date}_{end_date}{suffix}.csv"


def get_api_key(api_key=None):
    return (api_key or os.environ.get("ADANOS_API_KEY", "")).strip()


def enabled(api_key=None):
    return bool(get_api_key(api_key))


def _parse_date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime().date()
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def _default_fetcher(url, headers, timeout=10):
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as response:  # nosec: B310 - optional public API client
        return json.loads(response.read().decode("utf-8"))


def _normalize_payload(payload):
    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        return payload["data"]
    return payload if isinstance(payload, dict) else {}


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _extract_daily_trend(payload):
    normalized = _normalize_payload(payload)
    if not normalized.get("found", True):
        return []

    daily_trend = normalized.get("daily_trend") or []
    if not isinstance(daily_trend, list):
        return []

    results = []
    for item in daily_trend:
        if not isinstance(item, dict) or "date" not in item:
            continue

        sentiment_score = _safe_float(
            item.get("sentiment_score", item.get("sentiment"))
        )
        if sentiment_score is None:
            continue

        results.append(
            {
                "date": str(item["date"])[:10],
                "mentions": _safe_int(item.get("mentions")),
                "sentiment_score": sentiment_score,
            }
        )

    return results


def _fetch_source_daily_trend(
    symbol,
    source,
    days,
    api_key,
    base_url=DEFAULT_BASE_URL,
    fetcher=None,
):
    if fetcher is None:
        fetcher = _default_fetcher

    params = urlencode({"days": days})
    url = "{}/{}/stocks/v1/stock/{}?{}".format(
        base_url.rstrip("/"),
        source,
        symbol.upper(),
        params,
    )
    headers = {"X-API-Key": api_key, "Accept": "application/json"}

    try:
        payload = fetcher(url, headers)
    except (HTTPError, URLError, TimeoutError, ValueError):
        return []

    return _extract_daily_trend(payload)


def _aggregate_period_signal(source, daily_trend, start_date, end_date):
    window = [
        item
        for item in daily_trend
        if start_date <= _parse_date(item["date"]) <= end_date
    ]
    total_mentions = sum(item["mentions"] for item in window)

    if not window or total_mentions <= 0:
        return {
            "source": source,
            "available": False,
            "average_sentiment_score": None,
            "total_mentions": 0,
        }

    average_sentiment = sum(item["sentiment_score"] for item in window) / len(window)
    return {
        "source": source,
        "available": True,
        "average_sentiment_score": round(average_sentiment, 3),
        "total_mentions": total_mentions,
    }


def _sentiment_label(value):
    if value is None:
        return "unavailable"
    if value >= 0.2:
        return "bullish"
    if value <= -0.2:
        return "bearish"
    return "mixed"


def _source_alignment(scores):
    if not scores:
        return "unavailable"
    if len(scores) == 1:
        return "single-source"

    spread = max(scores) - min(scores)
    if spread <= 0.15:
        return "aligned"
    if spread <= 0.4:
        return "mixed"
    return "divergent"


def summarize_market_sentiment(signals):
    available = [signal for signal in signals if signal["available"]]
    if not available:
        return {}

    scores = [signal["average_sentiment_score"] for signal in available]
    total_mentions = sum(signal["total_mentions"] for signal in available)
    average_sentiment = sum(scores) / len(scores)

    return {
        "available": True,
        "average_sentiment_score": round(average_sentiment, 3),
        "average_sentiment_label": _sentiment_label(average_sentiment),
        "total_mentions": total_mentions,
        "coverage": len(available),
        "source_alignment": _source_alignment(scores),
        "sources": {signal["source"]: signal for signal in available},
    }


def enrich_recent_market_sentiment(
    data,
    symbol,
    api_key=None,
    base_url=DEFAULT_BASE_URL,
    fetcher=None,
    today=None,
    sources=SUPPORTED_SOURCES,
):
    enriched = data.copy()
    enriched["MarketSentiment"] = [json.dumps({})] * len(enriched)

    api_key = get_api_key(api_key)
    if not api_key or enriched.empty:
        return enriched

    today = _parse_date(today or date.today())
    cutoff = today - timedelta(days=MAX_LOOKBACK_DAYS - 1)

    candidate_starts = []
    for _, row in enriched.iterrows():
        start_date = _parse_date(row["Start Date"])
        end_date = _parse_date(row["End Date"])
        if end_date < cutoff or start_date > today:
            continue
        candidate_starts.append(max(start_date, cutoff))

    if not candidate_starts:
        return enriched

    earliest_date = min(candidate_starts)
    days = min(MAX_LOOKBACK_DAYS, max(1, (today - earliest_date).days + 1))

    source_trends = {}
    for source in sources:
        source_trends[source] = _fetch_source_daily_trend(
            symbol=symbol,
            source=source,
            days=days,
            api_key=api_key,
            base_url=base_url,
            fetcher=fetcher,
        )

    market_sentiment = []
    for _, row in enriched.iterrows():
        start_date = _parse_date(row["Start Date"])
        end_date = _parse_date(row["End Date"])

        if end_date < cutoff or start_date > today:
            market_sentiment.append(json.dumps({}))
            continue

        period_start = max(start_date, cutoff)
        period_end = min(end_date, today)
        signals = [
            _aggregate_period_signal(source, source_trends.get(source, []), period_start, period_end)
            for source in sources
        ]
        market_sentiment.append(json.dumps(summarize_market_sentiment(signals)))

    enriched["MarketSentiment"] = market_sentiment
    return enriched


def format_market_sentiment_prompt(value):
    if not value:
        return ""

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return ""

    if not isinstance(value, dict) or not value.get("available"):
        return ""

    average_sentiment = value.get("average_sentiment_score")
    header = "[Market Sentiment Signals]:"
    lines = [
        header,
        "",
        "Average sentiment score: {:.3f} ({})".format(
            average_sentiment,
            value.get("average_sentiment_label", _sentiment_label(average_sentiment)),
        ),
        "Source coverage: {}/{}".format(value.get("coverage", 0), len(SUPPORTED_SOURCES)),
        "Source alignment: {}".format(value.get("source_alignment", "unavailable")),
        "Total sentiment activity: {}".format(value.get("total_mentions", 0)),
    ]

    for source in SUPPORTED_SOURCES:
        source_signal = value.get("sources", {}).get(source)
        if not source_signal:
            continue
        lines.append(
            "- {}: sentiment {:.3f}, mentions {}".format(
                source.capitalize(),
                source_signal["average_sentiment_score"],
                source_signal["total_mentions"],
            )
        )

    return "\n".join(lines)

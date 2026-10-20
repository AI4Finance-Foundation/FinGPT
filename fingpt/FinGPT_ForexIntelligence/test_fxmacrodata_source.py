"""
Tests for the FXMacroData data source (HTTP is mocked)
"""

import sys
from datetime import datetime
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fingpt.FinGPT_ForexIntelligence import data_sources
from fingpt.FinGPT_ForexIntelligence.config import DATA_SOURCES
from fingpt.FinGPT_ForexIntelligence.data_sources import (
    DataSourceException,
    EconomicEvent,
    ForexDataSourceManager,
    FXMacroDataSource,
)

TEST_KEY = "test-key-123"

# 2026-10-02 12:30 UTC (released) and 2026-10-14 12:30 UTC (upcoming)
RELEASED_AT = 1790944200
UPCOMING_AT = 1791981000

CALENDAR_USD = {
    "currency": "USD",
    "data": [
        {
            "announcement_datetime": RELEASED_AT,
            "release": "unemployment",
            "calendar_event_id": f"usd_unemployment_{RELEASED_AT}",
            "name": "Unemployment Rate",
            "event_importance": "high",
            "market_tier": 1,
        },
        {
            "announcement_datetime": UPCOMING_AT,
            "release": "inflation",
            "calendar_event_id": f"usd_inflation_{UPCOMING_AT}",
            "name": "Inflation (CPI)",
            "event_importance": "medium",
            "market_tier": 2,
        },
        {
            "announcement_datetime": UPCOMING_AT,
            "release": "initial_jobless_claims",
            "calendar_event_id": f"usd_initial_jobless_claims_{UPCOMING_AT}",
            "name": "Initial Jobless Claims",
            "event_importance": "low",
            "market_tier": 3,
        },
    ],
}

ANNOUNCEMENTS = {
    "unemployment": {
        "data": [
            {"date": "2026-09-30", "val": 4.2, "previous_value": 4.1, "announcement_datetime": RELEASED_AT},
            {"date": "2026-08-31", "val": 4.1, "previous_value": 4.0, "announcement_datetime": RELEASED_AT - 2419200},
        ]
    },
    "inflation": {
        "data": [
            {"date": "2026-08-31", "val": 3.4, "previous_value": 3.4, "announcement_datetime": 1789129800},
            {"date": "2026-07-31", "val": 3.4, "previous_value": None, "announcement_datetime": 1786537800},
        ]
    },
    "initial_jobless_claims": {"data": []},
}


def _response(payload, status=200):
    response = mock.Mock()
    response.status_code = status
    response.json.return_value = payload
    if status >= 400:
        response.raise_for_status.side_effect = data_sources.requests.exceptions.HTTPError(f"{status} Error")
    else:
        response.raise_for_status.return_value = None
    return response


def _fake_get(url, params=None, headers=None, timeout=None, allow_redirects=False):
    if url.endswith("/calendar/USD"):
        return _response(CALENDAR_USD)
    if "/announcements/USD/" in url:
        return _response(ANNOUNCEMENTS[url.rsplit("/", 1)[1]])
    if url.endswith("/calendar/EUR"):
        return _response({"currency": "EUR", "data": []})
    return _response({"detail": "not found"}, status=404)


@pytest.fixture
def source(monkeypatch):
    monkeypatch.delenv("FXMACRODATA_API_KEY", raising=False)
    config = DATA_SOURCES["fxmacrodata"]
    src = FXMacroDataSource(config)
    src.rate_limit = 100000  # no sleeping between mocked requests
    return src


@pytest.fixture
def keyed_source(monkeypatch):
    monkeypatch.setenv("FXMACRODATA_API_KEY", TEST_KEY)
    src = FXMacroDataSource(DATA_SOURCES["fxmacrodata"])
    src.rate_limit = 100000
    return src


def test_config_and_manager_registration(monkeypatch):
    monkeypatch.delenv("FXMACRODATA_API_KEY", raising=False)
    config = DATA_SOURCES["fxmacrodata"]
    assert config.base_url.startswith("https://")
    assert config.api_key_env == "FXMACRODATA_API_KEY"

    manager = ForexDataSourceManager(sources=["fxmacrodata", "yfinance"])
    assert isinstance(manager.data_sources["fxmacrodata"], FXMacroDataSource)
    assert list(manager.data_sources) == ["fxmacrodata", "yfinance"]


def test_http_base_url_is_rejected():
    with pytest.raises(DataSourceException):
        FXMacroDataSource({"name": "FXMacroData", "base_url": "http://api.fxmacrodata.com/v1"})


def test_calendar_mapping(source):
    with mock.patch.object(data_sources.requests, "get", side_effect=_fake_get) as get:
        events = source.get_economic_calendar(
            datetime(2026, 10, 1), datetime(2026, 10, 20), currencies=["USD"]
        )

    calendar_call = get.call_args_list[0]
    assert calendar_call.args[0] == "https://api.fxmacrodata.com/v1/calendar/USD"
    assert calendar_call.kwargs["params"] == {"start_date": "2026-10-01", "end_date": "2026-10-20"}

    assert len(events) == 3
    assert all(isinstance(event, EconomicEvent) for event in events)

    released, upcoming, claims = events
    assert released.event_id == f"usd_unemployment_{RELEASED_AT}"
    assert released.name == "Unemployment Rate"
    assert released.currency == "USD"
    assert released.date == datetime(2026, 10, 2, 12, 30)
    assert released.timestamp == released.date
    assert (released.impact, released.tier) == ("HIGH", "HIGH")
    assert released.actual == 4.2
    assert released.previous == 4.1
    assert released.forecast is None

    assert (upcoming.impact, upcoming.tier) == ("MEDIUM", "MEDIUM")
    assert upcoming.actual is None
    assert upcoming.previous == 3.4
    assert upcoming.forecast is None

    assert (claims.impact, claims.tier) == ("LOW", "LOW")
    assert claims.actual is None and claims.previous is None

    announcement_calls = [c for c in get.call_args_list if "/announcements/" in c.args[0]]
    assert len(announcement_calls) == 3
    assert all(c.kwargs["params"] == {"limit": 2} for c in announcement_calls)


def test_calendar_without_actuals(monkeypatch):
    monkeypatch.delenv("FXMACRODATA_API_KEY", raising=False)
    src = FXMacroDataSource(DATA_SOURCES["fxmacrodata"], include_actuals=False)
    src.rate_limit = 100000
    with mock.patch.object(data_sources.requests, "get", side_effect=_fake_get) as get:
        events = src.get_economic_calendar(datetime(2026, 10, 1), datetime(2026, 10, 20), ["USD"])

    assert get.call_count == 1
    assert len(events) == 3
    assert all(event.actual is None and event.previous is None for event in events)


def test_keyless_usd_only(source):
    with mock.patch.object(data_sources.requests, "get", side_effect=_fake_get) as get:
        events = source.get_economic_calendar(
            datetime(2026, 10, 1), datetime(2026, 10, 20), currencies=["USD", "EUR", "JPY"]
        )

    urls = [c.args[0] for c in get.call_args_list]
    assert not any("/EUR" in url or "/JPY" in url for url in urls)
    assert len(events) == 3
    for call in get.call_args_list:
        assert "X-API-Key" not in call.kwargs["headers"]
        assert "Authorization" not in call.kwargs["headers"]


def test_api_key_sent_only_in_header(keyed_source):
    with mock.patch.object(data_sources.requests, "get", side_effect=_fake_get) as get:
        keyed_source.get_economic_calendar(
            datetime(2026, 10, 1), datetime(2026, 10, 20), currencies=["USD", "EUR"]
        )

    assert get.call_count >= 2
    for call in get.call_args_list:
        url = call.args[0]
        headers = call.kwargs["headers"]
        params = call.kwargs["params"] or {}
        assert url.startswith("https://")
        assert headers["X-API-Key"] == TEST_KEY
        assert "Authorization" not in headers
        assert TEST_KEY not in url
        assert TEST_KEY not in str(params.values())
        assert "api_key" not in params


def test_market_data_parsing_and_pagination(keyed_source):
    first_page = {
        "data": [
            {"date": "2026-10-02", "val": 1.1050, "close": 1.1050},
            {"date": "2026-10-01", "val": 1.1000, "close": 1.1000},
        ],
        "pagination": {"limit": 100, "offset": 0, "returned_count": 2, "total_count": 3,
                       "has_more": True, "next_offset": 2},
    }
    second_page = {
        "data": [{"date": "2026-09-30", "val": 1.0890}],
        "pagination": {"limit": 100, "offset": 2, "returned_count": 1, "total_count": 3,
                       "has_more": False, "next_offset": None},
    }
    responses = [_response(first_page), _response(second_page)]

    with mock.patch.object(data_sources.requests, "get", side_effect=responses) as get:
        data = keyed_source.get_market_data("EUR/USD", datetime(2026, 9, 30), datetime(2026, 10, 2))

    assert get.call_args_list[0].args[0] == "https://api.fxmacrodata.com/v1/forex/EUR/USD"
    assert get.call_args_list[0].kwargs["params"]["offset"] == 0
    assert get.call_args_list[1].kwargs["params"]["offset"] == 2
    assert get.call_args_list[0].kwargs["headers"]["X-API-Key"] == TEST_KEY

    assert [d.timestamp for d in data] == [datetime(2026, 9, 30), datetime(2026, 10, 1), datetime(2026, 10, 2)]
    assert [d.mid for d in data] == [1.0890, 1.1000, 1.1050]
    assert data[0].change is None
    assert data[1].change == pytest.approx(0.0110)
    assert data[2].change_percent == pytest.approx(0.0050 / 1.1 * 100)
    assert all(d.symbol == "EUR/USD" for d in data)


def test_symbol_formats(source):
    assert source._convert_to_fxmd_pair("EUR/USD") == ("EUR", "USD")
    assert source._convert_to_fxmd_pair("eurusd") == ("EUR", "USD")
    assert source._convert_to_fxmd_pair("USDJPY=X") == ("USD", "JPY")
    with pytest.raises(DataSourceException):
        source._convert_to_fxmd_pair("EURO")


def test_market_data_needs_key(source):
    with mock.patch.object(data_sources.requests, "get") as get:
        assert source.get_market_data("EUR/USD", datetime(2026, 9, 1), datetime(2026, 10, 1)) == []
    get.assert_not_called()


def test_news_sentiment_falls_through(source):
    assert source.get_news_sentiment("EUR/USD", datetime(2026, 9, 1), datetime(2026, 10, 1)) == {}


def test_redirect_is_rejected_without_parsing_body(keyed_source):
    response = _response({"data": []}, status=302)
    with mock.patch.object(data_sources.requests, "get", return_value=response) as get:
        with pytest.raises(DataSourceException):
            keyed_source._make_request("/calendar/USD")
    assert get.call_args.kwargs["allow_redirects"] is False
    response.json.assert_not_called()


@pytest.mark.parametrize("payload", [None, "text", {"detail": "fixture-private-value"}, {"error": "fixture-private-value"}])
def test_invalid_payload_is_rejected_without_echo(keyed_source, payload):
    with mock.patch.object(data_sources.requests, "get", return_value=_response(payload)):
        with pytest.raises(DataSourceException) as raised:
            keyed_source._make_request("/calendar/USD")
    assert "fixture-private-value" not in str(raised.value)


def test_transport_private_text_is_not_logged_or_chained(keyed_source, caplog):
    import traceback
    error = data_sources.requests.exceptions.ConnectionError("fixture-private-value")
    with mock.patch.object(data_sources.requests, "get", side_effect=error):
        with pytest.raises(DataSourceException) as raised:
            keyed_source._make_request("/calendar/USD")
    assert "fixture-private-value" not in caplog.text
    assert "fixture-private-value" not in "".join(traceback.format_exception(raised.value))


def test_invalid_key_is_rejected_without_echo(monkeypatch):
    monkeypatch.setenv("FXMACRODATA_API_KEY", "fixture-key\r\nX-Other: value")
    with pytest.raises(ValueError) as raised:
        FXMacroDataSource(DATA_SOURCES["fxmacrodata"])
    assert "fixture-key" not in str(raised.value)

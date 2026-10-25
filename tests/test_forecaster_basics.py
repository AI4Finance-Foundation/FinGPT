"""Regression tests for point-in-time basic financials (issue #280)."""

import sys
from pathlib import Path

FORECASTER_DIR = Path(__file__).resolve().parents[1] / "fingpt" / "FinGPT_Forecaster"
if str(FORECASTER_DIR) not in sys.path:
    sys.path.insert(0, str(FORECASTER_DIR))

from basics import (  # noqa: E402
    DEFAULT_PUBLICATION_LAG_DAYS,
    available_date,
    select_published_basic,
)

QUARTERS = [
    {"period": "2024-03-31", "eps": 1.0},
    {"period": "2024-06-30", "eps": 2.0},
    {"period": "2024-09-30", "eps": 3.0},
]


def test_default_lag_matches_sec_10q_deadline():
    assert DEFAULT_PUBLICATION_LAG_DAYS == 45
    assert available_date("2024-06-30") == "2024-08-14"


def test_quarter_is_hidden_until_published():
    # Q2 ended 2024-06-30 but is not public until 2024-08-14.
    assert select_published_basic(QUARTERS, "2024-07-07")["period"] == "2024-03-31"
    assert select_published_basic(QUARTERS, "2024-08-14")["period"] == "2024-03-31"
    assert select_published_basic(QUARTERS, "2024-08-15")["period"] == "2024-06-30"


def test_window_only_reports_freshly_published_quarter():
    # Q2 became public 2024-08-14: inside a window starting 2024-08-04 ...
    picked = select_published_basic(QUARTERS, "2024-08-18", window_start="2024-08-04")
    assert picked["period"] == "2024-06-30"
    # ... but not inside one starting after it.
    assert select_published_basic(QUARTERS, "2024-09-01", window_start="2024-08-18") == {}


def test_nothing_published_yet_returns_empty():
    assert select_published_basic(QUARTERS, "2024-04-01") == {}
    assert select_published_basic([], "2024-12-31") == {}


def test_custom_lag():
    assert select_published_basic(QUARTERS, "2024-07-07", publication_lag_days=0)[
        "period"
    ] == "2024-06-30"

"""Point-in-time selection of quarterly basic financials.

Finnhub's quarterly ``basic_financials`` series is keyed by ``period``, the
fiscal quarter's END date. Companies publish a quarter weeks after it ends,
so selecting by ``period`` alone puts figures into a prompt before the market
had them (see issue #280). Until an exact publication date is wired in, a
quarter is treated as public ``publication_lag_days`` after its period end.
"""

from datetime import datetime, timedelta

# SEC 10-Q deadline: 40 days after quarter end for large accelerated and
# accelerated filers, 45 days for all others. Earnings releases (8-K item
# 2.02) usually come earlier, so 45 days is a conservative default for
# quarterly figures.
DEFAULT_PUBLICATION_LAG_DAYS = 45

_DATE_FMT = "%Y-%m-%d"


def available_date(period, publication_lag_days=DEFAULT_PUBLICATION_LAG_DAYS):
    """Return the date (YYYY-MM-DD) from which a quarter ending at
    ``period`` is assumed to be public."""
    period_end = datetime.strptime(period[:10], _DATE_FMT)
    return (period_end + timedelta(days=publication_lag_days)).strftime(_DATE_FMT)


def select_published_basic(
    basic_list,
    decision_date,
    window_start=None,
    publication_lag_days=DEFAULT_PUBLICATION_LAG_DAYS,
):
    """Pick the latest quarter that was public strictly before ``decision_date``.

    ``basic_list`` must be sorted by ``period`` ascending. If ``window_start``
    is given, the quarter must also have become public on or after it,
    otherwise an empty dict is returned (no fresh quarter to report).
    """
    for basic in reversed(basic_list):
        public_on = available_date(basic["period"], publication_lag_days)
        if public_on >= decision_date:
            continue
        if window_start is None or public_on >= window_start:
            return basic
        # Older quarters became public even earlier, so none can qualify.
        return {}
    return {}

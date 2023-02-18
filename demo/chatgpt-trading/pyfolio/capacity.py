from __future__ import division

import empyrical as ep
import numpy as np
import pandas as pd

from . import pos


def daily_txns_with_bar_data(transactions, market_data):
    """
    Sums the absolute value of shares traded in each name on each day.
    Adds columns containing the closing price and total daily volume for
    each day-ticker combination.

    Parameters
    ----------
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
        - See full explanation in tears.create_full_tear_sheet
    market_data : pd.DataFrame
        Daily market_data
        - DataFrame has a multi-index index, one level is dates and another is
        market_data contains volume & price, equities as columns

    Returns
    -------
    txn_daily : pd.DataFrame
        Daily totals for transacted shares in each traded name.
        price and volume columns for close price and daily volume for
        the corresponding ticker, respectively.
    """

    transactions.index.name = 'date'
    txn_daily = pd.DataFrame(transactions.assign(
        amount=abs(transactions.amount)).groupby(
        ['symbol', pd.Grouper(freq='D')]).sum()['amount'])
    txn_daily['price'] = market_data.xs('price', level=1).unstack()
    txn_daily['volume'] = market_data.xs('volume', level=1).unstack()

    txn_daily = txn_daily.reset_index().set_index('date')

    return txn_daily


def days_to_liquidate_positions(positions, market_data,
                                max_bar_consumption=0.2,
                                capital_base=1e6,
                                mean_volume_window=5):
    """
    Compute the number of days that would have been required
    to fully liquidate each position on each day based on the
    trailing n day mean daily bar volume and a limit on the proportion
    of a daily bar that we are allowed to consume.

    This analysis uses portfolio allocations and a provided capital base
    rather than the dollar values in the positions DataFrame to remove the
    effect of compounding on days to liquidate. In other words, this function
    assumes that the net liquidation portfolio value will always remain
    constant at capital_base.

    Parameters
    ----------
    positions: pd.DataFrame
        Contains daily position values including cash
        - See full explanation in tears.create_full_tear_sheet
    market_data : pd.DataFrame
        Daily market_data
        - DataFrame has a multi-index index, one level is dates and another is
        market_data contains volume & price, equities as columns
    max_bar_consumption : float
        Max proportion of a daily bar that can be consumed in the
        process of liquidating a position.
    capital_base : integer
        Capital base multiplied by portfolio allocation to compute
        position value that needs liquidating.
    mean_volume_window : float
        Trailing window to use in mean volume calculation.

    Returns
    -------
    days_to_liquidate : pd.DataFrame
        Number of days required to fully liquidate daily positions.
        Datetime index, symbols as columns.
    """

    DV = market_data.xs('volume', level=1) * market_data.xs('price', level=1)
    roll_mean_dv = DV.rolling(window=mean_volume_window,
                              center=False).mean().shift()
    roll_mean_dv = roll_mean_dv.replace(0, np.nan)

    positions_alloc = pos.get_percent_alloc(positions)
    positions_alloc = positions_alloc.drop('cash', axis=1)

    days_to_liquidate = (positions_alloc * capital_base) / \
        (max_bar_consumption * roll_mean_dv)

    return days_to_liquidate.iloc[mean_volume_window:]


def get_max_days_to_liquidate_by_ticker(positions, market_data,
                                        max_bar_consumption=0.2,
                                        capital_base=1e6,
                                        mean_volume_window=5,
                                        last_n_days=None):
    """
    Finds the longest estimated liquidation time for each traded
    name over the course of backtest (or last n days of the backtest).

    Parameters
    ----------
    positions: pd.DataFrame
        Contains daily position values including cash
        - See full explanation in tears.create_full_tear_sheet
    market_data : pd.DataFrame
        Daily market_data
        - DataFrame has a multi-index index, one level is dates and another is
        market_data contains volume & price, equities as columns
    max_bar_consumption : float
        Max proportion of a daily bar that can be consumed in the
        process of liquidating a position.
    capital_base : integer
        Capital base multiplied by portfolio allocation to compute
        position value that needs liquidating.
    mean_volume_window : float
        Trailing window to use in mean volume calculation.
    last_n_days : integer
        Compute for only the last n days of the passed backtest data.

    Returns
    -------
    days_to_liquidate : pd.DataFrame
        Max Number of days required to fully liquidate each traded name.
        Index of symbols. Columns for days_to_liquidate and the corresponding
        date and position_alloc on that day.
    """

    dtlp = days_to_liquidate_positions(positions, market_data,
                                       max_bar_consumption=max_bar_consumption,
                                       capital_base=capital_base,
                                       mean_volume_window=mean_volume_window)

    if last_n_days is not None:
        dtlp = dtlp.loc[dtlp.index.max() - pd.Timedelta(days=last_n_days):]

    pos_alloc = pos.get_percent_alloc(positions)
    pos_alloc = pos_alloc.drop('cash', axis=1)

    liq_desc = pd.DataFrame()
    liq_desc['days_to_liquidate'] = dtlp.unstack()
    liq_desc['pos_alloc_pct'] = pos_alloc.unstack() * 100
    liq_desc.index.levels[0].name = 'symbol'
    liq_desc.index.levels[1].name = 'date'

    worst_liq = liq_desc.reset_index().sort_values(
        'days_to_liquidate', ascending=False).groupby('symbol').first()

    return worst_liq


def get_low_liquidity_transactions(transactions, market_data,
                                   last_n_days=None):
    """
    For each traded name, find the daily transaction total that consumed
    the greatest proportion of available daily bar volume.

    Parameters
    ----------
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
         - See full explanation in create_full_tear_sheet.
    market_data : pd.DataFrame
        Daily market_data
        - DataFrame has a multi-index index, one level is dates and another is
        market_data contains volume & price, equities as columns
    last_n_days : integer
        Compute for only the last n days of the passed backtest data.
    """

    txn_daily_w_bar = daily_txns_with_bar_data(transactions, market_data)
    txn_daily_w_bar.index.name = 'date'
    txn_daily_w_bar = txn_daily_w_bar.reset_index()

    if last_n_days is not None:
        md = txn_daily_w_bar.date.max() - pd.Timedelta(days=last_n_days)
        txn_daily_w_bar = txn_daily_w_bar[txn_daily_w_bar.date > md]

    bar_consumption = txn_daily_w_bar.assign(
        max_pct_bar_consumed=(
            txn_daily_w_bar.amount/txn_daily_w_bar.volume)*100
    ).sort_values('max_pct_bar_consumed', ascending=False)
    max_bar_consumption = bar_consumption.groupby('symbol').first()

    return max_bar_consumption[['date', 'max_pct_bar_consumed']]


def apply_slippage_penalty(returns, txn_daily, simulate_starting_capital,
                           backtest_starting_capital, impact=0.1):
    """
    Applies quadratic volumeshare slippage model to daily returns based
    on the proportion of the observed historical daily bar dollar volume
    consumed by the strategy's trades. Scales the size of trades based
    on the ratio of the starting capital we wish to test to the starting
    capital of the passed backtest data.

    Parameters
    ----------
    returns : pd.Series
        Time series of daily returns.
    txn_daily : pd.Series
        Daily transaciton totals, closing price, and daily volume for
        each traded name. See price_volume_daily_txns for more details.
    simulate_starting_capital : integer
        capital at which we want to test
    backtest_starting_capital: capital base at which backtest was
        origionally run. impact: See Zipline volumeshare slippage model
    impact : float
        Scales the size of the slippage penalty.

    Returns
    -------
    adj_returns : pd.Series
        Slippage penalty adjusted daily returns.
    """

    mult = simulate_starting_capital / backtest_starting_capital
    simulate_traded_shares = abs(mult * txn_daily.amount)
    simulate_traded_dollars = txn_daily.price * simulate_traded_shares
    simulate_pct_volume_used = simulate_traded_shares / txn_daily.volume

    penalties = simulate_pct_volume_used**2 \
        * impact * simulate_traded_dollars

    daily_penalty = penalties.resample('D').sum()
    daily_penalty = daily_penalty.reindex(returns.index).fillna(0)

    # Since we are scaling the numerator of the penalties linearly
    # by capital base, it makes the most sense to scale the denominator
    # similarly. In other words, since we aren't applying compounding to
    # simulate_traded_shares, we shouldn't apply compounding to pv.
    portfolio_value = ep.cum_returns(
        returns, starting_value=backtest_starting_capital) * mult

    adj_returns = returns - (daily_penalty / portfolio_value)

    return adj_returns

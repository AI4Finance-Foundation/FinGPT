#
# Copyright 2016 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import division

import pandas as pd
import numpy as np
import warnings

try:
    from zipline.assets import Equity, Future
    ZIPLINE = True
except ImportError:
    ZIPLINE = False
    warnings.warn(
        'Module "zipline.assets" not found; multipliers will not be applied'
        ' to position notionals.'
    )


def get_percent_alloc(values):
    """
    Determines a portfolio's allocations.

    Parameters
    ----------
    values : pd.DataFrame
        Contains position values or amounts.

    Returns
    -------
    allocations : pd.DataFrame
        Positions and their allocations.
    """

    return values.divide(
        values.sum(axis='columns'),
        axis='rows'
    )


def get_top_long_short_abs(positions, top=10):
    """
    Finds the top long, short, and absolute positions.

    Parameters
    ----------
    positions : pd.DataFrame
        The positions that the strategy takes over time.
    top : int, optional
        How many of each to find (default 10).

    Returns
    -------
    df_top_long : pd.DataFrame
        Top long positions.
    df_top_short : pd.DataFrame
        Top short positions.
    df_top_abs : pd.DataFrame
        Top absolute positions.
    """

    positions = positions.drop('cash', axis='columns')
    df_max = positions.max()
    df_min = positions.min()
    df_abs_max = positions.abs().max()
    df_top_long = df_max[df_max > 0].nlargest(top)
    df_top_short = df_min[df_min < 0].nsmallest(top)
    df_top_abs = df_abs_max.nlargest(top)
    return df_top_long, df_top_short, df_top_abs


def get_max_median_position_concentration(positions):
    """
    Finds the max and median long and short position concentrations
    in each time period specified by the index of positions.

    Parameters
    ----------
    positions : pd.DataFrame
        The positions that the strategy takes over time.

    Returns
    -------
    pd.DataFrame
        Columns are max long, max short, median long, and median short
        position concentrations. Rows are timeperiods.
    """

    expos = get_percent_alloc(positions)
    expos = expos.drop('cash', axis=1)

    longs = expos.where(expos.applymap(lambda x: x > 0))
    shorts = expos.where(expos.applymap(lambda x: x < 0))

    alloc_summary = pd.DataFrame()
    alloc_summary['max_long'] = longs.max(axis=1)
    alloc_summary['median_long'] = longs.median(axis=1)
    alloc_summary['median_short'] = shorts.median(axis=1)
    alloc_summary['max_short'] = shorts.min(axis=1)

    return alloc_summary


def extract_pos(positions, cash):
    """
    Extract position values from backtest object as returned by
    get_backtest() on the Quantopian research platform.

    Parameters
    ----------
    positions : pd.DataFrame
        timeseries containing one row per symbol (and potentially
        duplicate datetime indices) and columns for amount and
        last_sale_price.
    cash : pd.Series
        timeseries containing cash in the portfolio.

    Returns
    -------
    pd.DataFrame
        Daily net position values.
         - See full explanation in tears.create_full_tear_sheet.
    """

    positions = positions.copy()
    positions['values'] = positions.amount * positions.last_sale_price

    cash.name = 'cash'

    values = positions.reset_index().pivot_table(index='index',
                                                 columns='sid',
                                                 values='values')

    if ZIPLINE:
        for asset in values.columns:
            if type(asset) in [Equity, Future]:
                values[asset] = values[asset] * asset.price_multiplier

    values = values.join(cash).fillna(0)

    # NOTE: Set name of DataFrame.columns to sid, to match the behavior
    # of DataFrame.join in earlier versions of pandas.
    values.columns.name = 'sid'

    return values


def get_sector_exposures(positions, symbol_sector_map):
    """
    Sum position exposures by sector.

    Parameters
    ----------
    positions : pd.DataFrame
        Contains position values or amounts.
        - Example
            index         'AAPL'         'MSFT'        'CHK'        cash
            2004-01-09    13939.380     -15012.993    -403.870      1477.483
            2004-01-12    14492.630     -18624.870    142.630       3989.610
            2004-01-13    -13853.280    13653.640     -100.980      100.000
    symbol_sector_map : dict or pd.Series
        Security identifier to sector mapping.
        Security ids as keys/index, sectors as values.
        - Example:
            {'AAPL' : 'Technology'
             'MSFT' : 'Technology'
             'CHK' : 'Natural Resources'}

    Returns
    -------
    sector_exp : pd.DataFrame
        Sectors and their allocations.
        - Example:
            index         'Technology'    'Natural Resources' cash
            2004-01-09    -1073.613       -403.870            1477.4830
            2004-01-12    -4132.240       142.630             3989.6100
            2004-01-13    -199.640        -100.980            100.0000
    """

    cash = positions['cash']
    positions = positions.drop('cash', axis=1)

    unmapped_pos = np.setdiff1d(positions.columns.values,
                                list(symbol_sector_map.keys()))
    if len(unmapped_pos) > 0:
        warn_message = """Warning: Symbols {} have no sector mapping.
        They will not be included in sector allocations""".format(
            ", ".join(map(str, unmapped_pos)))
        warnings.warn(warn_message, UserWarning)

    sector_exp = positions.groupby(
        by=symbol_sector_map, axis=1).sum()

    sector_exp['cash'] = cash

    return sector_exp


def get_long_short_pos(positions):
    """
    Determines the long and short allocations in a portfolio.

    Parameters
    ----------
    positions : pd.DataFrame
        The positions that the strategy takes over time.

    Returns
    -------
    df_long_short : pd.DataFrame
        Long and short allocations as a decimal
        percentage of the total net liquidation
    """

    pos_wo_cash = positions.drop('cash', axis=1)
    longs = pos_wo_cash[pos_wo_cash > 0].sum(axis=1).fillna(0)
    shorts = pos_wo_cash[pos_wo_cash < 0].sum(axis=1).fillna(0)
    cash = positions.cash
    net_liquidation = longs + shorts + cash
    df_pos = pd.DataFrame({'long': longs.divide(net_liquidation, axis='index'),
                           'short': shorts.divide(net_liquidation,
                                                  axis='index')})
    df_pos['net exposure'] = df_pos['long'] + df_pos['short']
    return df_pos

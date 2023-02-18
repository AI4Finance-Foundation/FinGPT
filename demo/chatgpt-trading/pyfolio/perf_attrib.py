#
# Copyright 2017 Quantopian, Inc.
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
import warnings

from collections import OrderedDict
import empyrical as ep
import pandas as pd
import matplotlib.pyplot as plt

from .pos import get_percent_alloc
from .txn import get_turnover
from .utils import print_table, configure_legend

PERF_ATTRIB_TURNOVER_THRESHOLD = 0.25


def perf_attrib(returns,
                positions,
                factor_returns,
                factor_loadings,
                transactions=None,
                pos_in_dollars=True):
    """
    Attributes the performance of a returns stream to a set of risk factors.

    Preprocesses inputs, and then calls empyrical.perf_attrib. See
    empyrical.perf_attrib for more info.

    Performance attribution determines how much each risk factor, e.g.,
    momentum, the technology sector, etc., contributed to total returns, as
    well as the daily exposure to each of the risk factors. The returns that
    can be attributed to one of the given risk factors are the
    `common_returns`, and the returns that _cannot_ be attributed to a risk
    factor are the `specific_returns`, or the alpha. The common_returns and
    specific_returns summed together will always equal the total returns.

    Parameters
    ----------
    returns : pd.Series
        Returns for each day in the date range.
        - Example:
            2017-01-01   -0.017098
            2017-01-02    0.002683
            2017-01-03   -0.008669

    positions: pd.DataFrame
        Daily holdings (in dollars or percentages), indexed by date.
        Will be converted to percentages if positions are in dollars.
        Short positions show up as cash in the 'cash' column.
        - Examples:
                        AAPL  TLT  XOM  cash
            2017-01-01    34   58   10     0
            2017-01-02    22   77   18     0
            2017-01-03   -15   27   30    15

                            AAPL       TLT       XOM  cash
            2017-01-01  0.333333  0.568627  0.098039   0.0
            2017-01-02  0.188034  0.658120  0.153846   0.0
            2017-01-03  0.208333  0.375000  0.416667   0.0

    factor_returns : pd.DataFrame
        Returns by factor, with date as index and factors as columns
        - Example:
                        momentum  reversal
            2017-01-01  0.002779 -0.005453
            2017-01-02  0.001096  0.010290

    factor_loadings : pd.DataFrame
        Factor loadings for all days in the date range, with date and ticker as
        index, and factors as columns.
        - Example:
                               momentum  reversal
            dt         ticker
            2017-01-01 AAPL   -1.592914  0.852830
                       TLT     0.184864  0.895534
                       XOM     0.993160  1.149353
            2017-01-02 AAPL   -0.140009 -0.524952
                       TLT    -1.066978  0.185435
                       XOM    -1.798401  0.761549


    transactions : pd.DataFrame, optional
        Executed trade volumes and fill prices. Used to check the turnover of
        the algorithm. Default is None, in which case the turnover check is
        skipped.

        - One row per trade.
        - Trades on different names that occur at the
          same time will have identical indicies.
        - Example:
            index                  amount   price    symbol
            2004-01-09 12:18:01    483      324.12   'AAPL'
            2004-01-09 12:18:01    122      83.10    'MSFT'
            2004-01-13 14:12:23    -75      340.43   'AAPL'

    pos_in_dollars : bool
        Flag indicating whether `positions` are in dollars or percentages
        If True, positions are in dollars.

    Returns
    -------
    tuple of (risk_exposures_portfolio, perf_attribution)

    risk_exposures_portfolio : pd.DataFrame
        df indexed by datetime, with factors as columns
        - Example:
                        momentum  reversal
            dt
            2017-01-01 -0.238655  0.077123
            2017-01-02  0.821872  1.520515

    perf_attribution : pd.DataFrame
        df with factors, common returns, and specific returns as columns,
        and datetimes as index
        - Example:
                        momentum  reversal  common_returns  specific_returns
            dt
            2017-01-01  0.249087  0.935925        1.185012          1.185012
            2017-01-02 -0.003194 -0.400786       -0.403980         -0.403980
    """
    (returns,
     positions,
     factor_returns,
     factor_loadings) = _align_and_warn(returns,
                                        positions,
                                        factor_returns,
                                        factor_loadings,
                                        transactions=transactions,
                                        pos_in_dollars=pos_in_dollars)

    # Note that we convert positions to percentages *after* the checks
    # above, since get_turnover() expects positions in dollars.
    positions = _stack_positions(positions, pos_in_dollars=pos_in_dollars)

    return ep.perf_attrib(returns, positions, factor_returns, factor_loadings)


def compute_exposures(positions, factor_loadings, stack_positions=True,
                      pos_in_dollars=True):
    """
    Compute daily risk factor exposures.

    Normalizes positions (if necessary) and calls ep.compute_exposures.
    See empyrical.compute_exposures for more info.

    Parameters
    ----------
    positions: pd.DataFrame or pd.Series
        Daily holdings (in dollars or percentages), indexed by date, OR
        a series of holdings indexed by date and ticker.
        - Examples:
                        AAPL  TLT  XOM  cash
            2017-01-01    34   58   10     0
            2017-01-02    22   77   18     0
            2017-01-03   -15   27   30    15

                            AAPL       TLT       XOM  cash
            2017-01-01  0.333333  0.568627  0.098039   0.0
            2017-01-02  0.188034  0.658120  0.153846   0.0
            2017-01-03  0.208333  0.375000  0.416667   0.0

            dt          ticker
            2017-01-01  AAPL      0.417582
                        TLT       0.010989
                        XOM       0.571429
            2017-01-02  AAPL      0.202381
                        TLT       0.535714
                        XOM       0.261905

    factor_loadings : pd.DataFrame
        Factor loadings for all days in the date range, with date and ticker as
        index, and factors as columns.
        - Example:
                               momentum  reversal
            dt         ticker
            2017-01-01 AAPL   -1.592914  0.852830
                       TLT     0.184864  0.895534
                       XOM     0.993160  1.149353
            2017-01-02 AAPL   -0.140009 -0.524952
                       TLT    -1.066978  0.185435
                       XOM    -1.798401  0.761549

    stack_positions : bool
        Flag indicating whether `positions` should be converted to long format.

    pos_in_dollars : bool
        Flag indicating whether `positions` are in dollars or percentages
        If True, positions are in dollars.

    Returns
    -------
    risk_exposures_portfolio : pd.DataFrame
        df indexed by datetime, with factors as columns.
        - Example:
                        momentum  reversal
            dt
            2017-01-01 -0.238655  0.077123
            2017-01-02  0.821872  1.520515
    """
    if stack_positions:
        positions = _stack_positions(positions, pos_in_dollars=pos_in_dollars)

    return ep.compute_exposures(positions, factor_loadings)


def create_perf_attrib_stats(perf_attrib, risk_exposures):
    """
    Takes perf attribution data over a period of time and computes annualized
    multifactor alpha, multifactor sharpe, risk exposures.
    """
    summary = OrderedDict()
    total_returns = perf_attrib['total_returns']
    specific_returns = perf_attrib['specific_returns']
    common_returns = perf_attrib['common_returns']

    summary['Annualized Specific Return'] =\
        ep.annual_return(specific_returns)
    summary['Annualized Common Return'] =\
        ep.annual_return(common_returns)
    summary['Annualized Total Return'] =\
        ep.annual_return(total_returns)

    summary['Specific Sharpe Ratio'] =\
        ep.sharpe_ratio(specific_returns)

    summary['Cumulative Specific Return'] =\
        ep.cum_returns_final(specific_returns)
    summary['Cumulative Common Return'] =\
        ep.cum_returns_final(common_returns)
    summary['Total Returns'] =\
        ep.cum_returns_final(total_returns)

    summary = pd.Series(summary, name='')

    annualized_returns_by_factor = [ep.annual_return(perf_attrib[c])
                                    for c in risk_exposures.columns]
    cumulative_returns_by_factor = [ep.cum_returns_final(perf_attrib[c])
                                    for c in risk_exposures.columns]

    risk_exposure_summary = pd.DataFrame(
        data=OrderedDict([
            (
                'Average Risk Factor Exposure',
                risk_exposures.mean(axis='rows')
            ),
            ('Annualized Return', annualized_returns_by_factor),
            ('Cumulative Return', cumulative_returns_by_factor),
        ]),
        index=risk_exposures.columns,
    )

    return summary, risk_exposure_summary


def show_perf_attrib_stats(returns,
                           positions,
                           factor_returns,
                           factor_loadings,
                           transactions=None,
                           pos_in_dollars=True):
    """
    Calls `perf_attrib` using inputs, and displays outputs using
    `utils.print_table`.
    """
    risk_exposures, perf_attrib_data = perf_attrib(
        returns,
        positions,
        factor_returns,
        factor_loadings,
        transactions,
        pos_in_dollars=pos_in_dollars,
    )

    perf_attrib_stats, risk_exposure_stats =\
        create_perf_attrib_stats(perf_attrib_data, risk_exposures)

    percentage_formatter = '{:.2%}'.format
    float_formatter = '{:.2f}'.format

    summary_stats = perf_attrib_stats.loc[['Annualized Specific Return',
                                           'Annualized Common Return',
                                           'Annualized Total Return',
                                           'Specific Sharpe Ratio']]

    # Format return rows in summary stats table as percentages.
    for col_name in (
        'Annualized Specific Return',
        'Annualized Common Return',
        'Annualized Total Return',
    ):
        summary_stats[col_name] = percentage_formatter(summary_stats[col_name])

    # Display sharpe to two decimal places.
    summary_stats['Specific Sharpe Ratio'] = float_formatter(
        summary_stats['Specific Sharpe Ratio']
    )

    print_table(summary_stats, name='Summary Statistics')

    print_table(
        risk_exposure_stats,
        name='Exposures Summary',
        # In exposures table, format exposure column to 2 decimal places, and
        # return columns  as percentages.
        formatters={
            'Average Risk Factor Exposure': float_formatter,
            'Annualized Return': percentage_formatter,
            'Cumulative Return': percentage_formatter,
        },
    )


def plot_returns(perf_attrib_data, cost=None, ax=None):
    """
    Plot total, specific, and common returns.

    Parameters
    ----------
    perf_attrib_data : pd.DataFrame
        df with factors, common returns, and specific returns as columns,
        and datetimes as index. Assumes the `total_returns` column is NOT
        cost adjusted.
        - Example:
                        momentum  reversal  common_returns  specific_returns
            dt
            2017-01-01  0.249087  0.935925        1.185012          1.185012
            2017-01-02 -0.003194 -0.400786       -0.403980         -0.403980

    cost : pd.Series, optional
        if present, gets subtracted from `perf_attrib_data['total_returns']`,
        and gets plotted separately

    ax :  matplotlib.axes.Axes
        axes on which plots are made. if None, current axes will be used

    Returns
    -------
    ax :  matplotlib.axes.Axes
    """

    if ax is None:
        ax = plt.gca()

    returns = perf_attrib_data['total_returns']
    total_returns_label = 'Total returns'

    cumulative_returns_less_costs = _cumulative_returns_less_costs(
        returns,
        cost
    )
    if cost is not None:
        total_returns_label += ' (adjusted)'

    specific_returns = perf_attrib_data['specific_returns']
    common_returns = perf_attrib_data['common_returns']

    ax.plot(cumulative_returns_less_costs, color='b',
            label=total_returns_label)
    ax.plot(ep.cum_returns(specific_returns), color='g',
            label='Cumulative specific returns')
    ax.plot(ep.cum_returns(common_returns), color='r',
            label='Cumulative common returns')

    if cost is not None:
        ax.plot(-ep.cum_returns(cost), color='k',
                label='Cumulative cost spent')

    ax.set_title('Time series of cumulative returns')
    ax.set_ylabel('Returns')

    configure_legend(ax)

    return ax


def plot_alpha_returns(alpha_returns, ax=None):
    """
    Plot histogram of daily multi-factor alpha returns (specific returns).

    Parameters
    ----------
    alpha_returns : pd.Series
        series of daily alpha returns indexed by datetime

    ax :  matplotlib.axes.Axes
        axes on which plots are made. if None, current axes will be used

    Returns
    -------
    ax :  matplotlib.axes.Axes
    """
    if ax is None:
        ax = plt.gca()

    ax.hist(alpha_returns, color='g', label='Multi-factor alpha')
    ax.set_title('Histogram of alphas')
    ax.axvline(0, color='k', linestyle='--', label='Zero')

    avg = alpha_returns.mean()
    ax.axvline(avg, color='b', label='Mean = {: 0.5f}'.format(avg))
    configure_legend(ax)

    return ax


def plot_factor_contribution_to_perf(
        perf_attrib_data,
        ax=None,
        title='Cumulative common returns attribution',
):
    """
    Plot each factor's contribution to performance.

    Parameters
    ----------
    perf_attrib_data : pd.DataFrame
        df with factors, common returns, and specific returns as columns,
        and datetimes as index
        - Example:
                        momentum  reversal  common_returns  specific_returns
            dt
            2017-01-01  0.249087  0.935925        1.185012          1.185012
            2017-01-02 -0.003194 -0.400786       -0.403980         -0.403980

    ax :  matplotlib.axes.Axes
        axes on which plots are made. if None, current axes will be used

    title : str, optional
        title of plot

    Returns
    -------
    ax :  matplotlib.axes.Axes
    """
    if ax is None:
        ax = plt.gca()

    factors_to_plot = perf_attrib_data.drop(
        ['total_returns', 'common_returns', 'tilt_returns', 'timing_returns'],
        axis='columns', errors='ignore'
    )

    factors_cumulative = pd.DataFrame()
    for factor in factors_to_plot:
        factors_cumulative[factor] = ep.cum_returns(factors_to_plot[factor])

    for col in factors_cumulative:
        ax.plot(factors_cumulative[col])

    ax.axhline(0, color='k')
    configure_legend(ax, change_colors=True)

    ax.set_ylabel('Cumulative returns by factor')
    ax.set_title(title)

    return ax


def plot_risk_exposures(exposures, ax=None,
                        title='Daily risk factor exposures'):
    """
    Parameters
    ----------
    exposures : pd.DataFrame
        df indexed by datetime, with factors as columns
        - Example:
                        momentum  reversal
            dt
            2017-01-01 -0.238655  0.077123
            2017-01-02  0.821872  1.520515

    ax :  matplotlib.axes.Axes
        axes on which plots are made. if None, current axes will be used

    Returns
    -------
    ax :  matplotlib.axes.Axes
    """
    if ax is None:
        ax = plt.gca()

    for col in exposures:
        ax.plot(exposures[col])

    configure_legend(ax, change_colors=True)
    ax.set_ylabel('Factor exposures')
    ax.set_title(title)

    return ax


def _align_and_warn(returns,
                    positions,
                    factor_returns,
                    factor_loadings,
                    transactions=None,
                    pos_in_dollars=True):
    """
    Make sure that all inputs have matching dates and tickers,
    and raise warnings if necessary.
    """
    missing_stocks = positions.columns.difference(
        factor_loadings.index.get_level_values(1).unique()
    )

    # cash will not be in factor_loadings
    num_stocks = len(positions.columns) - 1
    missing_stocks = missing_stocks.drop('cash')
    num_stocks_covered = num_stocks - len(missing_stocks)
    missing_ratio = round(len(missing_stocks) / num_stocks, ndigits=3)

    if num_stocks_covered == 0:
        raise ValueError("Could not perform performance attribution. "
                         "No factor loadings were available for this "
                         "algorithm's positions.")

    if len(missing_stocks) > 0:

        if len(missing_stocks) > 5:

            missing_stocks_displayed = (
                " {} assets were missing factor loadings, including: {}..{}"
            ).format(len(missing_stocks),
                     ', '.join(missing_stocks[:5].map(str)),
                     missing_stocks[-1])
            avg_allocation_msg = "selected missing assets"

        else:
            missing_stocks_displayed = (
                "The following assets were missing factor loadings: {}."
            ).format(list(missing_stocks))
            avg_allocation_msg = "missing assets"

        missing_stocks_warning_msg = (
            "Could not determine risk exposures for some of this algorithm's "
            "positions. Returns from the missing assets will not be properly "
            "accounted for in performance attribution.\n"
            "\n"
            "{}. "
            "Ignoring for exposure calculation and performance attribution. "
            "Ratio of assets missing: {}. Average allocation of {}:\n"
            "\n"
            "{}.\n"
        ).format(
            missing_stocks_displayed,
            missing_ratio,
            avg_allocation_msg,
            positions[missing_stocks[:5].union(missing_stocks[[-1]])].mean(),
        )

        warnings.warn(missing_stocks_warning_msg)

        positions = positions.drop(missing_stocks, axis='columns',
                                   errors='ignore')

    missing_factor_loadings_index = positions.index.difference(
        factor_loadings.index.get_level_values(0).unique()
    )

    if len(missing_factor_loadings_index) > 0:

        if len(missing_factor_loadings_index) > 5:
            missing_dates_displayed = (
                "(first missing is {}, last missing is {})"
            ).format(
                missing_factor_loadings_index[0],
                missing_factor_loadings_index[-1]
            )
        else:
            missing_dates_displayed = list(missing_factor_loadings_index)

        warning_msg = (
            "Could not find factor loadings for {} dates: {}. "
            "Truncating date range for performance attribution. "
        ).format(len(missing_factor_loadings_index), missing_dates_displayed)

        warnings.warn(warning_msg)

        positions = positions.drop(missing_factor_loadings_index,
                                   errors='ignore')
        returns = returns.drop(missing_factor_loadings_index, errors='ignore')
        factor_returns = factor_returns.drop(missing_factor_loadings_index,
                                             errors='ignore')

    if transactions is not None and pos_in_dollars:
        turnover = get_turnover(positions, transactions).mean()
        if turnover > PERF_ATTRIB_TURNOVER_THRESHOLD:
            warning_msg = (
                "This algorithm has relatively high turnover of its "
                "positions. As a result, performance attribution might not be "
                "fully accurate.\n"
                "\n"
                "Performance attribution is calculated based "
                "on end-of-day holdings and does not account for intraday "
                "activity. Algorithms that derive a high percentage of "
                "returns from buying and selling within the same day may "
                "receive inaccurate performance attribution.\n"
            )
            warnings.warn(warning_msg)

    return (returns, positions, factor_returns, factor_loadings)


def _stack_positions(positions, pos_in_dollars=True):
    """
    Convert positions to percentages if necessary, and change them
    to long format.

    Parameters
    ----------
    positions: pd.DataFrame
        Daily holdings (in dollars or percentages), indexed by date.
        Will be converted to percentages if positions are in dollars.
        Short positions show up as cash in the 'cash' column.

    pos_in_dollars : bool
        Flag indicating whether `positions` are in dollars or percentages
        If True, positions are in dollars.
    """
    if pos_in_dollars:
        # convert holdings to percentages
        positions = get_percent_alloc(positions)

    # remove cash after normalizing positions
    positions = positions.drop('cash', axis='columns')

    # convert positions to long format
    positions = positions.stack()
    positions.index = positions.index.set_names(['dt', 'ticker'])

    return positions


def _cumulative_returns_less_costs(returns, costs):
    """
    Compute cumulative returns, less costs.
    """
    if costs is None:
        return ep.cum_returns(returns)
    return ep.cum_returns(returns - costs)

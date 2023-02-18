#
# Copyright 2018 Quantopian, Inc.
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

import datetime
from collections import OrderedDict
from functools import wraps

import empyrical as ep
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import scipy as sp
from matplotlib import figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.ticker import FuncFormatter

from . import _seaborn as sns
from . import capacity
from . import pos
from . import timeseries
from . import txn
from . import utils
from .utils import (APPROX_BDAYS_PER_MONTH,
                    MM_DISPLAY_UNIT)


def customize(func):
    """
    Decorator to set plotting context and axes style during function call.
    """
    @wraps(func)
    def call_w_context(*args, **kwargs):
        set_context = kwargs.pop('set_context', True)
        if set_context:
            with plotting_context(), axes_style():
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return call_w_context


def plotting_context(context='notebook', font_scale=1.5, rc=None):
    """
    Create pyfolio default plotting style context.

    Under the hood, calls and returns seaborn.plotting_context() with
    some custom settings. Usually you would use in a with-context.

    Parameters
    ----------
    context : str, optional
        Name of seaborn context.
    font_scale : float, optional
        Scale font by factor font_scale.
    rc : dict, optional
        Config flags.
        By default, {'lines.linewidth': 1.5}
        is being used and will be added to any
        rc passed in, unless explicitly overriden.

    Returns
    -------
    seaborn plotting context

    Example
    -------
    >>> with pyfolio.plotting.plotting_context(font_scale=2):
    >>>    pyfolio.create_full_tear_sheet(..., set_context=False)

    See also
    --------
    For more information, see seaborn.plotting_context().

    """
    if rc is None:
        rc = {}

    rc_default = {'lines.linewidth': 1.5}

    # Add defaults if they do not exist
    for name, val in rc_default.items():
        rc.setdefault(name, val)

    return sns.plotting_context(context=context, font_scale=font_scale, rc=rc)


def axes_style(style='darkgrid', rc=None):
    """
    Create pyfolio default axes style context.

    Under the hood, calls and returns seaborn.axes_style() with
    some custom settings. Usually you would use in a with-context.

    Parameters
    ----------
    style : str, optional
        Name of seaborn style.
    rc : dict, optional
        Config flags.

    Returns
    -------
    seaborn plotting context

    Example
    -------
    >>> with pyfolio.plotting.axes_style(style='whitegrid'):
    >>>    pyfolio.create_full_tear_sheet(..., set_context=False)

    See also
    --------
    For more information, see seaborn.plotting_context().

    """
    if rc is None:
        rc = {}

    rc_default = {}

    # Add defaults if they do not exist
    for name, val in rc_default.items():
        rc.setdefault(name, val)

    return sns.axes_style(style=style, rc=rc)


def plot_monthly_returns_heatmap(returns, ax=None, **kwargs):
    """
    Plots a heatmap of returns by month.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to seaborn plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    monthly_ret_table = ep.aggregate_returns(returns, 'monthly')
    monthly_ret_table = monthly_ret_table.unstack().round(3)

    sns.heatmap(
        monthly_ret_table.fillna(0) *
        100.0,
        annot=True,
        annot_kws={"size": 9},
        alpha=1.0,
        center=0.0,
        cbar=False,
        cmap=matplotlib.cm.RdYlGn,
        ax=ax, **kwargs)
    ax.set_ylabel('Year')
    ax.set_xlabel('Month')
    ax.set_title("Monthly returns (%)")
    return ax


def plot_annual_returns(returns, ax=None, **kwargs):
    """
    Plots a bar graph of returns by year.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    x_axis_formatter = FuncFormatter(utils.percentage)
    ax.xaxis.set_major_formatter(FuncFormatter(x_axis_formatter))
    ax.tick_params(axis='x', which='major')

    ann_ret_df = pd.DataFrame(
        ep.aggregate_returns(
            returns,
            'yearly'))

    ax.axvline(
        100 *
        ann_ret_df.values.mean(),
        color='steelblue',
        linestyle='--',
        lw=4,
        alpha=0.7)
    (100 * ann_ret_df.sort_index(ascending=False)
     ).plot(ax=ax, kind='barh', alpha=0.70, **kwargs)
    ax.axvline(0.0, color='black', linestyle='-', lw=3)

    ax.set_ylabel('Year')
    ax.set_xlabel('Returns')
    ax.set_title("Annual returns")
    ax.legend(['Mean'], frameon=True, framealpha=0.5)
    return ax


def plot_monthly_returns_dist(returns, ax=None, **kwargs):
    """
    Plots a distribution of monthly returns.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    x_axis_formatter = FuncFormatter(utils.percentage)
    ax.xaxis.set_major_formatter(FuncFormatter(x_axis_formatter))
    ax.tick_params(axis='x', which='major')

    monthly_ret_table = ep.aggregate_returns(returns, 'monthly')

    ax.hist(
        100 * monthly_ret_table,
        color='orangered',
        alpha=0.80,
        bins=20,
        **kwargs)

    ax.axvline(
        100 * monthly_ret_table.mean(),
        color='gold',
        linestyle='--',
        lw=4,
        alpha=1.0)

    ax.axvline(0.0, color='black', linestyle='-', lw=3, alpha=0.75)
    ax.legend(['Mean'], frameon=True, framealpha=0.5)
    ax.set_ylabel('Number of months')
    ax.set_xlabel('Returns')
    ax.set_title("Distribution of monthly returns")
    return ax


def plot_holdings(returns, positions, legend_loc='best', ax=None, **kwargs):
    """
    Plots total amount of stocks with an active position, either short
    or long. Displays daily total, daily average per month, and
    all-time daily average.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    positions : pd.DataFrame, optional
        Daily net position values.
         - See full explanation in tears.create_full_tear_sheet.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    positions = positions.copy().drop('cash', axis='columns')
    df_holdings = positions.replace(0, np.nan).count(axis=1)
    df_holdings_by_month = df_holdings.resample('1M').mean()
    df_holdings.plot(color='steelblue', alpha=0.6, lw=0.5, ax=ax, **kwargs)
    df_holdings_by_month.plot(
        color='orangered',
        lw=2,
        ax=ax,
        **kwargs)
    ax.axhline(
        df_holdings.values.mean(),
        color='steelblue',
        ls='--',
        lw=3)

    ax.set_xlim((returns.index[0], returns.index[-1]))

    leg = ax.legend(['Daily holdings',
                     'Average daily holdings, by month',
                     'Average daily holdings, overall'],
                    loc=legend_loc, frameon=True,
                    framealpha=0.5)
    leg.get_frame().set_edgecolor('black')

    ax.set_title('Total holdings')
    ax.set_ylabel('Holdings')
    ax.set_xlabel('')
    return ax


def plot_long_short_holdings(returns, positions,
                             legend_loc='upper left', ax=None, **kwargs):
    """
    Plots total amount of stocks with an active position, breaking out
    short and long into transparent filled regions.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    positions : pd.DataFrame, optional
        Daily net position values.
         - See full explanation in tears.create_full_tear_sheet.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.

    """

    if ax is None:
        ax = plt.gca()

    positions = positions.drop('cash', axis='columns')
    positions = positions.replace(0, np.nan)
    df_longs = positions[positions > 0].count(axis=1)
    df_shorts = positions[positions < 0].count(axis=1)
    lf = ax.fill_between(df_longs.index, 0, df_longs.values,
                         color='g', alpha=0.5, lw=2.0)
    sf = ax.fill_between(df_shorts.index, 0, df_shorts.values,
                         color='r', alpha=0.5, lw=2.0)

    bf = patches.Rectangle([0, 0], 1, 1, color='darkgoldenrod')
    leg = ax.legend([lf, sf, bf],
                    ['Long (max: %s, min: %s)' % (df_longs.max(),
                                                  df_longs.min()),
                     'Short (max: %s, min: %s)' % (df_shorts.max(),
                                                   df_shorts.min()),
                     'Overlap'], loc=legend_loc, frameon=True,
                    framealpha=0.5)
    leg.get_frame().set_edgecolor('black')

    ax.set_xlim((returns.index[0], returns.index[-1]))
    ax.set_title('Long and short holdings')
    ax.set_ylabel('Holdings')
    ax.set_xlabel('')
    return ax


def plot_drawdown_periods(returns, top=10, ax=None, **kwargs):
    """
    Plots cumulative returns highlighting top drawdown periods.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    top : int, optional
        Amount of top drawdowns periods to plot (default 10).
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(utils.two_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    df_cum_rets = ep.cum_returns(returns, starting_value=1.0)
    df_drawdowns = timeseries.gen_drawdown_table(returns, top=top)

    df_cum_rets.plot(ax=ax, **kwargs)

    lim = ax.get_ylim()
    colors = sns.cubehelix_palette(len(df_drawdowns))[::-1]
    for i, (peak, recovery) in df_drawdowns[
            ['Peak date', 'Recovery date']].iterrows():
        if pd.isnull(recovery):
            recovery = returns.index[-1]
        ax.fill_between((peak, recovery),
                        lim[0],
                        lim[1],
                        alpha=.4,
                        color=colors[i])
    ax.set_ylim(lim)
    ax.set_title('Top %i drawdown periods' % top)
    ax.set_ylabel('Cumulative returns')
    ax.legend(['Portfolio'], loc='upper left',
              frameon=True, framealpha=0.5)
    ax.set_xlabel('')
    return ax


def plot_drawdown_underwater(returns, ax=None, **kwargs):
    """
    Plots how far underwaterr returns are over time, or plots current
    drawdown vs. date.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(utils.percentage)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    df_cum_rets = ep.cum_returns(returns, starting_value=1.0)
    running_max = np.maximum.accumulate(df_cum_rets)
    underwater = -100 * ((running_max - df_cum_rets) / running_max)
    (underwater).plot(ax=ax, kind='area', color='coral', alpha=0.7, **kwargs)
    ax.set_ylabel('Drawdown')
    ax.set_title('Underwater plot')
    ax.set_xlabel('')
    return ax


def plot_perf_stats(returns, factor_returns, ax=None):
    """
    Create box plot of some performance metrics of the strategy.
    The width of the box whiskers is determined by a bootstrap.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series
        Daily noncumulative returns of the benchmark factor to which betas are
        computed. Usually a benchmark such as market returns.
         - This is in the same style as returns.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    bootstrap_values = timeseries.perf_stats_bootstrap(returns,
                                                       factor_returns,
                                                       return_stats=False)
    bootstrap_values = bootstrap_values.drop('Kurtosis', axis='columns')

    sns.boxplot(data=bootstrap_values, orient='h', ax=ax)

    return ax


STAT_FUNCS_PCT = [
    'Annual return',
    'Cumulative returns',
    'Annual volatility',
    'Max drawdown',
    'Daily value at risk',
    'Daily turnover'
]


def show_perf_stats(returns, factor_returns=None, positions=None,
                    transactions=None, turnover_denom='AGB',
                    live_start_date=None, bootstrap=False,
                    header_rows=None):
    """
    Prints some performance metrics of the strategy.

    - Shows amount of time the strategy has been run in backtest and
      out-of-sample (in live trading).

    - Shows Omega ratio, max drawdown, Calmar ratio, annual return,
      stability, Sharpe ratio, annual volatility, alpha, and beta.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series, optional
        Daily noncumulative returns of the benchmark factor to which betas are
        computed. Usually a benchmark such as market returns.
         - This is in the same style as returns.
    positions : pd.DataFrame, optional
        Daily net position values.
         - See full explanation in create_full_tear_sheet.
    transactions : pd.DataFrame, optional
        Prices and amounts of executed trades. One row per trade.
        - See full explanation in tears.create_full_tear_sheet
    turnover_denom : str, optional
        Either AGB or portfolio_value, default AGB.
        - See full explanation in txn.get_turnover.
    live_start_date : datetime, optional
        The point in time when the strategy began live trading, after
        its backtest period.
    bootstrap : boolean, optional
        Whether to perform bootstrap analysis for the performance
        metrics.
         - For more information, see timeseries.perf_stats_bootstrap
    header_rows : dict or OrderedDict, optional
        Extra rows to display at the top of the displayed table.
    """

    if bootstrap:
        perf_func = timeseries.perf_stats_bootstrap
    else:
        perf_func = timeseries.perf_stats

    perf_stats_all = perf_func(
        returns,
        factor_returns=factor_returns,
        positions=positions,
        transactions=transactions,
        turnover_denom=turnover_denom)

    date_rows = OrderedDict()
    if len(returns.index) > 0:
        date_rows['Start date'] = returns.index[0].strftime('%Y-%m-%d')
        date_rows['End date'] = returns.index[-1].strftime('%Y-%m-%d')

    if live_start_date is not None:
        live_start_date = ep.utils.get_utc_timestamp(live_start_date)
        returns_is = returns[returns.index < live_start_date]
        returns_oos = returns[returns.index >= live_start_date]

        positions_is = None
        positions_oos = None
        transactions_is = None
        transactions_oos = None

        if positions is not None:
            positions_is = positions[positions.index < live_start_date]
            positions_oos = positions[positions.index >= live_start_date]
            if transactions is not None:
                transactions_is = transactions[(transactions.index <
                                                live_start_date)]
                transactions_oos = transactions[(transactions.index >
                                                 live_start_date)]

        perf_stats_is = perf_func(
            returns_is,
            factor_returns=factor_returns,
            positions=positions_is,
            transactions=transactions_is,
            turnover_denom=turnover_denom)

        perf_stats_oos = perf_func(
            returns_oos,
            factor_returns=factor_returns,
            positions=positions_oos,
            transactions=transactions_oos,
            turnover_denom=turnover_denom)
        if len(returns.index) > 0:
            date_rows['In-sample months'] = int(len(returns_is) /
                                                APPROX_BDAYS_PER_MONTH)
            date_rows['Out-of-sample months'] = int(len(returns_oos) /
                                                    APPROX_BDAYS_PER_MONTH)

        perf_stats = pd.concat(OrderedDict([
            ('In-sample', perf_stats_is),
            ('Out-of-sample', perf_stats_oos),
            ('All', perf_stats_all),
        ]), axis=1)
    else:
        if len(returns.index) > 0:
            date_rows['Total months'] = int(len(returns) /
                                            APPROX_BDAYS_PER_MONTH)
        perf_stats = pd.DataFrame(perf_stats_all, columns=['Backtest'])

    for column in perf_stats.columns:
        for stat, value in perf_stats[column].iteritems():
            if stat in STAT_FUNCS_PCT:
                perf_stats.loc[stat, column] = str(np.round(value * 100,
                                                            3)) + '%'
    if header_rows is None:
        header_rows = date_rows
    else:
        header_rows = OrderedDict(header_rows)
        header_rows.update(date_rows)

    utils.print_table(
        perf_stats,
        float_format='{0:.2f}'.format,
        header_rows=header_rows,
    )


def plot_returns(returns,
                 live_start_date=None,
                 ax=None):
    """
    Plots raw returns over time.

    Backtest returns are in green, and out-of-sample (live trading)
    returns are in red.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    live_start_date : datetime, optional
        The date when the strategy began live trading, after
        its backtest period. This date should be normalized.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    ax.set_label('')
    ax.set_ylabel('Returns')

    if live_start_date is not None:
        live_start_date = ep.utils.get_utc_timestamp(live_start_date)
        is_returns = returns.loc[returns.index < live_start_date]
        oos_returns = returns.loc[returns.index >= live_start_date]
        is_returns.plot(ax=ax, color='g')
        oos_returns.plot(ax=ax, color='r')

    else:
        returns.plot(ax=ax, color='g')

    return ax


def plot_rolling_returns(returns,
                         factor_returns=None,
                         live_start_date=None,
                         logy=False,
                         cone_std=None,
                         legend_loc='best',
                         volatility_match=False,
                         cone_function=timeseries.forecast_cone_bootstrap,
                         ax=None, **kwargs):
    """
    Plots cumulative rolling returns versus some benchmarks'.

    Backtest returns are in green, and out-of-sample (live trading)
    returns are in red.

    Additionally, a non-parametric cone plot may be added to the
    out-of-sample returns region.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series, optional
        Daily noncumulative returns of the benchmark factor to which betas are
        computed. Usually a benchmark such as market returns.
         - This is in the same style as returns.
    live_start_date : datetime, optional
        The date when the strategy began live trading, after
        its backtest period. This date should be normalized.
    logy : bool, optional
        Whether to log-scale the y-axis.
    cone_std : float, or tuple, optional
        If float, The standard deviation to use for the cone plots.
        If tuple, Tuple of standard deviation values to use for the cone plots
         - See timeseries.forecast_cone_bounds for more details.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    volatility_match : bool, optional
        Whether to normalize the volatility of the returns to those of the
        benchmark returns. This helps compare strategies with different
        volatilities. Requires passing of benchmark_rets.
    cone_function : function, optional
        Function to use when generating forecast probability cone.
        The function signiture must follow the form:
        def cone(in_sample_returns (pd.Series),
                 days_to_project_forward (int),
                 cone_std= (float, or tuple),
                 starting_value= (int, or float))
        See timeseries.forecast_cone_bootstrap for an example.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    ax.set_xlabel('')
    ax.set_ylabel('Cumulative returns')
    ax.set_yscale('log' if logy else 'linear')

    if volatility_match and factor_returns is None:
        raise ValueError('volatility_match requires passing of '
                         'factor_returns.')
    elif volatility_match and factor_returns is not None:
        bmark_vol = factor_returns.loc[returns.index].std()
        returns = (returns / returns.std()) * bmark_vol

    cum_rets = ep.cum_returns(returns, 1.0)

    y_axis_formatter = FuncFormatter(utils.two_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    if factor_returns is not None:
        cum_factor_returns = ep.cum_returns(
            factor_returns[cum_rets.index], 1.0)
        cum_factor_returns.plot(lw=2, color='gray',
                                label=factor_returns.name, alpha=0.60,
                                ax=ax, **kwargs)

    if live_start_date is not None:
        live_start_date = ep.utils.get_utc_timestamp(live_start_date)
        is_cum_returns = cum_rets.loc[cum_rets.index < live_start_date]
        oos_cum_returns = cum_rets.loc[cum_rets.index >= live_start_date]
    else:
        is_cum_returns = cum_rets
        oos_cum_returns = pd.Series([])

    is_cum_returns.plot(lw=3, color='forestgreen', alpha=0.6,
                        label='Backtest', ax=ax, **kwargs)

    if len(oos_cum_returns) > 0:
        oos_cum_returns.plot(lw=4, color='red', alpha=0.6,
                             label='Live', ax=ax, **kwargs)

        if cone_std is not None:
            if isinstance(cone_std, (float, int)):
                cone_std = [cone_std]

            is_returns = returns.loc[returns.index < live_start_date]
            cone_bounds = cone_function(
                is_returns,
                len(oos_cum_returns),
                cone_std=cone_std,
                starting_value=is_cum_returns[-1])

            cone_bounds = cone_bounds.set_index(oos_cum_returns.index)
            for std in cone_std:
                ax.fill_between(cone_bounds.index,
                                cone_bounds[float(std)],
                                cone_bounds[float(-std)],
                                color='steelblue', alpha=0.5)

    if legend_loc is not None:
        ax.legend(loc=legend_loc, frameon=True, framealpha=0.5)
    ax.axhline(1.0, linestyle='--', color='black', lw=2)

    return ax


def plot_rolling_beta(returns, factor_returns, legend_loc='best',
                      ax=None, **kwargs):
    """
    Plots the rolling 6-month and 12-month beta versus date.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series
        Daily noncumulative returns of the benchmark factor to which betas are
        computed. Usually a benchmark such as market returns.
         - This is in the same style as returns.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(utils.two_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    ax.set_title("Rolling portfolio beta to " + str(factor_returns.name))
    ax.set_ylabel('Beta')
    rb_1 = timeseries.rolling_beta(
        returns, factor_returns, rolling_window=APPROX_BDAYS_PER_MONTH * 6)
    rb_1.plot(color='steelblue', lw=3, alpha=0.6, ax=ax, **kwargs)
    rb_2 = timeseries.rolling_beta(
        returns, factor_returns, rolling_window=APPROX_BDAYS_PER_MONTH * 12)
    rb_2.plot(color='grey', lw=3, alpha=0.4, ax=ax, **kwargs)
    ax.axhline(rb_1.mean(), color='steelblue', linestyle='--', lw=3)
    ax.axhline(0.0, color='black', linestyle='-', lw=2)

    ax.set_xlabel('')
    ax.legend(['6-mo',
               '12-mo'],
              loc=legend_loc, frameon=True, framealpha=0.5)
    ax.set_ylim((-1.0, 1.0))
    return ax


def plot_rolling_volatility(returns, factor_returns=None,
                            rolling_window=APPROX_BDAYS_PER_MONTH * 6,
                            legend_loc='best', ax=None, **kwargs):
    """
    Plots the rolling volatility versus date.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series, optional
        Daily noncumulative returns of the benchmark factor for which the
        benchmark rolling volatility is computed. Usually a benchmark such
        as market returns.
         - This is in the same style as returns.
    rolling_window : int, optional
        The days window over which to compute the volatility.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(utils.two_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    rolling_vol_ts = timeseries.rolling_volatility(
        returns, rolling_window)
    rolling_vol_ts.plot(alpha=.7, lw=3, color='orangered', ax=ax,
                        **kwargs)
    if factor_returns is not None:
        rolling_vol_ts_factor = timeseries.rolling_volatility(
            factor_returns, rolling_window)
        rolling_vol_ts_factor.plot(alpha=.7, lw=3, color='grey', ax=ax,
                                   **kwargs)

    ax.set_title('Rolling volatility (6-month)')
    ax.axhline(
        rolling_vol_ts.mean(),
        color='steelblue',
        linestyle='--',
        lw=3)

    ax.axhline(0.0, color='black', linestyle='-', lw=2)

    ax.set_ylabel('Volatility')
    ax.set_xlabel('')
    if factor_returns is None:
        ax.legend(['Volatility', 'Average volatility'],
                  loc=legend_loc, frameon=True, framealpha=0.5)
    else:
        ax.legend(['Volatility', 'Benchmark volatility', 'Average volatility'],
                  loc=legend_loc, frameon=True, framealpha=0.5)
    return ax


def plot_rolling_sharpe(returns, factor_returns=None,
                        rolling_window=APPROX_BDAYS_PER_MONTH * 6,
                        legend_loc='best', ax=None, **kwargs):
    """
    Plots the rolling Sharpe ratio versus date.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series, optional
        Daily noncumulative returns of the benchmark factor for
        which the benchmark rolling Sharpe is computed. Usually
        a benchmark such as market returns.
         - This is in the same style as returns.
    rolling_window : int, optional
        The days window over which to compute the sharpe ratio.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(utils.two_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    rolling_sharpe_ts = timeseries.rolling_sharpe(
        returns, rolling_window)
    rolling_sharpe_ts.plot(alpha=.7, lw=3, color='orangered', ax=ax,
                           **kwargs)

    if factor_returns is not None:
        rolling_sharpe_ts_factor = timeseries.rolling_sharpe(
            factor_returns, rolling_window)
        rolling_sharpe_ts_factor.plot(alpha=.7, lw=3, color='grey', ax=ax,
                                      **kwargs)

    ax.set_title('Rolling Sharpe ratio (6-month)')
    ax.axhline(
        rolling_sharpe_ts.mean(),
        color='steelblue',
        linestyle='--',
        lw=3)
    ax.axhline(0.0, color='black', linestyle='-', lw=3)

    ax.set_ylabel('Sharpe ratio')
    ax.set_xlabel('')
    if factor_returns is None:
        ax.legend(['Sharpe', 'Average'],
                  loc=legend_loc, frameon=True, framealpha=0.5)
    else:
        ax.legend(['Sharpe', 'Benchmark Sharpe', 'Average'],
                  loc=legend_loc, frameon=True, framealpha=0.5)

    return ax


def plot_gross_leverage(returns, positions, ax=None, **kwargs):
    """
    Plots gross leverage versus date.

    Gross leverage is the sum of long and short exposure per share
    divided by net asset value.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    positions : pd.DataFrame
        Daily net position values.
         - See full explanation in create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()
    gl = timeseries.gross_lev(positions)
    gl.plot(lw=0.5, color='limegreen', legend=False, ax=ax, **kwargs)

    ax.axhline(gl.mean(), color='g', linestyle='--', lw=3)

    ax.set_title('Gross leverage')
    ax.set_ylabel('Gross leverage')
    ax.set_xlabel('')
    return ax


def plot_exposures(returns, positions, ax=None, **kwargs):
    """
    Plots a cake chart of the long and short exposure.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    positions_alloc : pd.DataFrame
        Portfolio allocation of positions. See
        pos.get_percent_alloc.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    pos_no_cash = positions.drop('cash', axis=1)
    l_exp = pos_no_cash[pos_no_cash > 0].sum(axis=1) / positions.sum(axis=1)
    s_exp = pos_no_cash[pos_no_cash < 0].sum(axis=1) / positions.sum(axis=1)
    net_exp = pos_no_cash.sum(axis=1) / positions.sum(axis=1)

    ax.fill_between(l_exp.index,
                    0,
                    l_exp.values,
                    label='Long', color='green', alpha=0.5)
    ax.fill_between(s_exp.index,
                    0,
                    s_exp.values,
                    label='Short', color='red', alpha=0.5)
    ax.plot(net_exp.index, net_exp.values,
            label='Net', color='black', linestyle='dotted')

    ax.set_xlim((returns.index[0], returns.index[-1]))
    ax.set_title("Exposure")
    ax.set_ylabel('Exposure')
    ax.legend(loc='lower left', frameon=True, framealpha=0.5)
    ax.set_xlabel('')
    return ax


def show_and_plot_top_positions(returns, positions_alloc,
                                show_and_plot=2, hide_positions=False,
                                legend_loc='real_best', ax=None,
                                **kwargs):
    """
    Prints and/or plots the exposures of the top 10 held positions of
    all time.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    positions_alloc : pd.DataFrame
        Portfolio allocation of positions. See pos.get_percent_alloc.
    show_and_plot : int, optional
        By default, this is 2, and both prints and plots.
        If this is 0, it will only plot; if 1, it will only print.
    hide_positions : bool, optional
        If True, will not output any symbol names.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
        By default, the legend will display below the plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes, conditional
        The axes that were plotted on.

    """
    positions_alloc = positions_alloc.copy()
    positions_alloc.columns = positions_alloc.columns.map(utils.format_asset)

    df_top_long, df_top_short, df_top_abs = pos.get_top_long_short_abs(
        positions_alloc)

    if show_and_plot == 1 or show_and_plot == 2:
        utils.print_table(pd.DataFrame(df_top_long * 100, columns=['max']),
                          float_format='{0:.2f}%'.format,
                          name='Top 10 long positions of all time')

        utils.print_table(pd.DataFrame(df_top_short * 100, columns=['max']),
                          float_format='{0:.2f}%'.format,
                          name='Top 10 short positions of all time')

        utils.print_table(pd.DataFrame(df_top_abs * 100, columns=['max']),
                          float_format='{0:.2f}%'.format,
                          name='Top 10 positions of all time')

    if show_and_plot == 0 or show_and_plot == 2:

        if ax is None:
            ax = plt.gca()

        positions_alloc[df_top_abs.index].plot(
            title='Portfolio allocation over time, only top 10 holdings',
            alpha=0.5, ax=ax, **kwargs)

        # Place legend below plot, shrink plot by 20%
        if legend_loc == 'real_best':
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                             box.width, box.height * 0.9])

            # Put a legend below current axis
            ax.legend(loc='upper center', frameon=True, framealpha=0.5,
                      bbox_to_anchor=(0.5, -0.14), ncol=5)
        else:
            ax.legend(loc=legend_loc)

        ax.set_xlim((returns.index[0], returns.index[-1]))
        ax.set_ylabel('Exposure by holding')

        if hide_positions:
            ax.legend_.remove()

        return ax


def plot_max_median_position_concentration(positions, ax=None, **kwargs):
    """
    Plots the max and median of long and short position concentrations
    over the time.

    Parameters
    ----------
    positions : pd.DataFrame
        The positions that the strategy takes over time.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    alloc_summary = pos.get_max_median_position_concentration(positions)
    colors = ['mediumblue', 'steelblue', 'tomato', 'firebrick']
    alloc_summary.plot(linewidth=1, color=colors, alpha=0.6, ax=ax)

    ax.legend(loc='center left', frameon=True, framealpha=0.5)
    ax.set_ylabel('Exposure')
    ax.set_title('Long/short max and median position concentration')

    return ax


def plot_sector_allocations(returns, sector_alloc, ax=None, **kwargs):
    """
    Plots the sector exposures of the portfolio over time.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    sector_alloc : pd.DataFrame
        Portfolio allocation of positions. See pos.get_sector_alloc.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    sector_alloc.plot(title='Sector allocation over time',
                      alpha=0.5, ax=ax, **kwargs)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', frameon=True, framealpha=0.5,
              bbox_to_anchor=(0.5, -0.14), ncol=5)

    ax.set_xlim((sector_alloc.index[0], sector_alloc.index[-1]))
    ax.set_ylabel('Exposure by sector')
    ax.set_xlabel('')

    return ax


def plot_return_quantiles(returns, live_start_date=None, ax=None, **kwargs):
    """
    Creates a box plot of daily, weekly, and monthly return
    distributions.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    live_start_date : datetime, optional
        The point in time when the strategy began live trading, after
        its backtest period.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to seaborn plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    is_returns = returns if live_start_date is None \
        else returns.loc[returns.index < live_start_date]
    is_weekly = ep.aggregate_returns(is_returns, 'weekly')
    is_monthly = ep.aggregate_returns(is_returns, 'monthly')
    sns.boxplot(data=[is_returns, is_weekly, is_monthly],
                palette=["#4c72B0", "#55A868", "#CCB974"],
                ax=ax, **kwargs)

    if live_start_date is not None:
        oos_returns = returns.loc[returns.index >= live_start_date]
        oos_weekly = ep.aggregate_returns(oos_returns, 'weekly')
        oos_monthly = ep.aggregate_returns(oos_returns, 'monthly')

        sns.swarmplot(data=[oos_returns, oos_weekly, oos_monthly], ax=ax,
                      color="red",
                      marker="d", **kwargs)
        red_dots = matplotlib.lines.Line2D([], [], color="red", marker="d",
                                           label="Out-of-sample data",
                                           linestyle='')
        ax.legend(handles=[red_dots], frameon=True, framealpha=0.5)
    ax.set_xticklabels(['Daily', 'Weekly', 'Monthly'])
    ax.set_title('Return quantiles')

    return ax


def plot_turnover(returns, transactions, positions, turnover_denom='AGB',
                  legend_loc='best', ax=None, **kwargs):
    """
    Plots turnover vs. date.

    Turnover is the number of shares traded for a period as a fraction
    of total shares.

    Displays daily total, daily average per month, and all-time daily
    average.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
         - See full explanation in tears.create_full_tear_sheet.
    positions : pd.DataFrame
        Daily net position values.
         - See full explanation in tears.create_full_tear_sheet.
    turnover_denom : str, optional
        Either AGB or portfolio_value, default AGB.
        - See full explanation in txn.get_turnover.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(utils.two_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    df_turnover = txn.get_turnover(positions, transactions, turnover_denom)
    df_turnover_by_month = df_turnover.resample("M").mean()
    df_turnover.plot(color='steelblue', alpha=1.0, lw=0.5, ax=ax, **kwargs)
    df_turnover_by_month.plot(
        color='orangered',
        alpha=0.5,
        lw=2,
        ax=ax,
        **kwargs)
    ax.axhline(
        df_turnover.mean(), color='steelblue', linestyle='--', lw=3, alpha=1.0)
    ax.legend(['Daily turnover',
               'Average daily turnover, by month',
               'Average daily turnover, net'],
              loc=legend_loc, frameon=True, framealpha=0.5)
    ax.set_title('Daily turnover')
    ax.set_xlim((returns.index[0], returns.index[-1]))
    ax.set_ylim((0, 2))
    ax.set_ylabel('Turnover')
    ax.set_xlabel('')
    return ax


def plot_slippage_sweep(returns, positions, transactions,
                        slippage_params=(3, 8, 10, 12, 15, 20, 50),
                        ax=None, **kwargs):
    """
    Plots equity curves at different per-dollar slippage assumptions.

    Parameters
    ----------
    returns : pd.Series
        Timeseries of portfolio returns to be adjusted for various
        degrees of slippage.
    positions : pd.DataFrame
        Daily net position values.
         - See full explanation in tears.create_full_tear_sheet.
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
         - See full explanation in tears.create_full_tear_sheet.
    slippage_params: tuple
        Slippage pameters to apply to the return time series (in
        basis points).
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to seaborn plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    slippage_sweep = pd.DataFrame()
    for bps in slippage_params:
        adj_returns = txn.adjust_returns_for_slippage(returns, positions,
                                                      transactions, bps)
        label = str(bps) + " bps"
        slippage_sweep[label] = ep.cum_returns(adj_returns, 1)

    slippage_sweep.plot(alpha=1.0, lw=0.5, ax=ax)

    ax.set_title('Cumulative returns given additional per-dollar slippage')
    ax.set_ylabel('')

    ax.legend(loc='center left', frameon=True, framealpha=0.5)

    return ax


def plot_slippage_sensitivity(returns, positions, transactions,
                              ax=None, **kwargs):
    """
    Plots curve relating per-dollar slippage to average annual returns.

    Parameters
    ----------
    returns : pd.Series
        Timeseries of portfolio returns to be adjusted for various
        degrees of slippage.
    positions : pd.DataFrame
        Daily net position values.
         - See full explanation in tears.create_full_tear_sheet.
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
         - See full explanation in tears.create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to seaborn plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    avg_returns_given_slippage = pd.Series()
    for bps in range(1, 100):
        adj_returns = txn.adjust_returns_for_slippage(returns, positions,
                                                      transactions, bps)
        avg_returns = ep.annual_return(adj_returns)
        avg_returns_given_slippage.loc[bps] = avg_returns

    avg_returns_given_slippage.plot(alpha=1.0, lw=2, ax=ax)

    ax.set_title('Average annual returns given additional per-dollar slippage')
    ax.set_xticks(np.arange(0, 100, 10))
    ax.set_ylabel('Average annual return')
    ax.set_xlabel('Per-dollar slippage (bps)')

    return ax


def plot_capacity_sweep(returns, transactions, market_data,
                        bt_starting_capital,
                        min_pv=100000,
                        max_pv=300000000,
                        step_size=1000000,
                        ax=None):
    txn_daily_w_bar = capacity.daily_txns_with_bar_data(transactions,
                                                        market_data)

    captial_base_sweep = pd.Series()
    for start_pv in range(min_pv, max_pv, step_size):
        adj_ret = capacity.apply_slippage_penalty(returns,
                                                  txn_daily_w_bar,
                                                  start_pv,
                                                  bt_starting_capital)
        sharpe = ep.sharpe_ratio(adj_ret)
        if sharpe < -1:
            break
        captial_base_sweep.loc[start_pv] = sharpe
    captial_base_sweep.index = captial_base_sweep.index / MM_DISPLAY_UNIT

    if ax is None:
        ax = plt.gca()

    captial_base_sweep.plot(ax=ax)
    ax.set_xlabel('Capital base ($mm)')
    ax.set_ylabel('Sharpe ratio')
    ax.set_title('Capital base performance sweep')

    return ax


def plot_daily_turnover_hist(transactions, positions, turnover_denom='AGB',
                             ax=None, **kwargs):
    """
    Plots a histogram of daily turnover rates.

    Parameters
    ----------
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
         - See full explanation in tears.create_full_tear_sheet.
    positions : pd.DataFrame
        Daily net position values.
         - See full explanation in tears.create_full_tear_sheet.
    turnover_denom : str, optional
        Either AGB or portfolio_value, default AGB.
        - See full explanation in txn.get_turnover.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to seaborn plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()
    turnover = txn.get_turnover(positions, transactions, turnover_denom)
    sns.distplot(turnover, ax=ax, **kwargs)
    ax.set_title('Distribution of daily turnover rates')
    ax.set_xlabel('Turnover rate')
    return ax


def plot_daily_volume(returns, transactions, ax=None, **kwargs):
    """
    Plots trading volume per day vs. date.

    Also displays all-time daily average.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
         - See full explanation in tears.create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()
    daily_txn = txn.get_txn_vol(transactions)
    daily_txn.txn_shares.plot(alpha=1.0, lw=0.5, ax=ax, **kwargs)
    ax.axhline(daily_txn.txn_shares.mean(), color='steelblue',
               linestyle='--', lw=3, alpha=1.0)
    ax.set_title('Daily trading volume')
    ax.set_xlim((returns.index[0], returns.index[-1]))
    ax.set_ylabel('Amount of shares traded')
    ax.set_xlabel('')
    return ax


def plot_txn_time_hist(transactions, bin_minutes=5, tz='America/New_York',
                       ax=None, **kwargs):
    """
    Plots a histogram of transaction times, binning the times into
    buckets of a given duration.

    Parameters
    ----------
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
         - See full explanation in tears.create_full_tear_sheet.
    bin_minutes : float, optional
        Sizes of the bins in minutes, defaults to 5 minutes.
    tz : str, optional
        Time zone to plot against. Note that if the specified
        zone does not apply daylight savings, the distribution
        may be partially offset.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    txn_time = transactions.copy()

    txn_time.index = txn_time.index.tz_convert(pytz.timezone(tz))
    txn_time.index = txn_time.index.map(lambda x: x.hour * 60 + x.minute)
    txn_time['trade_value'] = (txn_time.amount * txn_time.price).abs()
    txn_time = txn_time.groupby(level=0).sum().reindex(index=range(570, 961))
    txn_time.index = (txn_time.index / bin_minutes).astype(int) * bin_minutes
    txn_time = txn_time.groupby(level=0).sum()

    txn_time['time_str'] = txn_time.index.map(lambda x:
                                              str(datetime.time(int(x / 60),
                                                                x % 60))[:-3])

    trade_value_sum = txn_time.trade_value.sum()
    txn_time.trade_value = txn_time.trade_value.fillna(0) / trade_value_sum

    ax.bar(txn_time.index, txn_time.trade_value, width=bin_minutes, **kwargs)

    ax.set_xlim(570, 960)
    ax.set_xticks(txn_time.index[::int(30 / bin_minutes)])
    ax.set_xticklabels(txn_time.time_str[::int(30 / bin_minutes)])
    ax.set_title('Transaction time distribution')
    ax.set_ylabel('Proportion')
    ax.set_xlabel('')
    return ax


def show_worst_drawdown_periods(returns, top=5):
    """
    Prints information about the worst drawdown periods.

    Prints peak dates, valley dates, recovery dates, and net
    drawdowns.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    top : int, optional
        Amount of top drawdowns periods to plot (default 5).
    """

    drawdown_df = timeseries.gen_drawdown_table(returns, top=top)
    utils.print_table(
        drawdown_df.sort_values('Net drawdown in %', ascending=False),
        name='Worst drawdown periods',
        float_format='{0:.2f}'.format,
    )


def plot_monthly_returns_timeseries(returns, ax=None, **kwargs):
    """
    Plots monthly returns as a timeseries.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to seaborn plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    def cumulate_returns(x):
        return ep.cum_returns(x)[-1]

    if ax is None:
        ax = plt.gca()

    monthly_rets = returns.resample('M').apply(lambda x: cumulate_returns(x))
    monthly_rets = monthly_rets.to_period()

    sns.barplot(x=monthly_rets.index,
                y=monthly_rets.values,
                color='steelblue')

    _, labels = plt.xticks()
    plt.setp(labels, rotation=90)

    # only show x-labels on year boundary
    xticks_coord = []
    xticks_label = []
    count = 0
    for i in monthly_rets.index:
        if i.month == 1:
            xticks_label.append(i)
            xticks_coord.append(count)
            # plot yearly boundary line
            ax.axvline(count, color='gray', ls='--', alpha=0.3)

        count += 1

    ax.axhline(0.0, color='darkgray', ls='-')
    ax.set_xticks(xticks_coord)
    ax.set_xticklabels(xticks_label)

    return ax


def plot_round_trip_lifetimes(round_trips, disp_amount=16, lsize=18, ax=None):
    """
    Plots timespans and directions of a sample of round trip trades.

    Parameters
    ----------
    round_trips : pd.DataFrame
        DataFrame with one row per round trip trade.
        - See full explanation in round_trips.extract_round_trips
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.subplot()

    symbols_sample = round_trips.symbol.unique()
    np.random.seed(1)
    sample = np.random.choice(round_trips.symbol.unique(), replace=False,
                              size=min(disp_amount, len(symbols_sample)))
    sample_round_trips = round_trips[round_trips.symbol.isin(sample)]

    symbol_idx = pd.Series(np.arange(len(sample)), index=sample)

    for symbol, sym_round_trips in sample_round_trips.groupby('symbol'):
        for _, row in sym_round_trips.iterrows():
            c = 'b' if row.long else 'r'
            y_ix = symbol_idx[symbol] + 0.05
            ax.plot([row['open_dt'], row['close_dt']],
                    [y_ix, y_ix], color=c,
                    linewidth=lsize, solid_capstyle='butt')

    ax.set_yticks(range(disp_amount))
    ax.set_yticklabels([utils.format_asset(s) for s in sample])

    ax.set_ylim((-0.5, min(len(sample), disp_amount) - 0.5))
    blue = patches.Rectangle([0, 0], 1, 1, color='b', label='Long')
    red = patches.Rectangle([0, 0], 1, 1, color='r', label='Short')
    leg = ax.legend(handles=[blue, red], loc='lower left',
                    frameon=True, framealpha=0.5)
    leg.get_frame().set_edgecolor('black')
    ax.grid(False)

    return ax


def show_profit_attribution(round_trips):
    """
    Prints the share of total PnL contributed by each
    traded name.

    Parameters
    ----------
    round_trips : pd.DataFrame
        DataFrame with one row per round trip trade.
        - See full explanation in round_trips.extract_round_trips
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    total_pnl = round_trips['pnl'].sum()
    pnl_attribution = round_trips.groupby('symbol')['pnl'].sum() / total_pnl
    pnl_attribution.name = ''

    pnl_attribution.index = pnl_attribution.index.map(utils.format_asset)
    utils.print_table(
        pnl_attribution.sort_values(
            inplace=False,
            ascending=False,
        ),
        name='Profitability (PnL / PnL total) per name',
        float_format='{:.2%}'.format,
    )


def plot_prob_profit_trade(round_trips, ax=None):
    """
    Plots a probability distribution for the event of making
    a profitable trade.

    Parameters
    ----------
    round_trips : pd.DataFrame
        DataFrame with one row per round trip trade.
        - See full explanation in round_trips.extract_round_trips
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    x = np.linspace(0, 1., 500)

    round_trips['profitable'] = round_trips.pnl > 0

    dist = sp.stats.beta(round_trips.profitable.sum(),
                         (~round_trips.profitable).sum())
    y = dist.pdf(x)
    lower_perc = dist.ppf(.025)
    upper_perc = dist.ppf(.975)

    lower_plot = dist.ppf(.001)
    upper_plot = dist.ppf(.999)

    if ax is None:
        ax = plt.subplot()

    ax.plot(x, y)
    ax.axvline(lower_perc, color='0.5')
    ax.axvline(upper_perc, color='0.5')

    ax.set_xlabel('Probability of making a profitable decision')
    ax.set_ylabel('Belief')
    ax.set_xlim(lower_plot, upper_plot)
    ax.set_ylim((0, y.max() + 1.))

    return ax


def plot_cones(name, bounds, oos_returns, num_samples=1000, ax=None,
               cone_std=(1., 1.5, 2.), random_seed=None, num_strikes=3):
    """
    Plots the upper and lower bounds of an n standard deviation
    cone of forecasted cumulative returns. Redraws a new cone when
    cumulative returns fall outside of last cone drawn.

    Parameters
    ----------
    name : str
        Account name to be used as figure title.
    bounds : pandas.core.frame.DataFrame
        Contains upper and lower cone boundaries. Column names are
        strings corresponding to the number of standard devations
        above (positive) or below (negative) the projected mean
        cumulative returns.
    oos_returns : pandas.core.frame.DataFrame
        Non-cumulative out-of-sample returns.
    num_samples : int
        Number of samples to draw from the in-sample daily returns.
        Each sample will be an array with length num_days.
        A higher number of samples will generate a more accurate
        bootstrap cone.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    cone_std : list of int/float
        Number of standard devations to use in the boundaries of
        the cone. If multiple values are passed, cone bounds will
        be generated for each value.
    random_seed : int
        Seed for the pseudorandom number generator used by the pandas
        sample method.
    num_strikes : int
        Upper limit for number of cones drawn. Can be anything from 0 to 3.

    Returns
    -------
    Returns are either an ax or fig option, but not both. If a
    matplotlib.Axes instance is passed in as ax, then it will be modified
    and returned. This allows for users to plot interactively in jupyter
    notebook. When no ax object is passed in, a matplotlib.figure instance
    is generated and returned. This figure can then be used to save
    the plot as an image without viewing it.

    ax : matplotlib.Axes
        The axes that were plotted on.
    fig : matplotlib.figure
        The figure instance which contains all the plot elements.
    """

    if ax is None:
        fig = figure.Figure(figsize=(10, 8))
        FigureCanvasAgg(fig)
        axes = fig.add_subplot(111)
    else:
        axes = ax

    returns = ep.cum_returns(oos_returns, starting_value=1.)
    bounds_tmp = bounds.copy()
    returns_tmp = returns.copy()
    cone_start = returns.index[0]
    colors = ["green", "orange", "orangered", "darkred"]

    for c in range(num_strikes + 1):
        if c > 0:
            tmp = returns.loc[cone_start:]
            bounds_tmp = bounds_tmp.iloc[0:len(tmp)]
            bounds_tmp = bounds_tmp.set_index(tmp.index)
            crossing = (tmp < bounds_tmp[float(-2.)].iloc[:len(tmp)])
            if crossing.sum() <= 0:
                break
            cone_start = crossing.loc[crossing].index[0]
            returns_tmp = returns.loc[cone_start:]
            bounds_tmp = (bounds - (1 - returns.loc[cone_start]))
        for std in cone_std:
            x = returns_tmp.index
            y1 = bounds_tmp[float(std)].iloc[:len(returns_tmp)]
            y2 = bounds_tmp[float(-std)].iloc[:len(returns_tmp)]
            axes.fill_between(x, y1, y2, color=colors[c], alpha=0.5)

    # Plot returns line graph
    label = 'Cumulative returns = {:.2f}%'.format((returns.iloc[-1] - 1) * 100)
    axes.plot(returns.index, returns.values, color='black', lw=3.,
              label=label)

    if name is not None:
        axes.set_title(name)
    axes.axhline(1, color='black', alpha=0.2)
    axes.legend(frameon=True, framealpha=0.5)

    if ax is None:
        return fig
    else:
        return axes

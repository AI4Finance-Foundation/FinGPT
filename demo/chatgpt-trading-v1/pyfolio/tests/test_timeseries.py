from __future__ import division

import os
from unittest import TestCase
from nose_parameterized import parameterized
from numpy.testing import assert_allclose, assert_almost_equal
from pandas.util.testing import assert_series_equal

import numpy as np
import pandas as pd

from .. import timeseries
from pyfolio.utils import to_utc, to_series
import gzip


DECIMAL_PLACES = 8


class TestDrawdown(TestCase):
    drawdown_list = np.array(
        [100, 90, 75]
    ) / 10.
    dt = pd.date_range('2000-1-3', periods=3, freq='D')

    drawdown_serie = pd.Series(drawdown_list, index=dt)

    @parameterized.expand([
        (drawdown_serie,)
    ])
    def test_get_max_drawdown_begins_first_day(self, px):
        rets = px.pct_change()
        drawdowns = timeseries.gen_drawdown_table(rets, top=1)
        self.assertEqual(drawdowns.loc[0, 'Net drawdown in %'], 25)

    drawdown_list = np.array(
        [100, 110, 120, 150, 180, 200, 100, 120,
         160, 180, 200, 300, 400, 500, 600, 800,
         900, 1000, 650, 600]
    ) / 10.
    dt = pd.date_range('2000-1-3', periods=20, freq='D')

    drawdown_serie = pd.Series(drawdown_list, index=dt)

    @parameterized.expand([
        (drawdown_serie,
         pd.Timestamp('2000-01-08'),
         pd.Timestamp('2000-01-09'),
         pd.Timestamp('2000-01-13'),
         50,
         pd.Timestamp('2000-01-20'),
         pd.Timestamp('2000-01-22'),
         None,
         40
         )
    ])
    def test_gen_drawdown_table_relative(
            self, px,
            first_expected_peak, first_expected_valley,
            first_expected_recovery, first_net_drawdown,
            second_expected_peak, second_expected_valley,
            second_expected_recovery, second_net_drawdown
    ):

        rets = px.pct_change()

        drawdowns = timeseries.gen_drawdown_table(rets, top=2)

        self.assertEqual(np.round(drawdowns.loc[0, 'Net drawdown in %']),
                         first_net_drawdown)
        self.assertEqual(drawdowns.loc[0, 'Peak date'],
                         first_expected_peak)
        self.assertEqual(drawdowns.loc[0, 'Valley date'],
                         first_expected_valley)
        self.assertEqual(drawdowns.loc[0, 'Recovery date'],
                         first_expected_recovery)

        self.assertEqual(np.round(drawdowns.loc[1, 'Net drawdown in %']),
                         second_net_drawdown)
        self.assertEqual(drawdowns.loc[1, 'Peak date'],
                         second_expected_peak)
        self.assertEqual(drawdowns.loc[1, 'Valley date'],
                         second_expected_valley)
        self.assertTrue(pd.isnull(drawdowns.loc[1, 'Recovery date']))

    px_list_1 = np.array(
        [100, 120, 100, 80, 70, 110, 180, 150]) / 100.  # Simple
    px_list_2 = np.array(
        [100, 120, 100, 80, 70, 80, 90, 90]) / 100.  # Ends in drawdown
    dt = pd.date_range('2000-1-3', periods=8, freq='D')

    @parameterized.expand([
        (pd.Series(px_list_1,
                   index=dt),
         pd.Timestamp('2000-1-4'),
         pd.Timestamp('2000-1-7'),
         pd.Timestamp('2000-1-9')),
        (pd.Series(px_list_2,
                   index=dt),
         pd.Timestamp('2000-1-4'),
         pd.Timestamp('2000-1-7'),
         None)
    ])
    def test_get_max_drawdown(
            self, px, expected_peak, expected_valley, expected_recovery):
        rets = px.pct_change().iloc[1:]

        peak, valley, recovery = timeseries.get_max_drawdown(rets)
        # Need to use isnull because the result can be NaN, NaT, etc.
        self.assertTrue(
            pd.isnull(peak)) if expected_peak is None else self.assertEqual(
                peak,
                expected_peak)
        self.assertTrue(
            pd.isnull(valley)) if expected_valley is None else \
            self.assertEqual(
                valley,
                expected_valley)
        self.assertTrue(
            pd.isnull(recovery)) if expected_recovery is None else \
            self.assertEqual(
                recovery,
                expected_recovery)

    @parameterized.expand([
        (pd.Series(px_list_2,
                   index=dt),
         pd.Timestamp('2000-1-4'),
         pd.Timestamp('2000-1-7'),
         None,
         None),
        (pd.Series(px_list_1,
                   index=dt),
         pd.Timestamp('2000-1-4'),
         pd.Timestamp('2000-1-7'),
         pd.Timestamp('2000-1-9'),
         4)
    ])
    def test_gen_drawdown_table(self, px, expected_peak,
                                expected_valley, expected_recovery,
                                expected_duration):
        rets = px.pct_change().iloc[1:]

        drawdowns = timeseries.gen_drawdown_table(rets, top=1)
        self.assertTrue(
            pd.isnull(
                drawdowns.loc[
                    0,
                    'Peak date'])) if expected_peak is None \
            else self.assertEqual(drawdowns.loc[0, 'Peak date'],
                                  expected_peak)
        self.assertTrue(
            pd.isnull(
                drawdowns.loc[0, 'Valley date'])) \
            if expected_valley is None else self.assertEqual(
                drawdowns.loc[0, 'Valley date'],
                expected_valley)
        self.assertTrue(
            pd.isnull(
                drawdowns.loc[0, 'Recovery date'])) \
            if expected_recovery is None else self.assertEqual(
                drawdowns.loc[0, 'Recovery date'],
                expected_recovery)
        self.assertTrue(
            pd.isnull(drawdowns.loc[0, 'Duration'])) \
            if expected_duration is None else self.assertEqual(
                drawdowns.loc[0, 'Duration'], expected_duration)

    def test_drawdown_overlaps(self):
        rand = np.random.RandomState(1337)
        n_samples = 252 * 5
        spy_returns = pd.Series(
            rand.standard_t(3.1, n_samples),
            pd.date_range('2005-01-02', periods=n_samples),
        )
        spy_drawdowns = timeseries.gen_drawdown_table(
            spy_returns,
            top=20).sort_values(by='Peak date')
        # Compare the recovery date of each drawdown with the peak of the next
        # Last pair might contain a NaT if drawdown didn't finish, so ignore it
        pairs = list(zip(spy_drawdowns['Recovery date'],
                         spy_drawdowns['Peak date'].shift(-1)))[:-1]
        self.assertGreater(len(pairs), 0)
        for recovery, peak in pairs:
            if not pd.isnull(recovery):
                self.assertLessEqual(recovery, peak)

    @parameterized.expand([
        (pd.Series(px_list_1,
                   index=dt),
         1,
         [(pd.Timestamp('2000-01-03 00:00:00'),
           pd.Timestamp('2000-01-03 00:00:00'),
           pd.Timestamp('2000-01-03 00:00:00'))])
    ])
    def test_top_drawdowns(self, returns, top, expected):
        self.assertEqual(
            timeseries.get_top_drawdowns(
                returns,
                top=top),
            expected)


class TestVariance(TestCase):

    @parameterized.expand([
        (1e7, 0.5, 1, 1, -10000000.0)
    ])
    def test_var_cov_var_normal(self, P, c, mu, sigma, expected):
        self.assertEqual(
            timeseries.var_cov_var_normal(
                P,
                c,
                mu,
                sigma),
            expected)


class TestNormalize(TestCase):
    dt = pd.date_range('2000-1-3', periods=8, freq='D')
    px_list = [1.0, 1.2, 1.0, 0.8, 0.7, 0.8, 0.8, 0.8]

    @parameterized.expand([
        (pd.Series(np.array(px_list) * 100, index=dt),
         pd.Series(px_list, index=dt))
    ])
    def test_normalize(self, returns, expected):
        self.assertTrue(timeseries.normalize(returns).equals(expected))


class TestStats(TestCase):
    simple_rets = pd.Series(
        [0.1] * 3 + [0] * 497,
        pd.date_range(
            '2000-1-3',
            periods=500,
            freq='D'))

    simple_week_rets = pd.Series(
        [0.1] * 3 + [0] * 497,
        pd.date_range(
            '2000-1-31',
            periods=500,
            freq='W'))

    simple_month_rets = pd.Series(
        [0.1] * 3 + [0] * 497,
        pd.date_range(
            '2000-1-31',
            periods=500,
            freq='M'))

    simple_benchmark = pd.Series(
        [0.03] * 4 + [0] * 496,
        pd.date_range(
            '2000-1-1',
            periods=500,
            freq='D'))
    px_list = np.array(
        [10, -10, 10]) / 100.  # Ends in drawdown
    dt = pd.date_range('2000-1-3', periods=3, freq='D')

    px_list_2 = [1.0, 1.2, 1.0, 0.8, 0.7, 0.8, 0.8, 0.8]
    dt_2 = pd.date_range('2000-1-3', periods=8, freq='D')

    @parameterized.expand([
        (simple_rets[:5], 2, [np.nan, np.inf, np.inf, 11.224972160321, np.inf])
    ])
    def test_sharpe_2(self, returns, rolling_sharpe_window, expected):
        np.testing.assert_array_almost_equal(
            timeseries.rolling_sharpe(returns,
                                      rolling_sharpe_window).values,
            np.asarray(expected))

    @parameterized.expand([
        (simple_rets[:5], simple_benchmark, 2, 0)
    ])
    def test_beta(self, returns, benchmark_rets, rolling_window, expected):
        actual = timeseries.rolling_beta(
            returns,
            benchmark_rets,
            rolling_window=rolling_window,
        ).values.tolist()[2]

        np.testing.assert_almost_equal(actual, expected)


class TestCone(TestCase):
    def test_bootstrap_cone_against_linear_cone_normal_returns(self):
        random_seed = 100
        np.random.seed(random_seed)
        days_forward = 200
        cone_stdevs = (1., 1.5, 2.)
        mu = .005
        sigma = .002
        rets = pd.Series(np.random.normal(mu, sigma, 10000))

        midline = np.cumprod(1 + (rets.mean() * np.ones(days_forward)))
        stdev = rets.std() * midline * np.sqrt(np.arange(days_forward)+1)

        normal_cone = pd.DataFrame(columns=pd.Float64Index([]))
        for s in cone_stdevs:
            normal_cone[s] = midline + s * stdev
            normal_cone[-s] = midline - s * stdev

        bootstrap_cone = timeseries.forecast_cone_bootstrap(
            rets, days_forward, cone_stdevs, starting_value=1,
            random_seed=random_seed, num_samples=10000)

        for col, vals in bootstrap_cone.iteritems():
            expected = normal_cone[col].values
            assert_allclose(vals.values, expected, rtol=.005)


class TestBootstrap(TestCase):
    @parameterized.expand([
        (0., 1., 1000),
        (1., 2., 500),
        (-1., 0.1, 10),
    ])
    def test_calc_bootstrap(self, true_mean, true_sd, n):
        """Compare bootstrap distribution of the mean to sampling distribution
        of the mean.

        """
        np.random.seed(123)
        func = np.mean
        returns = pd.Series((np.random.randn(n) * true_sd) +
                            true_mean)

        samples = timeseries.calc_bootstrap(func, returns,
                                            n_samples=10000)

        # Calculate statistics of sampling distribution of the mean
        mean_of_mean = np.mean(returns)
        sd_of_mean = np.std(returns) / np.sqrt(n)

        assert_almost_equal(
            np.mean(samples),
            mean_of_mean,
            3,
            'Mean of bootstrap does not match theoretical mean of'
            'sampling distribution')

        assert_almost_equal(
            np.std(samples),
            sd_of_mean,
            3,
            'SD of bootstrap does not match theoretical SD of'
            'sampling distribution')


class TestGrossLev(TestCase):
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    test_pos = to_utc(pd.read_csv(
        gzip.open(__location__ + '/test_data/test_pos.csv.gz'),
        index_col=0, parse_dates=True))
    test_gross_lev = pd.read_csv(
        gzip.open(
            __location__ + '/test_data/test_gross_lev.csv.gz'),
        index_col=0, parse_dates=True)
    test_gross_lev = to_series(to_utc(test_gross_lev))

    def test_gross_lev_calculation(self):
        assert_series_equal(
            timeseries.gross_lev(self.test_pos)['2004-02-01':],
            self.test_gross_lev['2004-02-01':], check_names=False)

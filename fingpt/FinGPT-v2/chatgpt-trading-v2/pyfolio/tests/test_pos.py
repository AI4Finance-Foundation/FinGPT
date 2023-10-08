from unittest import TestCase
from nose_parameterized import parameterized
from collections import OrderedDict
import os
import gzip

from pandas import (
    Series,
    DataFrame,
    date_range,
    Timestamp,
    read_csv
)
from pandas.util.testing import assert_frame_equal

from numpy import (
    arange,
    zeros_like,
    nan,
)

import warnings

from pyfolio.utils import (to_utc, to_series, check_intraday,
                           detect_intraday, estimate_intraday)
from pyfolio.pos import (get_percent_alloc,
                         extract_pos,
                         get_sector_exposures,
                         get_max_median_position_concentration)


class PositionsTestCase(TestCase):
    dates = date_range(start='2015-01-01', freq='D', periods=20)

    def test_get_percent_alloc(self):
        raw_data = arange(15, dtype=float).reshape(5, 3)
        # Make the first column negative to test absolute magnitudes.
        raw_data[:, 0] *= -1

        frame = DataFrame(
            raw_data,
            index=date_range('01-01-2015', freq='D', periods=5),
            columns=['A', 'B', 'C']
        )

        result = get_percent_alloc(frame)
        expected_raw = zeros_like(raw_data)
        for idx, row in enumerate(raw_data):
            expected_raw[idx] = row / row.sum()

        expected = DataFrame(
            expected_raw,
            index=frame.index,
            columns=frame.columns,
        )

        assert_frame_equal(result, expected)

    def test_extract_pos(self):
        index_dup = [Timestamp('2015-06-08', tz='UTC'),
                     Timestamp('2015-06-08', tz='UTC'),
                     Timestamp('2015-06-09', tz='UTC'),
                     Timestamp('2015-06-09', tz='UTC')]
        index = [Timestamp('2015-06-08', tz='UTC'),
                 Timestamp('2015-06-09', tz='UTC')]

        positions = DataFrame(
            {'amount': [100., 200., 300., 400.],
             'last_sale_price': [10., 20., 30., 40.],
             'sid': [1, 2, 1, 2]},
            index=index_dup
        )
        cash = Series([100., 200.], index=index)

        result = extract_pos(positions, cash)

        expected = DataFrame(OrderedDict([
            (1, [100.*10., 300.*30.]),
            (2, [200.*20., 400.*40.]),
            ('cash', [100., 200.])]),
            index=index
        )
        expected.index.name = 'index'
        expected.columns.name = 'sid'

        assert_frame_equal(result, expected)

    @parameterized.expand([
        (DataFrame([[1.0, 2.0, 3.0, 10.0]]*len(dates),
                   columns=[0, 1, 2, 'cash'], index=dates),
         {0: 'A', 1: 'B', 2: 'A'},
         DataFrame([[4.0, 2.0, 10.0]]*len(dates),
                   columns=['A', 'B', 'cash'], index=dates),
         False),
        (DataFrame([[1.0, 2.0, 3.0, 10.0]]*len(dates),
                   columns=[0, 1, 2, 'cash'], index=dates),
         Series(index=[0, 1, 2], data=['A', 'B', 'A']),
         DataFrame([[4.0, 2.0, 10.0]]*len(dates),
                   columns=['A', 'B', 'cash'], index=dates),
         False),
        (DataFrame([[1.0, 2.0, 3.0, 10.0]]*len(dates),
                   columns=[0, 1, 2, 'cash'], index=dates),
         {0: 'A', 1: 'B'},
         DataFrame([[1.0, 2.0, 10.0]]*len(dates),
                   columns=['A', 'B', 'cash'], index=dates),
         True)
    ])
    def test_sector_exposure(self, positions, mapping,
                             expected_sector_exposure,
                             warning_expected):
        """
        Tests sector exposure mapping and rollup.

        """
        with warnings.catch_warnings(record=True) as w:
            result_sector_exposure = get_sector_exposures(positions,
                                                          mapping)

            assert_frame_equal(result_sector_exposure,
                               expected_sector_exposure)
            if warning_expected:
                self.assertEqual(len(w), 1)
            else:
                self.assertEqual(len(w), 0)

    @parameterized.expand([
        (DataFrame([[1.0, 2.0, 3.0, 14.0]]*len(dates),
                   columns=[0, 1, 2, 'cash'], index=dates),
         DataFrame([[0.15, 0.1, nan, nan]]*len(dates),
                   columns=['max_long', 'median_long',
                            'median_short', 'max_short'], index=dates)),
        (DataFrame([[1.0, -2.0, -13.0, 15.0]]*len(dates),
                   columns=[0, 1, 2, 'cash'], index=dates),
         DataFrame([[1.0, 1.0, -7.5, -13.0]]*len(dates),
                   columns=['max_long', 'median_long',
                            'median_short', 'max_short'], index=dates)),
        (DataFrame([[nan, 2.0, nan, 8.0]]*len(dates),
                   columns=[0, 1, 2, 'cash'], index=dates),
         DataFrame([[0.2, 0.2, nan, nan]]*len(dates),
                   columns=['max_long', 'median_long',
                            'median_short', 'max_short'], index=dates))
    ])
    def test_max_median_exposure(self, positions, expected):
        alloc_summary = get_max_median_position_concentration(positions)
        assert_frame_equal(expected, alloc_summary)

    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    test_returns = read_csv(
        gzip.open(
            __location__ + '/test_data/test_returns.csv.gz'),
        index_col=0, parse_dates=True)
    test_returns = to_series(to_utc(test_returns))
    test_txn = to_utc(read_csv(
        gzip.open(
            __location__ + '/test_data/test_txn.csv.gz'),
        index_col=0, parse_dates=True))
    test_pos = to_utc(read_csv(
        gzip.open(__location__ + '/test_data/test_pos.csv.gz'),
        index_col=0, parse_dates=True))

    @parameterized.expand([
        (test_pos, test_txn, False),
        (test_pos.resample('1W').last(), test_txn, True)
    ])
    def test_detect_intraday(self, positions, transactions, expected):
        detected = detect_intraday(positions, transactions, threshold=0.25)
        assert detected == expected

    @parameterized.expand([
        ('infer', test_returns, test_pos, test_txn, test_pos),
        (False, test_returns, test_pos, test_txn, test_pos)
    ])
    def test_check_intraday(self, estimate, returns,
                            positions, transactions, expected):
        detected = check_intraday(estimate, returns, positions, transactions)
        assert_frame_equal(detected, expected)

    @parameterized.expand([
        (test_returns, test_pos, test_txn, (1506, 8)),
        (test_returns, test_pos.resample('1W').last(), test_txn, (1819, 8))
    ])
    def test_estimate_intraday(self, returns, positions,
                               transactions, expected):
        intraday_pos = estimate_intraday(returns, positions, transactions)
        assert intraday_pos.shape == expected

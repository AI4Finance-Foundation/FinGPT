from nose_parameterized import parameterized

from unittest import TestCase

from pandas import (
    Series,
    DataFrame,
    DatetimeIndex,
    date_range,
    Timedelta,
    read_csv
)
from pandas.util.testing import (assert_frame_equal)

import os
import gzip

from pyfolio.round_trips import (extract_round_trips,
                                 add_closing_transactions,
                                 _groupby_consecutive,
                                 )


class RoundTripTestCase(TestCase):
    dates = date_range(start='2015-01-01', freq='D', periods=20)
    dates_intraday = date_range(start='2015-01-01',
                                freq='2BH', periods=8)

    @parameterized.expand([
        (DataFrame(data=[[2, 10., 'A'],
                         [2, 20., 'A'],
                         [-2, 20., 'A'],
                         [-2, 10., 'A'],
                         ],
                   columns=['amount', 'price', 'symbol'],
                   index=dates_intraday[:4]),
         DataFrame(data=[[4, 15., 'A'],
                         [-4, 15., 'A'],
                         ],
                   columns=['amount', 'price', 'symbol'],
                   index=dates_intraday[[0, 2]])
         .rename_axis('dt', axis='index')
         ),
        (DataFrame(data=[[2, 10., 'A'],
                         [2, 20., 'A'],
                         [2, 20., 'A'],
                         [2, 10., 'A'],
                         ],
                   columns=['amount', 'price', 'symbol'],
                   index=dates_intraday[[0, 1, 4, 5]]),
         DataFrame(data=[[4, 15., 'A'],
                         [4, 15., 'A'],
                         ],
                   columns=['amount', 'price', 'symbol'],
                   index=dates_intraday[[0, 4]])
         .rename_axis('dt', axis='index')
         ),
    ])
    def test_groupby_consecutive(self, transactions, expected):
        grouped_txn = _groupby_consecutive(transactions)
        assert_frame_equal(grouped_txn.sort_index(axis='columns'),
                           expected.sort_index(axis='columns'))

    @parameterized.expand([
        # Simple round-trip
        (DataFrame(data=[[2, 10., 'A'],
                         [-2, 15., 'A']],
                   columns=['amount', 'price', 'symbol'],
                   index=dates[:2]),
         DataFrame(data=[[dates[0], dates[1],
                          Timedelta(days=1), 10., .5,
                          True, 'A']],
                   columns=['open_dt', 'close_dt',
                            'duration', 'pnl', 'rt_returns',
                            'long', 'symbol'],
                   index=[0])
         ),
        # Round-trip with left-over txn that shouldn't be counted
        (DataFrame(data=[[2, 10., 'A'],
                         [2, 15., 'A'],
                         [-9, 10., 'A']],
                   columns=['amount', 'price', 'symbol'],
                   index=dates[:3]),
         DataFrame(data=[[dates[0], dates[2],
                          Timedelta(days=2), -10., -.2,
                          True, 'A']],
                   columns=['open_dt', 'close_dt',
                            'duration', 'pnl', 'rt_returns',
                            'long', 'symbol'],
                   index=[0])
         ),
        # Round-trip with sell that crosses 0 and should be split
        (DataFrame(data=[[2, 10., 'A'],
                         [-4, 15., 'A'],
                         [3, 20., 'A']],
                   columns=['amount', 'price', 'symbol'],
                   index=dates[:3]),
         DataFrame(data=[[dates[0], dates[1],
                          Timedelta(days=1), 10., .5,
                          True, 'A'],
                         [dates[1], dates[2],
                          Timedelta(days=1),
                          -10, (-1. / 3),
                          False, 'A']],
                   columns=['open_dt', 'close_dt',
                            'duration', 'pnl', 'rt_returns',
                            'long', 'symbol'],
                   index=[0, 1])
         ),
        # Round-trip that does not cross 0
        (DataFrame(data=[[4, 10., 'A'],
                         [-2, 15., 'A'],
                         [2, 20., 'A']],
                   columns=['amount', 'price', 'symbol'],
                   index=dates[:3]),
         DataFrame(data=[[dates[0], dates[1],
                          Timedelta(days=1), 10., .5,
                          True, 'A']],
                   columns=['open_dt', 'close_dt',
                            'duration', 'pnl', 'rt_returns',
                            'long', 'symbol'],
                   index=[0])
         ),
        # Round-trip that does not cross 0 and has portfolio value
        (DataFrame(data=[[4, 10., 'A'],
                         [-2, 15., 'A'],
                         [2, 20., 'A']],
                   columns=['amount', 'price', 'symbol'],
                   index=dates[:3]),
         DataFrame(data=[[dates[0], dates[1],
                          Timedelta(days=1), 10., .5,
                          True, 'A', 0.1]],
                   columns=['open_dt', 'close_dt',
                            'duration', 'pnl', 'rt_returns',
                            'long', 'symbol', 'returns'],
                   index=[0]),
         Series([100., 100., 100.], index=dates[:3]),
         ),

    ])
    def test_extract_round_trips(self, transactions, expected,
                                 portfolio_value=None):
        round_trips = extract_round_trips(transactions,
                                          portfolio_value=portfolio_value)

        assert_frame_equal(round_trips.sort_index(axis='columns'),
                           expected.sort_index(axis='columns'))

    def test_add_closing_trades(self):
        dates = date_range(start='2015-01-01', periods=20)
        transactions = DataFrame(data=[[2, 10, 'A'],
                                       [-5, 10, 'A'],
                                       [-1, 10, 'B']],
                                 columns=['amount', 'price', 'symbol'],
                                 index=dates[:3])
        positions = DataFrame(data=[[20, 10, 0],
                                    [-30, 10, 30],
                                    [-60, 0, 30]],
                              columns=['A', 'B', 'cash'],
                              index=dates[:3])

        expected_ix = dates[:3].append(DatetimeIndex([dates[2] +
                                                      Timedelta(seconds=1)]))
        expected = DataFrame(data=[[2, 10, 'A'],
                                   [-5, 10, 'A'],
                                   [-1, 10., 'B'],
                                   [3, 20., 'A']],
                             columns=['amount', 'price', 'symbol'],
                             index=expected_ix)

        transactions_closed = add_closing_transactions(positions, transactions)
        assert_frame_equal(transactions_closed, expected)

    def test_txn_pnl_matches_round_trip_pnl(self):
        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__)))

        test_txn = read_csv(gzip.open(
                            __location__ + '/test_data/test_txn.csv.gz'),
                            index_col=0, parse_dates=True)
        test_pos = read_csv(gzip.open(
                            __location__ + '/test_data/test_pos.csv.gz'),
                            index_col=0, parse_dates=True)

        transactions_closed = add_closing_transactions(test_pos, test_txn)
        transactions_closed['txn_dollars'] = transactions_closed.amount * \
            -1. * transactions_closed.price
        round_trips = extract_round_trips(transactions_closed)

        self.assertAlmostEqual(round_trips.pnl.sum(),
                               transactions_closed.txn_dollars.sum())

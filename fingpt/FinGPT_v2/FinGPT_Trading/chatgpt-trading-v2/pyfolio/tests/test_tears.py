from matplotlib.testing.decorators import cleanup

from unittest import TestCase
from nose_parameterized import parameterized

import os
import gzip

from pandas import read_csv

from pyfolio.utils import (to_utc, to_series)
from pyfolio.tears import (create_full_tear_sheet,
                           create_simple_tear_sheet,
                           create_returns_tear_sheet,
                           create_position_tear_sheet,
                           create_txn_tear_sheet,
                           create_round_trip_tear_sheet,
                           create_interesting_times_tear_sheet,)


class PositionsTestCase(TestCase):
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

    @parameterized.expand([({},),
                           ({'slippage': 1},),
                           ({'live_start_date': test_returns.index[-20]},),
                           ({'round_trips': True},),
                           ({'hide_positions': True},),
                           ({'cone_std': 1},),
                           ({'bootstrap': True},),
                           ])
    @cleanup
    def test_create_full_tear_sheet_breakdown(self, kwargs):
        create_full_tear_sheet(self.test_returns,
                               positions=self.test_pos,
                               transactions=self.test_txn,
                               benchmark_rets=self.test_returns,
                               **kwargs
                               )

    @parameterized.expand([({},),
                           ({'slippage': 1},),
                           ({'live_start_date': test_returns.index[-20]},),
                           ])
    @cleanup
    def test_create_simple_tear_sheet_breakdown(self, kwargs):
        create_simple_tear_sheet(self.test_returns,
                                 positions=self.test_pos,
                                 transactions=self.test_txn,
                                 **kwargs
                                 )

    @parameterized.expand([({},),
                           ({'live_start_date':
                             test_returns.index[-20]},),
                           ({'cone_std': 1},),
                           ({'bootstrap': True},),
                           ])
    @cleanup
    def test_create_returns_tear_sheet_breakdown(self, kwargs):
        create_returns_tear_sheet(self.test_returns,
                                  benchmark_rets=self.test_returns,
                                  **kwargs
                                  )

    @parameterized.expand([({},),
                           ({'hide_positions': True},),
                           ({'show_and_plot_top_pos': 0},),
                           ({'show_and_plot_top_pos': 1},),
                           ])
    @cleanup
    def test_create_position_tear_sheet_breakdown(self, kwargs):
        create_position_tear_sheet(self.test_returns,
                                   self.test_pos,
                                   **kwargs
                                   )

    @parameterized.expand([({},),
                           ({'unadjusted_returns': test_returns},),
                           ])
    @cleanup
    def test_create_txn_tear_sheet_breakdown(self, kwargs):
        create_txn_tear_sheet(self.test_returns,
                              self.test_pos,
                              self.test_txn,
                              **kwargs
                              )

    @parameterized.expand([({},),
                           ({'sector_mappings': {}},),
                           ])
    @cleanup
    def test_create_round_trip_tear_sheet_breakdown(self, kwargs):
        create_round_trip_tear_sheet(self.test_returns,
                                     self.test_pos,
                                     self.test_txn,
                                     **kwargs
                                     )

    @parameterized.expand([({},),
                           ({'legend_loc': 1},),
                           ])
    @cleanup
    def test_create_interesting_times_tear_sheet_breakdown(self,
                                                           kwargs):
        create_interesting_times_tear_sheet(self.test_returns,
                                            self.test_returns,
                                            **kwargs
                                            )

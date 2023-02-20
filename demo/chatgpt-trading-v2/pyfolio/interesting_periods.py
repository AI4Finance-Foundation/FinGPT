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

"""Generates a list of historical event dates that may have had
significant impact on markets.  See extract_interesting_date_ranges."""

import pandas as pd

from collections import OrderedDict

PERIODS = OrderedDict()
# Dotcom bubble
PERIODS['Dotcom'] = (pd.Timestamp('20000310'), pd.Timestamp('20000910'))

# Lehmann Brothers
PERIODS['Lehman'] = (pd.Timestamp('20080801'), pd.Timestamp('20081001'))

# 9/11
PERIODS['9/11'] = (pd.Timestamp('20010911'), pd.Timestamp('20011011'))

# 05/08/11  US down grade and European Debt Crisis 2011
PERIODS[
    'US downgrade/European Debt Crisis'] = (pd.Timestamp('20110805'),
                                            pd.Timestamp('20110905'))

# 16/03/11  Fukushima melt down 2011
PERIODS['Fukushima'] = (pd.Timestamp('20110316'), pd.Timestamp('20110416'))

# 01/08/03  US Housing Bubble 2003
PERIODS['US Housing'] = (
    pd.Timestamp('20030108'), pd.Timestamp('20030208'))

# 06/09/12  EZB IR Event 2012
PERIODS['EZB IR Event'] = (
    pd.Timestamp('20120910'), pd.Timestamp('20121010'))

# August 2007, March and September of 2008, Q1 & Q2 2009,
PERIODS['Aug07'] = (pd.Timestamp('20070801'), pd.Timestamp('20070901'))
PERIODS['Mar08'] = (pd.Timestamp('20080301'), pd.Timestamp('20080401'))
PERIODS['Sept08'] = (pd.Timestamp('20080901'), pd.Timestamp('20081001'))
PERIODS['2009Q1'] = (pd.Timestamp('20090101'), pd.Timestamp('20090301'))
PERIODS['2009Q2'] = (pd.Timestamp('20090301'), pd.Timestamp('20090601'))

# Flash Crash (May 6, 2010 + 1 week post),
PERIODS['Flash Crash'] = (
    pd.Timestamp('20100505'), pd.Timestamp('20100510'))

# April and October 2014).
PERIODS['Apr14'] = (pd.Timestamp('20140401'), pd.Timestamp('20140501'))
PERIODS['Oct14'] = (pd.Timestamp('20141001'), pd.Timestamp('20141101'))

# Market down-turn in August/Sept 2015
PERIODS['Fall2015'] = (pd.Timestamp('20150815'), pd.Timestamp('20150930'))

# Market regimes
PERIODS['Low Volatility Bull Market'] = (pd.Timestamp('20050101'),
                                         pd.Timestamp('20070801'))

PERIODS['GFC Crash'] = (pd.Timestamp('20070801'),
                        pd.Timestamp('20090401'))

PERIODS['Recovery'] = (pd.Timestamp('20090401'),
                       pd.Timestamp('20130101'))

PERIODS['New Normal'] = (pd.Timestamp('20130101'),
                         pd.Timestamp('today'))

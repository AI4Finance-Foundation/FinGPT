import sys
from pathlib import Path

import pytest


SENTIMENT_DIR = Path(__file__).resolve().parents[1] / "fingpt" / "FinGPT_Sentiment_Analysis_v3"
sys.path.insert(0, str(SENTIMENT_DIR))

from quantitative_sentiment import earnings_impact, earnings_surprise


def test_earnings_surprise_uses_absolute_consensus_for_misses():
    assert earnings_surprise(2.11, 1.90) == pytest.approx(0.11052631578947367)
    assert earnings_surprise(-0.08, -0.10) == pytest.approx(0.2)


def test_earnings_impact_classifies_surprise():
    assert earnings_impact(2.11, 1.90)["label"] == "positive"
    assert earnings_impact(1.91, 1.90)["label"] == "neutral"
    assert earnings_impact(1.70, 1.90)["label"] == "negative"


def test_invalid_earnings_inputs_are_rejected():
    for actual, consensus, threshold in ((1.0, 0.0, 0.05), (1.0, 1.0, -0.01)):
        try:
            earnings_impact(actual, consensus, threshold)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid earnings input should raise ValueError")
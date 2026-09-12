"""Simple quantitative context for financial sentiment outputs."""


def earnings_surprise(actual, consensus):
    """Return the percentage difference between actual and consensus EPS."""
    if consensus == 0:
        raise ValueError("consensus EPS must not be zero")
    return (actual - consensus) / abs(consensus)


def earnings_impact(actual, consensus, threshold=0.05):
    """Classify an earnings result as positive, neutral, or negative impact.

    The threshold is a configurable heuristic, not a trading signal. A model's
    linguistic sentiment should be retained separately from this classification.
    """
    if threshold < 0:
        raise ValueError("threshold must not be negative")

    surprise = earnings_surprise(actual, consensus)
    if surprise > threshold:
        label = "positive"
    elif surprise < -threshold:
        label = "negative"
    else:
        label = "neutral"

    return {"label": label, "surprise": surprise}
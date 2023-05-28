"""Wrapper module around seaborn to suppress warnings on import.

This should be removed when seaborn stops raising:

UserWarning: axes.color_cycle is deprecated and replaced with axes.prop_cycle;
please use the latter.
"""
import warnings


with warnings.catch_warnings():
    warnings.filterwarnings(
        'ignore',
        'axes.color_cycle is deprecated',
        UserWarning,
        'matplotlib',
    )
    from seaborn import *  # noqa

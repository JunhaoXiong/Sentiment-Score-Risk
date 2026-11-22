"""Rolling, strictly backward-looking standardisation.

z_t = (x_t - mean(x_{t-W..t-1})) / std(x_{t-W..t-1})

The baseline excludes x_t itself, so a large new value is measured against
the past only (financial news is negative on average, hence the baseline).
x_t here is already the *effective* value for day t (see align.py), so z_t
uses only information available at close(t-1).
"""
from __future__ import annotations

import pandas as pd


def rolling_z(x: pd.Series, window: int = 250, min_periods: int = 120) -> pd.Series:
    past = x.shift(1).rolling(window, min_periods=min_periods)
    return (x - past.mean()) / past.std(ddof=1)

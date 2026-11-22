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


def rolling_residual(y: pd.Series, X: pd.DataFrame, window: int = 250,
                     min_periods: int = 120) -> pd.Series:
    """Out-of-sample residual y_t - X_t b_{t}, b_t estimated by OLS (with
    intercept) on the previous ``window`` rows only (rows t-window .. t-1).

    X_t must itself be known when y_t is; the coefficients never see row t.
    """
    import numpy as np

    d = pd.concat([y.rename("_y"), X], axis=1).dropna()
    Y = d["_y"].to_numpy()
    Z = np.column_stack([np.ones(len(d)), d.drop(columns="_y").to_numpy()])
    out = np.full(len(d), np.nan)
    for t in range(min_periods, len(d)):
        lo = max(0, t - window)
        b, *_ = np.linalg.lstsq(Z[lo:t], Y[lo:t], rcond=None)
        out[t] = Y[t] - Z[t] @ b
    return pd.Series(out, index=d.index).reindex(y.index)

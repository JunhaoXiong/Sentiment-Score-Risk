"""Rule-based VaR overlays (management add-ons) driven by news sentiment.

step   : VaR * m            when z_t < threshold, else VaR
smooth : VaR * (1 + k * max(0, -z_t - c))
flat   : VaR * (1 + u)      every day (the "same-capital" benchmark)

z_t is the effective tone z (known at close t-1), so the overlay adds no
look-ahead. Parameters must be chosen on a calibration period only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def step(var: pd.Series, z: pd.Series, m: float, threshold: float = -2.0) -> pd.Series:
    return var * np.where(z.reindex(var.index) < threshold, m, 1.0)


def smooth(var: pd.Series, z: pd.Series, k: float, c: float = 1.0) -> pd.Series:
    zz = z.reindex(var.index).fillna(0.0)
    return var * (1 + k * np.maximum(0.0, -zz - c))


def flat(var: pd.Series, u: float) -> pd.Series:
    return var * (1 + u)


def matching_flat_uplift(base: pd.Series, adjusted: pd.Series) -> float:
    """u such that flat(base, u) has the same mean as ``adjusted``."""
    return float(adjusted.mean() / base.mean() - 1)

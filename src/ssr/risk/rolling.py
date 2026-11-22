"""Rolling out-of-sample one-day VaR/ES for GJR-GARCH-X variants.

At each refit day i (every ``refit_every`` days) parameters are estimated on
the W returns r[i-W : i]. With those parameters fixed, the variance is
filtered forward through the training window and on to the next refit; the
forecast for day t uses r up to t-1 and X_t (known at close t-1). Nothing
from day t or later enters the day-t forecast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import garchx


def rolling_var(r: pd.Series, X: pd.DataFrame | None, W: int = 500,
                refit_every: int = 21, alphas=(0.95, 0.99),
                oos_start: int | None = None) -> tuple[pd.DataFrame, list[dict]]:
    r = r.astype(float)
    n = len(r)
    Xv = None if X is None or X.shape[1] == 0 else X.to_numpy(float)
    i0 = W if oos_start is None else max(oos_start, W)
    rows, fits = [], []
    prev = None
    for i in range(i0, n, refit_every):
        tr = slice(i - W, i)
        f = garchx.fit(r.values[tr], None if Xv is None else Xv[tr], start=prev)
        if not f.converged:  # retry from default start values
            f2 = garchx.fit(r.values[tr], None if Xv is None else Xv[tr])
            f = f2 if f2.loglik > f.loglik else f
        prev = f
        j = min(i + refit_every, n)
        span = slice(i - W, j)
        sig = garchx.sigma_path(f, r.values[span], None if Xv is None else Xv[span],
                                float(r.values[tr].var()))[W:]
        out = {"sigma": sig, "mu": np.full(len(sig), f.mu)}
        for a in alphas:
            q, es = garchx.t_quantile_es(a, f.nu)
            out[f"var{int(a*100)}"] = -(f.mu + sig * q)
            out[f"es{int(a*100)}"] = -(f.mu - sig * es)
        rows.append(pd.DataFrame(out, index=r.index[i:j]))
        fits.append({"date": r.index[i], **{k: getattr(f, k) for k in
                     ("mu", "omega", "alpha", "gamma", "beta", "nu", "loglik", "converged")},
                     **{f"delta{k}": d for k, d in enumerate(f.delta)}})
    fc = pd.concat(rows)
    fc["r"] = r.reindex(fc.index)
    fc["loss"] = -fc["r"]
    for a in alphas:
        fc[f"hit{int(a*100)}"] = (fc["loss"] > fc[f"var{int(a*100)}"]).astype(int)
    return fc, fits


def tick_loss(r: pd.Series, var: pd.Series, alpha: float) -> pd.Series:
    """Quantile (pinball) loss of the return quantile -VaR at level p = 1-alpha."""
    p = 1 - alpha
    q = -var
    return (p - (r < q).astype(float)) * (r - q)


def diebold_mariano(l1: pd.Series, l2: pd.Series, lags: int = 5) -> dict:
    """DM test on mean(l1 - l2) with Newey-West variance; negative = model 1 better."""
    d = (l1 - l2).dropna().to_numpy()
    n = len(d)
    m = d.mean()
    u = d - m
    s = u @ u / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        s += 2 * w * (u[k:] @ u[:-k]) / n
    stat = m / np.sqrt(s / n)
    from scipy.stats import norm
    return {"mean_diff": float(m), "dm": float(stat), "p": float(2 * norm.sf(abs(stat)))}

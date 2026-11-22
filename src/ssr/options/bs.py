"""Black-Scholes straddle price and Greeks (q = 0).

Price/vega follow JunhaoXiong/Heston-Option-Risk ``heston_risk.bs``; Greeks
added here. Underlying = dividend-adjusted SPY (total-return series), so q = 0
is the consistent choice.

Units: S, K in $; T in years; r, sigma annual decimals.
Greeks are per ONE straddle (one call + one put on one share):
  delta  dV/dS
  gamma  d2V/dS2
  vega   dV/dsigma per 1.00 of vol (divide by 100 for per vol point)
  theta  dV/dt per year (divide by 365 for per calendar day)
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

SQRT2PI = np.sqrt(2 * np.pi)


def _d1d2(S, K, T, r, sigma):
    w = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / w
    return d1, d1 - w


def straddle(S, K, T, r, sigma) -> dict:
    """Price and Greeks of a call + put; T <= 0 returns intrinsic, zero Greeks."""
    S, K, T, r, sigma = np.broadcast_arrays(*(np.asarray(a, float) for a in (S, K, T, r, sigma)))
    live = (T > 0) & (sigma > 0)
    Ts = np.where(live, T, 1.0)
    ss = np.where(live, sigma, 1.0)
    d1, d2 = _d1d2(S, K, Ts, r, ss)
    df = np.exp(-r * Ts)
    pdf = np.exp(-0.5 * d1**2) / SQRT2PI
    call = S * ndtr(d1) - K * df * ndtr(d2)
    put = K * df * ndtr(-d2) - S * ndtr(-d1)
    price = call + put
    delta = 2 * ndtr(d1) - 1
    gamma = 2 * pdf / (S * ss * np.sqrt(Ts))
    vega = 2 * S * pdf * np.sqrt(Ts)
    # theta (calendar time decay): call + put thetas
    common = -S * pdf * ss / (2 * np.sqrt(Ts))
    theta = 2 * common + r * K * df * (ndtr(-d2) - ndtr(d2))
    intrinsic = np.abs(S - K)
    out = {
        "price": np.where(live, price, intrinsic),
        "delta": np.where(live, delta, np.sign(S - K)),
        "gamma": np.where(live, gamma, 0.0),
        "vega": np.where(live, vega, 0.0),
        "theta": np.where(live, theta, 0.0),
    }
    return {k: (v[()] if v.ndim == 0 else v) for k, v in out.items()}

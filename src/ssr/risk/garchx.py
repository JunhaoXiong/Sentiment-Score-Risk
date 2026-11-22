"""GJR-GARCH-X with Student-t innovations, estimated by MLE.

    r_t = mu + eps_t,   eps_t = sigma_t z_t,   z_t ~ unit-variance Student-t(nu)
    sigma2_t = omega + (alpha + gamma 1[eps_{t-1} < 0]) eps_{t-1}^2
               + beta sigma2_{t-1} + sum_k delta_k x_{k,t}

x_{k,t} >= 0 are exogenous regressors that must be known at the close of
t-1 (the caller guarantees this; see ssr.sentiment.align). Using delta >= 0
and x >= 0 keeps the variance positive. Regressors used in this project:
  vix   : (VIX_{t-1}/100)^2 / 252  -- implied daily variance
  neg_z : max(0, -z_t) * 1e-4     -- only *negative* news-tone surprises
          (1e-4 = variance of a 1% daily move, so delta is O(0.1))

Returns are scaled by 100 for numerical stability.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import t as student_t

SCALE = 100.0


@dataclass(frozen=True)
class Fit:
    mu: float
    omega: float
    alpha: float
    gamma: float
    beta: float
    delta: tuple[float, ...]
    nu: float
    loglik: float
    n: int
    converged: bool

    @property
    def k(self) -> int:
        return 6 + len(self.delta)

    @property
    def aic(self) -> float:
        return 2 * self.k - 2 * self.loglik


def filter_variance(r: np.ndarray, X: np.ndarray, mu, omega, alpha, gamma, beta,
                    delta, sigma2_0: float) -> np.ndarray:
    """sigma2_t for t = 0..n-1 (r, X in scaled units). sigma2_t uses r_{t-1}, X_t."""
    n = len(r)
    s2 = np.empty(n)
    s2[0] = sigma2_0
    eps = r - mu
    xd = X @ np.asarray(delta) if X.shape[1] else np.zeros(n)
    for t in range(1, n):
        e = eps[t - 1]
        s2[t] = (omega + (alpha + gamma * (e < 0)) * e * e + beta * s2[t - 1] + xd[t])
    return s2


def _t_logpdf_unit(z, nu):
    c = (nu - 2.0)
    return (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * c)
            - (nu + 1) / 2 * np.log1p(z * z / c))


def _negll(theta, r, X, s2_0):
    k = X.shape[1]
    mu, omega, alpha, gamma, beta = theta[:5]
    delta = theta[5:5 + k]
    nu = theta[5 + k]
    if alpha + beta + 0.5 * gamma >= 0.9999:
        return 1e10
    s2 = filter_variance(r, X, mu, omega, alpha, gamma, beta, delta, s2_0)
    if not np.all(np.isfinite(s2)) or np.any(s2 <= 0):
        return 1e10
    z = (r - mu) / np.sqrt(s2)
    ll = _t_logpdf_unit(z, nu) - 0.5 * np.log(s2)
    return -float(ll.sum())


def fit(r: np.ndarray, X: np.ndarray | None = None, start: Fit | None = None) -> Fit:
    """MLE on returns r (unscaled) and regressors X (n x k, unscaled, >= 0)."""
    r = np.asarray(r, float) * SCALE
    X = np.zeros((len(r), 0)) if X is None else np.asarray(X, float) * SCALE ** 2
    k = X.shape[1]
    v = r.var()
    s2_0 = v
    if start is not None and len(start.delta) == k:
        x0 = [start.mu * SCALE, start.omega * SCALE ** 2, start.alpha, start.gamma,
              start.beta, *start.delta, start.nu]
    else:
        x0 = [r.mean(), 0.05 * v, 0.02, 0.12, 0.85, *([0.05] * k), 8.0]
    bounds = ([(-1.0, 1.0), (1e-6 * v, 2 * v), (0.0, 0.5), (0.0, 0.8), (0.0, 0.999)]
              + [(0.0, 50.0)] * k + [(2.05, 100.0)])
    res = minimize(_negll, x0, args=(r, X, s2_0), method="L-BFGS-B", bounds=bounds)
    th = res.x
    ll = -res.fun - len(r) * np.log(SCALE)  # log-lik of unscaled returns
    return Fit(mu=th[0] / SCALE, omega=th[1] / SCALE ** 2, alpha=th[2], gamma=th[3],
               beta=th[4], delta=tuple(th[5:5 + k]), nu=th[5 + k], loglik=float(ll),
               n=len(r), converged=bool(res.success))


def sigma_path(f: Fit, r: np.ndarray, X: np.ndarray | None, sigma2_0: float) -> np.ndarray:
    """Conditional vol (return units) for every day in r given fixed params."""
    rs = np.asarray(r, float) * SCALE
    Xs = np.zeros((len(rs), 0)) if X is None else np.asarray(X, float) * SCALE ** 2
    s2 = filter_variance(rs, Xs, f.mu * SCALE, f.omega * SCALE ** 2, f.alpha, f.gamma,
                         f.beta, f.delta, sigma2_0 * SCALE ** 2)
    return np.sqrt(s2) / SCALE


def t_quantile_es(alpha: float, nu: float) -> tuple[float, float]:
    """Left-tail quantile q (negative) and ES (positive) of unit-variance t(nu)."""
    c = np.sqrt((nu - 2) / nu)
    p = 1 - alpha
    ta = student_t.ppf(p, nu)
    q = ta * c
    es = c * student_t.pdf(ta, nu) / p * (nu + ta ** 2) / (nu - 1)
    return float(q), float(es)

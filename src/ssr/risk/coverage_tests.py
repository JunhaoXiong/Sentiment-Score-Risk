# Copied from JunhaoXiong/Market-Risk-Engine src/mre/backtest/coverage_tests.py (c9f5ecb).
"""VaR backtesting: Kupiec (1995) proportion-of-failures and Christoffersen (1998)
independence and conditional-coverage likelihood-ratio tests.

``hits`` is a 0/1 sequence: 1 when the realised loss exceeded the VaR forecast.
``p`` is the expected violation probability (1 - confidence level).

Kupiec POF:        LR_pof = -2 [ log L(p) - log L(x/n) ]                     ~ chi2(1)
Christoffersen ind: first-order Markov alternative with pi01, pi11 vs i.i.d. pi  ~ chi2(1)
Conditional coverage (exact form on the n-1 transitions):
                   LR_cc = -2 [ log L(p) - log L(pi01, pi11) ]               ~ chi2(2)
The common shortcut LR_pof + LR_ind differs from LR_cc only by the first observation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import xlogy
from scipy.stats import chi2


def _hits(h) -> np.ndarray:
    h = np.asarray(h)
    if h.ndim != 1 or h.size < 2:
        raise ValueError("hits must be a 1-D sequence with at least 2 observations")
    if not np.isin(h, (0, 1, True, False)).all():
        raise ValueError("hits must contain only 0/1")
    return h.astype(int)


def _bern_ll(n0: int, n1: int, q: float) -> float:
    """log-likelihood of n0 zeros and n1 ones under Bernoulli(q), with 0*log0 = 0."""
    return float(xlogy(n0, 1 - q) + xlogy(n1, q))


@dataclass(frozen=True)
class TestResult:
    name: str
    statistic: float
    p_value: float
    df: int

    def reject(self, significance: float) -> bool:
        return self.p_value < significance


def kupiec_pof(hits, p: float) -> TestResult:
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    h = _hits(hits)
    n, x = h.size, int(h.sum())
    lr = -2.0 * (_bern_ll(n - x, x, p) - _bern_ll(n - x, x, x / n))
    lr = max(lr, 0.0)
    return TestResult("Kupiec POF", lr, float(chi2.sf(lr, 1)), 1)


def transition_counts(hits) -> tuple[int, int, int, int]:
    h = _hits(hits)
    prev, cur = h[:-1], h[1:]
    n00 = int(((prev == 0) & (cur == 0)).sum())
    n01 = int(((prev == 0) & (cur == 1)).sum())
    n10 = int(((prev == 1) & (cur == 0)).sum())
    n11 = int(((prev == 1) & (cur == 1)).sum())
    return n00, n01, n10, n11


def _markov_ll(n00, n01, n10, n11) -> float:
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    return _bern_ll(n00, n01, pi01) + _bern_ll(n10, n11, pi11)


def christoffersen_independence(hits) -> TestResult:
    n00, n01, n10, n11 = transition_counts(hits)
    total = n00 + n01 + n10 + n11
    pi = (n01 + n11) / total
    lr = -2.0 * (_bern_ll(n00 + n10, n01 + n11, pi) - _markov_ll(n00, n01, n10, n11))
    lr = max(lr, 0.0)
    return TestResult("Christoffersen independence", lr, float(chi2.sf(lr, 1)), 1)


def christoffersen_cc(hits, p: float) -> TestResult:
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    n00, n01, n10, n11 = transition_counts(hits)
    lr = -2.0 * (_bern_ll(n00 + n10, n01 + n11, p) - _markov_ll(n00, n01, n10, n11))
    lr = max(lr, 0.0)
    return TestResult("Christoffersen conditional coverage", lr, float(chi2.sf(lr, 2)), 2)


def backtest_summary(hits, p: float) -> dict:
    h = _hits(hits)
    pof, ind, cc = kupiec_pof(h, p), christoffersen_independence(h), christoffersen_cc(h, p)
    n00, n01, n10, n11 = transition_counts(h)
    return {
        "n": int(h.size),
        "violations": int(h.sum()),
        "expected": float(p * h.size),
        "violation_rate": float(h.mean()),
        "clustered (n11)": n11,
        "kupiec_lr": pof.statistic, "kupiec_p": pof.p_value,
        "ind_lr": ind.statistic, "ind_p": ind.p_value,
        "cc_lr": cc.statistic, "cc_p": cc.p_value,
    }

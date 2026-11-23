import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from ssr.options import backtest, bs


def test_straddle_price_matches_closed_form():
    S, K, T, r, s = 100.0, 95.0, 0.25, 0.03, 0.2
    d1 = (np.log(S / K) + (r + s * s / 2) * T) / (s * np.sqrt(T)); d2 = d1 - s * np.sqrt(T)
    c = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    p = c - S + K * np.exp(-r * T)
    assert bs.straddle(S, K, T, r, s)["price"] == pytest.approx(c + p, rel=1e-12)


@pytest.mark.parametrize("greek,arg,h", [("delta", 0, 1e-3), ("vega", 4, 1e-5)])
def test_first_order_greeks_by_finite_difference(greek, arg, h):
    x = [100.0, 102.0, 0.1, 0.02, 0.25]
    up, dn = list(x), list(x)
    up[arg] += h; dn[arg] -= h
    fd = (bs.straddle(*up)["price"] - bs.straddle(*dn)["price"]) / (2 * h)
    assert bs.straddle(*x)[greek] == pytest.approx(fd, rel=1e-5)


def test_gamma_and_theta_by_finite_difference():
    x = [100.0, 102.0, 0.1, 0.02, 0.25]
    h = 1e-2
    p = lambda S, T: bs.straddle(S, 102.0, T, 0.02, 0.25)["price"]
    gfd = (p(100 + h, .1) - 2 * p(100, .1) + p(100 - h, .1)) / h**2
    assert bs.straddle(*x)["gamma"] == pytest.approx(gfd, rel=1e-4)
    tfd = -(p(100, .1 + 1e-6) - p(100, .1 - 1e-6)) / 2e-6
    assert bs.straddle(*x)["theta"] == pytest.approx(tfd, rel=1e-4)


def test_expired_is_intrinsic():
    g = bs.straddle(100.0, 90.0, 0.0, 0.02, 0.2)
    assert g["price"] == 10.0 and g["gamma"] == 0.0


def _market(sig_real, vix, n=260, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    lr = rng.standard_normal(n) * sig_real / np.sqrt(252) - 0.5 * sig_real**2 / 252
    lr[0] = 0
    return pd.DataFrame({"S": 100 * np.exp(np.cumsum(lr)), "vix": vix, "r": 0.0}, index=dates)


def test_bookkeeping_pnl_sums_to_final_wealth():
    m = _market(0.2, 20.0)
    cfg = backtest.Config(iv_mult=1.0, option_cost=0.01, hedge_cost_bp=1.0)
    tr = backtest.run_trade(5, -1, m.S.values, m.vix.values, m.r.values + 0.03, m.index, cfg)
    assert tr["pnl"].sum() + tr["cost"].sum() == pytest.approx(tr["final_wealth"], abs=1e-6)


def test_hedged_long_straddle_earns_realised_minus_implied():
    cfg = backtest.Config(iv_mult=1.0, option_cost=0.0, hedge_cost_bp=0.0)
    tot_hi, tot_lo = [], []
    for seed in range(40):
        hi = _market(0.30, 20.0, seed=seed)   # realised 30 > implied 20
        lo = _market(0.10, 20.0, seed=seed)   # realised 10 < implied 20
        tot_hi.append(backtest.run_trade(0, 1, hi.S.values, hi.vix.values, hi.r.values, hi.index, cfg)["total"])
        tot_lo.append(backtest.run_trade(0, 1, lo.S.values, lo.vix.values, lo.r.values, lo.index, cfg)["total"])
    assert np.mean(tot_hi) > 0 > np.mean(tot_lo)


def test_flat_signal_and_no_lookahead():
    m = _market(0.2, 20.0, n=120)
    cfg = backtest.Config(entry_every=21)
    sig = pd.Series(-1, index=m.index); sig.iloc[21] = 0
    daily, trades = backtest.run(sig, m, cfg)
    assert trades.loc[trades.entry == m.index[21], "total"].iloc[0] == 0.0
    m2 = m.copy(); m2.iloc[100:, 0] *= 1.5          # shock after the first trades expire
    d2, _ = backtest.run(sig, m2, cfg)
    first_exp = trades["expiry"].dropna().iloc[0]
    pd.testing.assert_series_equal(daily.loc[:first_exp], d2.loc[:first_exp])

import numpy as np
import pandas as pd
import pytest

from ssr.data import option_snapshots
from ssr.options import bs, forward


def make_chain(day, S, vol, expiries, strikes, spread=0.1, r=0.04, vix=20.0):
    rows = []
    for e in expiries:
        T = (pd.Timestamp(e) - day.normalize()).days / 365
        for K in strikes:
            g = bs.straddle(S, K, T, r, vol)
            # split straddle into call/put via parity-consistent BS pieces
            d1 = (np.log(S / K) + (r + vol**2 / 2) * T) / (vol * np.sqrt(T))
            from scipy.stats import norm
            call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d1 - vol * np.sqrt(T))
            put = g["price"] - call
            for kind, px in (("call", call), ("put", put)):
                rows.append({"snapshot_utc": (day + pd.Timedelta(hours=20)).tz_localize("UTC").isoformat(),
                             "expiry": e, "kind": kind, "strike": K, "bid": px - spread / 2,
                             "ask": px + spread / 2, "spot": S, "vix": vix, "vix3m": vix,
                             "quote_ok": True})
    return pd.DataFrame(rows)


def test_atm_iv_recovers_flat_vol():
    day = pd.Timestamp("2026-01-05")
    ch = make_chain(day, 100.0, 0.2, ["2026-01-23", "2026-02-20"], [95.0, 100.0, 105.0], spread=0.0)
    out = option_snapshots.atm_iv_30d(ch, r=0.04)
    assert out["ok"] and out["atm_iv_30d"] == pytest.approx(0.2, abs=0.005)


def test_paper_short_straddle_flat_market_earns_theta_minus_spread():
    days = pd.bdate_range("2026-01-05", "2026-02-20")
    chains = {d: make_chain(d, 100.0, 0.2, ["2026-02-06", "2026-02-20"], [100.0], spread=0.2) for d in days}
    sig = pd.Series(-1, index=days)
    res = forward.paper_trade(chains, sig, entry_every=1000)
    assert res["event"].iloc[0] == "open" and "close" in set(res["event"])
    assert res.loc[res.event == "open", "pnl"].iloc[0] < 0      # paid half spread
    assert res["pnl"].sum() > 0                                 # spot never moved: keeps premium


def test_flat_signal_never_trades():
    days = pd.bdate_range("2026-01-05", periods=10)
    chains = {d: make_chain(d, 100.0, 0.2, ["2026-02-13"], [100.0]) for d in days}
    res = forward.paper_trade(chains, pd.Series(0, index=days))
    assert (res["event"] == "flat").all() and res["pnl"].sum() == 0

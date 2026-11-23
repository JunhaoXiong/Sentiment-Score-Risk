"""Forward (paper) test of the straddle signals on REAL stored option quotes.

One trade at a time per signal: on an entry day pick the expiry closest to 30
calendar days (>= 20) and the strike nearest spot. Short trades are filled at
the bids, long trades at the asks (conservative); marks use mids; the hedge is
the BS delta at the IV implied by that day's straddle mid. At the last
snapshot on/before expiry the straddle settles at |S - K|.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from . import bs


def _quote(chain: pd.DataFrame, expiry: str, K: float) -> dict | None:
    g = chain[(chain["expiry"] == expiry) & (chain["strike"] == K) & chain["quote_ok"]]
    if set(g["kind"]) != {"call", "put"}:
        return None
    c, p = g[g.kind == "call"].iloc[0], g[g.kind == "put"].iloc[0]
    return {"bid": c.bid + p.bid, "ask": c.ask + p.ask, "mid": (c.bid + c.ask + p.bid + p.ask) / 2}


def _years(expiry: str, day: pd.Timestamp) -> float:
    return max((pd.Timestamp(expiry) - day.normalize()).days, 0) / 365


def _delta(S, K, T, r, mid) -> float:
    if T <= 0:
        return float(np.sign(S - K))
    try:
        iv = brentq(lambda s: bs.straddle(S, K, T, r, s)["price"] - mid, 1e-4, 5.0)
    except ValueError:
        return 0.0
    return float(bs.straddle(S, K, T, r, iv)["delta"])


def paper_trade(chains: dict[pd.Timestamp, pd.DataFrame], signal: pd.Series,
                r: float = 0.04, notional: float = 1e6, entry_every: int = 21) -> pd.DataFrame:
    """Daily P&L rows for one signal. ``signal`` indexed by snapshot day (-1/0/+1)."""
    days = sorted(chains)
    rows, trade, last_entry = [], None, -10**9
    for i, d in enumerate(days):
        ch = chains[d]
        S = float(ch["spot"].iloc[0])
        pnl = 0.0
        if trade is not None:
            dS = S - trade["S_prev"]
            closed = pd.Timestamp(trade["expiry"]) <= d.normalize()
            if closed:  # settle at intrinsic on/after expiry
                V = abs(S - trade["K"])
            else:       # mark at mid; keep the last mark if the quote is missing
                q = _quote(ch, trade["expiry"], trade["K"])
                V = q["mid"] if q else trade["V_prev"]
            pnl = trade["n"] * (V - trade["V_prev"]) + trade["hedge"] * dS
            trade["V_prev"], trade["S_prev"] = V, S
            if not closed:
                T = _years(trade["expiry"], d)
                trade["hedge"] = -trade["n"] * _delta(S, trade["K"], T, r, V)
            rows.append({"date": d, "pnl": pnl, "event": "close" if closed else "mark"})
            if closed:
                trade = None
            continue
        sig = int(signal.get(d, 0))
        if sig != 0 and i - last_entry >= entry_every:
            exps = sorted(ch["expiry"].unique(), key=lambda e: abs((pd.Timestamp(e) - d).days - 30))
            exps = [e for e in exps if (pd.Timestamp(e) - d).days >= 20]
            if exps:
                e = exps[0]
                strikes = ch[ch.expiry == e]["strike"].unique()
                K = float(strikes[np.argmin(np.abs(strikes - S))])
                q = _quote(ch, e, K)
                if q:
                    n = sig * notional / S
                    fill = q["ask"] if sig > 0 else q["bid"]
                    slip = n * (q["mid"] - fill)  # negative: paid half-spread
                    T = _years(e, d)
                    trade = {"expiry": e, "K": K, "n": n, "V_prev": q["mid"], "S_prev": S,
                             "hedge": -n * _delta(S, K, T, r, q["mid"])}
                    last_entry = i
                    rows.append({"date": d, "pnl": slip, "event": "open", "K": K,
                                 "expiry": e, "sign": sig, "entry_mid": q["mid"]})
                    continue
        rows.append({"date": d, "pnl": 0.0, "event": "flat"})
    return pd.DataFrame(rows)

"""Straddle backtest with optional daily delta hedging.

Timing (all at daily closes, no look-ahead):
  * a position opened at close t uses S_t, sigma_t = iv_mult * VIX_t / 100,
    r_t = ^IRX_t / 100, and a signal computed from data known at close t;
  * expiry = last trading day within ``tenor_days`` calendar days of t;
  * each day the straddle is re-marked with that day's sigma (BS as a
    quoting device; see report), and the hedge is re-set to -position*delta;
  * the cash account (premium, hedge purchases, costs) earns r_t daily;
  * at expiry the straddle settles at |S_T - K|.

Position size: ``notional / S_t`` straddles (one per share), sign from the
signal: -1 short, +1 long, 0 flat.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import bs


@dataclass
class Config:
    tenor_days: int = 30
    entry_every: int = 21          # trading days between entries (21 = monthly, 1 = daily ladder)
    iv_mult: float = 0.9           # ATM IV ~= iv_mult * VIX
    option_cost: float = 0.01      # fraction of straddle value paid at open and at close
    hedge_cost_bp: float = 1.0     # per $ traded in SPY
    notional: float = 1_000_000.0
    hedge: bool = True


def run_trade(t0: int, sign: int, S: np.ndarray, vix: np.ndarray, r: np.ndarray,
              dates: pd.DatetimeIndex, cfg: Config, scale: float = 1.0) -> dict:
    """Simulate one trade from index t0 to its expiry; return daily P&L path."""
    exp_date = dates[t0] + pd.Timedelta(days=cfg.tenor_days)
    if exp_date > dates[-1]:  # would expire after the data ends: not traded
        return {}
    t1 = int(np.searchsorted(dates.values, exp_date.to_datetime64(), side="right")) - 1
    if t1 <= t0:
        return {}
    K = S[t0]
    n = sign * scale * cfg.notional / S[t0]  # signed number of straddles
    idx = np.arange(t0, t1 + 1)
    T = np.array([(exp_date - dates[i]).days / 365.0 for i in idx])
    T[-1] = 0.0  # settle at intrinsic on the last trading day
    sig = cfg.iv_mult * vix[idx] / 100.0
    g = bs.straddle(S[idx], K, T, r[idx], sig)
    V = n * g["price"]
    hedge_sh = -n * g["delta"] if cfg.hedge else np.zeros(len(idx))
    hedge_sh[-1] = 0.0  # unwind at expiry
    pnl = np.zeros(len(idx))
    cost = np.zeros(len(idx))
    cost[0] += cfg.option_cost * abs(V[0])
    cost[-1] += cfg.option_cost * abs(V[-1])
    traded = np.abs(np.diff(np.r_[0.0, hedge_sh]))
    cost += cfg.hedge_cost_bp * 1e-4 * traded * S[idx]
    # mark-to-market P&L of option + hedge, plus financing of the cash account
    cash = -V[0] - hedge_sh[0] * S[t0]   # receive premium (V<0 when short), buy hedge
    for j in range(1, len(idx)):
        i, ip = idx[j], idx[j - 1]
        dt = (dates[i] - dates[ip]).days / 365.0
        fin = cash * (np.exp(r[ip] * dt) - 1)
        pnl[j] = (V[j] - V[j - 1]) + hedge_sh[j - 1] * (S[i] - S[ip]) + fin
        cash += fin - (hedge_sh[j] - hedge_sh[j - 1]) * S[i]
    pnl -= cost
    final_wealth = cash + V[-1] + hedge_sh[-1] * S[t1]  # bookkeeping check value
    return {"final_wealth": float(final_wealth),"dates": dates[idx], "pnl": pnl, "cost": cost, "K": K, "n": n,
            "entry": dates[t0], "expiry": dates[t1], "sign": sign,
            "premium": abs(V[0]), "iv0": sig[0], "vega0": n * g["vega"][0] / 100,
            "gamma0": n * g["gamma"][0], "theta0_day": n * g["theta"][0] / 365,
            "realized_vol": float(np.std(np.diff(np.log(S[idx])), ddof=0) * np.sqrt(252)),
            "total": float(pnl.sum())}


def run(signal: pd.Series, mkt: pd.DataFrame, cfg: Config) -> tuple[pd.Series, pd.DataFrame]:
    """signal: index = entry-eligible dates, values in {-1, 0, 1} known at that close.
    mkt: columns S, vix, r (decimal) on trading days. Returns daily P&L, trade table."""
    dates = mkt.index
    S, vix, r = (mkt[c].to_numpy(float) for c in ("S", "vix", "r"))
    pos = {d: i for i, d in enumerate(dates)}
    entries = [d for d in signal.index[::cfg.entry_every] if d in pos]
    # daily ladder: ~21 overlapping trades share the notional
    scale = 1.0 / 21 if cfg.entry_every == 1 else 1.0
    daily = pd.Series(0.0, index=dates)
    rows = []
    for d in entries:
        s = int(signal.loc[d])
        if s == 0:
            rows.append({"entry": d, "sign": 0, "total": 0.0})
            continue
        tr = run_trade(pos[d], s, S, vix, r, dates, cfg, scale)
        if not tr:
            continue
        daily.loc[tr["dates"][1:]] += tr["pnl"][1:]
        daily.loc[tr["dates"][0]] += tr["pnl"][0]  # opening cost
        rows.append({k: v for k, v in tr.items()
                     if k not in ("dates", "pnl", "cost", "final_wealth")})
    return daily, pd.DataFrame(rows)

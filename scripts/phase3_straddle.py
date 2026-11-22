"""Phase 3: sentiment x implied-vol SPY straddle strategies.

Signals (decided at close t; z_t is the effective tone-z, known at close t-1):
  A always short
  B short unless VIX/VIX3M > 1 (inverted term structure) -> flat
  C short unless z < -2 -> flat
  D short unless (inverted or z < -2) -> flat
  E short, but LONG when z < -2
  F short, but LONG when inverted   (term-structure analogue of E)
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ssr.data import market  # noqa: E402
from ssr.options import backtest, bs  # noqa: E402

cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
P3 = cfg["phase3"]
prim = cfg["gdelt"]["primary"]
REP, FIG = ROOT / "reports", ROOT / "reports/figures"

px = market.load(ROOT / "data/raw/market.csv")
irx = pd.read_csv(ROOT / "data/raw/irx.csv", index_col="date", parse_dates=True)["IRX"]
panel = pd.read_csv(ROOT / "data/processed/panel.csv", index_col=0, parse_dates=True)
mkt = pd.DataFrame({"S": px["SPY"], "vix": px["^VIX"], "r": irx.reindex(px.index).ffill() / 100})
mkt = mkt.loc[cfg["sample"]["start"]:cfg["sample"]["end"]].dropna()
z = panel[f"z_{prim}"].reindex(mkt.index)
ratio = (px["^VIX"] / px["^VIX3M"]).reindex(mkt.index)
elig = z.dropna().index  # entries only once z exists
inv = ratio.loc[elig] > P3["term_ratio_skip"]
neg = z.loc[elig] < P3["z_skip"]
SIG = {
    "A always short": pd.Series(-1, index=elig),
    "B skip if inverted": pd.Series(np.where(inv, 0, -1), index=elig),
    "C skip if z<-2": pd.Series(np.where(neg, 0, -1), index=elig),
    "D skip if either": pd.Series(np.where(inv | neg, 0, -1), index=elig),
    "E long if z<-2": pd.Series(np.where(neg, 1, -1), index=elig),
    "F long if inverted": pd.Series(np.where(inv, 1, -1), index=elig),
}
base = backtest.Config(tenor_days=P3["tenor_days"], entry_every=P3["entry_every"],
                       iv_mult=P3["iv_mult"], option_cost=P3["option_cost"],
                       hedge_cost_bp=P3["hedge_cost_bp"], notional=P3["notional"])


def stats(daily: pd.Series, trades: pd.DataFrame) -> dict:
    d = daily.loc[elig[0]:]
    yrs = len(d) / 252
    eq = d.cumsum()
    dd = (eq - eq.cummax()).min()
    live = trades[trades["sign"] != 0] if len(trades) else trades
    tot = live["total"] if len(live) else pd.Series(dtype=float)
    q01 = d.quantile(0.01)
    return {"ann_pnl": d.sum() / yrs, "ann_vol": d.std() * np.sqrt(252),
            "sharpe": d.mean() / d.std() * np.sqrt(252) if d.std() > 0 else None,
            "max_dd": dd, "worst_day": d.min(), "worst_day_date": str(d.idxmin().date()),
            "var99_hist": -q01, "es99_hist": -d[d <= q01].mean(),
            "n_trades": int(len(live)), "n_flat": int((trades["sign"] == 0).sum()) if len(trades) else 0,
            "n_long": int((live["sign"] > 0).sum()) if len(live) else 0,
            "mean_trade": tot.mean() if len(tot) else None,
            "t_mean_trade": tot.mean() / tot.std() * np.sqrt(len(tot)) if len(tot) > 1 else None,
            "win_rate": (tot > 0).mean() if len(tot) else None,
            "worst_trade": tot.min() if len(tot) else None}


def run_all(c: backtest.Config) -> dict:
    out = {}
    for k, s in SIG.items():
        d, t = backtest.run(s, mkt, c)
        out[k] = (d, t)
    return out


summ: dict = {"sample": [str(elig[0].date()), str(mkt.index[-1].date())],
              "config": {k: getattr(base, k) for k in ("tenor_days", "entry_every", "iv_mult",
                                                       "option_cost", "hedge_cost_bp", "notional")},
              "n_entry_days_inverted": int(inv.sum()), "n_entry_days_z_lt_-2": int(neg.sum()),
              "n_entry_days": len(elig)}
primary = run_all(base)
summ["primary"] = {k: stats(*v) for k, v in primary.items()}

# what did the skipped trades of A earn? (the value of each filter)
tA = primary["A always short"][1].set_index("entry")
for name, mask in {"inverted": inv, "z<-2": neg}.items():
    ent = [d for d in tA.index if mask.get(d, False)]
    x = tA.loc[ent, "total"]
    summ.setdefault("A_trades_when", {})[name] = {"n": len(x), "mean": x.mean() if len(x) else None,
                                                  "sum": x.sum(), "values": [round(v) for v in x]}
x = tA["total"]
summ["A_trade_dist"] = {"n": len(x), "mean": x.mean(), "median": x.median(), "min": x.min(),
                        "p05": x.quantile(.05), "max": x.max()}
summ["A_vrp"] = {"mean_iv0": tA["iv0"].mean(), "mean_realized": tA["realized_vol"].mean(),
                 "share_iv_gt_rv": float((tA["iv0"] > tA["realized_vol"]).mean())}

# sensitivities (A..F for each variant)
variants = {"unhedged": replace(base, hedge=False),
            "daily ladder": replace(base, entry_every=1),
            "iv_mult 0.85": replace(base, iv_mult=0.85), "iv_mult 1.0": replace(base, iv_mult=1.0),
            "cost 0.5%": replace(base, option_cost=0.005), "cost 2%": replace(base, option_cost=0.02)}
summ["sensitivity"] = {}
ladder = None
for vn, vc in variants.items():
    res = run_all(vc)
    if vn == "daily ladder":
        ladder = res
    summ["sensitivity"][vn] = {k: {m: stats(*v)[m] for m in ("ann_pnl", "sharpe", "max_dd", "n_trades")}
                               for k, v in res.items()}
    print(vn, "done", flush=True)

# paired daily P&L differences (Newey-West t), daily ladder has ~2,000 trades
from ssr.risk.rolling import diebold_mariano  # noqa: E402
summ["paired_diff_tstats"] = {}
for lab, res in (("monthly", primary), ("daily ladder", ladder)):
    dd_ = {k: v[0].loc[elig[0]:] for k, v in res.items()}
    summ["paired_diff_tstats"][lab] = {
        f"{a[:1]} - {b[:1]}": diebold_mariano(-dd_[a], -dd_[b], lags=21)
        for a, b in (("C skip if z<-2", "A always short"), ("B skip if inverted", "A always short"),
                     ("C skip if z<-2", "B skip if inverted"), ("D skip if either", "B skip if inverted"),
                     ("E long if z<-2", "A always short"), ("E long if z<-2", "F long if inverted"))}

# Greeks of a typical short position at open (primary A)
summ["greeks_at_open_A_mean"] = {c: tA[c].mean() for c in ("vega0", "gamma0", "theta0_day", "premium")}

# ---------------- stress: one fresh short straddle, $1M notional ----------------
S0, T0 = 100.0, 30 / 365
r0 = float(mkt["r"].iloc[-1])
vix_med = float(mkt["vix"].median())
sig0 = P3["iv_mult"] * vix_med / 100
n = -P3["notional"] / S0
V0 = n * bs.straddle(S0, S0, T0, r0, sig0)["price"]
hedge = -n * bs.straddle(S0, S0, T0, r0, sig0)["delta"]
rows = []
for x in (-0.10, -0.05, 0.0, 0.05, 0.10):
    S1 = S0 * np.exp(x)
    vols = {"vol unchanged": sig0, "vol +20 pts": sig0 + 0.20, "vol -30% (crush)": sig0 * 0.7}
    for hn, hp in (("Heston link (unconstrained)", P3["heston_unconstrained"]),
                   ("Heston link (Feller)", P3["heston_feller"])):
        # E[dv | dlnS = x] = rho*sigma_v*x (first order). For up-moves this linear
        # map drives variance towards 0, so the new vol is floored at 0.7*sigma0
        # (the same size as the "crush" scenario).
        v1 = sig0**2 + hp["rho"] * hp["sigma"] * x
        vols[hn] = max(np.sqrt(max(v1, 0.0)), 0.7 * sig0)
    for vn, s1 in vols.items():
        V1 = n * bs.straddle(S1, S0, T0 - 1 / 365, r0, s1)["price"]
        rows.append({"spot_move": x, "vol_scenario": vn, "new_vol": s1,
                     "pnl_hedged": (V1 - V0) + hedge * (S1 - S0),
                     "pnl_unhedged": V1 - V0})
stress = pd.DataFrame(rows)
stress.to_csv(REP / "phase3_stress.csv", index=False, float_format="%.6g")
summ["stress_setup"] = {"vix_median": vix_med, "sigma0": sig0, "premium": -V0}


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


(REP / "phase3_summary.json").write_text(json.dumps(clean(summ), indent=2, ensure_ascii=False))
pd.concat({k: v[0] for k, v in primary.items()}, axis=1).to_csv(
    ROOT / "data/processed/phase3_daily_pnl.csv", float_format="%.2f")
pd.concat({k: v[1] for k, v in primary.items()}).to_csv(REP / "phase3_trades.csv", float_format="%.6g")

fig, axs = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
for k, (d, _) in primary.items():
    axs[0].plot(d.loc[elig[0]:].cumsum() / 1e3, label=k, lw=1.1)
axs[0].set_ylabel("cum P&L, $k (monthly, hedged)"); axs[0].legend(fontsize=8, ncol=3)
for k, (d, _) in ladder.items():
    axs[1].plot(d.loc[elig[0]:].cumsum() / 1e3, label=k, lw=1.1)
axs[1].set_ylabel("cum P&L, $k (daily ladder)")
fig.tight_layout(); fig.savefig(FIG / "phase3_cum_pnl.png", dpi=130); plt.close(fig)
print(json.dumps(clean({k: summ[k] for k in ("primary", "A_trades_when", "A_trade_dist", "A_vrp")}), indent=1))

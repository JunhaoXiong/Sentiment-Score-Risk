"""Forward test report from stored snapshots.

Usage: python scripts/forward_report.py STORE_DIR
STORE_DIR is a checkout of the ``snapshots`` branch (chains/*.csv.gz, gdelt/).
Outputs reports/forward_summary.json and reports/forward_daily.csv.

1. Proxy check: real 30-day ATM IV vs VIX on every snapshot day
   (the backtest assumed ATM IV = 0.9 x VIX).
2. Paper trades of signals A-D on real quotes (short at bid, marks at mid).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ssr.data import gdelt, option_snapshots  # noqa: E402
from ssr.options import forward  # noqa: E402
from ssr.sentiment import align, standardize  # noqa: E402

store = Path(sys.argv[1])
cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
P3 = cfg["phase3"]
files = sorted((store / "chains").glob("spy_chain_*.csv.gz"))
chains = {pd.Timestamp(f.name[10:20]): pd.read_csv(f) for f in files}
days = pd.DatetimeIndex(sorted(chains))
r = 0.04

rows = []
for d, ch in chains.items():
    a = option_snapshots.atm_iv_30d(ch, r)
    rows.append({"date": d, **a, "quote_ok_share": float(ch["quote_ok"].mean())})
daily = pd.DataFrame(rows).set_index("date")
daily["ratio_iv_vix"] = daily["atm_iv_30d"] / (daily["vix"] / 100)
daily["term_ratio"] = daily["vix"] / daily["vix3m"]

# sentiment z on snapshot days (honest alignment), if recent GDELT exists
z = pd.Series(np.nan, index=days)
gd = store / "gdelt"
if (gd / "recent_timelinetone.json").exists() and (gd / "recent_timelinevolraw.json").exists():
    tone = gdelt.parse_timeline((gd / "recent_timelinetone.json").read_bytes())
    vol = gdelt.parse_timeline((gd / "recent_timelinevolraw.json").read_bytes())
    cal = pd.bdate_range(tone.index.min(), days.max() + pd.Timedelta(days=5))
    g = align.aggregate_gdelt(tone.join(vol), cal)
    z = standardize.rolling_z(g["tone"], cfg["sentiment"]["zscore_window"],
                              cfg["sentiment"]["zscore_min_periods"]).reindex(days)
daily["z"] = z
inv = daily["term_ratio"] > P3["term_ratio_skip"]
neg = daily["z"] < P3["z_skip"]
signals = {"A always short": pd.Series(-1, index=days),
           "B skip if inverted": pd.Series(np.where(inv, 0, -1), index=days),
           "C skip if z<-2": pd.Series(np.where(neg, 0, -1), index=days),
           "D skip if either": pd.Series(np.where(inv | neg, 0, -1), index=days)}
pnl = {}
for k, s in signals.items():
    res = forward.paper_trade(chains, s, r=r, notional=P3["notional"], entry_every=P3["entry_every"])
    pnl[k] = res.set_index("date")["pnl"] if len(res) else pd.Series(dtype=float)
    daily[f"pnl {k[:1]}"] = pnl[k]

out = {"n_snapshot_days": len(days),
       "first": str(days.min().date()) if len(days) else None,
       "last": str(days.max().date()) if len(days) else None,
       "iv_vix_ratio": {"mean": daily["ratio_iv_vix"].mean(), "median": daily["ratio_iv_vix"].median(),
                        "min": daily["ratio_iv_vix"].min(), "max": daily["ratio_iv_vix"].max(),
                        "assumed_in_backtest": P3["iv_mult"]},
       "paper_total_pnl": {k: float(v.sum()) for k, v in pnl.items()},
       "z_available_days": int(daily["z"].notna().sum())}
(ROOT / "reports").mkdir(exist_ok=True)
daily.to_csv(ROOT / "reports/forward_daily.csv", float_format="%.6g")
(ROOT / "reports/forward_summary.json").write_text(json.dumps(
    json.loads(pd.Series(out).to_json()), indent=2))
print(json.dumps(json.loads(pd.Series(out).to_json()), indent=1))

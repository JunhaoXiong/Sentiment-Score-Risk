"""Phase 2c: is the two-session lag what kills the sentiment signal?

Hourly GDELT bins (needed for an honest 16:00 ET cut) could not be downloaded:
from this environment short-window DOC API requests were rate-limited (HTTP 429)
for 25+ minutes without a single success, and the API ignores TIMEZONEADJ for
daily bins. So we run an UPPER-BOUND experiment instead:

  honest : UTC day d usable from the close of the next session (Phase 2)
  bound  : UTC day d treated as usable at close of d itself, i.e. one session
           earlier. This leaks the last 3-4 h of day-d news (16:00-20:00 ET)
           into the day-(d+1) forecast, so it is *optimistic* for sentiment.

If even the optimistic version does not beat GJR / GJR+VIX, a correctly-cut
intraday signal would not either, and the Phase 2 conclusion is robust to
timing. NOT a valid production signal.
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
from ssr.data import gdelt, market  # noqa: E402
from ssr.risk import coverage_tests as ct, garchx, rolling  # noqa: E402
from ssr.sentiment import align, standardize  # noqa: E402

cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
prim = cfg["gdelt"]["primary"]
W, EVERY = cfg["phase2"]["window"], cfg["phase2"]["refit_every"]
ZW, ZMP = cfg["sentiment"]["zscore_window"], cfg["sentiment"]["zscore_min_periods"]

px = market.load(ROOT / "data/raw/market.csv")
days = px.index
r = np.log(px["SPY"]).diff()
vix = (px["^VIX"].shift(1) / 100) ** 2 / 252
raw = gdelt.load(prim, ROOT / "data/raw/gdelt")

out: dict = {"note": "bound variant leaks 3-4h of post-close news; upper bound only"}
z = {}
for name, shift in {"honest": 0.0, "bound": -5.0}.items():
    g = align.aggregate_gdelt(raw, days, pub_shift_hours=shift)
    z[name] = standardize.rolling_z(g["tone"], ZW, ZMP).reindex(days)

zz = pd.DataFrame(z)
absr = r.abs()
out["corr_z_absr_same_day"] = {k: zz[k].corr(absr) for k in zz}
out["corr_z_absr_next_day"] = {k: zz[k].corr(absr.shift(-1)) for k in zz}

# common sample: both z available
d = pd.DataFrame({"r": r, "vix": vix,
                  "neg_honest": np.maximum(0, -zz["honest"]) * 1e-4,
                  "neg_bound": np.maximum(0, -zz["bound"]) * 1e-4}).loc[
    cfg["sample"]["start"]:cfg["sample"]["end"]].dropna()
out["sample"] = [str(d.index[0].date()), str(d.index[-1].date()), len(d)]

MODELS = {"M0 GJR": [], "M1 GJR+VIX": ["vix"],
          "M2h GJR+SENT(honest)": ["neg_honest"], "M2b GJR+SENT(bound)": ["neg_bound"],
          "M3h GJR+VIX+SENT(honest)": ["vix", "neg_honest"],
          "M3b GJR+VIX+SENT(bound)": ["vix", "neg_bound"]}
full, fcs = {}, {}
for m, cols in MODELS.items():
    f = garchx.fit(d["r"].values, d[cols].values if cols else None)
    full[m] = {"loglik": f.loglik, "delta": dict(zip(cols, f.delta)), "converged": f.converged}
    fcs[m], _ = rolling.rolling_var(d["r"], d[cols] if cols else None, W=W, refit_every=EVERY)
    print(m, "done", flush=True)
out["full_sample"] = full
out["oos"] = [str(fcs["M0 GJR"].index[0].date()), str(fcs["M0 GJR"].index[-1].date()),
              len(fcs["M0 GJR"])]

bt = {}
for m, fc in fcs.items():
    bt[m] = {}
    for a in (0.95, 0.99):
        tag = int(a * 100)
        s = ct.backtest_summary(fc[f"hit{tag}"].values, 1 - a)
        bt[m][tag] = {"violations": s["violations"], "expected": s["expected"],
                      "kupiec_p": s["kupiec_p"], "cc_p": s["cc_p"],
                      "tick_x1e4": rolling.tick_loss(fc["r"], fc[f"var{tag}"], a).mean() * 1e4}
out["backtests"] = bt
dm = {}
for a in (0.95, 0.99):
    tag = int(a * 100)
    L = {k: rolling.tick_loss(v["r"], v[f"var{tag}"], a) for k, v in fcs.items()}
    dm[tag] = {
        "M2b vs M0": rolling.diebold_mariano(L["M2b GJR+SENT(bound)"], L["M0 GJR"]),
        "M3b vs M1": rolling.diebold_mariano(L["M3b GJR+VIX+SENT(bound)"], L["M1 GJR+VIX"]),
        "M3h vs M1": rolling.diebold_mariano(L["M3h GJR+VIX+SENT(honest)"], L["M1 GJR+VIX"]),
    }
out["diebold_mariano"] = dm
w = slice("2025-03-21", "2025-04-03")  # 10 sessions before the 2025-04-04 low
out["tariff_pre10_var99_uplift"] = {
    "M3h over M1": fcs["M3h GJR+VIX+SENT(honest)"].loc[w, "var99"].mean() / fcs["M1 GJR+VIX"].loc[w, "var99"].mean() - 1,
    "M3b over M1": fcs["M3b GJR+VIX+SENT(bound)"].loc[w, "var99"].mean() / fcs["M1 GJR+VIX"].loc[w, "var99"].mean() - 1,
}


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else round(float(o), 6)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


(ROOT / "reports/phase2c_summary.json").write_text(json.dumps(clean(out), indent=2, ensure_ascii=False))
print(json.dumps(clean(out), indent=1, ensure_ascii=False))

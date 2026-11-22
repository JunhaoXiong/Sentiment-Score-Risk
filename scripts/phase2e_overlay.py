"""Phase 2e: does a sentiment-triggered VaR add-on beat a flat add-on of equal cost?

Base model: M1 GJR+VIX rolling forecasts from Phase 2.
Calibration: first half of the OOS period (choose m / k by mean tick loss).
Test: second half. Benchmark: flat uplift with the SAME average VaR in the test half.
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
from ssr.risk import coverage_tests as ct, overlay, rolling  # noqa: E402

cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
prim = cfg["gdelt"]["primary"]
p = pd.read_csv(ROOT / "data/processed/panel.csv", index_col=0, parse_dates=True)
z = p[f"z_{prim}"]
fc = pd.read_csv(ROOT / "data/processed/phase2_forecasts.csv", header=[0, 1], index_col=0,
                 parse_dates=True)["M1 GJR+VIX"]
r = fc["r"]
SPLIT = "2022-09-30"
cal, test = slice(None, SPLIT), slice(pd.Timestamp(SPLIT) + pd.Timedelta(days=1), None)
out: dict = {"base": "M1 GJR+VIX", "split": SPLIT,
             "cal": [str(fc.loc[cal].index[0].date()), str(fc.loc[cal].index[-1].date()), int(len(fc.loc[cal]))],
             "test": [str(fc.loc[test].index[0].date()), str(fc.loc[test].index[-1].date()), int(len(fc.loc[test]))]}


def evaluate(v: pd.Series, a: float, sl) -> dict:
    vv, rr = v.loc[sl], r.loc[sl]
    hits = (-rr > vv).astype(int).values
    s = ct.backtest_summary(hits, 1 - a)
    return {"hits": s["violations"], "expected": round(s["expected"], 1),
            "kupiec_p": s["kupiec_p"], "cc_p": s["cc_p"],
            "tick_x1e4": rolling.tick_loss(rr, vv, a).mean() * 1e4,
            "avg_var_pct": vv.mean() * 100}


res = {}
for a in (0.95, 0.99):
    tag = int(a * 100)
    base = fc[f"var{tag}"]
    grid_m = np.round(np.arange(1.0, 2.01, 0.05), 2)
    grid_k = np.round(np.arange(0.0, 1.01, 0.05), 2)
    tl_m = {m: rolling.tick_loss(r.loc[cal], overlay.step(base, z, m).loc[cal], a).mean() for m in grid_m}
    tl_k = {k: rolling.tick_loss(r.loc[cal], overlay.smooth(base, z, k).loc[cal], a).mean() for k in grid_k}
    m_star = min(tl_m, key=tl_m.get)
    k_star = min(tl_k, key=tl_k.get)
    step_v = overlay.step(base, z, m_star)
    smooth_v = overlay.smooth(base, z, k_star)
    # flat benchmark: in calibration chosen by tick loss; in test matched to each overlay's cost
    grid_u = np.round(np.arange(0.0, 0.61, 0.02), 2)
    tl_u = {u: rolling.tick_loss(r.loc[cal], overlay.flat(base, u).loc[cal], a).mean() for u in grid_u}
    u_star = min(tl_u, key=tl_u.get)
    u_step = overlay.matching_flat_uplift(base.loc[test], step_v.loc[test])
    u_smooth = overlay.matching_flat_uplift(base.loc[test], smooth_v.loc[test])
    rows = {
        "base": base, f"step m={m_star}": step_v, f"smooth k={k_star}": smooth_v,
        f"flat u={u_star} (cal-optimal)": overlay.flat(base, u_star),
        f"flat u={u_step:.3f} (same cost as step)": overlay.flat(base, u_step),
        f"flat u={u_smooth:.3f} (same cost as smooth)": overlay.flat(base, u_smooth),
    }
    res[tag] = {"m_star": m_star, "k_star": k_star, "u_star": u_star,
                "cal_tick_by_m": {str(k): v * 1e4 for k, v in tl_m.items()},
                "cal_tick_by_k": {str(k): v * 1e4 for k, v in tl_k.items()},
                "test": {name: evaluate(v, a, test) for name, v in rows.items()},
                "cal": {name: evaluate(v, a, cal) for name, v in rows.items()},
                "n_trigger_days_test": int((z.reindex(fc.loc[test].index) < -2).sum()),
                "n_trigger_days_cal": int((z.reindex(fc.loc[cal].index) < -2).sum())}
    L = {name: rolling.tick_loss(r.loc[test], v.loc[test], a) for name, v in rows.items()}
    res[tag]["dm_test"] = {
        "step vs same-cost flat": rolling.diebold_mariano(L[f"step m={m_star}"], L[f"flat u={u_step:.3f} (same cost as step)"]),
        "smooth vs same-cost flat": rolling.diebold_mariano(L[f"smooth k={k_star}"], L[f"flat u={u_smooth:.3f} (same cost as smooth)"]),
        "step vs base": rolling.diebold_mariano(L[f"step m={m_star}"], L["base"]),
    }
out["results"] = res
# hindsight check (NOT a valid choice): best m on the TEST half
a = 0.99
tl_test = {m: rolling.tick_loss(r.loc[test], overlay.step(fc["var99"], z, m).loc[test], a).mean() * 1e4
           for m in np.round(np.arange(1.0, 2.01, 0.05), 2)}
out["hindsight_best_m_test_99"] = min(tl_test, key=tl_test.get)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else round(float(o), 6)
    if isinstance(o, np.integer):
        return int(o)
    return o


(ROOT / "reports/phase2e_summary.json").write_text(json.dumps(clean(out), indent=2, ensure_ascii=False))
print(json.dumps(clean({k: v for k, v in out.items() if k != "results"}), ensure_ascii=False))
for tag, rr in out["results"].items():
    print(f"== {tag}%: m*={rr['m_star']} k*={rr['k_star']} u*={rr['u_star']} triggers cal/test={rr['n_trigger_days_cal']}/{rr['n_trigger_days_test']}")
    for part in ("cal", "test"):
        for name, e in rr[part].items():
            print(part, f"{name:40s}", e["hits"], e["expected"], round(e["kupiec_p"], 3), round(e["tick_x1e4"], 4), round(e["avg_var_pct"], 3))
    print(clean(rr["dm_test"]))

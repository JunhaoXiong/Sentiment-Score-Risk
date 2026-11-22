"""Phase 2d: does the part of news tone NOT explained by past market moves help?

Step 1  For each day t, regress tone-z on information known at close(t-1):
        r_{t-1..t-5}, |r|_{t-1..t-5}, dlogVIX_{t-1}, VIX_{t-1}
        using only the previous 250 days (rolling, out-of-sample). The
        residual is the "news surprise" s_t.
Step 2  Standardise s_t against its own past (rolling z) and put
        max(0, -s_t) * 1e-4 into GJR-GARCH-X, as in Phase 2.
If the surprise still adds nothing, the tone index carries no forward-looking
information beyond what returns and VIX already contain.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ssr.risk import coverage_tests as ct, garchx, rolling  # noqa: E402
from ssr.sentiment import standardize  # noqa: E402

cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
prim = cfg["gdelt"]["primary"]
W, EVERY = cfg["phase2"]["window"], cfg["phase2"]["refit_every"]
p = pd.read_csv(ROOT / "data/processed/panel.csv", index_col=0, parse_dates=True)
z = p[f"z_{prim}"]

# regressors known at close t-1
lag = pd.DataFrame(index=p.index)
for k in range(1, 6):
    lag[f"r_l{k}"] = p["r"].shift(k)
    lag[f"abs_l{k}"] = p["abs_r"].shift(k)
lag["dlogvix_l1"] = p["dlog_vix"].shift(1)
lag["vix_l1"] = p["vix_prev"]

out: dict = {}
# descriptive, in-sample: how much of z do past moves explain?
dd = pd.concat([z.rename("z"), lag], axis=1).dropna()
ins = sm.OLS(dd["z"], sm.add_constant(dd.drop(columns="z"))).fit()
out["in_sample_r2_z_on_past_market"] = ins.rsquared
out["n_in_sample"] = int(ins.nobs)

s = standardize.rolling_residual(z, lag, window=250, min_periods=120)
fitted = z - s
out["oos_corr_z_fitted"] = z.corr(fitted)
zs = standardize.rolling_z(s, 250, 120)
out["corr"] = {
    name: {"abs_r_t": x.corr(p["abs_r"]), "abs_r_t+1": x.corr(p["abs_r"].shift(-1)),
           "rv5_fwd": x.corr(p["rv5_fwd"])}
    for name, x in {"z": z, "surprise_z": zs}.items()}

d = pd.DataFrame({"r": p["r"], "vix": (p["vix_prev"] / 100) ** 2 / 252,
                  "neg_z": np.maximum(0, -z) * 1e-4,
                  "neg_s": np.maximum(0, -zs) * 1e-4}).dropna()
out["sample"] = [str(d.index[0].date()), str(d.index[-1].date()), len(d)]

MODELS = {"M0 GJR": [], "M1 GJR+VIX": ["vix"], "M3 GJR+VIX+SENT": ["vix", "neg_z"],
          "M4 GJR+SURPRISE": ["neg_s"], "M5 GJR+VIX+SURPRISE": ["vix", "neg_s"]}
full, fcs = {}, {}
for m, cols in MODELS.items():
    f = garchx.fit(d["r"].values, d[cols].values if cols else None)
    full[m] = {"loglik": f.loglik, "delta": dict(zip(cols, f.delta))}
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
        sm_ = ct.backtest_summary(fc[f"hit{tag}"].values, 1 - a)
        bt[m][tag] = {"violations": sm_["violations"], "expected": sm_["expected"],
                      "kupiec_p": sm_["kupiec_p"], "cc_p": sm_["cc_p"],
                      "tick_x1e4": rolling.tick_loss(fc["r"], fc[f"var{tag}"], a).mean() * 1e4}
out["backtests"] = bt
L = {a: {k: rolling.tick_loss(v["r"], v[f"var{int(a*100)}"], a) for k, v in fcs.items()}
     for a in (0.95, 0.99)}
out["diebold_mariano"] = {int(a * 100): {
    "M4 vs M0": rolling.diebold_mariano(L[a]["M4 GJR+SURPRISE"], L[a]["M0 GJR"]),
    "M5 vs M1": rolling.diebold_mariano(L[a]["M5 GJR+VIX+SURPRISE"], L[a]["M1 GJR+VIX"]),
    "M3 vs M1": rolling.diebold_mariano(L[a]["M3 GJR+VIX+SENT"], L[a]["M1 GJR+VIX"])}
    for a in (0.95, 0.99)}


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


(ROOT / "reports/phase2d_summary.json").write_text(json.dumps(clean(out), indent=2, ensure_ascii=False))
print(json.dumps(clean(out), indent=1, ensure_ascii=False))

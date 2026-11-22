"""Phase 2: does negative news tone improve one-day SPY VaR beyond GJR and VIX?

Models (all GJR-GARCH, Student-t, same days, same windows):
  M0 GJR            no exogenous term
  M1 GJR+VIX        x = (VIX_{t-1}/100)^2/252
  M2 GJR+SENT       x = max(0, -z_t) * 1e-4
  M3 GJR+VIX+SENT   both
Rolling W=500 (config), refit every 21 days, daily variance update.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy.stats import chi2  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ssr.risk import coverage_tests as ct, garchx, rolling  # noqa: E402

cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
P2 = cfg["phase2"]
REP, FIG, PROC = ROOT / "reports", ROOT / "reports/figures", ROOT / "data/processed"
prim = cfg["gdelt"]["primary"]

panel = pd.read_csv(PROC / "panel.csv", index_col=0, parse_dates=True)
z = panel[f"z_{prim}"]
d = pd.DataFrame({
    "r": panel["r"],
    "vix": (panel["vix_prev"] / 100) ** 2 / 252,
    "neg_z": np.maximum(0.0, -z) * 1e-4,
}).dropna()
print("modelling sample", d.index.min().date(), d.index.max().date(), len(d))

MODELS = {"M0 GJR": [], "M1 GJR+VIX": ["vix"], "M2 GJR+SENT": ["neg_z"],
          "M3 GJR+VIX+SENT": ["vix", "neg_z"]}
W, EVERY = P2["window"], P2["refit_every"]
summ: dict = {"sample_start": str(d.index.min().date()), "sample_end": str(d.index.max().date()),
              "n_sample": len(d), "window": W, "refit_every": EVERY}

# ---------------- full-sample fits (in-sample, descriptive) ----------------
full = {}
for name, cols in MODELS.items():
    f = garchx.fit(d["r"].values, d[cols].values if cols else None)
    full[name] = f
    summ.setdefault("full_sample", {})[name] = {
        "alpha": f.alpha, "gamma": f.gamma, "beta": f.beta, "nu": f.nu,
        "delta": dict(zip(cols, f.delta)), "loglik": f.loglik, "aic": f.aic,
        "converged": f.converged}


def lr_boundary(f_big, f_small):
    """LR for one delta on the boundary 0: p = 0.5 * P(chi2_1 > LR)."""
    lr = max(0.0, 2 * (f_big.loglik - f_small.loglik))
    return {"lr": lr, "p": 0.5 * chi2.sf(lr, 1)}


summ["lr_tests_in_sample"] = {
    "SENT vs GJR (M2 vs M0)": lr_boundary(full["M2 GJR+SENT"], full["M0 GJR"]),
    "SENT beyond VIX (M3 vs M1)": lr_boundary(full["M3 GJR+VIX+SENT"], full["M1 GJR+VIX"]),
    "VIX vs GJR (M1 vs M0)": lr_boundary(full["M1 GJR+VIX"], full["M0 GJR"]),
}

# ---------------- rolling out-of-sample ----------------
fcs, fits = {}, {}
for name, cols in MODELS.items():
    X = d[cols] if cols else None
    fcs[name], fits[name] = rolling.rolling_var(d["r"], X, W=W, refit_every=EVERY)
    print(name, "done", len(fcs[name]))
oos = fcs["M0 GJR"].index
summ["oos_start"], summ["oos_end"], summ["n_oos"] = str(oos[0].date()), str(oos[-1].date()), len(oos)
summ["nonconverged_refits"] = {k: int(sum(not f["converged"] for f in v)) for k, v in fits.items()}

bt = {}
for name, fc in fcs.items():
    bt[name] = {}
    for a in (0.95, 0.99):
        tag = int(a * 100)
        s = ct.backtest_summary(fc[f"hit{tag}"].values, 1 - a)
        tl = rolling.tick_loss(fc["r"], fc[f"var{tag}"], a)
        s["tick_loss_x1e4"] = tl.mean() * 1e4
        s["avg_var"] = fc[f"var{tag}"].mean()
        # ES check: average loss on hit days vs average ES forecast on those days
        h = fc[f"hit{tag}"] == 1
        s["es_ratio"] = (fc.loc[h, "loss"].mean() / fc.loc[h, f"es{tag}"].mean()) if h.any() else None
        bt[name][tag] = s
summ["backtests"] = bt

dm = {}
for a in (0.95, 0.99):
    tag = int(a * 100)
    L = {k: rolling.tick_loss(v["r"], v[f"var{tag}"], a) for k, v in fcs.items()}
    dm[tag] = {"M2 vs M0": rolling.diebold_mariano(L["M2 GJR+SENT"], L["M0 GJR"]),
               "M3 vs M1": rolling.diebold_mariano(L["M3 GJR+VIX+SENT"], L["M1 GJR+VIX"]),
               "M1 vs M0": rolling.diebold_mariano(L["M1 GJR+VIX"], L["M0 GJR"])}
summ["diebold_mariano_tick_loss"] = dm

# ---------------- crisis windows: does VaR rise earlier? ----------------
crises = {"COVID 2020": ("2020-02-10", "2020-04-15"),
          "2022 rate shock (Jun)": ("2022-05-25", "2022-06-30"),
          "Tariff shock 2025-04": ("2025-03-20", "2025-04-30")}
summ["crises"] = {}
for k, (a, b) in crises.items():
    w = {m: fc.loc[a:b] for m, fc in fcs.items()}
    worst = w["M0 GJR"]["r"].idxmin()
    pre = slice(None, worst)
    ent = {"worst_day": str(worst.date())}
    for m, x in w.items():
        p10 = x.loc[pre].iloc[-11:-1]  # 10 trading days before the worst day
        ent[m] = {"hits99": int(x["hit99"].sum()), "hits95": int(x["hit95"].sum()),
                  "var99_pre10_mean": p10["var99"].mean()}
    ent["sent_uplift_pre10_M2_over_M0"] = ent["M2 GJR+SENT"]["var99_pre10_mean"] / ent["M0 GJR"]["var99_pre10_mean"] - 1
    ent["sent_uplift_pre10_M3_over_M1"] = ent["M3 GJR+VIX+SENT"]["var99_pre10_mean"] / ent["M1 GJR+VIX"]["var99_pre10_mean"] - 1
    summ["crises"][k] = ent

# false alarms: average VaR uplift of M3 over M1 on calm days with large negative z
calm = fcs["M1 GJR+VIX"]["hit95"] == 0
ratio = fcs["M3 GJR+VIX+SENT"]["var99"] / fcs["M1 GJR+VIX"]["var99"] - 1
zz = z.reindex(ratio.index)
summ["uplift_M3_over_M1"] = {"mean": ratio.mean(), "p95": ratio.quantile(.95),
                            "max": ratio.max(), "max_day": str(ratio.idxmax().date()),
                            "mean_when_z_lt_-2": ratio[zz < -2].mean(),
                            "n_days_z_lt_-2": int((zz < -2).sum())}


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


(REP / "phase2_summary.json").write_text(json.dumps(clean(summ), indent=2, ensure_ascii=False))
pd.concat({k: v for k, v in fcs.items()}, axis=1).to_csv(PROC / "phase2_forecasts.csv", float_format="%.6g")
pd.concat({k: pd.DataFrame(v) for k, v in fits.items()}).to_csv(REP / "phase2_rolling_params.csv", float_format="%.6g")

# ---------------- figures ----------------
fig, axs = plt.subplots(len(crises), 1, figsize=(11, 9))
for ax, (k, (a, b)) in zip(axs, crises.items()):
    loss = fcs["M0 GJR"].loc[a:b, "loss"]
    ax.bar(loss.index, loss.clip(lower=0) * 100, color="0.75", label="SPY loss %")
    for m, c in zip(MODELS, ["k", "C0", "C3", "C1"]):
        ax.plot(fcs[m].loc[a:b].index, fcs[m].loc[a:b, "var99"] * 100, color=c, lw=1.2, label=m)
    ax.set_title(f"99% VaR, {k}", fontsize=10); ax.set_ylabel("%")
axs[0].legend(fontsize=8, ncol=5)
fig.tight_layout(); fig.savefig(FIG / "phase2_crisis_var.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(ratio.index, ratio * 100, lw=.7, color="C1")
ax.set_ylabel("% uplift"); ax.set_title("99% VaR: (GJR+VIX+SENT) / (GJR+VIX) - 1")
fig.tight_layout(); fig.savefig(FIG / "phase2_sent_uplift.png", dpi=130); plt.close(fig)
print(json.dumps(clean(summ), indent=1, ensure_ascii=False)[:6000])

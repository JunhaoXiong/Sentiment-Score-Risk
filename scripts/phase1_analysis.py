"""Phase 1: build the point-in-time sentiment panel and describe it.

Outputs
- data/processed/panel.csv          one row per trading day (effective signals)
- data/processed/fomc_scores.csv    LM scores per FOMC statement
- reports/phase1_summary.json       every number quoted in the report
- reports/figures/*.png

Descriptive only: correlations here are not risk-model evidence (Phase 2).
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
import statsmodels.api as sm  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ssr.data import gdelt, market  # noqa: E402
from ssr.sentiment import align, lexicon, standardize  # noqa: E402

cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
RAW, PROC, REP, FIG = (ROOT / "data/raw", ROOT / "data/processed",
                       ROOT / "reports", ROOT / "reports/figures")
for p in (PROC, FIG):
    p.mkdir(parents=True, exist_ok=True)
S0, S1 = pd.Timestamp(cfg["sample"]["start"]), pd.Timestamp(cfg["sample"]["end"])
W, MP = cfg["sentiment"]["zscore_window"], cfg["sentiment"]["zscore_min_periods"]

# ---------------------------------------------------------------- market
px = market.load(RAW / "market.csv")
days = px.index
mk = pd.DataFrame(index=days)
mk["r"] = np.log(px["SPY"]).diff()
mk["abs_r"] = mk["r"].abs()
mk["vix"] = px["^VIX"]
mk["dlog_vix"] = np.log(px["^VIX"]).diff()
mk["vix_slope"] = px["^VIX"] / px["^VIX3M"]          # >1 = inverted (stress)
mk["vix_prev"] = mk["vix"].shift(1)
# realised vol of the next 5 days (t..t+4), annualised: a *target*, never a feature
mk["rv5_fwd"] = (mk["r"] ** 2)[::-1].rolling(5).mean()[::-1] ** 0.5 * np.sqrt(252)

# ---------------------------------------------------------------- GDELT
gd_dir = RAW / "gdelt"
gpanel = {}
for name in cfg["gdelt"]["queries"]:
    try:
        d = gdelt.load(name, gd_dir)
    except FileNotFoundError:
        print(f"[warn] GDELT files missing for {name}; skipped")
        continue
    if "count" not in d:  # tone only: equal weights
        d["count"], d["total"] = 1.0, np.nan
    g = align.aggregate_gdelt(d, days)
    g["z"] = standardize.rolling_z(g["tone"], W, MP)
    gpanel[name] = g
prim = cfg["gdelt"]["primary"] if cfg["gdelt"]["primary"] in gpanel else next(iter(gpanel))

panel = mk.copy()
for name, g in gpanel.items():
    panel[f"tone_{name}"] = g["tone"]
    panel[f"z_{name}"] = g["z"]
    panel[f"vol_{name}"] = g["vol_share"]
panel = panel.loc[S0:S1]

# ---------------------------------------------------------------- FOMC (LM)
fo = pd.read_csv(RAW / "fomc/fomc_statements.csv", parse_dates=["date"])
fs = lexicon.score_frame(fo)
pub = align.et_time_to_utc(fs["date"], cfg["fomc"]["assumed_release_time_et"])
fs["effective"] = align.effective_day(pub, days).values
fs["z_net"] = standardize.rolling_z(fs["score_net"], window=16, min_periods=8)
fs.drop(columns=["url"]).to_csv(PROC / "fomc_scores.csv", index=False)
panel["fomc_net"] = fs.set_index("effective")["score_net"].reindex(panel.index)
panel.to_csv(PROC / "panel.csv", float_format="%.6g")

# ---------------------------------------------------------------- stats
summ: dict = {"primary_query": prim, "queries": cfg["gdelt"]["queries"],
              "zscore_window": W}
z = f"z_{prim}"
valid = panel.dropna(subset=[z, "r"])
summ["n_days_with_z"] = int(len(valid))
summ["z_first_day"] = str(valid.index.min().date())
for name, g in gpanel.items():
    t = g["tone"].loc[S0:S1]
    summ[f"tone_{name}"] = {"mean": t.mean(), "std": t.std(), "min": t.min(),
                            "min_day": str(t.idxmin().date()),
                            "pct_negative": float((t < 0).mean())}
if len(gpanel) == 2:
    a, b = list(gpanel)
    summ["corr_tone_queries"] = panel[f"tone_{a}"].corr(panel[f"tone_{b}"])


def hac_reg(y, X, lags=5):
    d = pd.concat([y, X], axis=1).dropna()
    m = sm.OLS(d.iloc[:, 0], sm.add_constant(d.iloc[:, 1:])).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"coef": m.params.drop("const").to_dict(),
            "t": m.tvalues.drop("const").to_dict(), "r2": m.rsquared, "n": int(m.nobs)}


summ["corr_z_with"] = {c: valid[z].corr(valid[c]) for c in
                       ["r", "abs_r", "dlog_vix", "vix_prev", "vix_slope", "rv5_fwd"]}
# lead-lag: corr(z_t, |r|_{t+k}); k<0 = z is computed AFTER the return was known
summ["leadlag_corr_z_absr"] = {int(k): valid[z].corr(panel["abs_r"].shift(-k).reindex(valid.index))
                               for k in range(-5, 6)}
# does z add anything to yesterday's VIX for today's |r| and next-5d RV? (teaser)
vix_d = panel["vix_prev"] / 100 / np.sqrt(252)
same = panel[z].notna()  # compare R^2 on the same days
summ["reg_absr_on_vix"] = hac_reg(panel["abs_r"][same], pd.DataFrame({"vix_prev_daily": vix_d})[same])
summ["reg_rv5_on_vix"] = hac_reg(panel["rv5_fwd"][same], pd.DataFrame({"vix_prev": panel["vix_prev"][same] / 100}))
summ["reg_absr_on_vix_z"] = hac_reg(panel["abs_r"], pd.DataFrame(
    {"vix_prev_daily": vix_d, "z": panel[z]}))
summ["reg_rv5_on_vix_z"] = hac_reg(panel["rv5_fwd"], pd.DataFrame(
    {"vix_prev": panel["vix_prev"] / 100, "z": panel[z]}))
# tail days: is z lower before the worst 1% SPY days?
q01 = valid["r"].quantile(0.01)
summ["z_mean_on_worst1pct_days"] = valid.loc[valid["r"] <= q01, z].mean()
summ["z_mean_other_days"] = valid.loc[valid["r"] > q01, z].mean()
summ["n_worst1pct"] = int((valid["r"] <= q01).sum())

summ["fomc"] = {
    "n": int(len(fs)), "words_median": float(fs["n_words"].median()),
    "net_mean": fs["score_net"].mean(), "net_min_date": str(fs.loc[fs["score_net"].idxmin(), "date"].date()),
    "net_max_date": str(fs.loc[fs["score_net"].idxmax(), "date"].date()),
    "unc_share_max_date": str(fs.loc[fs["unc_share"].idxmax(), "date"].date()),
    "corr_net_absr_effective_day": panel["fomc_net"].corr(panel["abs_r"]),
}
crises = {"Volmageddon 2018-02": ("2018-01-15", "2018-02-28"),
          "COVID 2020-02/03": ("2020-02-10", "2020-04-15"),
          "2022 rate shock (Jun)": ("2022-05-25", "2022-06-30"),
          "Tariff shock 2025-04": ("2025-03-20", "2025-04-30")}
summ["crises"] = {}
for k, (a, b) in crises.items():
    w = panel.loc[a:b]
    worst = w["r"].idxmin()
    zmin_day = w[z].idxmin()
    summ["crises"][k] = {"worst_day": str(worst.date()), "worst_r": w["r"].min(),
                         "z_min": w[z].min(), "z_min_day": str(zmin_day.date()),
                         "vix_max_day": str(w["vix"].idxmax().date()),
                         "z_below_-2_first": (str(w.index[w[z] < -2][0].date())
                                              if (w[z] < -2).any() else None)}


def tofloat(o):
    if isinstance(o, dict):
        return {k: tofloat(v) for k, v in o.items()}
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else round(float(o), 6)
    return o


(REP / "phase1_summary.json").write_text(json.dumps(tofloat(summ), indent=2, ensure_ascii=False))

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
ax[0].plot(panel.index, panel[f"tone_{prim}"], lw=.5, color="0.5", label="daily tone")
ax[0].plot(panel.index, panel[f"tone_{prim}"].rolling(21).mean(), color="C0", label="21d mean")
ax[0].set_ylabel("GDELT tone"); ax[0].legend(loc="lower left")
ax[1].plot(panel.index, panel[z], lw=.5, color="C3"); ax[1].axhline(-2, ls="--", c="k", lw=.7)
ax[1].set_ylabel(f"z (W={W}, past only)")
ax[2].plot(panel.index, panel["vix"], color="C2"); ax[2].set_ylabel("VIX")
fig.suptitle(f"GDELT stock-market news tone ({prim}), effective trading-day alignment")
fig.tight_layout(); fig.savefig(FIG / "tone_z_vix.png", dpi=130); plt.close(fig)

fig, axs = plt.subplots(2, 2, figsize=(12, 7))
for axx, (k, (a, b)) in zip(axs.ravel(), crises.items()):
    w = panel.loc[a:b]
    axx.plot(w.index, (1 + w["r"].fillna(0)).cumprod() * 100, color="k", label="SPY (=100)")
    axx.set_title(k, fontsize=10); axx.tick_params(axis="x", labelsize=7, rotation=30)
    a2 = axx.twinx(); a2.plot(w.index, w[z], color="C3", label="tone z")
    a2.axhline(-2, ls="--", c="C3", lw=.6); a2.set_ylabel("z", color="C3")
fig.tight_layout(); fig.savefig(FIG / "crisis_windows.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 3.5))
ll = summ["leadlag_corr_z_absr"]
ax.bar(list(ll), list(ll.values()), color=["0.6" if k < 0 else "C0" for k in ll])
ax.set_xlabel("k  (corr of z_t with |r|_{t+k});  grey: k<0 = return already known")
ax.set_ylabel("corr"); fig.tight_layout(); fig.savefig(FIG / "leadlag.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(fs["date"], fs["score_net"] * 100, marker="o", ms=3, label="LM net (pos-neg)/words, %")
ax.plot(fs["date"], fs["unc_share"] * 100, marker=".", ms=3, label="LM uncertainty share, %")
ax.axhline(0, c="k", lw=.5); ax.legend(); ax.set_title("FOMC statements, Loughran-McDonald scores")
fig.tight_layout(); fig.savefig(FIG / "fomc_lm.png", dpi=130); plt.close(fig)
print(json.dumps(tofloat(summ), indent=1, ensure_ascii=False)[:4000])

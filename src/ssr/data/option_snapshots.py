"""Daily SPY option-chain snapshots from Yahoo (forward test data).

Free historical option quotes do not exist, so we store our own: run once per
trading day ~15:45 ET (bid/ask are 0 outside market hours, which makes the
snapshot useless; such rows are kept but flagged). Stored per expiry within
``max_days`` calendar days, strikes within +-``moneyness`` of spot.

Columns: snapshot_utc, expiry, kind, strike, bid, ask, last, volume,
open_interest, yahoo_iv, spot, vix, vix3m, quote_ok
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def collect(max_days: int = 70, moneyness: float = 0.15) -> pd.DataFrame:
    import yfinance as yf

    now = pd.Timestamp.now(tz="UTC")
    spy = yf.Ticker("SPY")
    spot = float(spy.history(period="1d", interval="1m")["Close"].iloc[-1])
    vix = float(yf.Ticker("^VIX").history(period="1d")["Close"].iloc[-1])
    vix3m = float(yf.Ticker("^VIX3M").history(period="1d")["Close"].iloc[-1])
    rows = []
    for exp in spy.options:
        days = (pd.Timestamp(exp) - now.tz_localize(None).normalize()).days
        if days < 1 or days > max_days:
            continue
        ch = spy.option_chain(exp)
        for kind, df in (("call", ch.calls), ("put", ch.puts)):
            d = df[(df["strike"] >= spot * (1 - moneyness)) & (df["strike"] <= spot * (1 + moneyness))]
            rows.append(pd.DataFrame({
                "expiry": exp, "kind": kind, "strike": d["strike"], "bid": d["bid"],
                "ask": d["ask"], "last": d["lastPrice"], "volume": d["volume"],
                "open_interest": d["openInterest"], "yahoo_iv": d["impliedVolatility"]}))
    out = pd.concat(rows, ignore_index=True)
    out.insert(0, "snapshot_utc", now.isoformat())
    out["spot"], out["vix"], out["vix3m"] = spot, vix, vix3m
    out["quote_ok"] = (out["bid"] > 0) & (out["ask"] > out["bid"])
    return out


def save(df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    day = pd.Timestamp(df["snapshot_utc"].iloc[0]).tz_convert("America/New_York").strftime("%Y-%m-%d")
    f = out_dir / f"spy_chain_{day}.csv.gz"
    df.to_csv(f, index=False, compression={"method": "gzip", "mtime": 0})
    manifest = out_dir / "manifest.json"
    m = json.loads(manifest.read_text()) if manifest.exists() else {}
    m[f.name] = {"sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "rows": len(df),
                 "quote_ok_share": float(df["quote_ok"].mean())}
    manifest.write_text(json.dumps(m, indent=2, sort_keys=True))
    return f


def atm_iv_30d(chain: pd.DataFrame, r: float, target_days: float = 30.0) -> dict:
    """30-day ATM implied vol from one snapshot.

    For each expiry: straddle mid at the strike nearest spot, inverted with BS
    (q = 0) to one vol. Then interpolate total variance sigma^2 T linearly in
    T between the two expiries bracketing 30 days (the VIX convention).
    Only quotes with bid > 0 and ask > bid are used.
    """
    from scipy.optimize import brentq

    from ssr.options import bs

    c = chain[chain["quote_ok"]].copy()
    if c.empty:
        return {"ok": False}
    snap = pd.Timestamp(c["snapshot_utc"].iloc[0])
    S = float(c["spot"].iloc[0])
    pts = []
    for exp, g in c.groupby("expiry"):
        T = (pd.Timestamp(exp).tz_localize("America/New_York") + pd.Timedelta(hours=16)
             - snap.tz_convert("America/New_York")).total_seconds() / (365 * 86400)
        if T <= 2 / 365:
            continue
        both = g.pivot_table(index="strike", columns="kind", values=["bid", "ask"]).dropna()
        if both.empty:
            continue
        K = both.index[np.argmin(np.abs(both.index - S))]
        mid = ((both.loc[K, ("bid", "call")] + both.loc[K, ("ask", "call")]) / 2
               + (both.loc[K, ("bid", "put")] + both.loc[K, ("ask", "put")]) / 2)
        f = lambda s: bs.straddle(S, K, T, r, s)["price"] - mid
        try:
            iv = brentq(f, 1e-4, 5.0)
        except ValueError:
            continue
        pts.append((T, iv, K, mid))
    if len(pts) < 2:
        return {"ok": False}
    pts.sort()
    Tt = target_days / 365
    Ts = np.array([p[0] for p in pts]); w = np.array([p[1] ** 2 * p[0] for p in pts])
    if not (Ts[0] <= Tt <= Ts[-1]):
        return {"ok": False}
    tv = np.interp(Tt, Ts, w)
    return {"ok": True, "atm_iv_30d": float(np.sqrt(tv / Tt)), "spot": S,
            "n_expiries": len(pts), "vix": float(c["vix"].iloc[0]),
            "vix3m": float(c["vix3m"].iloc[0])}

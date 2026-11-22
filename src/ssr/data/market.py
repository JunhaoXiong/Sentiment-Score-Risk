"""SPY and VIX-family daily data from Yahoo (yfinance), with a sha256 snapshot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def download(tickers: list[str], start: str, end: str, out: Path) -> pd.DataFrame:
    import yfinance as yf

    # end is inclusive for us; yfinance's is exclusive
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    px = yf.download(tickers, start=start, end=end_excl, auto_adjust=True,
                     progress=False)["Close"]
    px = px[tickers]
    px.index.name = "date"
    out.parent.mkdir(parents=True, exist_ok=True)
    px.to_csv(out, float_format="%.6f")
    meta = {"tickers": tickers, "start": start, "end": end,
            "source": "Yahoo Finance via yfinance, auto_adjust=True",
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "downloaded_utc": pd.Timestamp.utcnow().isoformat()}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return px


def load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col="date", parse_dates=True)

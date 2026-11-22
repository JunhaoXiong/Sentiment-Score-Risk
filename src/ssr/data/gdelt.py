"""GDELT DOC 2.0 API: daily average news tone and article volume.

Facts this module relies on (checked 2026-11-22):
- Timeline modes cover 2017-01-01 to present (GDELT blog, 2018-05-18).
- A 2017-2025 window returns *daily* bins labelled by UTC date.
- The API rate-limits hard (HTTP 429, "one request every 5 seconds"); from a
  shared egress IP even that is not enough, so we back off generously.

The tone GDELT reports is its own dictionary-based tone (positive% - negative%
of words, averaged over matching articles). It is NOT the Loughran-McDonald
score; our own LM scoring is in :mod:`ssr.sentiment.lexicon`.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

API = "https://api.gdeltproject.org/api/v2/doc/doc"


def build_url(query: str, mode: str, start: str, end: str) -> str:
    return (
        f"{API}?query={quote(query)}&mode={mode}&format=json"
        f"&startdatetime={start}&enddatetime={end}"
    )


def fetch_raw(url: str, max_tries: int = 30, first_wait: float = 60.0,
              timeout: float = 180.0) -> bytes:
    """GET with exponential backoff on 429 / timeouts / non-JSON bodies."""
    wait = first_wait
    last = None
    for _ in range(max_tries):
        try:
            r = requests.get(url, timeout=timeout)
            body = r.content
            if r.status_code == 200 and body.lstrip().startswith(b"{"):
                return body
            last = f"HTTP {r.status_code}: {body[:120]!r}"
        except requests.RequestException as e:  # timeout, reset
            last = repr(e)
        time.sleep(wait)
        wait = min(wait * 1.5, 300)
    raise RuntimeError(f"GDELT fetch failed after {max_tries} tries: {last}")


def parse_timeline(body: bytes) -> pd.DataFrame:
    """Parse a timeline JSON into a DataFrame indexed by UTC date.

    timelinetone -> column 'tone'; timelinevolraw -> 'count' and 'total'
    ('norm' field = all monitored articles that day).
    """
    d = json.loads(body)
    series = d["timeline"][0]
    rows = []
    for p in series["data"]:
        row = {"date": pd.Timestamp(p["date"][:8])}
        if "norm" in p:
            row["count"] = float(p["value"])
            row["total"] = float(p["norm"])
        else:
            row["tone"] = float(p["value"])
        rows.append(row)
    return pd.DataFrame(rows).set_index("date").sort_index()


def download(queries: dict[str, str], start: str, end: str, out_dir: Path,
             pause: float = 60.0) -> None:
    """Download tone + volume per query; save raw JSON and a sha256 manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name, q in queries.items():
        for mode in ("timelinevolraw", "timelinetone"):
            f = out_dir / f"{name}_{mode}.json"
            if f.exists():
                continue
            url = build_url(q, mode, start, end)
            body = fetch_raw(url)
            parse_timeline(body)  # validate before saving
            f.write_bytes(body)
            manifest[f.name] = {
                "url": url,
                "sha256": hashlib.sha256(body).hexdigest(),
                "downloaded_utc": pd.Timestamp.utcnow().isoformat(),
            }
            manifest_path.write_text(json.dumps(manifest, indent=2))
            time.sleep(pause)


def load(name: str, raw_dir: Path) -> pd.DataFrame:
    """Daily (UTC-date) frame with tone, count, total for one query."""
    tone = parse_timeline((raw_dir / f"{name}_timelinetone.json").read_bytes())
    vf = raw_dir / f"{name}_timelinevolraw.json"
    if not vf.exists():  # volume optional: caller falls back to equal weights
        return tone
    return tone.join(parse_timeline(vf.read_bytes()), how="outer")

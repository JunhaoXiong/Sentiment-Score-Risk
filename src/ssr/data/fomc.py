"""FOMC policy statements (federalreserve.gov) as plain text.

Statement URLs look like /newsevents/pressreleases/monetaryYYYYMMDDa.htm. The
year pages also link other press releases with the same pattern, so we keep a
page only if its title says it is an FOMC statement.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://www.federalreserve.gov"
CURRENT = f"{BASE}/monetarypolicy/fomccalendars.htm"
LINK_RE = re.compile(r"/newsevents/pressreleases/monetary(\d{8})[a-z]\.htm")


def year_page(year: int) -> str:
    return f"{BASE}/monetarypolicy/fomchistorical{year}.htm"


def candidate_links(html: str) -> list[str]:
    return sorted(set(m.group(0) for m in LINK_RE.finditer(html)))


def is_statement(title: str) -> bool:
    """Policy statements only (not the 'Longer-Run Goals' strategy document)."""
    return "issues fomc statement" in title.lower()


def extract(html: str) -> tuple[str, str]:
    """Return (title, statement body text)."""
    soup = BeautifulSoup(html, "html.parser")
    title_el = soup.find("h3", class_="title") or soup.find("title")
    title = title_el.get_text(" ", strip=True) if title_el else ""
    art = soup.find(id="article")
    body = art.find("div", class_="col-xs-12 col-sm-8 col-md-8") if art else None
    if body is None:
        return title, ""
    text = body.get_text(" ", strip=True)
    # drop the trailing voting / implementation-note boilerplate
    for cut in ("Voting for the monetary policy action", "Voting for this action",
                "Implementation Note", "For media inquiries"):
        i = text.find(cut)
        if i > 0:
            text = text[:i]
    return title, re.sub(r"\s+", " ", text).strip()


def download(years: list[int], out_dir: Path, pause: float = 1.0) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = "research-client"
    links: set[str] = set()
    cur = s.get(CURRENT, timeout=60).text
    links.update(candidate_links(cur))
    for y in years:
        r = s.get(year_page(y), timeout=60)
        if r.status_code == 200:
            links.update(candidate_links(r.text))
        time.sleep(pause)
    rows = []
    for link in sorted(links):
        date = pd.Timestamp(LINK_RE.search(link).group(1))
        if date.year not in years:
            continue
        html = s.get(BASE + link, timeout=60).text
        time.sleep(pause)
        title, text = extract(html)
        if not is_statement(title) or not text:
            continue
        rows.append({"date": date, "url": BASE + link, "title": title,
                     "text": text})
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    f = out_dir / "fomc_statements.csv"
    df.to_csv(f, index=False)
    (out_dir / "fomc_statements.meta.json").write_text(json.dumps({
        "n": len(df), "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
        "downloaded_utc": pd.Timestamp.utcnow().isoformat()}, indent=2))
    return df

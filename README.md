# Sentiment-Score-Risk

News and Fed-communication sentiment as (1) a **market-risk early-warning
signal** (sentiment-augmented GARCH / VaR) and (2) a **volatility-trading
signal** for SPY options. Free data only; every number in `reports/` is
produced by the scripts here.

## Status
| Phase | Content | State |
|---|---|---|
| 1 | Point-in-time sentiment pipeline: GDELT news tone, LM dictionary on FOMC statements, rolling z-scores, close-time alignment | merged (PR #1) |
| 2 | Risk track: GJR-GARCH-X with sentiment vs GJR and GJR+VIX; rolling OOS VaR/ES (W=500); Kupiec, Christoffersen, Diebold-Mariano. **Result: no significant gain from news tone beyond GJR (in-sample delta = 0; OOS 99% DM p = 0.12)** | merged (PR #2) |
| 2c | Timing upper bound: shift GDELT one session earlier (leaks 3-4h); still delta = 0, DM p = 0.15 → negative result robust to timing | merged (PR #3) |
| 2b | Sentiment-triggered stress tests | planned |
| 3 | Quant track: delta-hedged 30d ATM SPY straddles (IV = 0.9×VIX, BS re-marked daily), six signals (always short / term-structure / sentiment filters, long flips), Greeks, VaR/ES, Heston spot-vol stress. **Filters cut drawdowns ~40–60%; sentiment ≈ term structure, edge not significant with 2,100 daily-ladder trades** | this PR |
| 4 | FinBERT challenger; forward test on stored live option-chain snapshots | planned |

## Data (all free)
| Source | What | Coverage |
|---|---|---|
| Yahoo (yfinance) | SPY adj. close, ^VIX, ^VIX3M, ^VIX9D | 2012–2025 used |
| GDELT DOC 2.0 API | daily average tone + article counts of matching news | 2017-01-01 onward |
| federalreserve.gov | FOMC policy statements (74, 2017–2025) | full text |
| Loughran–McDonald dictionary | positive / negative / uncertainty word lists | version bundled in `pysentiment2` |

Raw snapshots are in `data/raw/` with sha256 manifests.

## The no-look-ahead rule
An item published at time τ may enter the forecast for trading day *t* only
if τ ≤ 16:00 ET on day *t−1* (the close at which the day-*t* VaR is set).
GDELT daily bins are UTC days, which end at 19:00/20:00 ET, after the close,
so a UTC-day-*d* bin first affects the trading day two sessions later.
Weekend bins are merged with article-count weights. See
`src/ssr/sentiment/align.py` and `tests/test_phase1.py`.

## Reproduce
```bash
pip install -e .[test]
python scripts/download_gdelt.py      # slow: GDELT rate limits hard
python scripts/phase1_analysis.py
pytest
```

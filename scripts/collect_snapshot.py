"""Collect today's SPY option-chain snapshot (+ recent GDELT tone, best effort).

Usage: python scripts/collect_snapshot.py OUT_DIR
Meant to run on trading days ~15:45 ET (see .github/workflows/snapshot.yml).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ssr.data import option_snapshots  # noqa: E402

out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/snapshots"
df = option_snapshots.collect()
f = option_snapshots.save(df, out / "chains")
print(f"saved {f.name}: {len(df)} rows, quote_ok {df['quote_ok'].mean():.1%}")

# recent GDELT tone/volume (one request each, 400 days); failures are non-fatal
try:
    import pandas as pd
    import yaml

    from ssr.data import gdelt

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())["gdelt"]
    q = cfg["queries"][cfg["primary"]]
    end = pd.Timestamp.now(tz="UTC")
    start = (end - pd.Timedelta(days=400)).strftime("%Y%m%d%H%M%S")
    gd = out / "gdelt"
    gd.mkdir(parents=True, exist_ok=True)
    for mode in ("timelinetone", "timelinevolraw"):
        body = gdelt.fetch_raw(gdelt.build_url(q, mode, start, end.strftime("%Y%m%d%H%M%S")),
                               max_tries=4, first_wait=20)
        gdelt.parse_timeline(body)
        (gd / f"recent_{mode}.json").write_bytes(body)
    print("gdelt updated")
except Exception as e:  # noqa: BLE001
    print(f"gdelt skipped: {e}")

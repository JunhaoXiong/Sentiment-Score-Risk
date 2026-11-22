"""Download GDELT daily tone/volume for the configured queries (slow: rate limits)."""
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ssr.data import gdelt  # noqa: E402

cfg = yaml.safe_load((ROOT / "config.yaml").read_text())["gdelt"]
gdelt.download(cfg["queries"], cfg["start"], cfg["end"], ROOT / "data/raw/gdelt")
print("done")

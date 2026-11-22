"""Loughran-McDonald (LM) dictionary scoring — the user's word-count design.

score_net   = (N_pos - N_neg) / N_total            (user's design: share of text)
score_ratio = (N_pos - N_neg) / (N_pos + N_neg)    (polarity, length-free)
Also returned: pos/neg/uncertainty shares.

Dictionary source: the LM master dictionary CSV bundled in the PyPI package
``pysentiment2`` (an older LM release: 354 positive / 2,355 negative words).
The newest official release is on Google Drive (sraf.nd.edu) which this
environment cannot reach; swapping it in only changes the CSV path.

Simplifications (documented, not hidden): no negation handling ("not strong"
counts as positive), no stemming beyond LM's own inflected word list.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def default_dict_path() -> Path:
    import pysentiment2

    return Path(pysentiment2.__file__).parent / "static" / "LM.csv"


@lru_cache(maxsize=4)
def load_lm(path: str | None = None) -> dict[str, frozenset[str]]:
    p = Path(path) if path else default_dict_path()
    d = pd.read_csv(p, usecols=["Word", "Positive", "Negative", "Uncertainty"],
                    keep_default_na=False)
    w = d["Word"].astype(str).str.upper()
    return {
        "positive": frozenset(w[d["Positive"] > 0]),
        "negative": frozenset(w[d["Negative"] > 0]),
        "uncertainty": frozenset(w[d["Uncertainty"] > 0]),
    }


def tokenize(text: str) -> list[str]:
    return [t.upper() for t in TOKEN_RE.findall(text)]


def score(text: str, lm: dict[str, frozenset[str]] | None = None) -> dict[str, float]:
    lm = lm or load_lm()
    toks = tokenize(text)
    n = len(toks)
    pos = sum(t in lm["positive"] for t in toks)
    neg = sum(t in lm["negative"] for t in toks)
    unc = sum(t in lm["uncertainty"] for t in toks)
    return {
        "n_words": n, "n_pos": pos, "n_neg": neg, "n_unc": unc,
        "pos_share": pos / n if n else float("nan"),
        "neg_share": neg / n if n else float("nan"),
        "unc_share": unc / n if n else float("nan"),
        "score_net": (pos - neg) / n if n else float("nan"),
        "score_ratio": (pos - neg) / (pos + neg) if pos + neg else 0.0,
    }


def score_frame(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    lm = load_lm()
    s = pd.DataFrame([score(t, lm) for t in df[text_col]], index=df.index)
    return pd.concat([df.drop(columns=[text_col]), s], axis=1)

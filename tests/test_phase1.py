import json

import numpy as np
import pandas as pd
import pytest

from ssr.data import fomc, gdelt
from ssr.sentiment import align, lexicon, standardize

# March 2020: US DST starts Sun 2020-03-08
DAYS = pd.bdate_range("2020-03-02", "2020-03-20")


# ---------------- alignment ----------------
def test_fomc_at_close_is_used_next_day():
    pub = align.et_time_to_utc(pd.Series(pd.to_datetime(["2020-03-04"])), "16:00")
    assert align.effective_day(pub, DAYS).iloc[0] == pd.Timestamp("2020-03-05")


def test_one_second_after_close_skips_a_day():
    pub = align.et_time_to_utc(pd.Series(pd.to_datetime(["2020-03-04"])), "16:00")
    pub = pub + pd.Timedelta(seconds=1)
    assert align.effective_day(pub, DAYS).iloc[0] == pd.Timestamp("2020-03-06")


def test_sunday_release_not_used_on_monday():
    # emergency cut Sunday 2020-03-15: VaR for Mon 3/16 was set at Fri close
    pub = align.et_time_to_utc(pd.Series(pd.to_datetime(["2020-03-15"])), "17:00")
    assert align.effective_day(pub, DAYS).iloc[0] == pd.Timestamp("2020-03-17")


@pytest.mark.parametrize("utc_day,expected", [
    ("2020-03-09", "2020-03-11"),  # Mon bin ends Mon 20:00 EDT > close -> Wed
    ("2020-03-12", "2020-03-16"),  # Thu bin -> Mon
    ("2020-03-13", "2020-03-17"),  # Fri bin -> Tue
    ("2020-03-14", "2020-03-17"),  # Sat bin -> Tue
    ("2020-03-15", "2020-03-17"),  # Sun bin -> Tue
    ("2020-03-06", "2020-03-10"),  # Fri before DST change (EST) -> Tue
])
def test_gdelt_utc_bins(utc_day, expected):
    pub = align.gdelt_day_end_utc(pd.DatetimeIndex([utc_day]))
    assert align.effective_day(pub, DAYS).iloc[0] == pd.Timestamp(expected)


def test_no_item_is_used_before_it_exists():
    rng = np.random.default_rng(0)
    pub = pd.to_datetime("2020-03-02", utc=True) + pd.to_timedelta(
        rng.uniform(0, 14 * 86400, 500), unit="s")
    eff = align.effective_day(pd.Series(pub), DAYS)
    prev_close = align.closes_utc(DAYS)
    lookup = dict(zip(DAYS, [pd.NaT] + list(prev_close[:-1])))
    for p, e in zip(pub, eff):
        if pd.notna(e):
            assert p <= lookup[e]


def test_aggregate_weekend_weighted_by_count():
    idx = pd.DatetimeIndex(["2020-03-13", "2020-03-14", "2020-03-15"])
    daily = pd.DataFrame({"tone": [-1.0, -4.0, 2.0], "count": [100, 50, 50],
                          "total": [1000, 800, 800]}, index=idx)
    g = align.aggregate_gdelt(daily, DAYS)
    assert list(g.index) == [pd.Timestamp("2020-03-17")]
    assert g["tone"].iloc[0] == pytest.approx((-100 - 200 + 100) / 200)
    assert g["count"].iloc[0] == 200 and g["n_bins"].iloc[0] == 3


# ---------------- standardisation ----------------
def test_rolling_z_excludes_current_and_future():
    x = pd.Series(np.arange(10, dtype=float))
    z = standardize.rolling_z(x, window=3, min_periods=3)
    past = x.iloc[2:5]
    assert z.iloc[5] == pytest.approx((5 - past.mean()) / past.std(ddof=1))
    assert z.iloc[:3].isna().all()
    x2 = x.copy(); x2.iloc[7:] = 999.0
    z2 = standardize.rolling_z(x2, window=3, min_periods=3)
    pd.testing.assert_series_equal(z.iloc[:7], z2.iloc[:7])


# ---------------- lexicon ----------------
def test_lm_counts_known_words():
    lm = lexicon.load_lm()
    assert "LOSS" in lm["negative"] and "STRONG" in lm["positive"]
    assert "UNCERTAIN" in lm["uncertainty"]
    s = lexicon.score("Strong gains, but a loss and uncertain outlook. The end.", lm)
    assert s["n_words"] == 10
    assert s["n_pos"] >= 1 and s["n_neg"] >= 1
    assert s["score_net"] == pytest.approx((s["n_pos"] - s["n_neg"]) / 10)


def test_lm_empty_text():
    s = lexicon.score("")
    assert s["n_words"] == 0 and np.isnan(s["score_net"]) and s["score_ratio"] == 0


# ---------------- parsers ----------------
def test_gdelt_parse_tone_and_volume():
    tone = json.dumps({"timeline": [{"series": "Average Tone", "data": [
        {"date": "20170101T000000Z", "value": -0.5},
        {"date": "20170102T000000Z", "value": 0.25}]}]}).encode()
    vol = json.dumps({"timeline": [{"series": "Article Count", "data": [
        {"date": "20170101T000000Z", "value": 12, "norm": 3000}]}]}).encode()
    t = gdelt.parse_timeline(tone); v = gdelt.parse_timeline(vol)
    assert t.loc["2017-01-02", "tone"] == 0.25
    assert v.loc["2017-01-01", "count"] == 12 and v.loc["2017-01-01", "total"] == 3000


def test_fomc_extract_and_filter():
    html = ('<html><h3 class="title">Federal Reserve issues FOMC statement</h3>'
            '<div id="article"><div class="col-xs-12 col-sm-8 col-md-8">'
            '<p>Inflation remains elevated.</p><p>Voting for the monetary policy '
            'action were: Someone.</p></div></div></html>')
    title, text = fomc.extract(html)
    assert fomc.is_statement(title)
    assert text == "Inflation remains elevated."
    assert not fomc.is_statement("FOMC reaffirms its Statement on Longer-Run Goals")
    assert fomc.candidate_links('<a href="/newsevents/pressreleases/monetary20200315a.htm">') == [
        "/newsevents/pressreleases/monetary20200315a.htm"]


def test_upper_bound_shift_moves_bins_one_session_earlier():
    idx = pd.DatetimeIndex(["2020-03-09", "2020-03-13"])  # Mon, Fri (EDT)
    daily = pd.DataFrame({"tone": [-1.0, -2.0], "count": [10, 10], "total": [100, 100]}, index=idx)
    honest = align.aggregate_gdelt(daily, DAYS)
    bound = align.aggregate_gdelt(daily, DAYS, pub_shift_hours=-5)
    assert list(honest.index) == [pd.Timestamp("2020-03-11"), pd.Timestamp("2020-03-17")]
    assert list(bound.index) == [pd.Timestamp("2020-03-10"), pd.Timestamp("2020-03-16")]


def test_rolling_residual_uses_past_coefficients_only():
    rng = np.random.default_rng(9)
    n = 400
    x = pd.DataFrame({"x": rng.standard_normal(n)})
    y = 2 * x["x"] + 0.1 * rng.standard_normal(n)
    res = standardize.rolling_residual(y, x, window=100, min_periods=50)
    assert res.iloc[:50].isna().all()
    assert res.iloc[60:].abs().mean() < 0.2  # relationship removed
    y2 = y.copy(); y2.iloc[300:] += 100.0      # future shock
    res2 = standardize.rolling_residual(y2, x, window=100, min_periods=50)
    pd.testing.assert_series_equal(res.iloc[:300], res2.iloc[:300])

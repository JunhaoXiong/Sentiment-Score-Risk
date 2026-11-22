"""Point-in-time alignment of text-derived signals to trading days.

The single rule used everywhere (no look-ahead):

    An item published at time tau (US/Eastern) may be used in the forecast
    for trading day t  iff  tau <= close(t-1) = 16:00 ET on the previous
    trading day.

So the signal ``effective`` on day t is what a risk manager could know when
producing the day-t VaR at the previous close.

GDELT daily bins are UTC calendar days. UTC day d ends at 00:00 UTC of d+1,
i.e. 19:00 or 20:00 ET on d — after the 16:00 close. So UTC day d becomes
usable only at close(next trading day after d). This costs up to ~one day of
timeliness; finer (hourly) GDELT bins would remove it but the API's rate limit
made that impractical for 9 years of data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ET = "America/New_York"
CLOSE = pd.Timedelta(hours=16)


def closes_utc(trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """16:00 ET close of each trading day, as UTC timestamps (DST-aware)."""
    d = pd.DatetimeIndex(trading_days).normalize()
    return (d + CLOSE).tz_localize(ET).tz_convert("UTC")


def effective_day(pub_utc: pd.Series | pd.DatetimeIndex,
                  trading_days: pd.DatetimeIndex) -> pd.Series:
    """First trading day t whose previous close is >= the publication time.

    Returns NaT when the item is too late for the calendar provided.
    """
    days = pd.DatetimeIndex(trading_days).normalize().sort_values()
    c = closes_utc(days).as_unit("ns")
    pub = pd.DatetimeIndex(pd.to_datetime(pub_utc, utc=True)).as_unit("ns")
    # index of first close >= pub  -> that is day t-1; t is the next one
    k = np.searchsorted(c.asi8, pub.asi8, side="left") + 1
    out = np.full(len(pub), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    ok = k < len(days)
    out[ok] = days.values[k[ok]]
    return pd.Series(out, index=getattr(pub_utc, "index", None))


def gdelt_day_end_utc(utc_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Publication time of a GDELT daily bin = end of that UTC day."""
    return (pd.DatetimeIndex(utc_dates).normalize() + pd.Timedelta(days=1)).tz_localize("UTC")


def et_time_to_utc(dates: pd.Series, hhmm: str) -> pd.Series:
    h, m = (int(x) for x in hhmm.split(":"))
    ts = pd.to_datetime(dates).dt.normalize() + pd.Timedelta(hours=h, minutes=m)
    return ts.dt.tz_localize(ET).dt.tz_convert("UTC")


def aggregate_gdelt(daily: pd.DataFrame, trading_days: pd.DatetimeIndex,
                    pub_shift_hours: float = 0.0) -> pd.DataFrame:
    """Map UTC-day GDELT bins to the trading day they become usable for.

    Several UTC days can map to one trading day (weekends, holidays): tone is
    combined as an article-count-weighted mean, counts are summed.

    pub_shift_hours moves the assumed publication time. 0 (default) is the
    honest rule. A negative value (e.g. -5) pretends a UTC day is complete
    before the US close: that LEAKS the last 3-4 hours of news and is used only
    as an *upper bound* on what a timelier (intraday-cut) signal could add.
    """
    df = daily.copy()
    pub = gdelt_day_end_utc(df.index) + pd.Timedelta(hours=pub_shift_hours)
    df["eff"] = effective_day(pub, trading_days).values
    df = df.dropna(subset=["eff"])
    w = df["count"].where(df["count"] > 0)
    df["tone_x_w"] = df["tone"] * w
    df["w"] = w
    g = df.groupby("eff").agg(tone_x_w=("tone_x_w", "sum"), w=("w", "sum"),
                              count=("count", "sum"), total=("total", "sum"),
                              n_bins=("tone", "size"),
                              last_bin=("tone", lambda s: s.index.max()))
    g["tone"] = g["tone_x_w"] / g["w"]
    g["vol_share"] = g["count"] / g["total"]
    g.index.name = "date"
    return g[["tone", "count", "total", "vol_share", "n_bins", "last_bin"]]

"""
Causal OHLCV resampling.

Look-ahead bias frequently sneaks in through resampling: a "4H" bar that is
still forming contains information from the future relative to a 15m entry
bar. This module therefore:

* labels bars by their OPEN time (left-closed, left-labelled),
* optionally drops the last, incomplete bar (``drop_incomplete=True``),
* provides ``closed_bars_as_of`` which returns only HTF bars whose close time
  is <= the entry bar's close time — the primitive used by the MTF engine and
  the backtester.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from ai_trader.core.enums import Timeframe

_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def resample_ohlcv(df: pd.DataFrame, source: Timeframe, target: Timeframe,
                   drop_incomplete: bool = True) -> pd.DataFrame:
    if target.minutes < source.minutes:
        raise ValueError(f"Cannot resample {source.value} up to a smaller timeframe {target.value}")
    if target == source:
        return df.copy()
    rule = target.pandas_freq
    kwargs = {"label": "left", "closed": "left"}
    if target == Timeframe.W1:
        # trading weeks start Sunday 22:00 UTC (approx). Use Monday-based week label.
        rule = "W-SUN"
    out = df.resample(rule, **kwargs).agg(_AGG)
    out = out.dropna(subset=["open", "high", "low", "close"])
    if drop_incomplete and len(out) > 0:
        last_open = out.index[-1]
        expected_close = _bar_close_time(last_open, target)
        last_source_close = df.index[-1] + pd.Timedelta(minutes=source.minutes)
        if last_source_close < expected_close:
            out = out.iloc[:-1]
    out.index.name = "ts"
    return out


def _bar_close_time(open_ts: pd.Timestamp, tf: Timeframe) -> pd.Timestamp:
    if tf == Timeframe.MN1:
        return (open_ts + pd.offsets.MonthBegin(1)).normalize()
    if tf == Timeframe.W1:
        return open_ts + pd.Timedelta(days=7)
    return open_ts + pd.Timedelta(minutes=tf.minutes)


def closed_bars_as_of(df: pd.DataFrame, tf: Timeframe, as_of: datetime) -> pd.DataFrame:
    """Return only bars whose *close* time is <= as_of (strictly causal)."""
    as_of_ts = pd.Timestamp(as_of)
    if as_of_ts.tzinfo is None:
        as_of_ts = as_of_ts.tz_localize("UTC")
    closes = pd.Series([_bar_close_time(ts, tf) for ts in df.index], index=df.index)
    return df[closes <= as_of_ts]


def bar_close_time(open_ts: pd.Timestamp, tf: Timeframe) -> pd.Timestamp:
    return _bar_close_time(open_ts, tf)


def infer_timeframe(df: pd.DataFrame) -> Optional[Timeframe]:
    if len(df) < 3:
        return None
    delta = pd.Series(df.index[1:] - df.index[:-1]).mode().iloc[0]
    minutes = int(delta.total_seconds() // 60)
    for tf in Timeframe:
        if tf.minutes == minutes:
            return tf
    if minutes >= 28 * 1440:
        return Timeframe.MN1
    if minutes >= 7 * 1440 - 1440:
        return Timeframe.W1
    return None

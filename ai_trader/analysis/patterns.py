"""
Candlestick pattern recognition (single & multi-bar).

Every detector returns patterns for the *closed* bar at index ``i`` using
only bars ``<= i``. Strength is scaled by the bar's size relative to ATR so
that a pin bar inside a 0.2-ATR range does not count as much as a 1.5-ATR one.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from ai_trader.core.enums import Direction
from ai_trader.core.models import CandlePattern

from .indicators import atr as _atr


def _bar(df: pd.DataFrame, i: int):
    r = df.iloc[i]
    o, h, l, c = float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])
    body = abs(c - o)
    rng = h - l
    upper = h - max(o, c)
    lower = min(o, c) - l
    return o, h, l, c, body, rng, upper, lower


def detect_patterns(df: pd.DataFrame, i: Optional[int] = None, atr_period: int = 14,
                    trend_hint: Optional[Direction] = None) -> List[CandlePattern]:
    """Patterns present at bar i (default last bar)."""
    if i is None:
        i = len(df) - 1
    if i < 2:
        return []
    a_series = _atr(df, atr_period)
    a = float(a_series.iloc[i]) if not np.isnan(a_series.iloc[i]) else None
    ts = df.index[i].to_pydatetime()
    o, h, l, c, body, rng, upper, lower = _bar(df, i)
    o1, h1, l1, c1, body1, rng1, up1, lo1 = _bar(df, i - 1)
    o2, h2, l2, c2, body2, rng2, up2, lo2 = _bar(df, i - 2)
    if rng <= 0:
        return []
    size = (rng / a) if a else 1.0
    size_w = float(np.clip(size / 1.2, 0.3, 1.3))   # weight by relative size
    out: List[CandlePattern] = []

    # --- Pin bars / hammer / shooting star
    if lower >= 0.6 * rng and body <= 0.3 * rng and upper <= 0.2 * rng:
        name = "hammer" if (trend_hint is Direction.SHORT or c1 < o1) else "bullish_pin_bar"
        out.append(CandlePattern(name, i, ts, Direction.LONG, min(1.0, 0.7 * size_w), "long lower wick rejection"))
    if upper >= 0.6 * rng and body <= 0.3 * rng and lower <= 0.2 * rng:
        name = "shooting_star" if (trend_hint is Direction.LONG or c1 > o1) else "bearish_pin_bar"
        out.append(CandlePattern(name, i, ts, Direction.SHORT, min(1.0, 0.7 * size_w), "long upper wick rejection"))

    # --- Doji
    if body <= 0.1 * rng:
        out.append(CandlePattern("doji", i, ts, Direction.LONG if c1 < o1 else Direction.SHORT, 0.3, "indecision"))

    # --- Engulfing (body engulfs previous body, opposite colour)
    if c > o and c1 < o1 and c >= o1 and o <= c1 and body > body1 * 1.05:
        out.append(CandlePattern("bullish_engulfing", i, ts, Direction.LONG, min(1.0, 0.8 * size_w), "bullish engulfing"))
    if c < o and c1 > o1 and c <= o1 and o >= c1 and body > body1 * 1.05:
        out.append(CandlePattern("bearish_engulfing", i, ts, Direction.SHORT, min(1.0, 0.8 * size_w), "bearish engulfing"))

    # --- Inside bar (compression, breakout pending)
    if h <= h1 and l >= l1:
        out.append(CandlePattern("inside_bar", i, ts, Direction.LONG if c1 > o1 else Direction.SHORT, 0.35, "inside bar / compression"))

    # --- Morning / Evening star (3-bar)
    if (c2 < o2 and body2 > 0.5 * rng2 and body1 <= 0.3 * max(rng1, 1e-12)
            and c > o and c > (o2 + c2) / 2):
        out.append(CandlePattern("morning_star", i, ts, Direction.LONG, min(1.0, 0.85 * size_w), "3-bar bullish reversal"))
    if (c2 > o2 and body2 > 0.5 * rng2 and body1 <= 0.3 * max(rng1, 1e-12)
            and c < o and c < (o2 + c2) / 2):
        out.append(CandlePattern("evening_star", i, ts, Direction.SHORT, min(1.0, 0.85 * size_w), "3-bar bearish reversal"))

    # --- Marubozu-like displacement candle
    if body >= 0.8 * rng and size >= 1.3:
        out.append(CandlePattern("displacement", i, ts, Direction.LONG if c > o else Direction.SHORT, min(1.0, 0.6 * size_w), "strong momentum candle"))
    return out


def best_pattern(patterns: List[CandlePattern], direction: Direction) -> Optional[CandlePattern]:
    same = [p for p in patterns if p.direction is direction and p.name not in ("doji", "inside_bar")]
    return max(same, key=lambda p: p.strength) if same else None


def recent_patterns(df: pd.DataFrame, bars: int = 3, trend_hint: Optional[Direction] = None) -> List[CandlePattern]:
    out: List[CandlePattern] = []
    for i in range(max(2, len(df) - bars), len(df)):
        out += detect_patterns(df, i, trend_hint=trend_hint)
    return out

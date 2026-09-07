"""
Support / resistance, psychological and dynamic levels.

Horizontal levels are built by clustering confirmed swing points (price
tolerance = fraction of ATR); strength = touches × recency × timeframe weight.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ai_trader.core.enums import Timeframe
from ai_trader.core.models import Level, SwingPoint

from .indicators import atr as _atr, ema
from .structure import find_swings


def cluster_levels(swings: List[SwingPoint], tolerance: float, timeframe: Optional[Timeframe] = None,
                   n_bars: Optional[int] = None) -> List[Level]:
    if not swings:
        return []
    pts = sorted(swings, key=lambda s: s.price)
    clusters: List[List[SwingPoint]] = [[pts[0]]]
    for s in pts[1:]:
        if abs(s.price - np.mean([p.price for p in clusters[-1]])) <= tolerance:
            clusters[-1].append(s)
        else:
            clusters.append([s])
    levels: List[Level] = []
    for c in clusters:
        price = float(np.mean([p.price for p in c]))
        touches = len(c)
        kinds = {p.kind for p in c}
        kind = "resistance" if kinds == {"H"} else ("support" if kinds == {"L"} else "flip")
        recency = 1.0
        if n_bars:
            last_idx = max(p.index for p in c)
            recency = 0.5 + 0.5 * (last_idx / max(1, n_bars))
        strength = min(1.0, (0.35 + 0.2 * (touches - 1)) * recency)
        levels.append(Level(price=price, kind=kind, strength=float(strength), touches=touches, timeframe=timeframe))
    return levels


def horizontal_levels(df: pd.DataFrame, timeframe: Optional[Timeframe] = None, lookback: int = 3,
                      atr_period: int = 14, tol_atr: float = 0.5, upto: Optional[int] = None) -> List[Level]:
    swings = find_swings(df, lookback, upto)
    a = _atr(df, atr_period).iloc[-1 if upto is None else upto]
    tol = float(a) * tol_atr if not math.isnan(a) else float(df["close"].iloc[-1]) * 0.002
    n = len(df) if upto is None else upto + 1
    return cluster_levels([s for s in swings if s.confirmed_index < n], tol, timeframe, n)


def psychological_levels(price: float, pip_size: float, n_each_side: int = 3) -> List[Level]:
    """Round numbers: pick a step ~1-2% of price snapped to 1/2/5 × 10^k, plus half-steps."""
    if price <= 0:
        return []
    raw = price * 0.01
    exp = math.floor(math.log10(raw))
    base = 10 ** exp
    step = min([1 * base, 2 * base, 5 * base, 10 * base], key=lambda s: abs(s - raw))
    step = max(step, pip_size * 50)
    start = math.floor(price / step) * step
    out = []
    for k in range(-n_each_side, n_each_side + 1):
        lvl = start + k * step
        if lvl <= 0:
            continue
        out.append(Level(price=float(round(lvl, 8)), kind="psychological", strength=0.6 if k % 2 == 0 else 0.4))
        half = lvl + step / 2
        out.append(Level(price=float(round(half, 8)), kind="psychological", strength=0.3))
    return sorted(out, key=lambda l: l.price)


def dynamic_levels(df: pd.DataFrame) -> Dict[str, float]:
    c = df["close"]
    out = {}
    for p in (20, 50, 100, 200):
        if len(c) >= p:
            out[f"ema{p}"] = float(ema(c, p).iloc[-1])
    return out


def nearest_levels(levels: List[Level], price: float, above: int = 3, below: int = 3) -> Dict[str, List[Level]]:
    ups = sorted([l for l in levels if l.price > price], key=lambda l: l.price)[:above]
    downs = sorted([l for l in levels if l.price <= price], key=lambda l: -l.price)[:below]
    return {"above": ups, "below": downs}


def level_confluence(levels: List[Level], price: float, tolerance: float) -> List[Level]:
    return [l for l in levels if abs(l.price - price) <= tolerance]


def daily_weekly_levels(daily: pd.DataFrame, weekly: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """Previous day/week high-low + classic pivots — institutional reference points."""
    out: Dict[str, float] = {}
    if daily is not None and len(daily) >= 2:
        pd_bar = daily.iloc[-2]
        out.update({"pdh": float(pd_bar["high"]), "pdl": float(pd_bar["low"]), "pdc": float(pd_bar["close"])})
        p = (pd_bar["high"] + pd_bar["low"] + pd_bar["close"]) / 3
        out.update({"pivot": float(p), "r1": float(2 * p - pd_bar["low"]), "s1": float(2 * p - pd_bar["high"]),
                    "r2": float(p + (pd_bar["high"] - pd_bar["low"])), "s2": float(p - (pd_bar["high"] - pd_bar["low"]))})
    if weekly is not None and len(weekly) >= 2:
        pw = weekly.iloc[-2]
        out.update({"pwh": float(pw["high"]), "pwl": float(pw["low"])})
    return out

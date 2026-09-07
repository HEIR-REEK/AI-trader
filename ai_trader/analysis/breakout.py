"""
Breakout analysis: valid breakout vs fake breakout, retest detection,
and confirmation scoring.

A breakout is graded on:
  * close beyond the level (not just a wick)                        → base
  * displacement (bar range vs ATR)                                 → momentum
  * relative volume                                                 → participation
  * follow-through (next bars hold beyond the level)                → acceptance
  * prior compression (squeeze / narrow range before the break)     → energy
A *fake* breakout is a wick/close beyond the level followed by a close back
inside within ``fake_window`` bars — exactly the pattern liquidity-reversal
strategies want, and breakout strategies must avoid.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from ai_trader.core.enums import Direction

from .indicators import atr as _atr, relative_volume


@dataclass
class BreakoutAssessment:
    level: float
    direction: Direction
    index: int
    valid: bool
    fake: bool
    retested: bool
    score: float                      # 0..1 quality
    reasons: List[str] = field(default_factory=list)
    displacement: float = 0.0         # bar range / ATR
    rvol: float = 0.0
    follow_through_bars: int = 0


def assess_breakout(df: pd.DataFrame, level: float, direction: Direction, break_index: Optional[int] = None,
                    atr_period: int = 14, fake_window: int = 3, compression_lookback: int = 20) -> Optional[BreakoutAssessment]:
    """Assess the most recent close beyond ``level`` in ``direction`` (or a given bar)."""
    closes = df["close"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    n = len(df)
    if break_index is None:
        idxs = np.where(closes > level)[0] if direction is Direction.LONG else np.where(closes < level)[0]
        # first bar of the most recent run beyond the level
        if len(idxs) == 0:
            return None
        bi = int(idxs[-1])
        while bi - 1 in idxs and bi - 1 >= 0:
            bi -= 1
        break_index = bi
    i = break_index
    a_s = _atr(df, atr_period).to_numpy()
    a = a_s[i] if not np.isnan(a_s[i]) else (highs[i] - lows[i])
    rv_s = relative_volume(df).to_numpy()
    rv = float(rv_s[i]) if not np.isnan(rv_s[i]) else 1.0
    rng = highs[i] - lows[i]
    disp = rng / a if a > 0 else 1.0
    reasons: List[str] = []
    score = 0.0

    # base: closed beyond
    beyond = closes[i] > level if direction is Direction.LONG else closes[i] < level
    if not beyond:
        return None
    score += 0.2
    reasons.append(f"close beyond level {level:.5g}")

    # close location: strong close near extreme
    loc = (closes[i] - lows[i]) / rng if rng > 0 else 0.5
    if direction is Direction.SHORT:
        loc = 1 - loc
    if loc > 0.7:
        score += 0.1
        reasons.append("strong close in direction of break")

    # displacement
    if disp >= 1.5:
        score += 0.2
        reasons.append(f"displacement {disp:.1f}×ATR")
    elif disp >= 1.0:
        score += 0.1

    # volume
    if rv >= 1.5:
        score += 0.15
        reasons.append(f"relative volume {rv:.1f}×")
    elif rv >= 1.2:
        score += 0.08

    # compression before break
    if i >= compression_lookback:
        pre = df.iloc[i - compression_lookback:i]
        pre_range = pre["high"].max() - pre["low"].min()
        if a > 0 and pre_range < 3.0 * a:
            score += 0.15
            reasons.append("breakout from compression")

    # follow-through / fake detection
    ft = 0
    fake = False
    retested = False
    for j in range(i + 1, min(n, i + 1 + fake_window + 3)):
        inside = closes[j] < level if direction is Direction.LONG else closes[j] > level
        if inside and j <= i + fake_window:
            fake = True
            break
        if not inside:
            ft += 1
        touched = lows[j] <= level <= highs[j]
        if touched and not inside:
            retested = True
    if fake:
        reasons.append(f"closed back inside within {fake_window} bars → FAKE breakout")
        score = min(score, 0.25)
    else:
        if ft >= 2:
            score += 0.1
            reasons.append(f"{ft} bars of acceptance beyond level")
        if retested:
            score += 0.1
            reasons.append("level retested and held")
    return BreakoutAssessment(level=float(level), direction=direction, index=i, valid=(not fake and score >= 0.5),
                              fake=fake, retested=retested, score=float(min(1.0, score)), reasons=reasons,
                              displacement=float(disp), rvol=rv, follow_through_bars=ft)


def range_box(df: pd.DataFrame, lookback: int = 30) -> tuple[float, float]:
    d = df.tail(lookback)
    return float(d["high"].max()), float(d["low"].min())

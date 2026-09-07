"""
Market structure: swing points, HH/HL/LH/LL labelling, BOS / CHoCH / MSS.

Causality: a swing high at bar *i* with lookback *k* is only *known* at bar
``i + k`` (we need k bars to the right to confirm it). Every swing carries
``confirmed_index`` and every downstream consumer (backtester, MTF engine)
must only use swings whose ``confirmed_index <= current_bar``.

Definitions used
----------------
* BOS   – price closes beyond the most recent confirmed swing in the direction
          of the prevailing trend (continuation).
* CHoCH – first close beyond the most recent *counter*-trend swing (potential
          reversal; weaker than MSS).
* MSS   – a CHoCH that (a) follows a liquidity sweep of a prior swing and
          (b) happens with displacement (candle range > 1.5 × ATR). This is
          the highest-quality reversal signal in the structure module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ai_trader.core.enums import Direction, StructureEventType
from ai_trader.core.models import StructureEvent, SwingPoint

from .indicators import atr as _atr


# --------------------------------------------------------------------------
# Swings
# --------------------------------------------------------------------------
def find_swings(df: pd.DataFrame, lookback: int = 3, upto: Optional[int] = None) -> List[SwingPoint]:
    """Fractal swings: high[i] is the max of high[i-k..i+k] (ties broken by first)."""
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    n = len(df) if upto is None else min(len(df), upto + 1)
    k = lookback
    if n < 2 * k + 1:
        return []
    from numpy.lib.stride_tricks import sliding_window_view
    w = 2 * k + 1
    win_h = sliding_window_view(highs[:n], w)          # row j ↔ centre index i = j + k
    win_l = sliding_window_view(lows[:n], w)
    is_h = np.flatnonzero(np.argmax(win_h, axis=1) == k) + k   # first maximum must be the centre
    is_l = np.flatnonzero(np.argmin(win_l, axis=1) == k) + k
    idx = df.index
    swings: List[SwingPoint] = [SwingPoint(int(i), idx[i].to_pydatetime(), float(highs[i]), "H", int(i) + k) for i in is_h]
    swings += [SwingPoint(int(i), idx[i].to_pydatetime(), float(lows[i]), "L", int(i) + k) for i in is_l]
    swings.sort(key=lambda s: (s.index, s.kind))
    return _alternate(swings)


def _alternate(swings: List[SwingPoint]) -> List[SwingPoint]:
    """Enforce H/L alternation keeping the more extreme of consecutive same-kind swings."""
    out: List[SwingPoint] = []
    for s in swings:
        if out and out[-1].kind == s.kind:
            prev = out[-1]
            if (s.kind == "H" and s.price >= prev.price) or (s.kind == "L" and s.price <= prev.price):
                out[-1] = s
            continue
        out.append(s)
    return out


def label_swings(swings: List[SwingPoint]) -> List[Tuple[SwingPoint, str]]:
    """Attach HH/HL/LH/LL labels comparing each swing to the previous of the same kind."""
    last_h: Optional[SwingPoint] = None
    last_l: Optional[SwingPoint] = None
    out = []
    for s in swings:
        if s.kind == "H":
            lab = "H" if last_h is None else ("HH" if s.price > last_h.price else "LH")
            last_h = s
        else:
            lab = "L" if last_l is None else ("HL" if s.price > last_l.price else "LL")
            last_l = s
        out.append((s, lab))
    return out


# --------------------------------------------------------------------------
# Structure state machine
# --------------------------------------------------------------------------
@dataclass
class StructureState:
    trend: Direction | None = None          # LONG = bullish structure, SHORT = bearish, None = undefined
    last_high: Optional[SwingPoint] = None
    last_low: Optional[SwingPoint] = None
    events: List[StructureEvent] = field(default_factory=list)
    labels: List[Tuple[SwingPoint, str]] = field(default_factory=list)
    swings: List[SwingPoint] = field(default_factory=list)
    range_high: Optional[float] = None       # current dealing range
    range_low: Optional[float] = None

    @property
    def last_event(self) -> Optional[StructureEvent]:
        return self.events[-1] if self.events else None

    def recent(self, n: int = 3) -> List[StructureEvent]:
        return self.events[-n:]

    def bias_score(self) -> float:
        """-1..+1 summarising structure: label sequence + last events."""
        score = 0.0
        recent_labels = [lab for _, lab in self.labels[-6:]]
        for lab in recent_labels:
            score += {"HH": 1, "HL": 1, "LH": -1, "LL": -1}.get(lab, 0)
        for ev in self.events[-3:]:
            w = {StructureEventType.BOS: 1.0, StructureEventType.CHOCH: 1.5, StructureEventType.MSS: 2.0}[ev.kind]
            score += w * ev.direction.sign
        denom = max(1.0, len(recent_labels) + 3 * 2)
        return float(np.clip(score / denom, -1, 1))


def analyze_structure(df: pd.DataFrame, lookback: int = 3, atr_period: int = 14,
                      displacement_atr: float = 1.5, upto: Optional[int] = None) -> StructureState:
    """Walk bars causally; emit BOS/CHoCH/MSS as closes break confirmed swings."""
    n = len(df) if upto is None else min(len(df), upto + 1)
    closes = df["close"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    opens = df["open"].to_numpy()
    atr_s = _atr(df, atr_period).to_numpy()
    swings = find_swings(df, lookback, upto)
    state = StructureState(swings=swings, labels=label_swings(swings))

    # Pending swings become "active" once confirmed (bar >= confirmed_index)
    pending = list(swings)
    active_high: Optional[SwingPoint] = None
    active_low: Optional[SwingPoint] = None
    prev_high: Optional[SwingPoint] = None
    prev_low: Optional[SwingPoint] = None
    swept_high_recent = -10**9    # bar index of last sweep of a swing high (wick above, close below)
    swept_low_recent = -10**9

    for i in range(n):
        # activate confirmed swings
        while pending and pending[0].confirmed_index <= i:
            s = pending.pop(0)
            if s.kind == "H":
                prev_high, active_high = active_high, s
            else:
                prev_low, active_low = active_low, s
        if active_high is None or active_low is None:
            continue

        a = atr_s[i] if not np.isnan(atr_s[i]) else 0.0
        body_range = highs[i] - lows[i]
        displacement = a > 0 and body_range > displacement_atr * a and abs(closes[i] - opens[i]) > 0.6 * body_range

        # Sweep detection (wick beyond swing, close back inside)
        if highs[i] > active_high.price and closes[i] < active_high.price:
            swept_high_recent = i
        if lows[i] < active_low.price and closes[i] > active_low.price:
            swept_low_recent = i

        # Bullish break: close above active swing high
        if closes[i] > active_high.price and i > active_high.index:
            swept = (i - swept_low_recent) <= 10 * max(1, lookback)
            if state.trend in (Direction.LONG, None):
                kind = StructureEventType.BOS
            else:
                kind = StructureEventType.MSS if (swept and displacement) else StructureEventType.CHOCH
            state.events.append(StructureEvent(kind, Direction.LONG, i, df.index[i].to_pydatetime(),
                                               float(active_high.price), swept_liquidity=swept,
                                               displacement=bool(displacement),
                                               note=f"close {closes[i]:.5g} > swing high {active_high.price:.5g}"))
            state.trend = Direction.LONG
            # the broken high is consumed; use the next confirmed high when it arrives
            prev_high, active_high = active_high, SwingPoint(i, df.index[i].to_pydatetime(), float(highs[i]), "H", i)
        # Bearish break: close below active swing low
        elif closes[i] < active_low.price and i > active_low.index:
            swept = (i - swept_high_recent) <= 10 * max(1, lookback)
            if state.trend in (Direction.SHORT, None):
                kind = StructureEventType.BOS
            else:
                kind = StructureEventType.MSS if (swept and displacement) else StructureEventType.CHOCH
            state.events.append(StructureEvent(kind, Direction.SHORT, i, df.index[i].to_pydatetime(),
                                               float(active_low.price), swept_liquidity=swept,
                                               displacement=bool(displacement),
                                               note=f"close {closes[i]:.5g} < swing low {active_low.price:.5g}"))
            state.trend = Direction.SHORT
            prev_low, active_low = active_low, SwingPoint(i, df.index[i].to_pydatetime(), float(lows[i]), "L", i)
        else:
            # keep tracking running extremes for the current leg
            if state.trend is Direction.LONG and highs[i] > active_high.price and active_high.confirmed_index == active_high.index:
                active_high = SwingPoint(i, df.index[i].to_pydatetime(), float(highs[i]), "H", i)
            if state.trend is Direction.SHORT and lows[i] < active_low.price and active_low.confirmed_index == active_low.index:
                active_low = SwingPoint(i, df.index[i].to_pydatetime(), float(lows[i]), "L", i)

    state.last_high, state.last_low = active_high, active_low
    # Dealing range = last confirmed swing high/low pair
    confirmed = [s for s in swings if s.confirmed_index < n]
    hs = [s for s in confirmed if s.kind == "H"]
    ls = [s for s in confirmed if s.kind == "L"]
    if hs and ls:
        state.range_high = max(s.price for s in hs[-2:])
        state.range_low = min(s.price for s in ls[-2:])
    return state


def structure_summary(state: StructureState) -> dict:
    labels = [lab for _, lab in state.labels[-6:]]
    return {
        "trend": state.trend.value if state.trend else "UNDEFINED",
        "recent_labels": labels,
        "last_events": [f"{e.kind.value}:{e.direction.value}@{e.level:.5g}" + (" (sweep)" if e.swept_liquidity else "") for e in state.recent(3)],
        "bias_score": round(state.bias_score(), 3),
        "range_high": state.range_high,
        "range_low": state.range_low,
    }


def premium_discount(price: float, range_low: Optional[float], range_high: Optional[float]) -> Tuple[str, float]:
    """Return ('premium'|'discount'|'equilibrium', position 0..1 within the dealing range)."""
    if range_low is None or range_high is None or range_high <= range_low:
        return "unknown", 0.5
    pos = (price - range_low) / (range_high - range_low)
    if pos > 0.55:
        return "premium", float(pos)
    if pos < 0.45:
        return "discount", float(pos)
    return "equilibrium", float(pos)

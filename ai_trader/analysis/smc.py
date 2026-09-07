"""
Smart Money Concepts (SMC) — objective, testable definitions.

These concepts are treated as *hypotheses about where resting orders sit*,
not as guarantees. Each detector produces zones/pools with a strength score so
the confluence layer can weigh them; the backtester decides whether they earn
their weight on a given instrument.

Definitions
-----------
Liquidity pool     – cluster of ≥2 swing highs (buy-side) / lows (sell-side)
                     within a tolerance, or a single prominent swing; also
                     PDH/PDL/PWH/PWL and equal highs/lows.
Liquidity sweep    – wick trades through a pool and the bar closes back on the
                     original side (stop hunt / inducement).
Order block (OB)   – last opposite-coloured candle before a displacement move
                     that breaks structure. Bullish OB = last down candle
                     before an impulsive up move.
Breaker block      – an OB that failed (price closed through it) and is then
                     retested from the other side.
Mitigation block   – an OB whose zone has been revisited once (partially
                     mitigated); weaker than a fresh OB.
Fair Value Gap     – 3-candle imbalance: low[i+1] > high[i-1] (bullish) or
                     high[i+1] < low[i-1] (bearish). Filled when price trades
                     through the gap.
Premium/Discount   – position within the active dealing range (swing low →
                     swing high). Buy in discount, sell in premium.
OTE                – 0.618–0.786 retracement of the impulse leg.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ai_trader.core.enums import Direction, LiquidityType, Timeframe, ZoneType
from ai_trader.core.models import LiquidityPool, SwingPoint, Zone

from .indicators import atr as _atr
from .structure import find_swings


# --------------------------------------------------------------------------
# Liquidity
# --------------------------------------------------------------------------
def liquidity_pools(df: pd.DataFrame, swings: Optional[List[SwingPoint]] = None, lookback: int = 3,
                    tol_atr: float = 0.25, atr_period: int = 14, upto: Optional[int] = None,
                    extra_levels: Optional[Dict[str, float]] = None) -> List[LiquidityPool]:
    n = len(df) if upto is None else upto + 1
    swings = swings or find_swings(df, lookback, upto)
    swings = [s for s in swings if s.confirmed_index < n]
    a = _atr(df, atr_period).iloc[n - 1]
    tol = float(a) * tol_atr if not np.isnan(a) else float(df["close"].iloc[n - 1]) * 0.001
    pools: List[LiquidityPool] = []
    for kind, lt in (("H", LiquidityType.BUY_SIDE), ("L", LiquidityType.SELL_SIDE)):
        pts = [s for s in swings if s.kind == kind]
        used = set()
        for idx, s in enumerate(pts):
            if idx in used:
                continue
            group = [s]
            for j in range(idx + 1, len(pts)):
                if j not in used and abs(pts[j].price - s.price) <= tol:
                    group.append(pts[j])
                    used.add(j)
            level = float(np.max([g.price for g in group])) if kind == "H" else float(np.min([g.price for g in group]))
            note = "equal highs" if (kind == "H" and len(group) > 1) else ("equal lows" if len(group) > 1 else "swing " + ("high" if kind == "H" else "low"))
            pools.append(LiquidityPool(lt, level, max(g.index for g in group), group[-1].ts, touches=len(group), note=note))
    if extra_levels:
        price = float(df["close"].iloc[n - 1])
        ts = df.index[n - 1].to_pydatetime()
        for name, lvl in extra_levels.items():
            if lvl is None:
                continue
            lt = LiquidityType.BUY_SIDE if lvl > price else LiquidityType.SELL_SIDE
            pools.append(LiquidityPool(lt, float(lvl), n - 1, ts, touches=2, note=name.upper()))
    # mark sweeps
    highs, lows, closes = df["high"].to_numpy()[:n], df["low"].to_numpy()[:n], df["close"].to_numpy()[:n]
    for p in pools:
        start = p.index + 1
        if start >= n:
            continue
        if p.kind is LiquidityType.BUY_SIDE:
            hit = np.where(highs[start:] > p.level)[0]
            if len(hit):
                j = start + int(hit[0])
                if closes[j] < p.level:
                    p.swept, p.swept_index = True, j
        else:
            hit = np.where(lows[start:] < p.level)[0]
            if len(hit):
                j = start + int(hit[0])
                if closes[j] > p.level:
                    p.swept, p.swept_index = True, j
    return sorted(pools, key=lambda p: p.level)


def recent_sweep(pools: List[LiquidityPool], n_bars: int, within: int = 12) -> Optional[LiquidityPool]:
    """Most recent sweep within the last `within` bars."""
    cands = [p for p in pools if p.swept and p.swept_index is not None and n_bars - 1 - p.swept_index <= within]
    return max(cands, key=lambda p: p.swept_index) if cands else None


def nearest_pools(pools: List[LiquidityPool], price: float) -> Dict[str, Optional[LiquidityPool]]:
    above = [p for p in pools if p.level > price and not p.swept]
    below = [p for p in pools if p.level < price and not p.swept]
    return {"above": min(above, key=lambda p: p.level) if above else None,
            "below": max(below, key=lambda p: p.level) if below else None}


# --------------------------------------------------------------------------
# Fair value gaps
# --------------------------------------------------------------------------
def fair_value_gaps(df: pd.DataFrame, timeframe: Timeframe, min_size_atr: float = 0.3, atr_period: int = 14,
                    upto: Optional[int] = None, max_age: int = 300) -> List[Zone]:
    n = len(df) if upto is None else upto + 1
    highs, lows, closes = df["high"].to_numpy()[:n], df["low"].to_numpy()[:n], df["close"].to_numpy()[:n]
    a_s = _atr(df, atr_period).to_numpy()[:n]
    zones: List[Zone] = []
    start = max(2, n - max_age)
    for i in range(start, n):
        a = a_s[i] if not np.isnan(a_s[i]) else None
        # bullish FVG: gap between high[i-2] and low[i]
        if lows[i] > highs[i - 2]:
            size = lows[i] - highs[i - 2]
            if a is None or size >= min_size_atr * a:
                zones.append(Zone(ZoneType.FVG, Direction.LONG, float(highs[i - 2]), float(lows[i]), i - 1,
                                  df.index[i - 1].to_pydatetime(), timeframe, strength=float(min(1.0, size / (a or size))), note="bullish FVG"))
        if highs[i] < lows[i - 2]:
            size = lows[i - 2] - highs[i]
            if a is None or size >= min_size_atr * a:
                zones.append(Zone(ZoneType.FVG, Direction.SHORT, float(highs[i]), float(lows[i - 2]), i - 1,
                                  df.index[i - 1].to_pydatetime(), timeframe, strength=float(min(1.0, size / (a or size))), note="bearish FVG"))
    # mitigation: mark filled gaps (price traded fully through)
    for z in zones:
        after_lo = lows[z.index + 2:] if z.index + 2 < n else np.array([])
        after_hi = highs[z.index + 2:] if z.index + 2 < n else np.array([])
        if z.direction is Direction.LONG and len(after_lo) and after_lo.min() <= z.low:
            z.mitigated = True
        if z.direction is Direction.SHORT and len(after_hi) and after_hi.max() >= z.high:
            z.mitigated = True
        touched = ((after_lo <= z.high) & (after_hi >= z.low)).sum() if len(after_lo) else 0
        z.touches = int(touched)
    return zones


# --------------------------------------------------------------------------
# Order blocks / breakers / mitigation blocks
# --------------------------------------------------------------------------
def order_blocks(df: pd.DataFrame, timeframe: Timeframe, structure_events=None, displacement_atr: float = 1.2,
                 atr_period: int = 14, upto: Optional[int] = None, max_age: int = 300, lookahead: int = 3) -> List[Zone]:
    """Order block = last opposite candle before a displacement leg that breaks a swing.

    Without structure events we fall back to "displacement leg" definition:
    a run of `lookahead` bars whose net move ≥ displacement_atr × ATR.
    """
    n = len(df) if upto is None else upto + 1
    o, h, l, c = (df[k].to_numpy()[:n] for k in ("open", "high", "low", "close"))
    a_s = _atr(df, atr_period).to_numpy()[:n]
    zones: List[Zone] = []
    start = max(atr_period + 1, n - max_age)
    for i in range(start, n - lookahead):
        a = a_s[i]
        if np.isnan(a) or a <= 0:
            continue
        move_up = c[i + lookahead] - c[i]
        move_dn = c[i] - c[i + lookahead]
        # bullish OB: bearish candle at i followed by strong up displacement
        if c[i] < o[i] and move_up >= displacement_atr * a and h[i + 1:i + 1 + lookahead].max() > h[i]:
            zones.append(Zone(ZoneType.ORDER_BLOCK, Direction.LONG, float(l[i]), float(max(o[i], c[i])), i,
                              df.index[i].to_pydatetime(), timeframe, strength=float(min(1.0, move_up / (2 * a))), note="bullish OB"))
        if c[i] > o[i] and move_dn >= displacement_atr * a and l[i + 1:i + 1 + lookahead].min() < l[i]:
            zones.append(Zone(ZoneType.ORDER_BLOCK, Direction.SHORT, float(min(o[i], c[i])), float(h[i]), i,
                              df.index[i].to_pydatetime(), timeframe, strength=float(min(1.0, move_dn / (2 * a))), note="bearish OB"))
    # de-duplicate overlapping consecutive OBs (keep the latest in a cluster)
    zones = _dedupe(zones)
    # mitigation / breaker classification
    out: List[Zone] = []
    for z in zones:
        j0 = z.index + lookahead + 1
        if j0 >= n:
            out.append(z)
            continue
        lo_after, hi_after, cl_after = l[j0:], h[j0:], c[j0:]
        touched = (lo_after <= z.high) & (hi_after >= z.low)
        z.touches = int(touched.sum())
        if z.direction is Direction.LONG:
            broken = np.where(cl_after < z.low)[0]
        else:
            broken = np.where(cl_after > z.high)[0]
        if len(broken):
            # failed OB → breaker (flip direction)
            bidx = j0 + int(broken[0])
            out.append(Zone(ZoneType.BREAKER_BLOCK, z.direction.__class__("SHORT" if z.direction is Direction.LONG else "LONG"),
                            z.low, z.high, bidx, df.index[bidx].to_pydatetime(), timeframe, strength=z.strength * 0.8,
                            note=f"breaker (failed {z.note})"))
        elif z.touches > 0:
            z.kind = ZoneType.MITIGATION_BLOCK
            z.strength *= 0.7
            z.note = f"mitigated {z.note}"
            z.mitigated = True
            out.append(z)
        else:
            out.append(z)
    return out


def _dedupe(zones: List[Zone]) -> List[Zone]:
    zones = sorted(zones, key=lambda z: z.index)
    out: List[Zone] = []
    for z in zones:
        if out and out[-1].direction is z.direction and z.index - out[-1].index <= 2 and out[-1].contains(z.mid):
            out[-1] = z if z.strength >= out[-1].strength else out[-1]
        else:
            out.append(z)
    return out


def supply_demand_zones(df: pd.DataFrame, timeframe: Timeframe, upto: Optional[int] = None,
                        base_max_bars: int = 4, atr_period: int = 14, min_move_atr: float = 2.0) -> List[Zone]:
    """Rally-Base-Rally / Drop-Base-Rally style zones: consolidation followed by strong departure."""
    n = len(df) if upto is None else upto + 1
    o, h, l, c = (df[k].to_numpy()[:n] for k in ("open", "high", "low", "close"))
    a_s = _atr(df, atr_period).to_numpy()[:n]
    zones: List[Zone] = []
    i = atr_period + 1
    while i < n - base_max_bars - 3:
        a = a_s[i]
        if np.isnan(a):
            i += 1
            continue
        # base = up to base_max_bars small-bodied bars
        j = i
        while j < min(n - 3, i + base_max_bars) and abs(c[j] - o[j]) < 0.5 * a:
            j += 1
        if j == i:
            i += 1
            continue
        base_hi, base_lo = h[i:j].max(), l[i:j].min()
        depart = c[min(n - 1, j + 2)] - c[j - 1]
        if depart >= min_move_atr * a:
            zones.append(Zone(ZoneType.DEMAND, Direction.LONG, float(base_lo), float(base_hi), j - 1,
                              df.index[j - 1].to_pydatetime(), timeframe, strength=float(min(1.0, depart / (3 * a))), note="demand zone"))
            i = j + 3
        elif -depart >= min_move_atr * a:
            zones.append(Zone(ZoneType.SUPPLY, Direction.SHORT, float(base_lo), float(base_hi), j - 1,
                              df.index[j - 1].to_pydatetime(), timeframe, strength=float(min(1.0, -depart / (3 * a))), note="supply zone"))
            i = j + 3
        else:
            i += 1
    for z in zones:
        after = slice(z.index + 3, n)
        if z.index + 3 < n:
            touched = (l[after] <= z.high) & (h[after] >= z.low)
            z.touches = int(touched.sum())
            if (z.direction is Direction.LONG and (c[after] < z.low).any()) or (z.direction is Direction.SHORT and (c[after] > z.high).any()):
                z.mitigated = True
    return zones


# --------------------------------------------------------------------------
# Premium / discount & OTE
# --------------------------------------------------------------------------
def dealing_range(swings: List[SwingPoint], n_bars: int) -> Tuple[Optional[SwingPoint], Optional[SwingPoint]]:
    """Latest confirmed impulse leg (swing low → swing high or vice versa)."""
    conf = [s for s in swings if s.confirmed_index < n_bars]
    if len(conf) < 2:
        return None, None
    return conf[-2], conf[-1]


def ote_zone(leg_start: SwingPoint, leg_end: SwingPoint) -> Optional[Zone]:
    if leg_start is None or leg_end is None:
        return None
    lo, hi = min(leg_start.price, leg_end.price), max(leg_start.price, leg_end.price)
    rng = hi - lo
    if rng <= 0:
        return None
    if leg_end.kind == "H":  # up leg → look to buy 0.618-0.786 retrace
        return Zone(ZoneType.DEMAND, Direction.LONG, hi - 0.786 * rng, hi - 0.618 * rng, leg_end.index, leg_end.ts,
                    Timeframe.H1, strength=0.6, note="OTE (0.618-0.786 retracement of up-leg)")
    return Zone(ZoneType.SUPPLY, Direction.SHORT, lo + 0.618 * rng, lo + 0.786 * rng, leg_end.index, leg_end.ts,
                Timeframe.H1, strength=0.6, note="OTE (0.618-0.786 retracement of down-leg)")


def active_zones(zones: List[Zone], price: float, direction: Direction, max_distance: float,
                 include_mitigated: bool = False) -> List[Zone]:
    """Unmitigated zones of the requested side within `max_distance` of price (or containing it)."""
    out = []
    for z in zones:
        if z.direction is not direction:
            continue
        if z.mitigated and not include_mitigated and z.kind is not ZoneType.MITIGATION_BLOCK:
            continue
        # a demand zone should be at/below price, supply at/above
        if direction is Direction.LONG and z.low > price + max_distance:
            continue
        if direction is Direction.SHORT and z.high < price - max_distance:
            continue
        if z.distance(price) <= max_distance:
            out.append(z)
    return sorted(out, key=lambda z: (z.distance(price), -z.strength))


def zone_confluence(zones: List[Zone], price: float, tolerance: float) -> List[Zone]:
    return [z for z in zones if z.contains(price, tolerance)]


@dataclass
class SMCSnapshot:
    pools: List[LiquidityPool]
    fvgs: List[Zone]
    order_blocks: List[Zone]
    supply_demand: List[Zone]
    ote: Optional[Zone]
    pd_state: str
    pd_position: float
    recent_sweep: Optional[LiquidityPool]
    notes: List[str] = field(default_factory=list)

    def all_zones(self) -> List[Zone]:
        z = self.fvgs + self.order_blocks + self.supply_demand
        if self.ote:
            z.append(self.ote)
        return z


def smc_snapshot(df: pd.DataFrame, timeframe: Timeframe, lookback: int = 3, extra_levels: Optional[Dict[str, float]] = None,
                 upto: Optional[int] = None) -> SMCSnapshot:
    from .structure import premium_discount
    n = len(df) if upto is None else upto + 1
    swings = find_swings(df, lookback, upto)
    pools = liquidity_pools(df, swings, lookback, upto=upto, extra_levels=extra_levels)
    fvgs = fair_value_gaps(df, timeframe, upto=upto)
    obs = order_blocks(df, timeframe, upto=upto)
    sd = supply_demand_zones(df, timeframe, upto=upto)
    a, b = dealing_range(swings, n)
    ote = ote_zone(a, b) if a and b else None
    if ote:
        ote.timeframe = timeframe
    price = float(df["close"].iloc[n - 1])
    conf = [s for s in swings if s.confirmed_index < n]
    hs = [s for s in conf if s.kind == "H"]
    ls = [s for s in conf if s.kind == "L"]
    # Dealing range = current leg: last confirmed swing high/low extended by the
    # running (not yet confirmed) extreme since that swing — causal, uses past bars only.
    range_high = range_low = None
    if hs and ls:
        last_h, last_l = hs[-1], ls[-1]
        run_hi = float(df["high"].iloc[last_l.index:n].max())
        run_lo = float(df["low"].iloc[last_h.index:n].min())
        range_high = max(last_h.price, run_hi)
        range_low = min(last_l.price, run_lo)
    pd_state, pos = premium_discount(price, range_low, range_high)
    sweep = recent_sweep(pools, n)
    notes = []
    if sweep:
        notes.append(f"{sweep.kind.value.replace('_', '-').lower()} liquidity swept at {sweep.level:.5g} ({sweep.note})")
    notes.append(f"price in {pd_state} ({pos:.0%} of dealing range)")
    return SMCSnapshot(pools, fvgs, obs, sd, ote, pd_state, pos, sweep, notes)

"""
Deterministic, hand-scripted market scenarios.

Random synthetic paths contain no real liquidity structure, so the engine
(correctly) rarely finds high-confluence setups on them. These scripted
scenarios reproduce *textbook* sequences (impulsive HH/HL trend → pullback
into discount → equal-lows sweep → displacement MSS leaving an FVG → retrace
into the FVG → confirmation candle) so that:

* the TRADE path of the decision engine is exercised end-to-end in tests,
* demos show what a passing setup looks like,
* each confluence component can be switched off to check the score reacts.

Bars are M15 with the entry bar last. Higher timeframes are resampled by the
loader / ``FrameProvider`` (closed bars only, causal).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ScenarioConfig:
    start_price: float = 1800.0
    atr: float = 5.0                     # target M15 ATR (price units)
    cycles: int = 60                     # impulse+pullback cycles in the trend history
    impulse_bars: int = 100
    impulse_move: float = 45.0
    pullback_bars: int = 50
    pullback_move: float = -18.0
    final_pullback_bars: int = 40
    final_pullback_move: float = -30.0   # deeper than usual → discount of the last leg
    equal_lows: bool = True
    sweep: bool = True
    displacement: bool = True
    retrace_into_fvg: bool = True
    confirmation_candle: bool = True
    volume_confirms: bool = True
    seed: int = 1
    end: Optional[datetime] = None
    direction: str = "long"              # "long" or "short" (mirrored)


def _leg(prev: float, move: float, bars: int, rng: np.random.Generator, atr: float) -> List[float]:
    """Closes for one leg: linear drift + AR-ish noise, scaled so ATR ≈ atr."""
    noise = rng.normal(0, atr * 0.55, bars)
    drift = np.full(bars, move / bars)
    out = []
    c = prev
    for i in range(bars):
        c = c + drift[i] + noise[i]
        out.append(c)
    # pin the leg end to the intended level so structure is deterministic
    correction = (prev + move) - out[-1]
    return [x + correction * (i + 1) / bars for i, x in enumerate(out)]


def _wicks(opens: np.ndarray, closes: np.ndarray, rng: np.random.Generator, atr):
    atr = np.asarray(atr, dtype=float)
    up_w = np.abs(rng.normal(0, 1, len(closes))) * atr * 0.25
    dn_w = np.abs(rng.normal(0, 1, len(closes))) * atr * 0.25
    return np.maximum(opens, closes) + up_w, np.minimum(opens, closes) - dn_w


def textbook_pullback_setup(cfg: ScenarioConfig = ScenarioConfig()) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    atr = cfg.atr
    closes: List[float] = [cfg.start_price]
    scale0 = cfg.start_price
    # ---- Phase A: impulsive HH/HL trend (moves & noise scale with price) ----
    for _ in range(cfg.cycles):
        k = closes[-1] / scale0
        closes += _leg(closes[-1], cfg.impulse_move * k, cfg.impulse_bars, rng, atr * k)
        k = closes[-1] / scale0
        closes += _leg(closes[-1], cfg.pullback_move * k, cfg.pullback_bars, rng, atr * k)
    k = closes[-1] / scale0
    closes += _leg(closes[-1], cfg.impulse_move * k, cfg.impulse_bars, rng, atr * k)   # end on an impulse (swing high)
    atr = atr * (closes[-1] / scale0)      # current ATR at the end of the trend
    closes_arr = np.array(closes)
    opens = np.concatenate([[closes_arr[0]], closes_arr[:-1]])
    highs, lows = _wicks(opens, closes_arr, rng, cfg.atr * closes_arr / scale0)
    vol = rng.gamma(2.0, 100, len(closes_arr)) + np.abs(closes_arr - opens) / (cfg.atr * closes_arr / scale0) * 150
    O, H, L, C, V = list(opens), list(highs), list(lows), list(closes_arr), list(vol)

    def add(o_, h_, l_, c_, v_):
        O.append(o_); H.append(max(h_, o_, c_)); L.append(min(l_, o_, c_)); C.append(c_); V.append(v_)

    top = C[-1]
    prev = top
    # ---- Phase B: final pullback into discount, ending with equal lows -----
    k = top / scale0
    pb_closes = _leg(prev, cfg.final_pullback_move * k, cfg.final_pullback_bars, rng, atr)
    low_target = top + cfg.final_pullback_move * k
    for i, c in enumerate(pb_closes):
        c = max(c, low_target + 0.3 * atr)
        if cfg.equal_lows and i in (len(pb_closes) - 9, len(pb_closes) - 3):
            add(prev, prev + 0.15 * atr, low_target, low_target + 0.35 * atr, 190)
            prev = low_target + 0.35 * atr
        else:
            add(prev, max(prev, c) + 0.15 * atr, min(prev, c) - 0.15 * atr, c, 150 + rng.gamma(2, 20))
            prev = c
    eq_low = low_target
    # ---- Phase C: sweep of the equal lows (wick through, close back above) --
    if cfg.sweep:
        sweep_low = eq_low - 0.7 * atr
        c = eq_low + 0.5 * atr
        add(prev, prev + 0.1 * atr, sweep_low, c, 650 if cfg.volume_confirms else 150)
        prev = c
    else:
        sweep_low = eq_low - 0.3 * atr
    # ---- Phase D: displacement up (MSS) leaving an FVG ---------------------
    if cfg.displacement:
        c1 = prev + 1.2 * atr
        add(prev, c1 + 0.05 * atr, prev - 0.05 * atr, c1, 600 if cfg.volume_confirms else 160)
        gap_low = c1 + 0.05 * atr                    # high of bar 1
        c2 = c1 + 1.6 * atr
        add(c1, c2 + 0.05 * atr, c1 - 0.03 * atr, c2, 850 if cfg.volume_confirms else 170)
        c3 = c2 + 1.1 * atr
        bar3_low = c2 - 0.05 * atr
        add(c2, c3 + 0.05 * atr, bar3_low, c3, 550 if cfg.volume_confirms else 160)
        gap_high = bar3_low                          # low of bar 3 > high of bar 1 → bullish FVG
        prev = c3
    else:
        for _ in range(3):
            c = prev + 0.3 * atr + rng.normal(0, atr * 0.2)
            add(prev, max(prev, c) + 0.15 * atr, min(prev, c) - 0.15 * atr, c, 170)
            prev = c
        gap_low, gap_high = prev - 1.0 * atr, prev - 0.5 * atr
    # ---- Phase E: retrace into the FVG + confirmation candle ---------------
    if cfg.retrace_into_fvg:
        target = gap_low + (gap_high - gap_low) * 0.5
        for i in range(3):
            c = prev - (prev - target) * (0.5 if i < 2 else 1.0)
            add(prev, prev + 0.12 * atr, c - 0.12 * atr, c, 120)
            prev = c
        if cfg.confirmation_candle:
            o_ = prev - 0.05 * atr
            c_ = prev + 0.8 * atr
            add(o_, c_ + 0.05 * atr, o_ - 0.5 * atr, c_, 520 if cfg.volume_confirms else 130)   # engulfing-style close inside/above gap
            prev = c_
    total = len(C)
    end_ts = pd.Timestamp(cfg.end or datetime(2026, 9, 4, 13, 45, tzinfo=timezone.utc)).floor("15min")
    idx = pd.date_range(end=end_ts, periods=total, freq="15min", tz="UTC")
    df = pd.DataFrame({"open": O, "high": H, "low": L, "close": C, "volume": V}, index=idx)
    df.index.name = "ts"
    if cfg.direction == "short":
        # mirror around a pivot above the whole path so every price stays positive
        pivot = float(df["high"].max()) * 0.6 + float(df["low"].min()) * 0.6
        m = df.copy()
        m["open"] = 2 * pivot - df["open"]
        m["close"] = 2 * pivot - df["close"]
        m["high"] = 2 * pivot - df["low"]
        m["low"] = 2 * pivot - df["high"]
        df = m
        sweep_low, eq_low, top = 2 * pivot - sweep_low, 2 * pivot - eq_low, 2 * pivot - top
        gap_low, gap_high = 2 * pivot - gap_high, 2 * pivot - gap_low
    df.attrs["scenario"] = {"sweep_low": float(sweep_low), "eq_low": float(eq_low), "fvg": (float(gap_low), float(gap_high)),
                            "trend_top": float(top), "direction": cfg.direction}
    return df


def range_fade_setup(bars: int = 9000, seed: int = 3, price: float = 1.1000, atr: float = 0.0006,
                     width_atr: float = 14.0, period: int = 128, noise: float = 0.45,
                     rejection: bool = True) -> pd.DataFrame:
    """Well-respected multi-week box range (slow oscillation + noise, hard-clipped at the
    edges so equal highs/lows build up); price drives into the lower edge over the last
    16 bars and prints a hammer that wicks through the edge and closes back inside →
    candidate for ``range_mean_reversion`` in a RANGE regime."""
    rng = np.random.default_rng(seed)
    width = width_atr * atr
    lo, hi = price - width / 2, price + width / 2
    t = np.arange(bars)
    core = price + (width / 2) * 0.92 * np.sin(2 * np.pi * t / period)
    closes = core + rng.normal(0, atr * noise, bars)
    closes = np.clip(closes, lo + 0.05 * atr, hi - 0.05 * atr)
    n_tail = 16
    closes[-n_tail:] = np.linspace(closes[-n_tail - 1], lo + 0.2 * atr, n_tail) + rng.normal(0, atr * 0.12, n_tail)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs, lows = _wicks(opens, closes, rng, atr)
    vol = rng.gamma(2, 100, bars)
    if rejection:  # hammer: long lower wick through the edge, close back inside above the open
        o_ = closes[-2] - 0.05 * atr
        c_ = closes[-2] + 0.15 * atr
        opens[-1], closes[-1], highs[-1], lows[-1], vol[-1] = o_, c_, c_ + 0.05 * atr, lo - 0.6 * atr, 420
    idx = pd.date_range(end=pd.Timestamp(datetime(2026, 9, 4, 13, 45, tzinfo=timezone.utc)), periods=bars, freq="15min", tz="UTC")
    df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": vol}, index=idx)
    df.index.name = "ts"
    df.attrs["scenario"] = {"range": (float(lo), float(hi)), "atr": atr}
    return df


def choppy_no_edge(bars: int = 9000, seed: int = 5, price: float = 1.1000, vol: float = 0.0004) -> pd.DataFrame:
    """Featureless choppy M15 market — the engine should return NO TRADE."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0, vol, bars)
    closes = price * np.exp(np.cumsum(r) - 0.0)
    # mild mean reversion to keep it in a band
    anchor = price
    out = []
    p = price
    for i in range(bars):
        p = p * float(np.exp(-0.02 * np.log(p / anchor) + r[i]))
        out.append(p)
    closes = np.array(out)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs, lows = _wicks(opens, closes, rng, price * vol * 2)
    idx = pd.date_range(end=pd.Timestamp(datetime(2026, 9, 4, 13, 45, tzinfo=timezone.utc)), periods=bars, freq="15min", tz="UTC")
    df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": rng.gamma(2, 100, bars)}, index=idx)
    df.index.name = "ts"
    return df

"""
Volatility assessment: state (dead / normal / elevated / extreme), expansion
vs contraction, squeeze, and the *tradeability* verdict used as a hard filter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .indicators import atr, atr_percentile, bb_squeeze, bollinger, historical_volatility


@dataclass
class VolatilityAssessment:
    atr: float
    atr_pct_of_price: float
    atr_percentile: float                 # 0..1 vs lookback
    state: str                            # DEAD | LOW | NORMAL | ELEVATED | EXTREME
    phase: str                            # EXPANSION | CONTRACTION | STABLE
    squeeze: bool
    bb_width_percentile: float
    hv_ratio: float                       # short HV / long HV
    atr_spike_ratio: float                # ATR now / median ATR over lookback
    tradeable: bool
    notes: List[str] = field(default_factory=list)

    def score(self, family: str) -> float:
        """0..1 suitability of current volatility for a strategy family."""
        s = 0.5
        if self.state in ("NORMAL", "ELEVATED"):
            s += 0.25
        if self.state == "EXTREME":
            s -= 0.4
        if self.state == "DEAD":
            s -= 0.3
        if family in ("breakout", "momentum", "trend_following") and self.phase == "EXPANSION":
            s += 0.2
        if family in ("mean_reversion", "range") and self.phase == "CONTRACTION":
            s += 0.15
        if family in ("mean_reversion", "range") and self.phase == "STABLE":
            s += 0.1
        if family in ("mean_reversion", "range") and self.phase == "EXPANSION":
            s -= 0.2
        if family == "breakout" and self.squeeze:
            s += 0.1
        return float(np.clip(s, 0, 1))

    def to_dict(self) -> Dict:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def assess_volatility(df: pd.DataFrame, atr_period: int = 14, extreme_pct: float = 0.97, dead_pct: float = 0.05,
                      lookback: Optional[int] = None) -> VolatilityAssessment:
    lookback = lookback or min(250, max(40, len(df) // 2))
    a = atr(df, atr_period)
    ap = atr_percentile(df, atr_period, lookback)
    last_atr = float(a.iloc[-1])
    price = float(df["close"].iloc[-1])
    pct = float(ap.iloc[-1]) if not np.isnan(ap.iloc[-1]) else 0.5
    bb = bollinger(df["close"], 20)
    bw = bb["bb_width"]
    bw_pct_series = bw.rolling(min(120, max(30, len(df) // 3)), min_periods=20).apply(
        lambda x: (x[:-1] < x[-1]).mean() if len(x) > 1 else np.nan, raw=True)
    bw_pct = float(bw_pct_series.iloc[-1]) if not np.isnan(bw_pct_series.iloc[-1]) else 0.5
    sq = bool(bb_squeeze(df).iloc[-1])
    hv_s = historical_volatility(df["close"], 10)
    hv_l = historical_volatility(df["close"], 50)
    hv_ratio = float(hv_s.iloc[-1] / hv_l.iloc[-1]) if hv_l.iloc[-1] and not np.isnan(hv_l.iloc[-1]) and hv_l.iloc[-1] > 0 else 1.0
    # expansion: ATR rising vs its own 20-bar mean and short HV > long HV
    atr_ratio = float(a.iloc[-1] / a.tail(20).mean()) if a.tail(20).mean() > 0 else 1.0
    med = float((a / df["close"]).tail(lookback).median())
    spike = float((a.iloc[-1] / price) / med) if med > 0 else 1.0
    if atr_ratio > 1.15 and hv_ratio > 1.1:
        phase = "EXPANSION"
    elif atr_ratio < 0.85 and hv_ratio < 0.9:
        phase = "CONTRACTION"
    else:
        phase = "STABLE"
    if pct >= extreme_pct and spike >= 1.8:
        state = "EXTREME"          # top-percentile AND ≥1.8× the median → news-like spike
    elif pct >= 0.8:
        state = "ELEVATED"
    elif pct <= dead_pct:
        state = "DEAD"
    elif pct <= 0.25:
        state = "LOW"
    else:
        state = "NORMAL"
    notes = [f"ATR {last_atr:.5g} ({last_atr / price:.2%} of price), percentile {pct:.0%} → {state}", f"phase {phase}"]
    if sq:
        notes.append("Bollinger/Keltner squeeze active")
    tradeable = state not in ("EXTREME", "DEAD")
    if not tradeable:
        notes.append("volatility outside tradeable band → new entries blocked")
    return VolatilityAssessment(last_atr, last_atr / price, pct, state, phase, sq, bw_pct, hv_ratio, spike, tradeable, notes)

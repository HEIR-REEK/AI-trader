"""
Strategy protocol + shared helpers.

A strategy receives the full ``MultiTimeframeContext`` and the
``RegimeAssessment`` and returns zero or more ``StrategySignal`` candidates.
Strategies are deliberately *narrow*: they encode one methodology each and
provide **evidence** (in ``signal.evidence``) that the confluence scorer turns
into points. They never decide alone.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from ai_trader.analysis.mtf import MultiTimeframeContext, TimeframeAnalysis
from ai_trader.core.enums import Bias, Direction, EntryType, StrategyFamily, Timeframe
from ai_trader.core.models import Instrument, RegimeAssessment, StrategySignal, Zone


class Strategy(ABC):
    name: str = "base"
    family: StrategyFamily = StrategyFamily.TREND_FOLLOWING
    description: str = ""

    @abstractmethod
    def generate(self, ctx: MultiTimeframeContext, regime: RegimeAssessment, instrument: Instrument) -> List[StrategySignal]:
        ...

    # ---------------------------------------------------------------- utils
    @staticmethod
    def dir_from_bias(bias: Bias) -> Optional[Direction]:
        return {Bias.BUY: Direction.LONG, Bias.SELL: Direction.SHORT}.get(bias)

    @staticmethod
    def targets_from_rr(entry: float, stop: float, direction: Direction, rrs=(1.5, 2.5, 4.0)) -> List[float]:
        risk = abs(entry - stop)
        return [entry + direction.sign * risk * r for r in rrs]

    @staticmethod
    def targets_from_levels(entry: float, stop: float, direction: Direction, candidates: List[float],
                            min_rr=(1.2, 2.0, 3.0)) -> List[float]:
        """Pick structural targets (liquidity pools / levels) that satisfy minimum R multiples,
        falling back to R-multiples when no structural level exists."""
        risk = abs(entry - stop)
        if risk <= 0:
            return []
        ahead = sorted([c for c in candidates if (c - entry) * direction.sign > 0], key=lambda c: abs(c - entry))
        out: List[float] = []
        for mr in min_rr:
            need = entry + direction.sign * risk * mr
            # first structural level at/after the required R-multiple, shaded 5% of risk in
            # front of it (targets *in front of* liquidity, never behind it)
            lvl = next((c for c in ahead if (c - need) * direction.sign >= 0), None)
            if lvl is not None:
                lvl = lvl - direction.sign * 0.05 * risk
            if lvl is None or (lvl - need) * direction.sign < 0:
                lvl = need
            # enforce strict monotonicity vs previous target (min 0.5R apart)
            if out and (lvl - out[-1]) * direction.sign < 0.5 * risk:
                lvl = out[-1] + direction.sign * 0.5 * risk
            out.append(float(lvl))
        return out

    @staticmethod
    def stop_beyond(level: float, direction: Direction, atr: float, buffer_atr: float = 0.25) -> float:
        return level - direction.sign * atr * buffer_atr

    @staticmethod
    def candidate_target_levels(ctx: MultiTimeframeContext, direction: Direction) -> List[float]:
        lv: List[float] = []
        for a in ctx.analyses.values():
            for p in a.smc.pools:
                if not p.swept:
                    lv.append(p.level)
            for l in a.levels:
                lv.append(l.price)
        for k, v in ctx.key_levels.items():
            if k in ("pdh", "pdl", "pwh", "pwl", "r1", "r2", "s1", "s2"):
                lv.append(v)
        return lv

    @staticmethod
    def zone_near_price(zones: List[Zone], price: float, direction: Direction, atr: float, max_atr: float = 1.0) -> Optional[Zone]:
        best = None
        for z in zones:
            if z.direction is not direction or z.mitigated:
                continue
            if z.distance(price) <= max_atr * atr:
                if best is None or z.distance(price) < best.distance(price) or (z.distance(price) == best.distance(price) and z.strength > best.strength):
                    best = z
        return best

    @staticmethod
    def entry_ta(ctx: MultiTimeframeContext) -> TimeframeAnalysis:
        return ctx.entry

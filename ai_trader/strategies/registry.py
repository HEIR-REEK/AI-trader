"""Strategy registry: which strategies exist and which are allowed per regime/instrument."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from ai_trader.core.enums import AssetClass, StrategyFamily
from ai_trader.core.models import Instrument, RegimeAssessment

from .base import Strategy
from .reversal import BreakoutRetestStrategy, LiquiditySweepReversalStrategy, RangeMeanReversionStrategy
from .trend import EMATrendStrategy, MomentumContinuationStrategy, TrendPullbackStrategy


class StrategyRegistry:
    def __init__(self, strategies: Optional[Iterable[Strategy]] = None):
        self._strategies: Dict[str, Strategy] = {}
        for s in (strategies or default_strategies()):
            self.register(s)

    def register(self, s: Strategy) -> None:
        self._strategies[s.name] = s

    def get(self, name: str) -> Strategy:
        return self._strategies[name]

    def all(self) -> List[Strategy]:
        return list(self._strategies.values())

    def for_regime(self, regime: RegimeAssessment, instrument: Optional[Instrument] = None) -> List[Strategy]:
        allowed = set(regime.allowed_families)
        out = [s for s in self._strategies.values() if s.family in allowed]
        if instrument is not None and instrument.is_synthetic:
            # synthetic indices: no macro, no sessions; ICT time concepts are meaningless,
            # liquidity sweeps still statistically meaningful (stop clusters), momentum less so
            out = [s for s in out if s.family in (StrategyFamily.MEAN_REVERSION, StrategyFamily.RANGE, StrategyFamily.LIQUIDITY_REVERSAL,
                                                  StrategyFamily.PULLBACK, StrategyFamily.BREAKOUT, StrategyFamily.TREND_FOLLOWING)]
        return out


def default_strategies() -> List[Strategy]:
    return [
        TrendPullbackStrategy(),
        EMATrendStrategy(),
        MomentumContinuationStrategy(),
        LiquiditySweepReversalStrategy(),
        RangeMeanReversionStrategy(),
        BreakoutRetestStrategy(),
    ]

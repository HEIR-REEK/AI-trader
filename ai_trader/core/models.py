"""
Core data structures. These are plain dataclasses (fast, no validation overhead)
used inside the engine. API schemas (pydantic) live in ai_trader/api/schemas.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from .enums import (
    AssetClass,
    Bias,
    DecisionType,
    Direction,
    EntryType,
    LiquidityType,
    Regime,
    Session,
    SetupGrade,
    StrategyFamily,
    StructureEventType,
    Timeframe,
    ZoneType,
)


# ---------------------------------------------------------------------------
# Instruments & market data
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Instrument:
    symbol: str
    name: str
    asset_class: AssetClass
    pip_size: float                # smallest quoted move used for "pips"
    point_value: float = 1.0       # P&L per 1.0 price move per 1 unit
    min_lot: float = 0.01
    lot_size: float = 1.0          # units per 1.0 lot
    sessions: tuple = (Session.ALWAYS,)
    quote_currency: str = "USD"
    macro_sensitive: bool = False
    is_synthetic: bool = False
    spread_points: float = 0.0     # typical spread (in price units) for cost modelling
    aliases: tuple = ()

    @property
    def digits(self) -> int:
        return _digits(self.pip_size) + 1

    def round_price(self, price: float) -> float:
        return round(float(price), self.digits)

    def to_pips(self, distance: float) -> float:
        return distance / self.pip_size if self.pip_size else distance


def _digits(pip: float) -> int:
    d = 0
    while pip < 1 and d < 10:
        pip *= 10
        d += 1
    return d


@dataclass
class MarketData:
    """Container of OHLCV frames keyed by timeframe (all UTC-indexed)."""
    symbol: str
    frames: Dict[Timeframe, pd.DataFrame]
    as_of: Optional[datetime] = None

    def get(self, tf: Timeframe) -> pd.DataFrame:
        if tf not in self.frames:
            raise KeyError(f"{self.symbol}: timeframe {tf.value} not loaded")
        return self.frames[tf]

    @property
    def timeframes(self) -> List[Timeframe]:
        return sorted(self.frames.keys(), key=lambda t: t.minutes)

    @property
    def last_price(self) -> float:
        tf = self.timeframes[0]
        return float(self.frames[tf]["close"].iloc[-1])


# ---------------------------------------------------------------------------
# Analysis primitives
# ---------------------------------------------------------------------------
@dataclass
class SwingPoint:
    index: int
    ts: datetime
    price: float
    kind: str  # "H" or "L"
    confirmed_index: int  # bar where the swing became known (causality)


@dataclass
class StructureEvent:
    kind: StructureEventType
    direction: Direction
    index: int
    ts: datetime
    level: float
    swept_liquidity: bool = False
    displacement: bool = False
    note: str = ""


@dataclass
class Zone:
    kind: ZoneType
    direction: Direction          # LONG = demand-side zone, SHORT = supply-side
    low: float
    high: float
    index: int
    ts: datetime
    timeframe: Timeframe
    strength: float = 0.5         # 0..1
    mitigated: bool = False
    touches: int = 0
    note: str = ""

    @property
    def mid(self) -> float:
        return (self.low + self.high) / 2.0

    def contains(self, price: float, tolerance: float = 0.0) -> bool:
        return (self.low - tolerance) <= price <= (self.high + tolerance)

    def distance(self, price: float) -> float:
        if self.contains(price):
            return 0.0
        return min(abs(price - self.low), abs(price - self.high))


@dataclass
class LiquidityPool:
    kind: LiquidityType
    level: float
    index: int
    ts: datetime
    touches: int = 1
    swept: bool = False
    swept_index: Optional[int] = None
    note: str = ""


@dataclass
class Level:
    price: float
    kind: str               # "support" / "resistance" / "psychological" / "pivot"
    strength: float = 0.5
    touches: int = 0
    timeframe: Optional[Timeframe] = None


@dataclass
class CandlePattern:
    name: str
    index: int
    ts: datetime
    direction: Direction
    strength: float = 0.5   # 0..1
    note: str = ""


# ---------------------------------------------------------------------------
# Regime
# ---------------------------------------------------------------------------
@dataclass
class RegimeAssessment:
    primary: Regime
    secondary: List[Regime] = field(default_factory=list)
    confidence: float = 0.0                  # 0..1
    scores: Dict[str, float] = field(default_factory=dict)
    features: Dict[str, float] = field(default_factory=dict)
    allowed_families: List[StrategyFamily] = field(default_factory=list)
    explanation: List[str] = field(default_factory=list)
    timeframe: Optional[Timeframe] = None

    @property
    def tradeable(self) -> bool:
        return bool(self.allowed_families)


# ---------------------------------------------------------------------------
# Strategy output
# ---------------------------------------------------------------------------
@dataclass
class StrategySignal:
    strategy: str
    family: StrategyFamily
    direction: Direction
    entry_low: float
    entry_high: float
    entry_type: EntryType
    stop: float
    targets: List[float]
    timeframe: Timeframe
    confidence: float = 0.5                  # strategy's own 0..1 confidence
    reasons: List[str] = field(default_factory=list)
    invalidation: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)  # feeds confluence scorer

    @property
    def entry_mid(self) -> float:
        return (self.entry_low + self.entry_high) / 2.0

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry_mid - self.stop)

    def rr(self, target_index: int = 1) -> float:
        if self.risk_per_unit <= 0 or target_index >= len(self.targets):
            return 0.0
        return abs(self.targets[target_index] - self.entry_mid) / self.risk_per_unit


# ---------------------------------------------------------------------------
# Confluence / decision
# ---------------------------------------------------------------------------
@dataclass
class ComponentScore:
    name: str
    points: float
    max_points: float
    reasons: List[str] = field(default_factory=list)

    @property
    def pct(self) -> float:
        return 0.0 if self.max_points == 0 else self.points / self.max_points


@dataclass
class ConfluenceResult:
    total: float
    grade: SetupGrade
    components: List[ComponentScore]
    conflicts: List[str] = field(default_factory=list)        # all conflicts, prefixed "MAJOR: " / "minor: "
    factors: List[str] = field(default_factory=list)
    major_conflicts: List[str] = field(default_factory=list)  # veto-class (counted against max_conflicts)
    minor_conflicts: List[str] = field(default_factory=list)  # informational, each costs MINOR_PENALTY points

    @property
    def breakdown(self) -> Dict[str, float]:
        return {c.name: round(c.points, 1) for c in self.components}

    @property
    def n_major(self) -> int:
        return len(self.major_conflicts)

    @property
    def n_minor(self) -> int:
        return len(self.minor_conflicts)


@dataclass
class RiskPlan:
    account_equity: float
    risk_pct: float
    risk_amount: float
    units: float
    lots: float
    stop_distance: float
    stop_distance_atr: float
    rr_tp1: float
    rr_tp2: float
    rr_tp3: float
    approved: bool
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class Explanation:
    why_this_trade: List[str]
    why_now: List[str]
    what_invalidates: List[str]
    what_could_make_it_fail: List[str]
    no_trade_conditions: List[str]


@dataclass
class TradePlan:
    instrument: str
    regime: RegimeAssessment
    bias: Bias
    direction: Direction
    entry_low: float
    entry_high: float
    entry_type: EntryType
    stop: float
    tp1: float
    tp2: float
    tp3: float
    rr: float
    score: float
    grade: SetupGrade
    strategy: str
    factors: List[str]
    invalidation: str
    explanation: Explanation
    risk: RiskPlan
    confluence: ConfluenceResult
    timeframe: Timeframe
    created_at: datetime
    warnings: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    @property
    def rr_text(self) -> str:
        return f"1:{self.rr:.1f}"


@dataclass
class Decision:
    instrument: str
    decision: DecisionType
    created_at: datetime
    message: str
    regime: Optional[RegimeAssessment] = None
    bias: Bias = Bias.NEUTRAL
    plan: Optional[TradePlan] = None
    reasons: List[str] = field(default_factory=list)
    candidates: List[Dict[str, Any]] = field(default_factory=list)   # rejected candidates + why
    context: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_trade(self) -> bool:
        return self.decision is DecisionType.TRADE

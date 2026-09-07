"""Core enumerations shared by every layer."""
from __future__ import annotations

from enum import Enum


class AssetClass(str, Enum):
    METAL = "metal"
    SYNTHETIC_INDEX = "synthetic_index"
    STOCK_INDEX = "stock_index"
    FOREX = "forex"
    CRYPTO = "crypto"


class Timeframe(str, Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
    MN1 = "1M"

    @property
    def minutes(self) -> int:
        return _TF_MINUTES[self]

    @property
    def pandas_freq(self) -> str:
        return _TF_PANDAS[self]

    @property
    def group(self) -> str:
        """HTF / STRUCTURAL / ENTRY grouping used by top-down analysis."""
        if self in (Timeframe.MN1, Timeframe.W1, Timeframe.D1):
            return "HIGH"
        if self in (Timeframe.H4, Timeframe.H1):
            return "STRUCTURAL"
        return "ENTRY"

    def __lt__(self, other: "Timeframe") -> bool:  # type: ignore[override]
        return self.minutes < other.minutes


_TF_MINUTES = {
    Timeframe.M1: 1,
    Timeframe.M5: 5,
    Timeframe.M15: 15,
    Timeframe.M30: 30,
    Timeframe.H1: 60,
    Timeframe.H4: 240,
    Timeframe.D1: 1440,
    Timeframe.W1: 10080,
    Timeframe.MN1: 43200,  # nominal
}

_TF_PANDAS = {
    Timeframe.M1: "1min",
    Timeframe.M5: "5min",
    Timeframe.M15: "15min",
    Timeframe.M30: "30min",
    Timeframe.H1: "1h",
    Timeframe.H4: "4h",
    Timeframe.D1: "1D",
    Timeframe.W1: "W-SUN",
    Timeframe.MN1: "MS",
}


class Bias(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"

    @property
    def sign(self) -> int:
        return {"BUY": 1, "SELL": -1, "NEUTRAL": 0}[self.value]

    @property
    def opposite(self) -> "Bias":
        return {Bias.BUY: Bias.SELL, Bias.SELL: Bias.BUY, Bias.NEUTRAL: Bias.NEUTRAL}[self]


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def sign(self) -> int:
        return 1 if self is Direction.LONG else -1

    @property
    def bias(self) -> Bias:
        return Bias.BUY if self is Direction.LONG else Bias.SELL


class Regime(str, Enum):
    STRONG_BULL = "STRONG_BULLISH_TREND"
    STRONG_BEAR = "STRONG_BEARISH_TREND"
    WEAK_TREND = "WEAK_TREND"
    RANGE = "RANGE_BOUND"
    ACCUMULATION = "ACCUMULATION"
    DISTRIBUTION = "DISTRIBUTION"
    BREAKOUT = "BREAKOUT"
    FALSE_BREAKOUT = "FALSE_BREAKOUT"
    HIGH_VOL = "HIGH_VOLATILITY"
    LOW_VOL = "LOW_VOLATILITY"
    EXPANSION = "EXPANSION"
    CONTRACTION = "CONTRACTION"
    REVERSAL = "REVERSAL"
    NEWS_DRIVEN = "NEWS_DRIVEN"
    UNKNOWN = "UNKNOWN"


class StrategyFamily(str, Enum):
    TREND_FOLLOWING = "trend_following"
    PULLBACK = "pullback"
    BREAKOUT = "breakout"
    MEAN_REVERSION = "mean_reversion"
    LIQUIDITY_REVERSAL = "liquidity_reversal"
    MOMENTUM = "momentum"
    RANGE = "range"


class EntryType(str, Enum):
    MARKET = "Market"
    LIMIT = "Limit"
    STOP = "Stop"


class DecisionType(str, Enum):
    TRADE = "TRADE"
    NO_TRADE = "NO_TRADE"


class SetupGrade(str, Enum):
    EXTREMELY_STRONG = "EXTREMELY STRONG SETUP — still probabilistic, requires risk validation"
    HIGH_QUALITY = "HIGH-QUALITY SETUP"
    MODERATE = "MODERATE SETUP"
    NO_TRADE = "NO TRADE"


class Session(str, Enum):
    ASIA = "ASIA"
    LONDON = "LONDON"
    NEW_YORK = "NEW_YORK"
    OFF_HOURS = "OFF_HOURS"
    ALWAYS = "ALWAYS"  # synthetic indices / crypto trade 24/7


class StructureEventType(str, Enum):
    BOS = "BOS"          # break of structure (continuation)
    CHOCH = "CHOCH"      # change of character (first counter-trend break)
    MSS = "MSS"          # market structure shift (displacement break after sweep)


class ZoneType(str, Enum):
    ORDER_BLOCK = "ORDER_BLOCK"
    BREAKER_BLOCK = "BREAKER_BLOCK"
    MITIGATION_BLOCK = "MITIGATION_BLOCK"
    FVG = "FAIR_VALUE_GAP"
    SUPPLY = "SUPPLY"
    DEMAND = "DEMAND"
    SUPPORT = "SUPPORT"
    RESISTANCE = "RESISTANCE"


class LiquidityType(str, Enum):
    BUY_SIDE = "BUY_SIDE"    # resting above highs (stops of shorts, breakout buys)
    SELL_SIDE = "SELL_SIDE"  # resting below lows


class TradeStatus(str, Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    PARTIAL = "PARTIAL"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


class ExecutionMode(str, Enum):
    SIGNAL_ONLY = "SIGNAL_ONLY"
    PAPER = "PAPER"
    LIVE = "LIVE"

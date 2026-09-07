"""
Hard filters — conditions under which the system refuses to look for entries
at all. Each returns a reason string when it *blocks*.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from ai_trader.analysis.ict import in_session
from ai_trader.analysis.mtf import MultiTimeframeContext
from ai_trader.config.settings import DecisionThresholds, get_settings
from ai_trader.core.enums import Regime
from ai_trader.core.models import Instrument, RegimeAssessment
from ai_trader.data.loader import LoadResult
from ai_trader.fundamentals.macro import NewsAssessment
from ai_trader.risk.manager import RiskManager


def data_quality_filter(load: LoadResult) -> List[str]:
    out = []
    if not load.ok:
        out.append("DATA QUALITY: " + "; ".join(load.issues[:4]))
    return out


def regime_filter(regime: RegimeAssessment, th: Optional[DecisionThresholds] = None) -> List[str]:
    th = th or get_settings().thresholds
    out = []
    if regime.primary is Regime.UNKNOWN:
        out.append("REGIME UNKNOWN: no market type scored above the evidence threshold")
    if regime.primary is Regime.NEWS_DRIVEN:
        out.append("NEWS-DRIVEN MARKET: high-impact event window — price is being driven by flows, not structure")
    if regime.primary is Regime.HIGH_VOL or Regime.HIGH_VOL in regime.secondary:
        out.append(f"EXTREME VOLATILITY: ATR percentile {regime.features.get('atr_percentile', 0):.0%}")
    if not regime.allowed_families:
        out.append(f"REGIME {regime.primary.value}: no strategy family permitted")
    return out


def volatility_filter(ctx: MultiTimeframeContext) -> List[str]:
    va = ctx.entry.volatility
    return [] if va.tradeable else [f"VOLATILITY {va.state}: {'; '.join(va.notes[-1:])}"]


def news_filter(news: Optional[NewsAssessment]) -> List[str]:
    if news is None or not news.in_blackout:
        return []
    e = news.blackout_event
    return [f"NEWS BLACKOUT: {e.title} ({e.currency}, {e.impact}) at {e.ts:%Y-%m-%d %H:%M} UTC"]


def session_filter(instrument: Instrument, now: datetime, enforce: bool = False) -> List[str]:
    if not enforce:
        return []
    return [] if in_session(now, instrument.sessions) else [f"SESSION: {instrument.symbol} outside its active sessions"]


def risk_filter(risk: RiskManager, now: datetime) -> List[str]:
    allowed, reasons = risk.trading_allowed(now)
    return [] if allowed else [f"RISK: {r}" for r in reasons]


def conflict_filter(ctx: MultiTimeframeContext) -> List[str]:
    # explicit HTF vs structural conflict = wait
    if any("HTF bias" in c for c in ctx.conflicts):
        return ["CONFLICT: higher-timeframe and structural biases disagree — wait for resolution"]
    return []

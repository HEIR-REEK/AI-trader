"""
XAUUSD specialised analysis module.

Gold has properties the generic engine should respect:
* it is a USD / real-yield instrument (DXY, US10Y, real yields, Fed pricing),
* it is a safe-haven (VIX / geopolitics can override macro correlations),
* it is *session-sensitive*: London open and the NY open / 8:30 ET data slot
  produce most of the displacement; Asia is usually range/accumulation,
* it loves liquidity sweeps of Asia range / PDH / PDL before the true move,
* it is extremely news-sensitive (NFP, CPI, FOMC → 1–3 % ranges in minutes).

This module does not generate signals directly. It produces a
``GoldContext`` that the decision engine uses to (a) adjust confidence,
(b) add explanation lines, (c) veto new entries during high-risk windows,
and (d) classify the setup archetype (continuation / pullback / sweep
reversal / breakout).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ai_trader.analysis.ict import ICTContext, ict_context
from ai_trader.analysis.mtf import MultiTimeframeContext
from ai_trader.core.enums import Bias, Direction, Regime, Session, StrategyFamily
from ai_trader.core.models import RegimeAssessment, StrategySignal
from ai_trader.core.timeutils import session_of
from ai_trader.fundamentals.macro import MacroContext, NewsAssessment, macro_context


@dataclass
class GoldContext:
    session: Session
    session_quality: float             # 0..1 multiplier for confidence
    macro: MacroContext
    news: Optional[NewsAssessment]
    ict: ICTContext
    daily_trend: Bias
    weekly_structure: str
    key_liquidity: Dict[str, float]
    asia_range_swept: Optional[str]    # "high" | "low" | None
    volatility_note: str
    setup_archetype: Optional[str] = None
    confidence_multiplier: float = 1.0
    veto: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "session": self.session.value, "session_quality": self.session_quality, "macro": self.macro.to_dict(),
            "news": None if self.news is None else self.news.to_dict(), "ict": self.ict.to_dict(),
            "daily_trend": self.daily_trend.value, "weekly_structure": self.weekly_structure,
            "key_liquidity": {k: round(v, 2) for k, v in self.key_liquidity.items()}, "asia_range_swept": self.asia_range_swept,
            "volatility_note": self.volatility_note, "setup_archetype": self.setup_archetype,
            "confidence_multiplier": round(self.confidence_multiplier, 3), "veto": self.veto, "notes": self.notes,
        }


SESSION_QUALITY = {Session.LONDON: 1.0, Session.NEW_YORK: 1.0, Session.ASIA: 0.7, Session.OFF_HOURS: 0.5, Session.ALWAYS: 0.9}


def gold_context(ctx: MultiTimeframeContext, regime: RegimeAssessment, now: datetime,
                 macro_series: Optional[Dict[str, pd.Series]] = None, news: Optional[NewsAssessment] = None,
                 signal: Optional[StrategySignal] = None) -> GoldContext:
    from ai_trader.core.enums import Timeframe
    en = ctx.entry
    daily = ctx.analyses.get(Timeframe.D1)
    weekly = ctx.analyses.get(Timeframe.W1)
    sess = session_of(now)
    ict = ict_context(en.df, daily.df if daily is not None else None, now)
    macro = macro_context("XAUUSD", macro_series)
    daily_trend = daily.trend_bias if daily is not None else ctx.htf_bias
    weekly_structure = "n/a"
    if weekly is not None:
        weekly_structure = f"{weekly.structure.trend.value if weekly.structure.trend else 'undefined'} / labels {' '.join(l for _, l in weekly.structure.labels[-4:])}"
    key = {k: v for k, v in ctx.key_levels.items() if k in ("pdh", "pdl", "pwh", "pwl")}
    if ict.asia_range:
        key["asia_high"], key["asia_low"] = ict.asia_range.high, ict.asia_range.low
    # Asia range sweep (classic London manipulation)
    swept = None
    if ict.asia_range:
        today = en.df[en.df.index >= pd.Timestamp(now).tz_convert("UTC").normalize() + pd.Timedelta(hours=7)]
        if len(today):
            if today["high"].max() > ict.asia_range.high and en.price < ict.asia_range.high:
                swept = "high"
            elif today["low"].min() < ict.asia_range.low and en.price > ict.asia_range.low:
                swept = "low"
    notes: List[str] = []
    mult = SESSION_QUALITY.get(sess, 0.8)
    notes.append(f"session {sess.value} (quality {mult:.1f})")
    if sess is Session.ASIA:
        notes.append("Asia: expect range/accumulation; prefer waiting for London sweep of Asia range")
    if ict.in_kill_zone:
        mult = min(1.05, mult + 0.05)
        notes.append(f"inside {ict.kill_zone}")
    if swept:
        notes.append(f"Asia range {swept} swept — watch for reversal/continuation confirmation")
    # macro alignment
    veto = None
    if signal is not None and macro.known:
        agree = (macro.bias is signal.direction.bias)
        if macro.bias is not Bias.NEUTRAL:
            if agree:
                mult *= 1.05
                notes.append(f"macro tailwind (DXY/yields) supports {signal.direction.value}")
            else:
                mult *= 0.85
                notes.append(f"macro headwind: DXY/yields favour {macro.bias.value} while trade is {signal.direction.value}")
    elif not macro.known:
        notes.append("macro context unknown (no DXY/yield series) — treat correlation as unverified")
    # news
    if news is not None:
        if news.in_blackout:
            veto = "high-impact USD news blackout"
        elif news.penalty > 0:
            mult *= (1 - news.penalty * 0.5)
    # volatility note
    va = en.volatility
    vol_note = f"ATR {va.atr:.2f} ({va.atr_pct_of_price:.2%}) {va.state}/{va.phase}"
    if va.state == "EXTREME":
        veto = veto or "extreme gold volatility (news-like conditions)"
    # archetype
    arche = None
    if signal is not None:
        if signal.family is StrategyFamily.LIQUIDITY_REVERSAL:
            arche = "liquidity_sweep_reversal"
        elif signal.family is StrategyFamily.PULLBACK:
            arche = "pullback_continuation"
        elif signal.family in (StrategyFamily.TREND_FOLLOWING, StrategyFamily.MOMENTUM):
            arche = "trend_continuation"
        elif signal.family is StrategyFamily.BREAKOUT:
            arche = "breakout"
        elif signal.family in (StrategyFamily.MEAN_REVERSION, StrategyFamily.RANGE):
            arche = "range_fade"
        # gold-specific: fading a strong daily trend in NY is a low-quality archetype
        if daily_trend is not Bias.NEUTRAL and signal.direction.bias is not daily_trend and arche in ("range_fade", "liquidity_sweep_reversal"):
            mult *= 0.85
            notes.append(f"counter-daily-trend {arche} — reduced confidence")
    return GoldContext(sess, SESSION_QUALITY.get(sess, 0.8), macro, news, ict, daily_trend, weekly_structure, key, swept,
                       vol_note, arche, float(np.clip(mult, 0.4, 1.1)), veto, notes)

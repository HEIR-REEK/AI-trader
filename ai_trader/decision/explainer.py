"""
Explainable output: WHY THIS TRADE? WHY NOW? WHAT INVALIDATES IT? WHAT COULD
MAKE IT FAIL? — assembled from the evidence collected along the pipeline.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ai_trader.analysis.mtf import MultiTimeframeContext
from ai_trader.core.enums import Direction, Regime, StrategyFamily
from ai_trader.core.models import ConfluenceResult, Explanation, RegimeAssessment, RiskPlan, StrategySignal
from ai_trader.fundamentals.macro import NewsAssessment


def build_explanation(sig: StrategySignal, ctx: MultiTimeframeContext, regime: RegimeAssessment, conf: ConfluenceResult,
                      risk: RiskPlan, news: Optional[NewsAssessment], extra_notes: Optional[List[str]] = None) -> Explanation:
    en, st = ctx.entry, ctx.structural
    d = sig.direction
    why_trade: List[str] = [f"Regime: {regime.primary.value} (confidence {regime.confidence:.0%}) permits {sig.family.value}"]
    why_trade += sig.reasons
    why_trade += [f for f in conf.factors if f not in sig.reasons][:8]

    why_now: List[str] = []
    ev = en.structure.last_event
    if ev is not None and ev.direction is d:
        why_now.append(f"{en.timeframe.value} {ev.kind.value} {d.value} {len(en.df) - 1 - ev.index} bars ago — confirmation is fresh")
    if en.patterns:
        p = en.patterns[-1]
        why_now.append(f"latest candle: {p.name} ({'with' if p.direction is d else 'against'} trade)")
    va = en.volatility
    why_now.append(f"volatility {va.state}/{va.phase} — {'expansion favours follow-through' if va.phase == 'EXPANSION' else 'stop distance calibrated to current ATR'}")
    if extra_notes:
        why_now += extra_notes
    if news is not None and news.next_high_impact is not None and news.minutes_to_next is not None:
        why_now.append(f"next high-impact event {news.next_high_impact.title} in {news.minutes_to_next / 60:.1f}h — manage before release")

    invalidates: List[str] = [sig.invalidation or f"close beyond stop {sig.stop:.5g}"]
    if d is Direction.LONG:
        invalidates.append(f"a {en.timeframe.value} close below {sig.stop:.5g} (structure would print a lower low)")
        if st.structure.last_low is not None:
            invalidates.append(f"loss of structural swing low {st.structure.last_low.price:.5g} flips the {st.timeframe.value} structure bearish")
    else:
        invalidates.append(f"a {en.timeframe.value} close above {sig.stop:.5g} (structure would print a higher high)")
        if st.structure.last_high is not None:
            invalidates.append(f"reclaim of structural swing high {st.structure.last_high.price:.5g} flips the {st.timeframe.value} structure bullish")
    invalidates.append(f"regime change away from {regime.primary.value} (e.g. ADX collapse or volatility spike) before entry fills")

    fail: List[str] = []
    if conf.conflicts:
        fail += [f"conflict: {c}" for c in conf.conflicts]
    fail += [
        "the sweep/zone may be *the* liquidity that a larger move targets (stop placed just beyond it can still be run)",
        "unscheduled news / headlines can gap through the stop — slippage risk is real",
        f"win probability is not 100%: with TP2 at 1:{risk.rr_tp2:.1f} the break-even hit-rate is {1 / (1 + max(risk.rr_tp2, 0.01)):.0%}",
    ]
    if sig.family in (StrategyFamily.MEAN_REVERSION, StrategyFamily.RANGE):
        fail.append("ranges resolve into breakouts eventually — the first breakout after a long range often traps faders")
    if sig.family in (StrategyFamily.TREND_FOLLOWING, StrategyFamily.PULLBACK, StrategyFamily.MOMENTUM):
        fail.append("late-trend entries risk buying the last leg before distribution — watch for climax volume / divergence")
    if regime.confidence < 0.6:
        fail.append(f"regime confidence is only {regime.confidence:.0%} — the market type itself is uncertain")

    no_trade: List[str] = [
        "price runs to the entry zone *after* invalidating (do not chase)",
        "entry zone is reached with a displacement candle *against* the trade (zone failing, not holding)",
        "a high-impact event enters the blackout window before the fill",
        "volatility becomes EXTREME or DEAD before entry",
        "daily/weekly loss limit or correlation limit is reached",
        f"the {en.timeframe.value} prints an opposing {'CHoCH' if ev else 'structure break'} before the fill",
    ]
    return Explanation(why_trade, why_now, invalidates, fail, no_trade)


def no_trade_reasons_summary(reasons: List[str], candidates: List[Dict]) -> List[str]:
    out = list(reasons)
    for c in candidates[:5]:
        out.append(f"{c['strategy']} {c['direction']}: {c['score']}/100 — {c['rejected_because']}")
    return out

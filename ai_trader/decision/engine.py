"""
DecisionEngine — orchestrates the full pipeline for one instrument:

  data → MTF context → regime → hard filters → strategies (allowed for regime)
  → confluence scoring → conflict/threshold check → risk plan → explanation
  → TradePlan | NO TRADE

The engine never forces a trade. Every rejection reason is recorded.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from ai_trader import NO_TRADE_MESSAGE, WAIT_MESSAGE
from ai_trader.analysis.mtf import MultiTimeframeContext, build_context
from ai_trader.config.instruments import get_instrument
from ai_trader.config.settings import Settings, get_settings
from ai_trader.core.enums import AssetClass, Bias, DecisionType, Direction, Regime, SetupGrade, Timeframe
from ai_trader.core.models import Decision, Instrument, MarketData, RegimeAssessment, StrategySignal, TradePlan
from ai_trader.data.calendar import EconomicCalendar
from ai_trader.data.loader import LoadResult, MarketDataLoader
from ai_trader.data.providers import DataProvider
from ai_trader.fundamentals.macro import NewsAssessment, assess_news
from ai_trader.regime.detector import RegimeDetector
from ai_trader.risk.manager import RiskManager
from ai_trader.specialized.gold import gold_context
from ai_trader.specialized.synthetic import synthetic_context
from ai_trader.strategies.registry import StrategyRegistry

from . import filters
from .confluence import ConfluenceScorer, grade_for
from .explainer import build_explanation


class DecisionEngine:
    def __init__(self, provider: DataProvider, settings: Optional[Settings] = None, calendar: Optional[EconomicCalendar] = None,
                 risk: Optional[RiskManager] = None, registry: Optional[StrategyRegistry] = None,
                 regime_detector: Optional[RegimeDetector] = None, macro_series: Optional[Dict[str, pd.Series]] = None,
                 setup_model=None, cache_analysis: bool = False):
        self.settings = settings or get_settings()
        # per-symbol cache of per-timeframe analyses (used by the backtester: HTF frames that
        # have not printed a new bar are not re-analysed — same result, far less work)
        self.cache_analysis = cache_analysis
        self._analysis_cache: Dict[str, Dict[Timeframe, tuple]] = {}
        self.provider = provider
        self.loader = MarketDataLoader(provider, self.settings.analysis)
        self.calendar = calendar or EconomicCalendar()
        self.risk = risk or RiskManager(self.settings.risk)
        self.registry = registry or StrategyRegistry()
        self.regime_detector = regime_detector or RegimeDetector(self.settings.regime_map)
        self.scorer = ConfluenceScorer(self.settings.scoring, self.settings.thresholds)
        self.macro_series = macro_series
        self.setup_model = setup_model  # optional ML meta-labeller with predict_proba(features)->float

    # ------------------------------------------------------------------ API
    def analyze(self, symbol: str, as_of: Optional[datetime] = None, timeframes: Optional[List[Timeframe]] = None) -> Decision:
        inst = get_instrument(symbol)
        load = self.loader.load(inst.symbol, timeframes=timeframes, as_of=as_of, sessions_24_7=inst.is_synthetic or inst.asset_class is AssetClass.CRYPTO)
        return self.decide(load, inst, as_of)

    def decide(self, load: LoadResult, inst: Instrument, as_of: Optional[datetime] = None) -> Decision:
        now = as_of or load.data.as_of or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        th = self.settings.thresholds
        context: Dict[str, Any] = {"as_of": now.isoformat(), "instrument": inst.symbol, "asset_class": inst.asset_class.value}

        # 0) data quality
        blocks = filters.data_quality_filter(load)
        if blocks:
            return self._no_trade(inst, now, blocks, context=context)

        # 1) multi-timeframe context
        ctx = build_context(load.data, inst, self.settings.analysis, cache=self._analysis_cache.setdefault(inst.symbol, {}) if self.cache_analysis else None)
        context["mtf"] = ctx.summary()

        # 2) news / macro
        news = assess_news(inst.symbol, inst, self.calendar, now, self.settings.news) if not inst.is_synthetic else None
        if news is not None:
            context["news"] = news.to_dict()

        # 3) regime (structural timeframe)
        regime = self.regime_detector.detect(ctx.structural, news.minutes_to_next if news else None,
                                             bool(news and news.recent_surprise))
        context["regime"] = {"primary": regime.primary.value, "secondary": [r.value for r in regime.secondary],
                             "confidence": round(regime.confidence, 3), "allowed_families": [f.value for f in regime.allowed_families],
                             "explanation": regime.explanation, "scores": regime.scores}

        # 4) specialised modules
        special_notes: List[str] = []
        special_mult = 1.0
        special_veto: Optional[str] = None
        allowed_override = None
        if inst.symbol == "XAUUSD":
            g = gold_context(ctx, regime, now, self.macro_series, news)
            context["gold"] = g.to_dict()
            special_notes += g.notes
            special_mult *= g.confidence_multiplier
            special_veto = g.veto
        elif inst.is_synthetic:
            s = synthetic_context(ctx, regime)
            context["synthetic"] = s.to_dict()
            special_notes += s.notes
            special_mult *= s.confidence_multiplier
            special_veto = s.veto
            allowed_override = s.allowed_families
        if allowed_override is not None:
            regime.allowed_families = allowed_override

        # 5) hard filters
        blocks = (filters.regime_filter(regime, th) + filters.volatility_filter(ctx) + filters.news_filter(news)
                  + filters.risk_filter(self.risk, now) + filters.conflict_filter(ctx))
        if special_veto:
            blocks.append(f"SPECIALISED VETO: {special_veto}")
        if blocks:
            return self._no_trade(inst, now, blocks, regime, ctx, context)

        # 6) strategies allowed for regime
        strategies = self.registry.for_regime(regime, inst)
        candidates: List[Dict[str, Any]] = []
        scored: List[tuple] = []
        news_pen = news.penalty if news else 0.0
        for strat in strategies:
            try:
                sigs = strat.generate(ctx, regime, inst)
            except Exception as e:  # a strategy bug must never crash the engine → treated as no signal
                candidates.append({"strategy": strat.name, "direction": "-", "score": 0, "rejected_because": f"strategy error: {e}"})
                continue
            for sig in sigs:
                conf = self.scorer.score(sig, ctx, regime, news_pen)
                total = conf.total * special_mult
                # optional ML meta-label: multiplicative, bounded, never additive
                ml_p = None
                if self.setup_model is not None:
                    try:
                        ml_p = float(self.setup_model.predict_setup_proba(sig, ctx, regime, conf))
                        total *= (0.8 + 0.4 * ml_p)   # p=0.5 → ×1.0, p=1 → ×1.2, p=0 → ×0.8
                    except Exception:
                        ml_p = None
                total = min(100.0, total)
                grade = grade_for(total, th)
                rejected = None
                major = conf.major_conflicts
                if total < th.effective_min_score:
                    rejected = f"score {total:.0f} below minimum {th.effective_min_score:.0f}"
                elif len(major) > th.max_conflicts:
                    rejected = f"{len(major)} major conflicting signal(s): {major[0]}"
                elif sig.rr(0) < th.min_rr_tp1 or sig.rr(1) < th.min_rr_tp2:
                    rejected = f"R:R TP1 1:{sig.rr(0):.1f} / TP2 1:{sig.rr(1):.1f} below minimum"
                candidates.append({"strategy": sig.strategy, "direction": sig.direction.value, "score": round(total, 1),
                                   "grade": grade.value, "rr_tp2": round(sig.rr(1), 2), "conflicts": conf.conflicts,
                                   "breakdown": conf.breakdown, "ml_probability": ml_p,
                                   "rejected_because": rejected or "passed",
                                   # geometry (lets the backtester evaluate *rejected* candidates too → threshold validation)
                                   "family": sig.family.value, "timeframe": sig.timeframe.value, "entry_type": sig.entry_type.value,
                                   "entry_low": float(sig.entry_low), "entry_high": float(sig.entry_high), "stop": float(sig.stop),
                                   "targets": [float(t) for t in sig.targets], "n_major": len(major), "n_minor": len(conf.minor_conflicts)})
                if rejected is None:
                    scored.append((total, sig, conf, grade, ml_p))
        context["candidates"] = candidates
        context["specialised_notes"] = special_notes
        if not scored:
            reasons = [f"no candidate reached the confluence threshold ({th.effective_min_score:.0f}) without conflicts"] if candidates else \
                      [f"no strategy in {[s.name for s in strategies]} produced a candidate in regime {regime.primary.value}"]
            return self._no_trade(inst, now, reasons, regime, ctx, context, message=WAIT_MESSAGE)

        # 7) direction conflict among passing candidates → wait
        dirs = {s.direction for _, s, _, _, _ in scored}
        if len(dirs) > 1:
            return self._no_trade(inst, now, ["passing candidates disagree on direction (long and short setups both qualify) — conflicting market"],
                                  regime, ctx, context, message=WAIT_MESSAGE)

        # 8) best candidate → risk validation
        scored.sort(key=lambda t: t[0], reverse=True)
        total, sig, conf, grade, ml_p = scored[0]
        rp = self.risk.plan(sig, inst, ctx.entry.atr, now)
        if not rp.approved:
            context["risk_rejection"] = rp.reasons
            return self._no_trade(inst, now, [f"RISK: {r}" for r in rp.reasons], regime, ctx, context)

        # 9) explanation & plan
        expl = build_explanation(sig, ctx, regime, conf, rp, news, special_notes)
        warnings = list(rp.warnings)
        if news and news.penalty > 0 and not news.in_blackout:
            warnings += news.notes
        if ml_p is not None:
            warnings.append(f"ML setup probability {ml_p:.0%} applied as a bounded multiplier (advisory only)")
        plan = TradePlan(
            instrument=inst.symbol, regime=regime, bias=sig.direction.bias, direction=sig.direction,
            entry_low=inst.round_price(sig.entry_low), entry_high=inst.round_price(sig.entry_high), entry_type=sig.entry_type,
            stop=inst.round_price(sig.stop), tp1=inst.round_price(sig.targets[0]), tp2=inst.round_price(sig.targets[1]),
            tp3=inst.round_price(sig.targets[2]), rr=round(sig.rr(1), 2), score=round(total, 1), grade=grade,
            strategy=sig.strategy, factors=conf.factors, invalidation=sig.invalidation, explanation=expl, risk=rp,
            confluence=conf, timeframe=sig.timeframe, created_at=now, warnings=warnings,
            context={"regime_confidence": regime.confidence, "htf_bias": ctx.htf_bias.value, "alignment": ctx.alignment,
                     "specialised_multiplier": special_mult, "ml_probability": ml_p},
        )
        return Decision(inst.symbol, DecisionType.TRADE, now, f"{grade.value} — {sig.direction.value} {inst.symbol} ({total:.0f}/100)",
                        regime, sig.direction.bias, plan, [], candidates, context)

    # -------------------------------------------------------------- helpers
    def _no_trade(self, inst: Instrument, now: datetime, reasons: List[str], regime: Optional[RegimeAssessment] = None,
                  ctx: Optional[MultiTimeframeContext] = None, context: Optional[Dict[str, Any]] = None,
                  message: str = NO_TRADE_MESSAGE) -> Decision:
        bias = ctx.dominant_bias() if ctx else Bias.NEUTRAL
        return Decision(inst.symbol, DecisionType.NO_TRADE, now, message, regime, bias, None, reasons,
                        (context or {}).get("candidates", []), context or {})

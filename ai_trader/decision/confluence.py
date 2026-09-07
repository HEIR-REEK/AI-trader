"""
Confluence scoring — turns a StrategySignal + context into a 0–100 score with
an itemised breakdown, list of confluence factors, and lists of conflicts.

Components (default weights, configurable in ScoringWeights):
  market_structure 20 · htf_trend 15 · liquidity 15 · price_action 10 ·
  volume 10 · volatility 10 · indicators 10 · risk_reward 10

Rules
* Each component returns a 0..1 ratio × its max points, with reasons.
* Not-applicable components (e.g. volume on a synthetic index with no volume
  feed) have their weight redistributed pro-rata.
* Conflicts are split by severity:
    MAJOR  → counted against ``thresholds.max_conflicts`` (default 0 → reject)
    MINOR  → each subtracts ``MINOR_PENALTY`` points and is reported
  Which is which is family-aware: e.g. "entry in premium" is MAJOR for a
  pullback/reversal entry (you are supposed to buy cheap) but only a note for a
  momentum/breakout entry (price is *always* at the premium of the current leg
  when a breakout happens).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ai_trader.analysis.mtf import MultiTimeframeContext
from ai_trader.config.settings import DecisionThresholds, ScoringWeights, get_settings
from ai_trader.core.enums import Bias, Direction, Regime, SetupGrade, StrategyFamily, StructureEventType, ZoneType
from ai_trader.core.models import ComponentScore, ConfluenceResult, RegimeAssessment, StrategySignal

MINOR_PENALTY = 2.0
VALUE_FAMILIES = (StrategyFamily.PULLBACK, StrategyFamily.LIQUIDITY_REVERSAL, StrategyFamily.MEAN_REVERSION, StrategyFamily.RANGE)
CONTINUATION_FAMILIES = (StrategyFamily.TREND_FOLLOWING, StrategyFamily.MOMENTUM, StrategyFamily.BREAKOUT)
COUNTER_TREND_FAMILIES = (StrategyFamily.MEAN_REVERSION, StrategyFamily.RANGE, StrategyFamily.LIQUIDITY_REVERSAL)


def grade_for(score: float, th: DecisionThresholds) -> SetupGrade:
    if score >= th.extremely_strong:
        return SetupGrade.EXTREMELY_STRONG
    if score >= th.high_quality:
        return SetupGrade.HIGH_QUALITY
    if score >= th.moderate:
        return SetupGrade.MODERATE
    return SetupGrade.NO_TRADE


class _Collector:
    """Accumulates reasons / conflicts for one component."""

    def __init__(self):
        self.why: List[str] = []
        self.major: List[str] = []
        self.minor: List[str] = []


class ConfluenceScorer:
    def __init__(self, weights: Optional[ScoringWeights] = None, thresholds: Optional[DecisionThresholds] = None):
        s = get_settings()
        self.weights = weights or s.scoring
        self.thresholds = thresholds or s.thresholds

    # ------------------------------------------------------------------ API
    def score(self, sig: StrategySignal, ctx: MultiTimeframeContext, regime: RegimeAssessment,
              news_penalty: float = 0.0) -> ConfluenceResult:
        w = self.weights.as_dict()
        parts: List[Tuple[str, float, _Collector, bool]] = []
        for name, fn in (("market_structure", self._structure), ("htf_trend", self._htf), ("liquidity", self._liquidity),
                         ("price_action", self._price_action), ("volume", self._volume), ("volatility", self._volatility),
                         ("indicators", self._indicators), ("risk_reward", self._risk_reward)):
            c = _Collector()
            res = fn(sig, ctx, regime, c)
            ratio, applicable = (res if isinstance(res, tuple) else (res, True))
            parts.append((name, float(np.clip(ratio, 0, 1)), c, applicable))

        na_weight = sum(w[n] for n, _, _, ap in parts if not ap)
        ap_weight = sum(w[n] for n, _, _, ap in parts if ap) or 1.0
        components: List[ComponentScore] = []
        total = 0.0
        major: List[str] = []
        minor: List[str] = []
        for name, ratio, c, ap in parts:
            max_pts = 0.0 if not ap else w[name] * (1 + na_weight / ap_weight)
            pts = ratio * max_pts
            total += pts
            components.append(ComponentScore(name, pts, max_pts, c.why if ap else ["not applicable — weight redistributed"]))
            major += c.major
            minor += c.minor
        total -= MINOR_PENALTY * len(minor)
        if news_penalty > 0:
            total *= (1 - news_penalty)
            minor.append(f"news proximity: score reduced by {news_penalty:.0%}")
        if regime.confidence < self.thresholds.min_regime_confidence:
            total *= 0.85
            major.append(f"regime confidence {regime.confidence:.0%} below {self.thresholds.min_regime_confidence:.0%}")
        total = float(np.clip(total, 0.0, 100.0))
        factors = [x for cmp in components for x in cmp.reasons if not x.startswith("not applicable")]
        # ConfluenceResult.conflicts = MAJOR (decision-relevant); minor ones are appended with a prefix for transparency
        conflicts = [f"MAJOR: {m}" for m in major] + [f"minor: {m}" for m in minor]
        return ConfluenceResult(total=round(total, 1), grade=grade_for(total, self.thresholds), components=components,
                                conflicts=conflicts, factors=factors, major_conflicts=major, minor_conflicts=minor)

    # --------------------------------------------------------- components
    def _structure(self, sig, ctx, regime, c: _Collector) -> float:
        en, st = ctx.entry, ctx.structural
        d = sig.direction
        r = 0.0
        sb = st.structure.bias_score()
        if sb * d.sign > 0.15:
            r += 0.35
            c.why.append(f"structural TF structure {'bullish' if d is Direction.LONG else 'bearish'} (bias {sb:+.2f})")
        elif sb * d.sign < -0.3:
            if sig.family in COUNTER_TREND_FAMILIES:
                c.minor.append(f"structural TF structure against trade (bias {sb:+.2f}) — counter-trend entry")
                r += 0.1
            else:
                c.major.append(f"structural TF structure against trade (bias {sb:+.2f})")
        ev = sig.evidence.get("structure_event") or en.structure.last_event
        bars_since = (len(en.df) - 1 - ev.index) if ev else 999
        if ev is not None and ev.direction is d and bars_since <= 10:
            r += {StructureEventType.MSS: 0.4, StructureEventType.CHOCH: 0.3, StructureEventType.BOS: 0.25}[ev.kind]
            c.why.append(f"entry TF {ev.kind.value} in trade direction" + (" (displacement)" if ev.displacement else ""))
        elif ev is not None and ev.direction is not d and bars_since <= 3:
            if sig.family in COUNTER_TREND_FAMILIES and ev.kind is StructureEventType.BOS:
                # a fade *expects* price to have just pushed into the extreme; only a displacement
                # break (breakout momentum) is a warning
                if ev.displacement:
                    c.minor.append(f"fading a fresh entry-TF {ev.kind.value} with displacement — breakout risk, rejection must hold")
                else:
                    c.why.append("approach BOS without displacement (no breakout momentum into the extreme)")
            else:
                c.major.append(f"most recent entry-TF structure event ({ev.kind.value}) is against the trade")
        # range / mean-reversion families: structure credit comes from a respected range and an extreme location
        rng_box = sig.evidence.get("range")
        rng_pos = sig.evidence.get("range_pos")
        if rng_box is not None and rng_pos is not None and sig.family in (StrategyFamily.MEAN_REVERSION, StrategyFamily.RANGE):
            at_extreme = (d is Direction.LONG and rng_pos <= 0.2) or (d is Direction.SHORT and rng_pos >= 0.8)
            if at_extreme:
                r += 0.35
                c.why.append(f"price at range extreme ({rng_pos:.0%} of {rng_box[0]:.5g}–{rng_box[1]:.5g})")
            if regime.primary in (Regime.RANGE, Regime.ACCUMULATION, Regime.DISTRIBUTION, Regime.LOW_VOL):
                r += 0.2
                c.why.append(f"{regime.primary.value} regime supports fading the extremes")
        pd_state = st.smc.pd_state
        good_pd = (d is Direction.LONG and pd_state == "discount") or (d is Direction.SHORT and pd_state == "premium")
        bad_pd = (d is Direction.LONG and pd_state == "premium") or (d is Direction.SHORT and pd_state == "discount")
        if good_pd:
            r += 0.25
            c.why.append(f"entry in {pd_state} of structural dealing range")
        elif pd_state == "equilibrium":
            r += 0.12
        elif bad_pd:
            if sig.family in VALUE_FAMILIES:
                c.major.append(f"value entry taken in {pd_state} (should be {'discount' if d is Direction.LONG else 'premium'})")
            else:
                r += 0.1   # continuation entries are expected at the leg's premium; neutral
                c.why.append(f"continuation entry at leg {pd_state} (acceptable for {sig.family.value})")
        if sig.family in regime.allowed_families:
            r += 0.1
        return min(1.0, r)

    def _htf(self, sig, ctx, regime, c: _Collector) -> float:
        d = sig.direction
        r = 0.0
        if sig.family in (StrategyFamily.MEAN_REVERSION, StrategyFamily.RANGE):
            # a range fade wants the higher timeframes to be *not trending* against it
            if ctx.htf_bias is Bias.NEUTRAL or ctx.htf_strength < 0.35:
                r = 0.7
                c.why.append(f"higher timeframes not trending ({ctx.htf_bias.value} {ctx.htf_strength:.0%}) — range environment confirmed")
            elif ctx.htf_bias is d.bias:
                r = 0.6 + 0.4 * ctx.htf_strength
                c.why.append(f"fade taken in the direction of the higher timeframe bias {ctx.htf_bias.value} ({ctx.htf_strength:.0%})")
            elif ctx.htf_strength < 0.6:
                r = 0.3
                c.minor.append(f"trade against higher timeframe bias {ctx.htf_bias.value} ({ctx.htf_strength:.0%}) — take partials early")
            else:
                c.major.append(f"fading a strong higher-timeframe trend ({ctx.htf_bias.value} {ctx.htf_strength:.0%})")
            if ctx.structural_bias is Bias.NEUTRAL or ctx.structural_bias is d.bias:
                r = min(1.0, r + 0.15)
            return r
        if ctx.htf_bias is d.bias:
            r += 0.6 + 0.4 * ctx.htf_strength
            c.why.append(f"higher timeframe bias {ctx.htf_bias.value} (strength {ctx.htf_strength:.0%})")
        elif ctx.htf_bias is Bias.NEUTRAL:
            r += 0.45
            c.why.append("higher timeframe neutral (no HTF headwind)")
        elif ctx.htf_strength < 0.35:
            r += 0.4
            c.why.append(f"higher timeframe bias {ctx.htf_bias.value} but weak ({ctx.htf_strength:.0%}) — no real headwind")
        else:
            if sig.family in COUNTER_TREND_FAMILIES and ctx.htf_strength < 0.6:
                r += 0.15
                c.minor.append(f"trade against higher timeframe bias {ctx.htf_bias.value} (weak, {ctx.htf_strength:.0%})")
            else:
                c.major.append(f"trade against higher timeframe bias {ctx.htf_bias.value} ({ctx.htf_strength:.0%})")
        if ctx.structural_bias is d.bias:
            r = min(1.0, r + 0.15)
            c.why.append(f"structural timeframe bias {ctx.structural_bias.value}")
        if abs(ctx.alignment) >= 0.4 and np.sign(ctx.alignment) == d.sign:
            c.why.append(f"multi-timeframe alignment {ctx.alignment:+.2f}")
        return min(1.0, r)

    def _liquidity(self, sig, ctx, regime, c: _Collector) -> float:
        en, st = ctx.entry, ctx.structural
        d = sig.direction
        r = 0.0
        sweep = sig.evidence.get("sweep") or en.smc.recent_sweep or st.smc.recent_sweep
        if sweep is not None:
            good = (d is Direction.LONG and sweep.kind.value == "SELL_SIDE") or (d is Direction.SHORT and sweep.kind.value == "BUY_SIDE")
            if good:
                r += 0.4
                c.why.append(f"liquidity sweep of {sweep.note} at {sweep.level:.5g} supports the trade")
            else:
                c.minor.append(f"recent sweep of {sweep.kind.value.lower().replace('_', '-')} liquidity works against the trade")
        zone = sig.evidence.get("zone")
        if zone is not None:
            base = {ZoneType.ORDER_BLOCK: 0.3, ZoneType.FVG: 0.25, ZoneType.DEMAND: 0.25, ZoneType.SUPPLY: 0.25,
                    ZoneType.BREAKER_BLOCK: 0.2, ZoneType.MITIGATION_BLOCK: 0.15}.get(zone.kind, 0.15)
            r += base * (0.6 + 0.4 * zone.strength)
            c.why.append(f"entry at {zone.timeframe.value} {zone.note} ({zone.kind.value.lower().replace('_', ' ')})")
        # significant pools = equal highs/lows, HTF/structural pools, key levels
        sig_pools = []
        for tf, a in ctx.analyses.items():
            n_tf = len(a.df)
            for p in a.smc.pools:
                if p.swept:
                    continue
                key_level = p.note.upper() in ("PDH", "PDL", "PWH", "PWL")
                if tf.group != "ENTRY" or key_level:
                    sig_pools.append(p)
                elif p.touches >= 2 and (n_tf - 1 - p.index) >= 12:
                    # equal highs/lows on the entry TF count only when they are older than the current leg
                    sig_pools.append(p)
        ahead = [p for p in sig_pools if (p.level - en.price) * d.sign > 0]
        rng_box = sig.evidence.get("range")
        range_fade = sig.family in (StrategyFamily.MEAN_REVERSION, StrategyFamily.RANGE) and rng_box is not None
        if range_fade:
            # pools *inside* the range are consumed on the way to the other side; the draw is the far edge
            lo_b, hi_b = min(rng_box), max(rng_box)
            inside = [p for p in ahead if lo_b - 1e-12 <= p.level <= hi_b + 1e-12]
            near = [p for p in inside if abs(p.level - sig.entry_mid) / max(sig.risk_per_unit, 1e-12) < 0.5]
            if near:
                c.minor.append(f"internal range liquidity ({near[0].note}) within 0.5R — expect a pause before the mid")
            far = max((abs(p.level - sig.entry_mid) / max(sig.risk_per_unit, 1e-12) for p in inside), default=0.0)
            if far >= 1.5:
                r += 0.2
                c.why.append(f"opposite range edge liquidity is the draw ({far:.1f}R)")
            ahead = []   # handled
        if ahead:
            nearest = min(ahead, key=lambda p: abs(p.level - en.price))
            dist_r = abs(nearest.level - sig.entry_mid) / max(sig.risk_per_unit, 1e-12)
            if dist_r >= 1.5:
                r += 0.2
                c.why.append(f"untapped {nearest.kind.value.lower().replace('_', '-')} liquidity ({nearest.note}) at {nearest.level:.5g} as draw ({dist_r:.1f}R)")
            elif dist_r < 0.5:
                c.major.append(f"significant opposing liquidity ({nearest.note}) only {dist_r:.1f}R away at {nearest.level:.5g}")
            elif dist_r < 1.0:
                c.minor.append(f"opposing liquidity ({nearest.note}) {dist_r:.1f}R away at {nearest.level:.5g} — TP1 may be capped")
        elif not range_fade:
            r += 0.1
        behind = [p for p in sig_pools if (sig.stop - p.level) * d.sign < 0 <= (sig.entry_mid - p.level) * d.sign
                  and abs(sig.stop - p.level) <= 0.3 * sig.risk_per_unit]
        if behind:
            c.minor.append("stop rests just beyond a liquidity pool (likely to be run) — consider placing beyond the pool")
        else:
            r += 0.1
        return min(1.0, r)

    def _price_action(self, sig, ctx, regime, c: _Collector) -> float:
        en = ctx.entry
        d = sig.direction
        r = 0.0
        pat = sig.evidence.get("pattern")
        if pat is None:
            from ai_trader.analysis.patterns import best_pattern
            pat = best_pattern(en.patterns, d)
        if pat is not None:
            r += 0.5 * pat.strength + 0.2
            c.why.append(f"{pat.name} on {en.timeframe.value} (strength {pat.strength:.2f})")
        opp = [p for p in en.patterns if p.direction is not d and p.strength >= 0.7 and p.name not in ("doji", "inside_bar")]
        if opp:
            latest_opp = opp[-1]
            # an opposing pattern on the *last* bar is major; older ones are minor
            if latest_opp.index == len(en.df) - 1 and latest_opp.strength >= 0.85:
                c.major.append(f"last candle is a strong opposing {latest_opp.name}")
            else:
                c.minor.append(f"opposing candle pattern in recent bars: {latest_opp.name}")
        br = sig.evidence.get("breakout")
        if br is not None:
            r += 0.4 * br.score
            c.why.append(f"breakout quality {br.score:.2f}" + (" with retest" if br.retested else ""))
        if sig.evidence.get("ema_reclaim"):
            r += 0.3
            c.why.append("EMA20 reclaim")
        last = en.df.iloc[-1]
        if (last["close"] - last["open"]) * d.sign > 0:
            r += 0.15
        return min(1.0, r)

    def _volume(self, sig, ctx, regime, c: _Collector):
        en = ctx.entry
        v = en.volume
        if not v.available:
            return 0.5, False
        s = v.score(sig.direction)
        if sig.family in (StrategyFamily.MEAN_REVERSION, StrategyFamily.RANGE) and v.price_vs_vwap in ("above", "below"):
            # for a fade the stretch *away* from VWAP is the edge and VWAP is the magnet/target
            same_side = v.price_vs_vwap == ("above" if sig.direction is Direction.LONG else "below")
            s += -0.1 if same_side else 0.1
            if not same_side:
                c.why.append(f"price {v.price_vs_vwap} session VWAP — VWAP acts as the reversion magnet")
            s = float(np.clip(s, 0, 1))
        notes = [n for n in v.notes if "confirms" in n or "spike" in n or "VWAP" in n]
        c.why.append(f"volume score {s:.2f}" + (": " + "; ".join(notes) if notes else ""))
        if v.confirms_direction is False and (v.rvol or 0) > 1.5:
            c.minor.append("elevated volume against the trade direction")
        if v.climax:
            c.minor.append("possible climax volume (exhaustion)")
        return s, True

    def _volatility(self, sig, ctx, regime, c: _Collector) -> float:
        en = ctx.entry
        va = en.volatility
        s = va.score(sig.family.value)
        if s >= 0.5:
            c.why.append(f"volatility {va.state}/{va.phase} suits {sig.family.value} ({s:.2f})")
        if va.state == "EXTREME":
            c.major.append("extreme volatility")
        if va.state == "DEAD":
            c.major.append("dead volatility (no follow-through expected)")
        stop_atr = sig.risk_per_unit / max(en.atr, 1e-12)
        if stop_atr < 0.5:
            c.major.append(f"stop only {stop_atr:.2f} ATR — inside noise")
            s *= 0.6
        elif stop_atr > 4:
            c.major.append(f"stop {stop_atr:.1f} ATR — too wide for the timeframe")
            s *= 0.7
        else:
            c.why.append(f"stop distance {stop_atr:.2f} ATR (outside noise, not oversized)")
        return float(np.clip(s, 0, 1))

    def _indicators(self, sig, ctx, regime, c: _Collector) -> float:
        en, st = ctx.entry, ctx.structural
        d = sig.direction
        l, ls = en.last, st.last
        votes, n = 0.0, 0
        mean_rev = sig.family in (StrategyFamily.MEAN_REVERSION, StrategyFamily.RANGE)
        rsi = l.get("rsi")
        if rsi is not None:
            n += 1
            if mean_rev:
                ok = (rsi <= 35 and d is Direction.LONG) or (rsi >= 65 and d is Direction.SHORT)
            else:
                ok = (45 <= rsi <= 72 and d is Direction.LONG) or (28 <= rsi <= 55 and d is Direction.SHORT)
                if (rsi > 78 and d is Direction.LONG) or (rsi < 22 and d is Direction.SHORT):
                    c.minor.append(f"RSI {rsi:.0f} exhausted for a continuation entry")
            votes += 1 if ok else 0
            if ok:
                c.why.append(f"RSI {rsi:.0f} consistent with {sig.family.value}")
        mh = l.get("macd_hist")
        if mh is not None:
            n += 1
            if mean_rev:
                hist = en.ind["macd_hist"].dropna() if "macd_hist" in en.ind else None
                turning = hist is not None and len(hist) >= 2 and (hist.iloc[-1] - hist.iloc[-2]) * d.sign > 0
                if turning:
                    votes += 1
                    c.why.append("MACD histogram turning (momentum into the extreme is fading)")
            elif mh * d.sign > 0:
                votes += 1
                c.why.append("MACD histogram aligned")
            elif sig.family in CONTINUATION_FAMILIES:
                c.minor.append("MACD histogram against the trade")
        adx = ls.get("adx")
        if adx is not None:
            n += 1
            ok = adx < 25 if mean_rev else adx >= 20
            votes += 1 if ok else 0
            if ok:
                c.why.append(f"ADX {adx:.0f} suits {sig.family.value}")
        pdi, mdi = ls.get("plus_di"), ls.get("minus_di")
        if pdi is not None and mdi is not None and not mean_rev:
            n += 1
            if (pdi > mdi) == (d is Direction.LONG):
                votes += 1
                c.why.append("DI+ / DI- aligned")
        e20, e50 = ls.get("ema20"), ls.get("ema50")
        if e20 is not None and e50 is not None and not mean_rev:
            n += 1
            if (e20 > e50) == (d is Direction.LONG):
                votes += 1
                c.why.append("EMA20/50 stack aligned on structural TF")
        stk = l.get("stoch_k")
        if stk is not None and mean_rev:
            n += 1
            if (stk <= 25 and d is Direction.LONG) or (stk >= 75 and d is Direction.SHORT):
                votes += 1
                c.why.append(f"Stochastic {stk:.0f} stretched")
        return votes / n if n else 0.5

    def _risk_reward(self, sig, ctx, regime, c: _Collector) -> float:
        rr1, rr2, rr3 = sig.rr(0), sig.rr(1), sig.rr(2)
        th = self.thresholds
        if rr2 <= 0:
            c.major.append("no valid risk/reward")
            return 0.0
        if rr1 < th.min_rr_tp1:
            c.major.append(f"TP1 R:R 1:{rr1:.2f} below minimum 1:{th.min_rr_tp1}")
        r = float(np.clip((rr2 - 0.8) / 2.4, 0, 1))  # 1:2 → 0.5, 1:2.5 → 0.7, ≥1:3.2 → 1.0
        if rr2 >= th.min_rr_tp2:
            c.why.append(f"risk/reward TP1 1:{rr1:.1f}, TP2 1:{rr2:.1f}, TP3 1:{rr3:.1f}")
        else:
            c.major.append(f"TP2 R:R 1:{rr2:.2f} below minimum 1:{th.min_rr_tp2}")
        width_r = (sig.entry_high - sig.entry_low) / max(sig.risk_per_unit, 1e-12)
        if width_r > 0.6:
            r *= 0.8
            c.minor.append(f"entry zone is {width_r:.0%} of the risk distance — imprecise entry")
        return r

"""
Multi-timeframe (top-down) context.

For each loaded timeframe we build a ``TimeframeAnalysis`` (indicators,
structure, levels, SMC snapshot, patterns, volume, volatility). The
``MultiTimeframeContext`` then derives:

* HTF bias   (Monthly/Weekly/Daily structure + EMA stack)
* Structural bias (4H/1H structure + last events)
* Entry TF state (latest sweep / MSS / pattern)
* Alignment score  (-1..+1) and a list of conflicts

Higher-timeframe frames are already causal (closed bars only) thanks to the
loader/resampler.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ai_trader.config.settings import AnalysisSettings, get_settings
from ai_trader.core.enums import Bias, Direction, Timeframe
from ai_trader.core.models import CandlePattern, Instrument, Level, MarketData

from .indicators import compute_indicator_frame, latest
from .levels import daily_weekly_levels, horizontal_levels, psychological_levels
from .patterns import recent_patterns
from .smc import SMCSnapshot, smc_snapshot
from .structure import StructureState, analyze_structure, structure_summary
from .volatility import VolatilityAssessment, assess_volatility
from .volume import VolumeAssessment, assess_volume


@dataclass
class TimeframeAnalysis:
    timeframe: Timeframe
    df: pd.DataFrame                       # raw OHLCV
    ind: pd.DataFrame                      # indicator frame
    structure: StructureState
    levels: List[Level]
    smc: SMCSnapshot
    patterns: List[CandlePattern]
    volume: VolumeAssessment
    volatility: VolatilityAssessment
    trend_bias: Bias
    trend_strength: float                  # 0..1
    notes: List[str] = field(default_factory=list)

    @property
    def last(self) -> Dict[str, float]:
        return latest(self.ind)

    @property
    def price(self) -> float:
        return float(self.df["close"].iloc[-1])

    @property
    def atr(self) -> float:
        return float(self.ind["atr"].iloc[-1])

    def summary(self) -> Dict:
        l = self.last
        return {
            "timeframe": self.timeframe.value,
            "price": self.price,
            "trend_bias": self.trend_bias.value,
            "trend_strength": round(self.trend_strength, 3),
            "structure": structure_summary(self.structure),
            "adx": None if l.get("adx") is None else round(l["adx"], 1),
            "rsi": None if l.get("rsi") is None else round(l["rsi"], 1),
            "ema_stack": self._ema_stack(),
            "atr": round(self.atr, 6),
            "volatility": self.volatility.state,
            "premium_discount": self.smc.pd_state,
            "recent_sweep": None if not self.smc.recent_sweep else f"{self.smc.recent_sweep.kind.value}@{self.smc.recent_sweep.level:.5g}",
            "patterns": [p.name for p in self.patterns[-3:]],
        }

    def _ema_stack(self) -> str:
        l = self.last
        e20, e50, e200 = l.get("ema20"), l.get("ema50"), l.get("ema200")
        if None in (e20, e50):
            return "n/a"
        if e200 is None:
            return "bullish" if e20 > e50 else "bearish"
        if e20 > e50 > e200:
            return "bullish"
        if e20 < e50 < e200:
            return "bearish"
        return "mixed"


def _trend_bias(ind: pd.DataFrame, structure: StructureState) -> tuple[Bias, float]:
    l = latest(ind)
    votes = 0.0
    weight = 0.0
    e20, e50, e200 = l.get("ema20"), l.get("ema50"), l.get("ema200")
    price = float(ind["close"].iloc[-1])
    if e20 is not None and e50 is not None:
        votes += 1.0 if e20 > e50 else -1.0
        weight += 1.0
        votes += 0.5 if price > e50 else -0.5
        weight += 0.5
    if e200 is not None:
        votes += 1.0 if price > e200 else -1.0
        weight += 1.0
    slope = l.get("ema50_slope")
    if slope is not None:
        votes += float(np.clip(slope * 10, -1, 1))
        weight += 1.0
    sb = structure.bias_score()
    votes += 2.0 * sb
    weight += 2.0
    adx = l.get("adx") or 0.0
    strength_raw = abs(votes / weight) if weight else 0.0
    strength = float(np.clip(0.6 * strength_raw + 0.4 * min(1.0, adx / 40.0), 0, 1))
    score = votes / weight if weight else 0.0
    if score > 0.25:
        return Bias.BUY, strength
    if score < -0.25:
        return Bias.SELL, strength
    return Bias.NEUTRAL, strength


def analyze_timeframe(df: pd.DataFrame, tf: Timeframe, instrument: Optional[Instrument] = None,
                      settings: Optional[AnalysisSettings] = None, extra_levels: Optional[Dict[str, float]] = None) -> TimeframeAnalysis:
    settings = settings or get_settings().analysis
    ind = compute_indicator_frame(df, settings.atr_period, settings.adx_period)
    structure = analyze_structure(df, settings.swing_lookback, settings.atr_period)
    lv = horizontal_levels(df, tf, settings.swing_lookback, settings.atr_period)
    if instrument is not None:
        lv += psychological_levels(float(df["close"].iloc[-1]), instrument.pip_size)
    smc = smc_snapshot(df, tf, settings.swing_lookback, extra_levels=extra_levels)
    trend_bias, strength = _trend_bias(ind, structure)
    hint = Direction.LONG if trend_bias is Bias.BUY else (Direction.SHORT if trend_bias is Bias.SELL else None)
    pats = recent_patterns(df, bars=3, trend_hint=hint)
    vol = assess_volume(df, hint, ind=ind)
    vola = assess_volatility(df, settings.atr_period, settings.extreme_volatility_atr_percentile, settings.dead_volatility_atr_percentile, ind=ind)
    return TimeframeAnalysis(tf, df, ind, structure, lv, smc, pats, vol, vola, trend_bias, strength)


@dataclass
class MultiTimeframeContext:
    symbol: str
    analyses: Dict[Timeframe, TimeframeAnalysis]
    htf_bias: Bias
    htf_strength: float
    structural_bias: Bias
    structural_strength: float
    entry_tf: Timeframe
    alignment: float                      # -1..+1 (sign = direction, magnitude = agreement)
    conflicts: List[str]
    key_levels: Dict[str, float]
    notes: List[str] = field(default_factory=list)

    @property
    def entry(self) -> TimeframeAnalysis:
        return self.analyses[self.entry_tf]

    @property
    def structural(self) -> TimeframeAnalysis:
        s = get_settings().analysis
        for tf in s.structural_timeframes:
            if tf in self.analyses:
                return self.analyses[tf]
        return self.entry

    @property
    def htf(self) -> Optional[TimeframeAnalysis]:
        s = get_settings().analysis
        for tf in s.htf_timeframes:
            if tf in self.analyses:
                return self.analyses[tf]
        return None

    @property
    def price(self) -> float:
        return self.entry.price

    def dominant_bias(self) -> Bias:
        if self.htf_bias is not Bias.NEUTRAL and self.structural_bias in (self.htf_bias, Bias.NEUTRAL):
            return self.htf_bias
        if self.htf_bias is Bias.NEUTRAL:
            return self.structural_bias
        return Bias.NEUTRAL  # explicit conflict

    def summary(self) -> Dict:
        return {
            "symbol": self.symbol,
            "htf_bias": self.htf_bias.value,
            "htf_strength": round(self.htf_strength, 3),
            "structural_bias": self.structural_bias.value,
            "structural_strength": round(self.structural_strength, 3),
            "dominant_bias": self.dominant_bias().value,
            "alignment": round(self.alignment, 3),
            "conflicts": self.conflicts,
            "key_levels": {k: round(v, 6) for k, v in self.key_levels.items()},
            "timeframes": {tf.value: a.summary() for tf, a in sorted(self.analyses.items(), key=lambda kv: -kv[0].minutes)},
        }


def _frame_key(df: pd.DataFrame, extra: Optional[Dict[str, float]]) -> tuple:
    return (len(df), int(df.index[0].value), int(df.index[-1].value), float(df["close"].iloc[-1]), float(df["high"].iloc[-1]),
            float(df["low"].iloc[-1]), tuple(sorted((k, round(v, 10)) for k, v in (extra or {}).items())))


def build_context(md: MarketData, instrument: Optional[Instrument] = None,
                  settings: Optional[AnalysisSettings] = None,
                  cache: Optional[Dict[Timeframe, tuple]] = None) -> MultiTimeframeContext:
    """Analyse every loaded timeframe and combine them.

    ``cache`` (optional, owned by the caller — e.g. the backtester) maps timeframe →
    (frame_key, TimeframeAnalysis). Analysis is a pure function of the frame, so when
    a higher timeframe has not printed a new bar since the previous call its analysis
    is reused verbatim. This changes nothing in the result, only the run time.
    """
    settings = settings or get_settings().analysis
    daily = md.frames.get(Timeframe.D1)
    weekly = md.frames.get(Timeframe.W1)
    key_levels = daily_weekly_levels(daily, weekly) if daily is not None else {}
    analyses: Dict[Timeframe, TimeframeAnalysis] = {}
    for tf in md.timeframes:
        extra = {k: v for k, v in key_levels.items() if k in ("pdh", "pdl", "pwh", "pwl")} if tf.group != "HIGH" else None
        frame = md.frames[tf]
        if cache is not None:
            key = _frame_key(frame, extra)
            hit = cache.get(tf)
            if hit is not None and hit[0] == key:
                analyses[tf] = hit[1]
                continue
            analyses[tf] = analyze_timeframe(frame, tf, instrument, settings, extra)
            cache[tf] = (key, analyses[tf])
        else:
            analyses[tf] = analyze_timeframe(frame, tf, instrument, settings, extra)

    def _combine(tfs: List[Timeframe]) -> tuple[Bias, float]:
        present = [analyses[t] for t in tfs if t in analyses]
        if not present:
            return Bias.NEUTRAL, 0.0
        score = 0.0
        w_sum = 0.0
        for a in present:
            w = np.log2(a.timeframe.minutes + 1)
            score += w * a.trend_bias.sign * max(a.trend_strength, 0.2)
            w_sum += w
        s = score / w_sum
        strength = float(np.mean([a.trend_strength for a in present]))
        if s > 0.15:
            return Bias.BUY, strength
        if s < -0.15:
            return Bias.SELL, strength
        return Bias.NEUTRAL, strength

    htf_bias, htf_strength = _combine(settings.htf_timeframes)
    st_bias, st_strength = _combine(settings.structural_timeframes)
    entry_tf = next((t for t in settings.entry_timeframes if t in analyses), md.timeframes[0])
    entry_bias = analyses[entry_tf].trend_bias

    # alignment: weighted agreement of all timeframe biases
    signs = []
    weights = []
    for tf, a in analyses.items():
        signs.append(a.trend_bias.sign * max(0.2, a.trend_strength))
        weights.append(np.log2(tf.minutes + 1))
    alignment = float(np.average(signs, weights=weights)) if signs else 0.0

    conflicts: List[str] = []
    if (htf_bias is not Bias.NEUTRAL and st_bias is not Bias.NEUTRAL and htf_bias is not st_bias
            and htf_strength >= 0.35 and st_strength >= 0.35):
        conflicts.append(f"HTF bias {htf_bias.value} vs structural bias {st_bias.value}")
    dom = htf_bias if htf_bias is not Bias.NEUTRAL else st_bias
    if dom is not Bias.NEUTRAL and entry_bias is not Bias.NEUTRAL and entry_bias is not dom:
        conflicts.append(f"entry TF {entry_tf.value} bias {entry_bias.value} against dominant {dom.value} (pullback or reversal?)")
    notes = [f"HTF {htf_bias.value} ({htf_strength:.0%}), structural {st_bias.value} ({st_strength:.0%}), alignment {alignment:+.2f}"]
    return MultiTimeframeContext(md.symbol, analyses, htf_bias, htf_strength, st_bias, st_strength, entry_tf,
                                 alignment, conflicts, key_levels, notes)

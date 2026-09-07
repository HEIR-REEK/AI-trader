"""
Market regime detection.

A *rule-based voting* detector (transparent, testable) that the ML layer can
later augment. It scores each candidate regime from a feature vector and
returns the primary regime, secondary regimes, a confidence and the strategy
families permitted in that regime.

Features (all causal, from the structural timeframe unless noted):
  adx, plus/minus DI, efficiency ratio, EMA-50 slope (ATR-normalised),
  EMA stack, structure bias (HH/HL vs LH/LL + BOS/CHoCH), ATR percentile,
  BB-width percentile, squeeze flag, HV ratio, choppiness, range position,
  recent breakout assessment, recent sweep/fake-break flags, news proximity.

Answering "what type of market are we trading?" *before* looking for entries
is what stops trend strategies firing in ranges and mean-reversion firing
into strong trends.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ai_trader.analysis.breakout import assess_breakout, range_box
from ai_trader.analysis.indicators import latest
from ai_trader.analysis.mtf import TimeframeAnalysis
from ai_trader.config.settings import RegimeStrategyMap, get_settings
from ai_trader.core.enums import Direction, Regime, StrategyFamily
from ai_trader.core.models import RegimeAssessment


def regime_features(a: TimeframeAnalysis, news_proximity_min: Optional[float] = None,
                    recent_surprise: bool = False) -> Dict[str, float]:
    l = latest(a.ind)
    df = a.df
    price = a.price
    hi, lo = range_box(df, 30)
    rng = hi - lo
    pos = (price - lo) / rng if rng > 0 else 0.5
    st = a.structure
    ev = st.last_event
    swept_recent = 1.0 if (a.smc.recent_sweep is not None) else 0.0
    # recent breakout of the 30-bar box (excluding the last bar's own contribution)
    prior_hi, prior_lo = range_box(df.iloc[:-6], 30) if len(df) > 40 else (hi, lo)
    br_up = assess_breakout(df, prior_hi, Direction.LONG) if price > prior_hi else None
    br_dn = assess_breakout(df, prior_lo, Direction.SHORT) if price < prior_lo else None
    br = br_up or br_dn
    # fake-out: recent close beyond prior box that has come back inside
    recent_closes = df["close"].tail(6)
    fake = 0.0
    if (recent_closes.iloc[:-1] > prior_hi).any() and price < prior_hi:
        fake = 1.0
    if (recent_closes.iloc[:-1] < prior_lo).any() and price > prior_lo:
        fake = 1.0
    # ADX slope (rising trend strength vs fading)
    adx_series = a.ind["adx"].dropna()
    adx_slope = float(adx_series.iloc[-1] - adx_series.iloc[-6]) if len(adx_series) > 6 else 0.0
    # net drift over 50 bars in ATR units
    atr_v = a.atr or 1e-12
    drift50 = float((df["close"].iloc[-1] - df["close"].iloc[-min(50, len(df) - 1)]) / atr_v)
    # volume trend (accumulation/distribution heuristic): OBV slope vs price slope
    obv = a.ind["obv"]
    obv_slope = float(obv.iloc[-1] - obv.iloc[-20]) if len(obv) > 20 and a.volume.available else 0.0
    price_slope20 = float(df["close"].iloc[-1] - df["close"].iloc[-20]) if len(df) > 20 else 0.0
    return {
        "adx": float(l.get("adx") or 0.0),
        "adx_slope": adx_slope,
        "plus_di": float(l.get("plus_di") or 0.0),
        "minus_di": float(l.get("minus_di") or 0.0),
        "er": float(l.get("er") or 0.0),
        "ema50_slope": float(l.get("ema50_slope") or 0.0),
        "chop": float(l.get("chop") or 50.0),
        "structure_bias": float(st.bias_score()),
        "atr_percentile": float(a.volatility.atr_percentile),
        "bb_width_pct": float(a.volatility.bb_width_percentile),
        "squeeze": 1.0 if a.volatility.squeeze else 0.0,
        "hv_ratio": float(a.volatility.hv_ratio),
        "atr_spike_ratio": float(a.volatility.atr_spike_ratio),
        "expansion": 1.0 if a.volatility.phase == "EXPANSION" else 0.0,
        "contraction": 1.0 if a.volatility.phase == "CONTRACTION" else 0.0,
        "range_pos": float(pos),
        "range_width_atr": float(rng / atr_v),
        "breakout_score": float(br.score) if br else 0.0,
        "breakout_dir": (1.0 if br and br.direction is Direction.LONG else (-1.0 if br else 0.0)),
        "fake_breakout": fake,
        "recent_sweep": swept_recent,
        "last_event_choch": 1.0 if ev and ev.kind.value in ("CHOCH", "MSS") else 0.0,
        "last_event_dir": float(ev.direction.sign) if ev else 0.0,
        "drift50_atr": drift50,
        "obv_price_divergence": float(np.sign(obv_slope) * -np.sign(price_slope20)) if obv_slope and price_slope20 else 0.0,
        "news_proximity_min": float(news_proximity_min) if news_proximity_min is not None else 1e9,
        "recent_surprise": 1.0 if recent_surprise else 0.0,
        "trend_strength": float(a.trend_strength),
        "trend_sign": float(a.trend_bias.sign),
    }


def score_regimes(f: Dict[str, float]) -> Dict[Regime, float]:
    s: Dict[Regime, float] = {r: 0.0 for r in Regime}
    adx, er, slope, sb = f["adx"], f["er"], f["ema50_slope"], f["structure_bias"]
    trend_sign = np.sign(slope) if abs(slope) > 1e-9 else np.sign(sb)
    trend_evidence = (min(adx, 50) / 50) * 0.35 + min(er / 0.5, 1.0) * 0.30 + min(abs(slope) / 0.15, 1.0) * 0.15 + min(abs(sb), 1.0) * 0.20
    di_agree = (f["plus_di"] > f["minus_di"]) == (trend_sign > 0)
    strong = trend_evidence * (1.1 if di_agree else 0.8)
    if trend_sign > 0:
        s[Regime.STRONG_BULL] = strong if adx >= 22 else strong * 0.4
    else:
        s[Regime.STRONG_BEAR] = strong if adx >= 22 else strong * 0.4
    s[Regime.WEAK_TREND] = max(0.0, 0.75 - abs(trend_evidence - 0.42) * 2.2) if 15 <= adx < 25 else 0.15 * (adx < 15)

    range_evidence = (1 - min(adx, 40) / 40) * 0.4 + (1 - min(er / 0.4, 1.0)) * 0.3 + max(0.0, (f["chop"] - 50) / 30) * 0.3
    s[Regime.RANGE] = range_evidence * (0.6 if f["range_width_atr"] < 3 else 1.0)
    # accumulation / distribution: range + drift + OBV/price divergence
    if range_evidence > 0.45:
        if f["obv_price_divergence"] > 0 or f["drift50_atr"] < -1.5:
            s[Regime.ACCUMULATION] = range_evidence * 0.7 * (1 + 0.5 * (f["range_pos"] < 0.4))
        if f["obv_price_divergence"] > 0 or f["drift50_atr"] > 1.5:
            s[Regime.DISTRIBUTION] = range_evidence * 0.7 * (1 + 0.5 * (f["range_pos"] > 0.6))

    s[Regime.BREAKOUT] = f["breakout_score"] * (1.2 if f["expansion"] else 0.9)
    # a failed break of a *range* — inside a strong trend a pullback through the
    # 30-bar box is normal behaviour, so damp by range evidence
    s[Regime.FALSE_BREAKOUT] = (0.85 * f["fake_breakout"] + 0.25 * f["recent_sweep"] * (f["fake_breakout"] > 0)) * (0.3 + 0.7 * range_evidence)
    # HIGH_VOL requires top-percentile rank AND an absolute spike vs the median
    spike_term = float(np.clip((f["atr_spike_ratio"] - 1.2) / 0.8, 0, 1))
    s[Regime.HIGH_VOL] = max(0.0, (f["atr_percentile"] - 0.85) / 0.15) * spike_term
    s[Regime.LOW_VOL] = max(0.0, (0.15 - f["atr_percentile"]) / 0.15)
    s[Regime.EXPANSION] = 0.6 * f["expansion"] + 0.3 * max(0.0, (f["hv_ratio"] - 1.0)) + 0.1 * f["breakout_score"]
    s[Regime.CONTRACTION] = 0.6 * f["contraction"] + 0.4 * f["squeeze"] + 0.2 * max(0.0, (0.3 - f["bb_width_pct"]) / 0.3)
    # Reversal needs a *trend to reverse*: CHoCH flips are routine inside ranges,
    # so the score is damped by range evidence and scaled by trend strength.
    reversal_base = 0.6 * f["last_event_choch"] + 0.3 * f["recent_sweep"] + (0.2 if f["adx_slope"] < -3 else 0.0)
    s[Regime.REVERSAL] = reversal_base * float(np.clip(1.0 - range_evidence, 0, 1)) * (0.4 + 0.6 * min(adx / 25.0, 1.0))
    if f["news_proximity_min"] <= 30 or f["recent_surprise"]:
        s[Regime.NEWS_DRIVEN] = 1.0 if f["news_proximity_min"] <= 15 or f["recent_surprise"] else 0.7
    return {k: float(np.clip(v, 0, 1.5)) for k, v in s.items()}


PRIMARY_CANDIDATES = [Regime.STRONG_BULL, Regime.STRONG_BEAR, Regime.WEAK_TREND, Regime.RANGE,
                      Regime.ACCUMULATION, Regime.DISTRIBUTION, Regime.BREAKOUT, Regime.FALSE_BREAKOUT, Regime.REVERSAL]
OVERLAY_CANDIDATES = [Regime.HIGH_VOL, Regime.LOW_VOL, Regime.EXPANSION, Regime.CONTRACTION, Regime.NEWS_DRIVEN]


class RegimeDetector:
    def __init__(self, regime_map: Optional[RegimeStrategyMap] = None, ml_model=None):
        self.regime_map = regime_map or get_settings().regime_map
        self.ml_model = ml_model  # optional: object with predict_proba(features_df) → {Regime: p}

    def detect(self, a: TimeframeAnalysis, news_proximity_min: Optional[float] = None,
               recent_surprise: bool = False) -> RegimeAssessment:
        f = regime_features(a, news_proximity_min, recent_surprise)
        scores = score_regimes(f)
        if self.ml_model is not None:
            try:
                ml = self.ml_model.predict_regime_proba(f)
                for r, p in ml.items():
                    scores[r] = 0.6 * scores.get(r, 0.0) + 0.4 * p
            except Exception:  # ML is advisory only — never let it break detection
                pass

        # Hard overlays first: news / extreme volatility dominate everything
        explanation: List[str] = []
        if scores[Regime.NEWS_DRIVEN] >= 0.7:
            primary = Regime.NEWS_DRIVEN
            explanation.append("high-impact news within blackout window or recent surprise → NEWS_DRIVEN")
        elif scores[Regime.HIGH_VOL] >= 0.8:
            primary = Regime.HIGH_VOL
            explanation.append(f"ATR percentile {f['atr_percentile']:.0%} with spike {f['atr_spike_ratio']:.1f}× median → HIGH_VOLATILITY")
        else:
            ranked = sorted(PRIMARY_CANDIDATES, key=lambda r: scores[r], reverse=True)
            primary = ranked[0]
            if scores[primary] < 0.3:
                primary = Regime.UNKNOWN
        secondary = [r for r in OVERLAY_CANDIDATES if scores[r] >= 0.5 and r is not primary]
        # add the runner-up primary if close
        ranked = sorted(PRIMARY_CANDIDATES, key=lambda r: scores[r], reverse=True)
        if primary in PRIMARY_CANDIDATES and len(ranked) > 1 and scores[ranked[1]] >= 0.8 * scores[ranked[0]] and scores[ranked[1]] > 0.3:
            secondary.insert(0, ranked[1])

        # confidence: margin between top-2 primary scores + absolute level
        top = scores[primary] if primary is not Regime.UNKNOWN else 0.0
        second = scores[ranked[1]] if primary in PRIMARY_CANDIDATES and len(ranked) > 1 else 0.0
        confidence = float(np.clip(0.5 * min(top, 1.0) + 0.5 * min(1.0, max(0.0, top - second) * 2), 0, 1))
        if primary in (Regime.NEWS_DRIVEN, Regime.HIGH_VOL):
            confidence = max(confidence, 0.8)

        allowed = list(self.regime_map.mapping.get(primary, []))
        # overlays can veto
        if Regime.HIGH_VOL in secondary or Regime.NEWS_DRIVEN in secondary:
            allowed = []
            explanation.append("volatility/news overlay vetoes new entries")
        if Regime.LOW_VOL in secondary:
            # quiet tape: momentum has nothing to feed on; value entries (pullback / sweep) and
            # range strategies remain valid, DEAD volatility is blocked separately by the volatility filter
            allowed = [fam for fam in allowed if fam is not StrategyFamily.MOMENTUM]
            explanation.append("low-volatility overlay: momentum family disabled")
        explanation += self._explain(primary, f)
        return RegimeAssessment(primary=primary, secondary=secondary, confidence=confidence,
                                scores={r.value: round(v, 3) for r, v in scores.items()}, features=f,
                                allowed_families=allowed, explanation=explanation, timeframe=a.timeframe)

    @staticmethod
    def _explain(primary: Regime, f: Dict[str, float]) -> List[str]:
        out = [f"ADX {f['adx']:.1f}, efficiency ratio {f['er']:.2f}, EMA50 slope {f['ema50_slope']:+.3f} ATR/bar, "
               f"structure bias {f['structure_bias']:+.2f}, ATR percentile {f['atr_percentile']:.0%}, chop {f['chop']:.0f}"]
        if primary in (Regime.STRONG_BULL, Regime.STRONG_BEAR):
            out.append("trend-following, pullback and momentum families enabled; mean-reversion disabled")
        elif primary in (Regime.RANGE, Regime.ACCUMULATION, Regime.DISTRIBUTION):
            out.append("range/mean-reversion and liquidity-reversal families enabled; trend-following disabled")
        elif primary is Regime.BREAKOUT:
            out.append(f"breakout score {f['breakout_score']:.2f} with expansion={bool(f['expansion'])}")
        elif primary is Regime.FALSE_BREAKOUT:
            out.append("recent close beyond range failed and returned inside → only liquidity-reversal permitted")
        elif primary is Regime.REVERSAL:
            out.append("CHoCH/MSS detected after a sweep → reversal candidates only, wait for confirmation")
        elif primary is Regime.UNKNOWN:
            out.append("no regime scored above the minimum evidence threshold → NO TRADE")
        return out

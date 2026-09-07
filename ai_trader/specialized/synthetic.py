"""
Synthetic / Volatility-index specialised module (Volatility 10/25/50/75/100).

These instruments are *engineered* random processes (constant target
volatility, 24/7, no sessions, no macro, no order book in the usual sense).
Therefore:

* Session, kill-zone, news and macro logic are DISABLED.
* Volume is unavailable → weight redistributed by the scorer.
* We measure the instrument's own statistical properties on the loaded data
  and expose them so strategy families are enabled/disabled per instrument:
    - Hurst exponent (rescaled range) → trending (>0.55) / mean-reverting (<0.45)
    - variance ratio (Lo-MacKinlay) → same question, different estimator
    - lag-1 autocorrelation of returns
    - realised vol vs nominal target (10/25/50/75/100 → annualised %),
    - ATR stability (coefficient of variation) — synthetic ATR should be stable;
      instability suggests feed problems.
* Strategy parameters must be backtested *independently per index*; the
  ``StatisticalProfile`` is what the backtester keys on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ai_trader.analysis.mtf import MultiTimeframeContext
from ai_trader.core.enums import StrategyFamily
from ai_trader.core.models import RegimeAssessment

NOMINAL_VOL = {"VOL10": 0.10, "VOL25": 0.25, "VOL50": 0.50, "VOL75": 0.75, "VOL100": 1.00}


def hurst_exponent(x: np.ndarray, min_chunk: int = 8) -> float:
    """Rescaled-range Hurst estimate on log returns (0.5 = random walk)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 64:
        return 0.5
    sizes = []
    rs_vals = []
    size = min_chunk
    while size <= n // 2:
        chunks = n // size
        rs = []
        for c in range(chunks):
            seg = x[c * size:(c + 1) * size]
            dev = np.cumsum(seg - seg.mean())
            r = dev.max() - dev.min()
            s = seg.std(ddof=1)
            if s > 0:
                rs.append(r / s)
        if rs:
            sizes.append(size)
            rs_vals.append(np.mean(rs))
        size *= 2
    if len(sizes) < 3:
        return 0.5
    slope = np.polyfit(np.log(sizes), np.log(rs_vals), 1)[0]
    return float(np.clip(slope, 0.0, 1.0))


def variance_ratio(returns: np.ndarray, q: int = 8) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < q * 10:
        return 1.0
    var1 = r.var(ddof=1)
    rq = pd.Series(r).rolling(q).sum().dropna().to_numpy()
    varq = rq.var(ddof=1) / q
    return float(varq / var1) if var1 > 0 else 1.0


@dataclass
class StatisticalProfile:
    symbol: str
    bars: int
    hurst: float
    variance_ratio: float
    autocorr_lag1: float
    realised_vol_annual: float
    nominal_vol_annual: Optional[float]
    vol_deviation_pct: Optional[float]
    atr_cv: float
    character: str                          # TRENDING | MEAN_REVERTING | RANDOM_WALK
    recommended_families: List[StrategyFamily]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        d = {k: v for k, v in self.__dict__.items() if k != "recommended_families"}
        d["recommended_families"] = [f.value for f in self.recommended_families]
        return d


def statistical_profile(symbol: str, df: pd.DataFrame, bar_minutes: int) -> StatisticalProfile:
    close = df["close"].to_numpy(dtype=float)
    r = np.diff(np.log(close))
    h = hurst_exponent(r)
    vr = variance_ratio(r, q=8)
    ac = float(pd.Series(r).autocorr(lag=1)) if len(r) > 30 else 0.0
    bars_per_year = 525_600 / bar_minutes
    rv = float(np.std(r, ddof=1) * np.sqrt(bars_per_year)) if len(r) > 2 else 0.0
    nominal = NOMINAL_VOL.get(symbol.upper())
    dev = None if not nominal else (rv / nominal - 1) * 100
    tr = (df["high"] - df["low"]).rolling(14).mean().dropna()
    atr_cv = float(tr.std() / tr.mean()) if len(tr) and tr.mean() > 0 else 0.0
    notes = []
    if h > 0.56 and vr > 1.1:
        character = "TRENDING"
        fams = [StrategyFamily.TREND_FOLLOWING, StrategyFamily.PULLBACK, StrategyFamily.BREAKOUT]
        notes.append(f"Hurst {h:.2f} / VR {vr:.2f}: persistent — trend & breakout families favoured")
    elif h < 0.45 and vr < 0.9:
        character = "MEAN_REVERTING"
        fams = [StrategyFamily.MEAN_REVERSION, StrategyFamily.RANGE, StrategyFamily.LIQUIDITY_REVERSAL]
        notes.append(f"Hurst {h:.2f} / VR {vr:.2f}: anti-persistent — mean-reversion families favoured")
    else:
        character = "RANDOM_WALK"
        fams = [StrategyFamily.LIQUIDITY_REVERSAL, StrategyFamily.PULLBACK]
        notes.append(f"Hurst {h:.2f} / VR {vr:.2f}: close to random walk — edge must come from structure + R:R, expect low signal density")
    if dev is not None:
        if abs(dev) > 25:
            notes.append(f"realised vol {rv:.0%} deviates {dev:+.0f}% from nominal {nominal:.0%} — check feed / timeframe assumptions")
        else:
            notes.append(f"realised vol {rv:.0%} ≈ nominal {nominal:.0%} (feed consistent)")
    if atr_cv > 0.5:
        notes.append(f"ATR coefficient of variation {atr_cv:.2f} is high for a synthetic index — volatility regime shifts present or bad data")
    notes.append("no sessions, no news, no macro: time-of-day and calendar filters disabled")
    return StatisticalProfile(symbol, len(df), h, vr, ac, rv, nominal, dev, atr_cv, character, fams, notes)


@dataclass
class SyntheticContext:
    profile: StatisticalProfile
    allowed_families: List[StrategyFamily]
    confidence_multiplier: float
    veto: Optional[str]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {"profile": self.profile.to_dict(), "allowed_families": [f.value for f in self.allowed_families],
                "confidence_multiplier": self.confidence_multiplier, "veto": self.veto, "notes": self.notes}


def synthetic_context(ctx: MultiTimeframeContext, regime: RegimeAssessment) -> SyntheticContext:
    st = ctx.structural
    prof = statistical_profile(ctx.symbol, st.df, st.timeframe.minutes)
    allowed = [f for f in regime.allowed_families if f in prof.recommended_families]
    mult = 1.0
    veto = None
    notes = list(prof.notes)
    if prof.character == "RANDOM_WALK":
        mult = 0.9
        notes.append("random-walk character → confidence ×0.9; only structure+liquidity setups with ≥1:2 R:R pass")
    if prof.vol_deviation_pct is not None and abs(prof.vol_deviation_pct) > 40:
        veto = "realised volatility inconsistent with index specification — data integrity concern"
    if not allowed and regime.allowed_families:
        notes.append(f"regime allows {[f.value for f in regime.allowed_families]} but statistical profile does not support them on this index")
    return SyntheticContext(prof, allowed, mult, veto, notes)

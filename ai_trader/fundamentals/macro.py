"""
Fundamental / macro context.

The system does not *predict* macro; it (1) knows when high-impact scheduled
events are near and reduces confidence or blocks entries, (2) turns available
macro series (DXY, US10Y, real yields, VIX…) into a directional *headwind /
tailwind* score per instrument, and (3) exposes everything in the explanation.

Macro series are supplied as a dict of pandas Series (daily) by the data
layer; when absent the context reports "unknown" rather than guessing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ai_trader.config.settings import NewsSettings, get_settings
from ai_trader.core.enums import Bias
from ai_trader.core.models import Instrument
from ai_trader.data.calendar import EconomicCalendar, EconomicEvent


@dataclass
class NewsAssessment:
    in_blackout: bool
    blackout_event: Optional[EconomicEvent]
    next_high_impact: Optional[EconomicEvent]
    minutes_to_next: Optional[float]
    recent_surprise: Optional[EconomicEvent]
    penalty: float                     # 0..1 confidence reduction
    calendar_known: bool
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        def ev(e):
            return None if e is None else {"ts": e.ts.isoformat(), "currency": e.currency, "impact": e.impact, "title": e.title}
        return {"in_blackout": self.in_blackout, "blackout_event": ev(self.blackout_event), "next_high_impact": ev(self.next_high_impact),
                "minutes_to_next": self.minutes_to_next, "recent_surprise": ev(self.recent_surprise), "penalty": self.penalty,
                "calendar_known": self.calendar_known, "notes": self.notes}


def assess_news(symbol: str, instrument: Instrument, calendar: EconomicCalendar, now: datetime,
                settings: Optional[NewsSettings] = None) -> NewsAssessment:
    settings = settings or get_settings().news
    if not instrument.macro_sensitive:
        return NewsAssessment(False, None, None, None, None, 0.0, True, ["instrument not macro-sensitive (no news filter)"])
    if not calendar.known:
        pen = settings.high_impact_penalty * 0.5 if settings.treat_missing_calendar_as_unknown else 0.0
        return NewsAssessment(False, None, None, None, None, pen, False,
                              [f"economic calendar unavailable — confidence reduced {pen:.0%}; verify news manually"])
    ev = calendar.in_blackout(symbol, now, settings.blackout_before_minutes, settings.blackout_after_minutes)
    nxt = calendar.next_high_impact(symbol, now, 48)
    mins = None if nxt is None else (nxt.ts - now).total_seconds() / 60
    surprises = calendar.recent_surprises(symbol, now, 6)
    surprise = surprises[-1] if surprises else None
    notes = []
    pen = 0.0
    if ev is not None:
        pen = 1.0
        notes.append(f"HIGH-IMPACT NEWS BLACKOUT: {ev.title} ({ev.currency}) at {ev.ts:%H:%M} UTC — no new entries")
    elif mins is not None and mins <= 120:
        pen = settings.high_impact_penalty
        notes.append(f"{nxt.title} ({nxt.currency}) in {mins:.0f} min — confidence reduced {pen:.0%}, avoid holding through release")
    elif nxt is not None:
        notes.append(f"next high-impact: {nxt.title} ({nxt.currency}) in {mins / 60:.1f}h")
    if surprise is not None and surprise.surprise() is not None:
        notes.append(f"recent surprise: {surprise.title} actual {surprise.actual} vs forecast {surprise.forecast} — expect repricing / erratic flows")
        pen = max(pen, settings.high_impact_penalty)
    return NewsAssessment(ev is not None, ev, nxt, mins, surprise, pen, True, notes)


# --------------------------------------------------------------------------
# Macro direction scores
# --------------------------------------------------------------------------
@dataclass
class MacroContext:
    bias: Bias
    score: float                       # -1..+1 tailwind for longs
    drivers: Dict[str, float]
    notes: List[str] = field(default_factory=list)
    known: bool = False

    def to_dict(self) -> Dict:
        return {"bias": self.bias.value, "score": round(self.score, 3), "drivers": {k: round(v, 3) for k, v in self.drivers.items()},
                "notes": self.notes, "known": self.known}


def _mom(series: Optional[pd.Series], days: int = 20) -> Optional[float]:
    if series is None or len(series.dropna()) < days + 1:
        return None
    s = series.dropna()
    return float(s.iloc[-1] / s.iloc[-days - 1] - 1)


def _z(series: Optional[pd.Series], days: int = 20, window: int = 120) -> Optional[float]:
    if series is None or len(series.dropna()) < window:
        return None
    s = series.dropna()
    chg = s.diff(days)
    return float((chg.iloc[-1] - chg.tail(window).mean()) / (chg.tail(window).std() or 1e-9))


# sensitivity of each instrument to macro drivers (sign = effect on price of a RISE in the driver)
MACRO_SENSITIVITY: Dict[str, Dict[str, float]] = {
    "XAUUSD": {"DXY": -0.35, "US10Y": -0.25, "US_REAL_YIELD": -0.3, "VIX": +0.1, "BREAKEVEN": +0.1},
    "XAGUSD": {"DXY": -0.35, "US10Y": -0.2, "US_REAL_YIELD": -0.25, "VIX": -0.1, "COPPER": +0.2},
    "EURUSD": {"DXY": -0.6, "US10Y": -0.2, "BUND_US_SPREAD": +0.2},
    "GBPUSD": {"DXY": -0.6, "US10Y": -0.2},
    "AUDUSD": {"DXY": -0.5, "VIX": -0.25, "COPPER": +0.25},
    "NZDUSD": {"DXY": -0.5, "VIX": -0.25},
    "USDJPY": {"DXY": +0.4, "US10Y": +0.4, "VIX": -0.2},
    "USDCAD": {"DXY": +0.5, "OIL": -0.3},
    "USDCHF": {"DXY": +0.6, "VIX": -0.2},
    "US500": {"US10Y": -0.3, "VIX": -0.5, "DXY": -0.1},
    "US100": {"US10Y": -0.4, "VIX": -0.5},
    "US30": {"US10Y": -0.25, "VIX": -0.5},
    "DE40": {"VIX": -0.5, "BUND10Y": -0.3},
    "UK100": {"VIX": -0.4, "DXY": +0.1},
    "JP225": {"VIX": -0.4, "USDJPY": +0.4},
    "BTCUSD": {"DXY": -0.3, "VIX": -0.3, "US10Y": -0.2},
    "ETHUSD": {"DXY": -0.3, "VIX": -0.3, "US10Y": -0.2},
}


def macro_context(symbol: str, series: Optional[Dict[str, pd.Series]]) -> MacroContext:
    sens = MACRO_SENSITIVITY.get(symbol.upper())
    if not sens:
        return MacroContext(Bias.NEUTRAL, 0.0, {}, ["no macro model for this instrument"], known=False)
    if not series:
        return MacroContext(Bias.NEUTRAL, 0.0, {}, ["macro series not supplied — macro context unknown (not assumed neutral)"], known=False)
    drivers: Dict[str, float] = {}
    notes: List[str] = []
    total, wsum = 0.0, 0.0
    for name, w in sens.items():
        z = _z(series.get(name))
        if z is None:
            continue
        contrib = float(np.clip(z / 2.0, -1, 1)) * w
        drivers[name] = contrib
        total += contrib
        wsum += abs(w)
        direction = "rising" if z > 0.5 else ("falling" if z < -0.5 else "flat")
        notes.append(f"{name} {direction} (20d z={z:+.1f}) → {'tailwind' if contrib > 0 else 'headwind' if contrib < 0 else 'neutral'} for longs")
    if wsum == 0:
        return MacroContext(Bias.NEUTRAL, 0.0, {}, ["macro series present but too short"], known=False)
    score = float(np.clip(total / wsum, -1, 1))
    bias = Bias.BUY if score > 0.2 else (Bias.SELL if score < -0.2 else Bias.NEUTRAL)
    return MacroContext(bias, score, drivers, notes, known=True)

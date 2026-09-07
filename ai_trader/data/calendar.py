"""
Economic calendar loader.

Sources (in order): explicit list → JSON file (``data/calendar.json``) →
empty calendar. The decision engine treats an *empty/unknown* calendar for a
macro-sensitive instrument as a confidence penalty (configurable) rather
than pretending there is no news.

JSON schema per event:
  {"ts": "2026-09-05T12:30:00Z", "currency": "USD", "impact": "HIGH",
   "title": "Non-Farm Payrolls", "actual": null, "forecast": 150, "previous": 142}
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional

import pandas as pd

IMPACT_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}

# Which currencies move which instruments (used to filter events by relevance).
INSTRUMENT_CURRENCIES = {
    "XAUUSD": ["USD"], "XAGUSD": ["USD"],
    "US500": ["USD"], "US100": ["USD"], "US30": ["USD"], "DE40": ["EUR"], "UK100": ["GBP"], "JP225": ["JPY"],
    "BTCUSD": ["USD"], "ETHUSD": ["USD"], "SOLUSD": ["USD"], "XRPUSD": ["USD"],
}
HIGH_IMPACT_TITLES = (
    "non-farm", "nfp", "cpi", "ppi", "fomc", "fed", "interest rate", "rate decision", "gdp",
    "unemployment", "ecb", "boe", "boj", "snb", "rba", "boc", "pce", "retail sales", "ism", "pmi",
)


@dataclass
class EconomicEvent:
    ts: datetime
    currency: str
    impact: str
    title: str
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None

    @property
    def impact_rank(self) -> int:
        return IMPACT_RANK.get(self.impact.upper(), 1)

    def surprise(self) -> Optional[float]:
        if self.actual is None or self.forecast is None:
            return None
        return self.actual - self.forecast


def currencies_for(symbol: str) -> List[str]:
    s = symbol.upper()
    if s in INSTRUMENT_CURRENCIES:
        return INSTRUMENT_CURRENCIES[s]
    if len(s) == 6:
        return [s[:3], s[3:]]
    return ["USD"]


@dataclass
class EconomicCalendar:
    events: List[EconomicEvent] = field(default_factory=list)
    source: str = "empty"

    @classmethod
    def from_json(cls, path: str) -> "EconomicCalendar":
        if not os.path.exists(path):
            return cls([], source="missing")
        with open(path) as f:
            raw = json.load(f)
        evs = []
        for r in raw:
            ts = pd.Timestamp(r["ts"])
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            evs.append(EconomicEvent(ts.to_pydatetime(), r.get("currency", "USD"), r.get("impact", "MEDIUM"),
                                     r.get("title", ""), r.get("actual"), r.get("forecast"), r.get("previous")))
        return cls(sorted(evs, key=lambda e: e.ts), source=path)

    @classmethod
    def from_events(cls, events: Iterable[EconomicEvent]) -> "EconomicCalendar":
        return cls(sorted(events, key=lambda e: e.ts), source="explicit")

    @property
    def known(self) -> bool:
        return self.source not in ("empty", "missing")

    def relevant(self, symbol: str, start: datetime, end: datetime, min_impact: str = "MEDIUM") -> List[EconomicEvent]:
        ccys = set(currencies_for(symbol))
        rank = IMPACT_RANK[min_impact.upper()]
        return [e for e in self.events if start <= e.ts <= end and e.currency in ccys and e.impact_rank >= rank]

    def next_high_impact(self, symbol: str, now: datetime, horizon_hours: int = 48) -> Optional[EconomicEvent]:
        evs = self.relevant(symbol, now, now + timedelta(hours=horizon_hours), "HIGH")
        return evs[0] if evs else None

    def in_blackout(self, symbol: str, now: datetime, before_min: int, after_min: int) -> Optional[EconomicEvent]:
        for e in self.relevant(symbol, now - timedelta(minutes=after_min), now + timedelta(minutes=before_min), "HIGH"):
            return e
        return None

    def recent_surprises(self, symbol: str, now: datetime, lookback_hours: int = 6) -> List[EconomicEvent]:
        return [e for e in self.relevant(symbol, now - timedelta(hours=lookback_hours), now, "HIGH") if e.surprise() is not None]

"""
ICT-style time & price concepts: kill zones, Judas swing, session ranges,
daily bias from PD arrays. All of it is treated as *context*, never as a
stand-alone trigger, and is validated through the backtester like any other
feature.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ai_trader.core.enums import Direction, Session
from ai_trader.core.timeutils import KILL_ZONES, SESSION_WINDOWS, kill_zone_of, session_of


@dataclass
class SessionRange:
    session: str
    high: float
    low: float
    start: datetime
    end: datetime

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2


@dataclass
class ICTContext:
    session: Session
    kill_zone: Optional[str]
    in_kill_zone: bool
    asia_range: Optional[SessionRange]
    london_range: Optional[SessionRange]
    judas_swing: Optional[Direction]      # direction of the *fake* move (opposite = expected true move)
    judas_note: str
    daily_open: Optional[float]
    price_vs_daily_open: Optional[str]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "session": self.session.value,
            "kill_zone": self.kill_zone,
            "in_kill_zone": self.in_kill_zone,
            "asia_range": None if not self.asia_range else {"high": self.asia_range.high, "low": self.asia_range.low},
            "london_range": None if not self.london_range else {"high": self.london_range.high, "low": self.london_range.low},
            "judas_swing": self.judas_swing.value if self.judas_swing else None,
            "judas_note": self.judas_note,
            "daily_open": self.daily_open,
            "price_vs_daily_open": self.price_vs_daily_open,
            "notes": self.notes,
        }


def _session_range(intraday: pd.DataFrame, day: pd.Timestamp, start: time, end: time, name: str) -> Optional[SessionRange]:
    s = pd.Timestamp.combine(day.date(), start).tz_localize("UTC")
    e = pd.Timestamp.combine(day.date(), end).tz_localize("UTC")
    seg = intraday[(intraday.index >= s) & (intraday.index < e)]
    if len(seg) < 2:
        return None
    return SessionRange(name, float(seg["high"].max()), float(seg["low"].min()), s.to_pydatetime(), e.to_pydatetime())


def ict_context(intraday: pd.DataFrame, daily: Optional[pd.DataFrame] = None, now: Optional[datetime] = None,
                always_open: bool = False) -> ICTContext:
    """Compute session/kill-zone context from intraday (≤1h) bars."""
    now = now or intraday.index[-1].to_pydatetime()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    day = pd.Timestamp(now).tz_convert("UTC").normalize()
    price = float(intraday["close"].iloc[-1])
    sess = Session.ALWAYS if always_open else session_of(now)
    kz = kill_zone_of(now)
    asia = _session_range(intraday, day, *SESSION_WINDOWS[Session.ASIA], "ASIA")
    london = _session_range(intraday, day, time(7, 0), time(12, 0), "LONDON")

    # Daily open (00:00 UTC) — ICT uses midnight NY; we use the UTC day open as a
    # deterministic proxy and expose it so it can be replaced per instrument.
    today = intraday[intraday.index >= day]
    daily_open = float(today["open"].iloc[0]) if len(today) else None
    pvdo = None
    if daily_open is not None:
        pvdo = "above" if price > daily_open else "below"

    # Judas swing: during London KZ, price first runs one side of the Asia range
    # then reverses back through the daily open. We flag the *fake* direction.
    judas, note = None, ""
    if asia and daily_open is not None and len(today) >= 3:
        lon = today[(today.index >= day + pd.Timedelta(hours=7))]
        if len(lon) >= 3:
            ran_high = lon["high"].max() > asia.high
            ran_low = lon["low"].min() < asia.low
            if ran_high and not ran_low and price < daily_open:
                judas, note = Direction.LONG, "London ran Asia high then reversed below daily open → bearish Judas swing"
            elif ran_low and not ran_high and price > daily_open:
                judas, note = Direction.SHORT, "London ran Asia low then reversed above daily open → bullish Judas swing"
    notes = []
    if kz:
        notes.append(f"inside {kz.replace('_', ' ').lower()} (higher-quality time window)")
    if asia:
        notes.append(f"Asia range {asia.low:.5g}–{asia.high:.5g}")
    if judas:
        notes.append(note)
    return ICTContext(sess, kz, kz is not None, asia, london, judas, note, daily_open, pvdo, notes)


def in_session(now: datetime, sessions) -> bool:
    if Session.ALWAYS in sessions:
        return True
    return session_of(now) in sessions

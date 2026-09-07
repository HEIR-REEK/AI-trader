"""Time helpers: UTC normalisation and trading-session classification."""
from __future__ import annotations

from datetime import datetime, time, timezone
from typing import Optional

import pandas as pd

from .enums import Session


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy whose DatetimeIndex is tz-aware UTC and sorted."""
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True)
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


# Session windows (UTC, approximate, ignoring DST shifts by design — the
# calendar layer can override these per instrument if needed).
SESSION_WINDOWS = {
    Session.ASIA: (time(0, 0), time(7, 0)),
    Session.LONDON: (time(7, 0), time(16, 0)),
    Session.NEW_YORK: (time(12, 0), time(21, 0)),
}

# ICT-style kill zones (UTC approximations).
KILL_ZONES = {
    "ASIA_KZ": (time(0, 0), time(3, 0)),
    "LONDON_KZ": (time(7, 0), time(10, 0)),
    "NY_AM_KZ": (time(12, 0), time(15, 0)),
    "NY_PM_KZ": (time(17, 30), time(20, 0)),
}


def session_of(ts: datetime) -> Session:
    t = ts.astimezone(timezone.utc).time() if ts.tzinfo else ts.time()
    if _in(t, *SESSION_WINDOWS[Session.LONDON]) and _in(t, *SESSION_WINDOWS[Session.NEW_YORK]):
        return Session.NEW_YORK  # overlap counted as NY (highest liquidity)
    for s in (Session.LONDON, Session.NEW_YORK, Session.ASIA):
        if _in(t, *SESSION_WINDOWS[s]):
            return s
    return Session.OFF_HOURS


def kill_zone_of(ts: datetime) -> Optional[str]:
    t = ts.astimezone(timezone.utc).time() if ts.tzinfo else ts.time()
    for name, (a, b) in KILL_ZONES.items():
        if _in(t, a, b):
            return name
    return None


def _in(t: time, a: time, b: time) -> bool:
    return a <= t < b

"""
Data-integrity checks. Bad data is the cheapest way to produce a confident,
wrong signal, so the decision engine refuses to trade on frames that fail.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd

from ai_trader.core.enums import Timeframe


@dataclass
class ValidationReport:
    ok: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    bars: int = 0
    gaps: int = 0
    stale_bars: float = 0.0

    def summary(self) -> str:
        status = "OK" if self.ok else "FAIL"
        return f"[{status}] bars={self.bars} gaps={self.gaps} issues={len(self.issues)} warnings={len(self.warnings)}"


def validate_ohlcv(df: pd.DataFrame, timeframe: Timeframe, min_bars: int = 50,
                   now: Optional[datetime] = None, max_stale_bars: Optional[int] = None,
                   max_gap_ratio: float = 0.05, sessions_24_7: bool = True) -> ValidationReport:
    rep = ValidationReport(ok=True, bars=len(df))
    if df is None or len(df) == 0:
        rep.ok = False
        rep.issues.append("empty frame")
        return rep
    if len(df) < min_bars:
        rep.ok = False
        rep.issues.append(f"insufficient bars: {len(df)} < {min_bars}")
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        rep.ok = False
        rep.issues.append("index must be tz-aware DatetimeIndex (UTC)")
        return rep
    if not df.index.is_monotonic_increasing:
        rep.ok = False
        rep.issues.append("index not sorted")
    if df.index.has_duplicates:
        rep.ok = False
        rep.issues.append("duplicate timestamps")

    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    bad_hl = np.sum((h < l) | (h < np.maximum(o, c)) | (l > np.minimum(o, c)))
    if bad_hl:
        rep.ok = False
        rep.issues.append(f"{int(bad_hl)} bars violate high>=max(o,c)>=min(o,c)>=low")
    if np.any(c <= 0) or np.any(~np.isfinite(c)):
        rep.ok = False
        rep.issues.append("non-positive or non-finite prices")

    # Gaps (only meaningful for continuous markets; weekends excluded for FX-style)
    if len(df) > 2:
        deltas = pd.Series(df.index[1:] - df.index[:-1])
        expected = pd.Timedelta(minutes=timeframe.minutes)
        if timeframe in (Timeframe.W1, Timeframe.MN1):
            gaps = 0
        else:
            big = deltas > expected * 1.5
            if not sessions_24_7:
                # allow weekend gaps (Fri close → Sun/Mon open) up to 3 days
                big &= deltas > pd.Timedelta(days=3)
            gaps = int(big.sum())
        rep.gaps = gaps
        if gaps / max(1, len(df)) > max_gap_ratio:
            rep.warnings.append(f"{gaps} gaps ({gaps / len(df):.1%} of bars)")

    # Outlier bars (single-bar returns > 15 sigma) → likely bad ticks
    rets = np.diff(np.log(c))
    if len(rets) > 30:
        sigma = np.std(rets[-500:]) or 1e-9
        spikes = int(np.sum(np.abs(rets) > 15 * sigma))
        if spikes:
            rep.warnings.append(f"{spikes} suspicious spike bars (>15σ)")

    # Zero-range bars
    zero_range = int(np.sum(h == l))
    if zero_range / len(df) > 0.2:
        rep.warnings.append(f"{zero_range / len(df):.0%} zero-range bars (illiquid / bad feed?)")

    # Staleness
    if now is not None and max_stale_bars is not None and timeframe not in (Timeframe.W1, Timeframe.MN1):
        now_ts = pd.Timestamp(now)
        if now_ts.tzinfo is None:
            now_ts = now_ts.tz_localize("UTC")
        last_close = df.index[-1] + pd.Timedelta(minutes=timeframe.minutes)
        stale = (now_ts - last_close) / pd.Timedelta(minutes=timeframe.minutes)
        rep.stale_bars = float(max(0.0, stale))
        if stale > max_stale_bars:
            rep.ok = False
            rep.issues.append(f"stale feed: last bar closed {stale:.1f} bars ago")
    return rep

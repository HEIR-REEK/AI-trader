"""
MarketDataLoader — assembles a ``MarketData`` object (all required timeframes)
for one instrument from a provider, resampling from the lowest available
timeframe where necessary and validating every frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from ai_trader.config.settings import AnalysisSettings, get_settings
from ai_trader.core.enums import Timeframe
from ai_trader.core.exceptions import DataError, ProviderError
from ai_trader.core.models import MarketData

from .providers import DataProvider
from .resampler import closed_bars_as_of, resample_ohlcv
from .validation import ValidationReport, validate_ohlcv

DEFAULT_LIMITS = {
    Timeframe.M1: 2000, Timeframe.M5: 1500, Timeframe.M15: 1200, Timeframe.M30: 1000,
    Timeframe.H1: 1000, Timeframe.H4: 800, Timeframe.D1: 500, Timeframe.W1: 260, Timeframe.MN1: 120,
}


@dataclass
class LoadResult:
    data: MarketData
    reports: Dict[Timeframe, ValidationReport] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.reports.values())

    @property
    def issues(self) -> List[str]:
        out = []
        for tf, r in self.reports.items():
            out += [f"{tf.value}: {i}" for i in r.issues]
        return out


class MarketDataLoader:
    def __init__(self, provider: DataProvider, settings: Optional[AnalysisSettings] = None):
        self.provider = provider
        self.settings = settings or get_settings().analysis

    def required_timeframes(self) -> List[Timeframe]:
        s = self.settings
        tfs = set(s.htf_timeframes) | set(s.structural_timeframes) | set(s.entry_timeframes)
        return sorted(tfs, key=lambda t: t.minutes)

    def load(self, symbol: str, timeframes: Optional[List[Timeframe]] = None, as_of: Optional[datetime] = None,
             limits: Optional[Dict[Timeframe, int]] = None, sessions_24_7: bool = True,
             resample_from_lowest: bool = True) -> LoadResult:
        tfs = timeframes or self.required_timeframes()
        limits = {**DEFAULT_LIMITS, **(limits or {})}
        frames: Dict[Timeframe, pd.DataFrame] = {}
        reports: Dict[Timeframe, ValidationReport] = {}

        lowest = min(tfs, key=lambda t: t.minutes)
        base = None
        for tf in tfs:
            df = None
            try:
                df = self.provider.get_ohlcv(symbol, tf, limit=limits[tf], end=as_of)
            except ProviderError:
                df = None
            if (df is None or len(df) == 0) and resample_from_lowest and tf != lowest:
                if base is None:
                    base = self.provider.get_ohlcv(symbol, lowest, limit=limits[lowest] * max(1, tf.minutes // lowest.minutes), end=as_of)
                df = resample_ohlcv(base.drop(columns=[c for c in base.columns if c == "label"]), lowest, tf, drop_incomplete=True)
            if df is None or len(df) == 0:
                raise DataError(f"{symbol}: no data for {tf.value}")
            df = df.drop(columns=[c for c in df.columns if c == "label"])
            if df.attrs:
                df = df.copy(deep=False)
                df.attrs = {}          # pandas deep-copies attrs on every op → measurable slowdown in analysis
            if as_of is not None:
                df = closed_bars_as_of(df, tf, as_of)
            min_bars = self.settings.min_bars_override.get(tf, self.settings.min_bars.get(tf.group, 50))
            reports[tf] = validate_ohlcv(df, tf, min_bars=min_bars, now=as_of,
                                         max_stale_bars=self.settings.max_data_staleness_bars if as_of else None,
                                         sessions_24_7=sessions_24_7)
            frames[tf] = df
        md = MarketData(symbol=symbol, frames=frames, as_of=as_of or frames[lowest].index[-1].to_pydatetime())
        return LoadResult(data=md, reports=reports)

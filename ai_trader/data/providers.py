"""
Market data providers.

All providers implement the same protocol and return **UTC-indexed OHLCV
DataFrames** with columns ``open, high, low, close, volume``. Providers never
interpret price; they only fetch/generate and normalise.

Providers:
  * CSVProvider        – local files ``{data_dir}/{SYMBOL}_{tf}.csv``
  * SyntheticProvider  – regime-labelled synthetic paths (tests, demos, stress)
  * TwelveDataProvider – REST provider (needs AITRADER_TWELVEDATA_API_KEY)
  * CompositeProvider  – tries providers in order (e.g. csv → twelvedata)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Protocol, Sequence

import numpy as np
import pandas as pd

from ai_trader.core.enums import Timeframe
from ai_trader.core.exceptions import DataError, ProviderError
from ai_trader.core.timeutils import ensure_utc_index

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class DataProvider(Protocol):
    name: str

    def get_ohlcv(self, symbol: str, timeframe: Timeframe, limit: int = 1000,
                  end: Optional[datetime] = None) -> pd.DataFrame: ...

    def available(self) -> bool: ...


def normalise_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case columns, enforce dtypes, UTC index, drop NaN rows."""
    out = df.copy()
    out.columns = [str(c).lower().strip() for c in out.columns]
    rename = {"time": "ts", "timestamp": "ts", "date": "ts", "datetime": "ts", "vol": "volume"}
    out = out.rename(columns=rename)
    if "ts" in out.columns:
        out["ts"] = pd.to_datetime(out["ts"], utc=True)
        out = out.set_index("ts")
    if "volume" not in out.columns:
        out["volume"] = 0.0
    missing = [c for c in OHLCV_COLUMNS if c not in out.columns]
    if missing:
        raise DataError(f"OHLCV frame missing columns: {missing}")
    out = out[OHLCV_COLUMNS].astype(float)
    out = ensure_utc_index(out)
    out = out.dropna(subset=["open", "high", "low", "close"])
    out.index.name = "ts"
    return out


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
@dataclass
class CSVProvider:
    data_dir: str = "data"
    name: str = "csv"

    def path_for(self, symbol: str, timeframe: Timeframe) -> str:
        return os.path.join(self.data_dir, f"{symbol.upper()}_{timeframe.value}.csv")

    def available(self) -> bool:
        return os.path.isdir(self.data_dir)

    def get_ohlcv(self, symbol: str, timeframe: Timeframe, limit: int = 1000,
                  end: Optional[datetime] = None) -> pd.DataFrame:
        path = self.path_for(symbol, timeframe)
        if not os.path.exists(path):
            raise ProviderError(f"CSV not found: {path}")
        df = normalise_ohlcv(pd.read_csv(path))
        if end is not None:
            df = df[df.index <= pd.Timestamp(end).tz_convert("UTC") if pd.Timestamp(end).tzinfo else pd.Timestamp(end, tz="UTC")]
        return df.tail(limit)

    def save(self, symbol: str, timeframe: Timeframe, df: pd.DataFrame) -> str:
        os.makedirs(self.data_dir, exist_ok=True)
        path = self.path_for(symbol, timeframe)
        normalise_ohlcv(df).to_csv(path, index_label="ts")
        return path


# ---------------------------------------------------------------------------
# Synthetic (regime-labelled) — used for unit tests, demos and stress tests.
# IMPORTANT: synthetic data validates *mechanics*, never edge. Strategies must
# still be validated on real historical data before any live use.
# ---------------------------------------------------------------------------
@dataclass
class RegimeSegment:
    bars: int
    kind: str            # "trend_up" | "trend_down" | "range" | "expansion" | "contraction" | "ou"
    vol: float = 0.001   # per-bar return std
    drift: float = 0.0   # per-bar drift (trend regimes)
    mean_revert: float = 0.05  # OU speed (range regimes)


@dataclass
class SyntheticProvider:
    """Generates ONE canonical base-timeframe path per symbol and resamples every
    requested timeframe from it, so all timeframes are mutually consistent.

    The generator stitches regime segments so tests can assert that the regime
    detector recovers the labelled regime; ``label`` holds the per-bar truth.
    Scripted ``segments`` are placed at the END of the path (front-padded with a
    quiet range) so the most recent bars are the interesting ones.
    """
    seed: int = 42
    start_price: float = 2000.0
    base_tf: Timeframe = Timeframe.M5
    segments: Optional[Sequence[RegimeSegment]] = None
    base_bars: int = 130_000            # ≈ 450 days of 5-minute bars → ~64 weekly bars
    name: str = "synthetic"
    _cache: Dict[str, pd.DataFrame] = None  # type: ignore[assignment]

    def __post_init__(self):
        self._cache = {}

    def available(self) -> bool:
        return True

    def default_segments(self, total_bars: int) -> List[RegimeSegment]:
        block = max(200, total_bars // 6)
        return [
            RegimeSegment(block, "range", vol=0.0006),
            RegimeSegment(block, "trend_up", vol=0.0009, drift=0.00025),
            RegimeSegment(block, "contraction", vol=0.0003),
            RegimeSegment(block, "expansion", vol=0.0020),
            RegimeSegment(block, "trend_down", vol=0.0009, drift=-0.00025),
            RegimeSegment(total_bars - 5 * block, "range", vol=0.0006),
        ]

    def generate(self, symbol: str, end: Optional[datetime] = None) -> pd.DataFrame:
        end_ts = pd.Timestamp(end or datetime.now(timezone.utc)).floor(self.base_tf.pandas_freq)
        key = f"{symbol}:{end_ts.isoformat()}"
        if key in self._cache:
            return self._cache[key]
        rng = np.random.default_rng(self.seed + (sum(ord(ch) for ch in symbol) % 10_000))
        segs = list(self.segments) if self.segments else self.default_segments(self.base_bars)
        total = sum(sg.bars for sg in segs)
        if total < self.base_bars:
            segs = [RegimeSegment(self.base_bars - total, "range", vol=segs[0].vol * 0.8, mean_revert=0.03)] + segs
        closes: List[float] = []
        labels: List[str] = []
        price = self.start_price
        anchor = price
        for seg in segs:
            if seg.kind in ("trend_up", "trend_down"):
                r = seg.drift + rng.normal(0, seg.vol, seg.bars)
                path = price * np.exp(np.cumsum(r))
            elif seg.kind == "expansion":
                r = rng.normal(0, seg.vol, seg.bars) * (1.0 + rng.random(seg.bars))
                path = price * np.exp(np.cumsum(r))
            elif seg.kind in ("range", "ou", "contraction"):
                path = np.empty(seg.bars)
                p = price
                noise = rng.normal(0, seg.vol, seg.bars)
                for i in range(seg.bars):
                    p = p * float(np.exp(-seg.mean_revert * np.log(p / anchor) + noise[i]))
                    path[i] = p
            else:
                r = rng.normal(0, seg.vol, seg.bars)
                path = price * np.exp(np.cumsum(r))
            closes.extend(path.tolist())
            labels.extend([seg.kind] * seg.bars)
            price = float(path[-1])
            anchor = price
        n = len(closes)
        closes_arr = np.array(closes)
        opens = np.concatenate([[self.start_price], closes_arr[:-1]])
        body = np.abs(closes_arr - opens)
        highs = np.maximum(opens, closes_arr) + np.abs(rng.normal(0, 1, n)) * body * 0.6 + closes_arr * 0.00005
        lows = np.minimum(opens, closes_arr) - np.abs(rng.normal(0, 1, n)) * body * 0.6 - closes_arr * 0.00005
        vol = (body / closes_arr * 1e6 + rng.gamma(2.0, 50, n)).round()
        idx = pd.date_range(end=end_ts, periods=n, freq=self.base_tf.pandas_freq, tz="UTC")
        df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes_arr, "volume": vol}, index=idx)
        df.index.name = "ts"
        df["label"] = labels
        self._cache[key] = df
        return df

    def get_ohlcv(self, symbol: str, timeframe: Timeframe, limit: int = 1000,
                  end: Optional[datetime] = None) -> pd.DataFrame:
        from .resampler import resample_ohlcv  # local import to avoid cycle
        base = self.generate(symbol, end)
        labels = base["label"]
        ohlc = base.drop(columns=["label"])
        if timeframe == self.base_tf:
            out = ohlc.tail(limit).copy()
            out["label"] = labels.reindex(out.index)
            return out
        rkey = f"{symbol}:{timeframe.value}:{base.index[-1].isoformat()}"
        if rkey not in self._cache:
            out = resample_ohlcv(ohlc, self.base_tf, timeframe, drop_incomplete=True)
            out["label"] = labels.resample(timeframe.pandas_freq if timeframe != Timeframe.W1 else "W-SUN",
                                           label="left", closed="left").last().reindex(out.index)
            self._cache[rkey] = out
        return self._cache[rkey].tail(limit)


# ---------------------------------------------------------------------------
# TwelveData REST (optional; only used when API key present)
# ---------------------------------------------------------------------------
_TD_INTERVAL = {
    Timeframe.M1: "1min", Timeframe.M5: "5min", Timeframe.M15: "15min", Timeframe.M30: "30min",
    Timeframe.H1: "1h", Timeframe.H4: "4h", Timeframe.D1: "1day", Timeframe.W1: "1week", Timeframe.MN1: "1month",
}
_TD_SYMBOL = {
    "XAUUSD": "XAU/USD", "XAGUSD": "XAG/USD", "EURUSD": "EUR/USD", "GBPUSD": "GBP/USD", "USDJPY": "USD/JPY",
    "AUDUSD": "AUD/USD", "USDCAD": "USD/CAD", "USDCHF": "USD/CHF", "NZDUSD": "NZD/USD", "EURGBP": "EUR/GBP",
    "EURJPY": "EUR/JPY", "GBPJPY": "GBP/JPY", "AUDJPY": "AUD/JPY", "BTCUSD": "BTC/USD", "ETHUSD": "ETH/USD",
    "SOLUSD": "SOL/USD", "XRPUSD": "XRP/USD", "US500": "SPX", "US100": "NDX", "US30": "DJI",
    "DE40": "DAX", "UK100": "FTSE", "JP225": "N225",
}


@dataclass
class TwelveDataProvider:
    api_key: Optional[str] = None
    base_url: str = "https://api.twelvedata.com"
    timeout: int = 15
    name: str = "twelvedata"

    def __post_init__(self):
        self.api_key = self.api_key or os.environ.get("AITRADER_TWELVEDATA_API_KEY") or os.environ.get("TWELVEDATA_API_KEY")

    def available(self) -> bool:
        return bool(self.api_key)

    def get_ohlcv(self, symbol: str, timeframe: Timeframe, limit: int = 1000,
                  end: Optional[datetime] = None) -> pd.DataFrame:
        if not self.available():
            raise ProviderError("TwelveData API key missing (AITRADER_TWELVEDATA_API_KEY)")
        try:
            import requests  # lazy
        except ImportError as e:  # pragma: no cover
            raise ProviderError("requests not installed") from e
        params = {
            "symbol": _TD_SYMBOL.get(symbol.upper(), symbol),
            "interval": _TD_INTERVAL[timeframe],
            "outputsize": min(limit, 5000),
            "apikey": self.api_key,
            "timezone": "UTC",
            "order": "ASC",
        }
        if end:
            params["end_date"] = pd.Timestamp(end).strftime("%Y-%m-%d %H:%M:%S")
        try:
            r = requests.get(f"{self.base_url}/time_series", params=params, timeout=self.timeout)
            r.raise_for_status()
            payload = r.json()
        except Exception as e:
            raise ProviderError(f"TwelveData request failed: {e}") from e
        if payload.get("status") == "error" or "values" not in payload:
            raise ProviderError(f"TwelveData error: {payload.get('message', payload)}")
        df = pd.DataFrame(payload["values"]).rename(columns={"datetime": "ts"})
        return normalise_ohlcv(df).tail(limit)


# ---------------------------------------------------------------------------
# In-memory frame provider (tests / scenarios / API uploads)
# ---------------------------------------------------------------------------
@dataclass
class FrameProvider:
    """Serves pre-built frames; missing timeframes are causally resampled from the
    lowest available one."""
    frames: Dict[str, Dict[Timeframe, pd.DataFrame]]
    name: str = "frames"

    def available(self) -> bool:
        return bool(self.frames)

    def get_ohlcv(self, symbol: str, timeframe: Timeframe, limit: int = 1000,
                  end: Optional[datetime] = None) -> pd.DataFrame:
        from .resampler import resample_ohlcv
        sym = symbol.upper()
        if sym not in self.frames:
            raise ProviderError(f"no frames for {sym}")
        tfs = self.frames[sym]
        if timeframe in tfs:
            df = tfs[timeframe]
        else:
            lowest = min(tfs, key=lambda t: t.minutes)
            if lowest.minutes > timeframe.minutes:
                raise ProviderError(f"cannot build {timeframe.value} from {lowest.value}")
            df = resample_ohlcv(tfs[lowest], lowest, timeframe, drop_incomplete=True)
        if end is not None:
            e = pd.Timestamp(end)
            e = e.tz_localize("UTC") if e.tzinfo is None else e.tz_convert("UTC")
            df = df[df.index <= e]
        return df.tail(limit)


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------
@dataclass
class CompositeProvider:
    providers: Sequence[DataProvider]
    name: str = "composite"

    def available(self) -> bool:
        return any(p.available() for p in self.providers)

    def get_ohlcv(self, symbol: str, timeframe: Timeframe, limit: int = 1000,
                  end: Optional[datetime] = None) -> pd.DataFrame:
        errors = []
        for p in self.providers:
            if not p.available():
                continue
            try:
                return p.get_ohlcv(symbol, timeframe, limit, end)
            except DataError as e:
                errors.append(f"{p.name}: {e}")
        raise ProviderError("All providers failed: " + "; ".join(errors))


def build_default_provider(data_dir: str = "data", prefer: str = "csv") -> DataProvider:
    csv = CSVProvider(data_dir=data_dir)
    td = TwelveDataProvider()
    syn = SyntheticProvider()
    if prefer == "synthetic":
        return syn
    if prefer == "twelvedata":
        return CompositeProvider([td, csv])
    return CompositeProvider([csv, td])

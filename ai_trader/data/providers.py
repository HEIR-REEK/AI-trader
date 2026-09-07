"""
Market data providers.

All providers implement the same protocol and return **UTC-indexed OHLCV
DataFrames** with columns ``open, high, low, close, volume``. Providers never
interpret price; they only fetch/generate and normalise.

Providers:
  * CSVProvider        – local files ``{data_dir}/{SYMBOL}_{tf}.csv``; accepts
                         comma/tab/semicolon/pipe delimiters, MT4/MT5-style
                         ``<OPEN>`` headers, split DATE+TIME columns, epoch
                         timestamps and common OHLCV aliases
  * SyntheticProvider  – regime-labelled synthetic paths (tests, demos, stress)
  * TwelveDataProvider – REST provider (needs AITRADER_TWELVEDATA_API_KEY)
  * CompositeProvider  – tries providers in order (e.g. csv → twelvedata)
"""
from __future__ import annotations

import os
import re
import warnings
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


# ---------------------------------------------------------------------------
# CSV parsing helpers — broker exports come in many dialects (MT4/MT5 angle
# brackets + tab/semicolon separators, OANDA single-letter OHLC, epoch
# timestamps, split date/time columns, ...). All of that is normalised here so
# the rest of the engine only ever sees clean UTC OHLCV frames.
# ---------------------------------------------------------------------------

_DELIMITERS = (",", "\t", ";", "|")  # detection priority order

# Timestamp candidates in preference order. Bar-OPEN flavours come first (the
# engine labels bars by open time); CLOSE flavours are a last resort.
_TS_PRIORITY = [
    "open_time", "opentime", "time_open", "start_time", "starttime",
    "ts", "timestamp", "datetime", "date_time", "bar_time", "bartime",
    "date", "time", "epoch",
    "close_time", "closetime", "time_close", "end_time", "endtime",
]
_TS_ALIASES = set(_TS_PRIORITY) | {
    "datetime_utc", "date_utc", "time_utc", "timestamp_utc", "open_time_utc",
}

_OHLCV_ALIASES = {
    "open": {"open", "o", "open_price", "opening"},
    "high": {"high", "h", "high_price", "highest"},
    "low": {"low", "l", "low_price", "lowest"},
    "close": {"close", "c", "close_price", "closing", "settle", "settlement",
              "adj_close", "adjclose", "adjusted_close"},
    "volume": {"volume", "vol", "tickvol", "tick_vol", "tickvolume", "tick_volume",
               "real_volume", "realvolume", "base_volume", "basevolume",
               "contracts", "qty", "quantity", "ticks"},
}


def _clean_column(name: object) -> str:
    s = str(name).strip().lower().replace("<", "").replace(">", "")  # MT4/MT5 <OPEN>
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9a-z_]", "", s)
    return re.sub(r"_+", "_", s).strip("_")


def _detect_delimiter(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            sample = "".join(fh.readline() for _ in range(5))
    except OSError as e:
        raise DataError(f"cannot read {path}: {e}") from e
    if not sample.strip():
        raise DataError(f"{path} is empty")
    counts = {d: sample.count(d) for d in _DELIMITERS}
    best = max(_DELIMITERS, key=lambda d: counts[d])  # ties → priority order
    if counts[best] == 0:
        raise DataError(f"{path}: no delimiter found (expected one of comma/tab/semicolon/pipe)")
    return best


def read_csv_smart(path: str) -> pd.DataFrame:
    """Read a broker CSV regardless of delimiter/encoding quirks.

    Handles comma/tab/semicolon/pipe separators and a UTF-8 BOM. Raises
    ``DataError`` (never a raw pandas error) on empty/unreadable files.
    """
    sep = _detect_delimiter(path)
    try:
        return pd.read_csv(path, sep=sep, encoding="utf-8-sig", skipinitialspace=True)
    except pd.errors.EmptyDataError as e:
        raise DataError(f"{path} is empty") from e
    except Exception as e:  # ParserError, UnicodeDecodeError, ...
        raise DataError(f"{path}: cannot parse CSV ({e})") from e


def _coerce_numeric_ts(vals: pd.Series) -> pd.Series:
    """Epoch numbers (s/ms/us/ns by magnitude) or YYYYMMDD integers → UTC."""
    ref = vals.dropna()
    if len(ref) == 0:
        raise DataError("timestamp column has no valid values")
    if bool(((ref % 1 == 0) & ref.between(19000101, 21001231)).all()):
        s = vals.astype("Int64").astype(str).where(vals.notna())
        return pd.to_datetime(s, format="%Y%m%d", utc=True, errors="coerce")
    mag = float(ref.abs().max())
    unit = "s" if mag < 1e11 else "ms" if mag < 1e14 else "us" if mag < 1e17 else "ns"
    return pd.to_datetime(vals, unit=unit, utc=True)


def _coerce_ts(series: pd.Series) -> pd.Series:
    """Parse a timestamp column: ISO strings, epoch numbers (s/ms/us/ns) or
    YYYYMMDD integers. Naive values are assumed UTC; aware ones converted."""
    s = series
    if pd.api.types.is_numeric_dtype(s.dtype) and not pd.api.types.is_datetime64_any_dtype(s.dtype):
        return _coerce_numeric_ts(pd.to_numeric(s, errors="coerce"))
    if pd.api.types.is_string_dtype(s.dtype) or s.dtype == object:
        non_null = s[s.notna()]
        if len(non_null) and non_null.astype(str).str.strip().str.fullmatch(r"[+-]?\d+(?:\.\d+)?").all():
            return _coerce_numeric_ts(pd.to_numeric(s, errors="coerce"))
    try:
        return pd.to_datetime(s, utc=True)
    except Exception:
        pass
    parsed = pd.to_datetime(s, utc=True, format="mixed", errors="coerce")
    if parsed.isna().all():
        raise DataError("cannot parse timestamps (no values understood; expected ISO datetimes or epoch numbers)")
    n_bad = int((parsed.isna() & s.notna()).sum())
    if n_bad:
        warnings.warn(f"dropping {n_bad} row(s) with unparseable timestamps")
    return parsed


def _combine_date_time(date_col: pd.Series, time_col: pd.Series) -> pd.Series:
    """Merge split DATE + TIME columns (MT4/MT5 style) into UTC timestamps."""
    d = date_col.astype(str).str.strip()
    t = time_col.astype(str).str.strip()
    if bool(d.str.fullmatch(r"\d{8}").all()) and bool(t.str.fullmatch(r"\d{6}").all()):
        return pd.to_datetime(d + " " + t, format="%Y%m%d %H%M%S", utc=True, errors="coerce")
    if bool(d.str.fullmatch(r"\d{8}").all()):
        base = pd.to_datetime(d, format="%Y%m%d", utc=True, errors="coerce")
        try:
            delta = pd.to_timedelta(t, errors="coerce")
        except Exception:
            delta = pd.Series(pd.NaT, index=t.index)
        return base + delta
    mask = date_col.notna() & time_col.notna()
    combined = pd.Series(pd.NA, index=date_col.index, dtype="string")
    combined[mask] = d[mask] + " " + t[mask]
    return _coerce_ts(combined)


def normalise_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Map broker aliases, enforce dtypes, UTC index, drop NaN rows.

    Accepts MT4/MT5-style headers (``<OPEN>`` ...), split ``DATE``+``TIME``
    columns, epoch timestamps and common OHLCV aliases. The result is sorted
    ascending with duplicate timestamps removed (keep last).
    """
    if df is None or len(df) == 0:
        raise DataError("OHLCV frame is empty")
    out = df.copy()
    out.columns = [_clean_column(c) for c in out.columns]
    out = out.loc[:, ~out.columns.duplicated(keep="first")]
    cols = set(out.columns)

    if "date" in cols and "time" in cols and not (cols & (_TS_ALIASES - {"date", "time"})):
        # split DATE + TIME with no combined alternative → merge them
        ts = _combine_date_time(out["date"], out["time"])
        out = out.drop(columns=["date", "time"])
    else:
        ts_col = next((c for c in _TS_PRIORITY if c in cols), None)
        if ts_col is None:
            if isinstance(out.index, pd.DatetimeIndex):
                ts = pd.Series(out.index, index=out.index)
            else:
                raise DataError(
                    "no timestamp column found (looked for "
                    + ", ".join(_TS_PRIORITY[:8]) + f", ...); have: {sorted(cols)}"
                )
        else:
            ts = out[ts_col]
            out = out.drop(columns=[ts_col])
        ts = _coerce_ts(ts)

    out["_ts"] = ts.to_numpy()
    out = out.dropna(subset=["_ts"]).set_index("_ts")
    out.index.name = "ts"
    if len(out) == 0:
        raise DataError("no rows with valid timestamps")

    rename = {}
    for canon, aliases in _OHLCV_ALIASES.items():
        hit = next((c for c in out.columns if c in aliases), None)
        if hit is not None:
            rename[hit] = canon
    out = out.rename(columns=rename)
    if "volume" not in out.columns:
        out["volume"] = 0.0
    missing = [c for c in OHLCV_COLUMNS if c not in out.columns]
    if missing:
        raise DataError(f"OHLCV frame missing columns: {missing} (have: {sorted(set(out.columns))})")
    try:
        out = out[OHLCV_COLUMNS].astype(float)
    except (ValueError, TypeError) as e:
        raise DataError(f"non-numeric OHLCV values ({e})") from e
    out = ensure_utc_index(out)
    out = out.dropna(subset=["open", "high", "low", "close"])
    if len(out) == 0:
        raise DataError("no valid OHLCV rows after cleaning")
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

    def existing(self, symbol: str, timeframes: Sequence[Timeframe]) -> Dict[Timeframe, str]:
        """Subset of ``timeframes`` that have a CSV file, mapped to their path."""
        found: Dict[Timeframe, str] = {}
        for tf in timeframes:
            p = self.path_for(symbol, tf)
            if os.path.exists(p):
                found[tf] = p
        return found

    def get_ohlcv(self, symbol: str, timeframe: Timeframe, limit: int = 1000,
                  end: Optional[datetime] = None) -> pd.DataFrame:
        path = self.path_for(symbol, timeframe)
        if not os.path.exists(path):
            raise ProviderError(
                f"CSV not found: {path} "
                f"(expected file {symbol.upper()}_{timeframe.value}.csv in {self.data_dir!r})"
            )
        try:
            df = normalise_ohlcv(read_csv_smart(path))
        except DataError as e:
            raise DataError(f"{path}: {e}") from e
        if end is not None:
            e = pd.Timestamp(end)
            e = e.tz_localize("UTC") if e.tzinfo is None else e.tz_convert("UTC")
            df = df[df.index <= e]
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
    _cache: Dict[tuple, pd.DataFrame] = None  # type: ignore[assignment]

    def __post_init__(self):
        self._cache = {}

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
            src = tfs[lowest]
            key = (sym, timeframe, id(src), len(src), int(src.index[-1].value))
            df = self._cache.get(key)
            if df is None:
                df = resample_ohlcv(src, lowest, timeframe, drop_incomplete=True)
                if len(self._cache) > 64:
                    self._cache.clear()
                self._cache[key] = df
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

"""
Vectorised technical indicators.

Every function is pure: ``Series/DataFrame in → Series/DataFrame out``, aligned
to the input index, NaN-padded at the start (never forward-looking).
Implemented in-house so that every formula is auditable and there is no
hidden repainting.

Nothing here is a signal. Indicators are *features* that the structure,
regime and confluence layers combine with price action, liquidity, volume,
volatility and risk/reward.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Moving averages
# --------------------------------------------------------------------------
def sma(s: pd.Series, period: int) -> pd.Series:
    return s.rolling(period, min_periods=period).mean()


def ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False, min_periods=period).mean()


def wma(s: pd.Series, period: int) -> pd.Series:
    w = np.arange(1, period + 1, dtype=float)
    return s.rolling(period).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def rma(s: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (used by RSI/ATR/ADX)."""
    return s.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


# --------------------------------------------------------------------------
# Volatility
# --------------------------------------------------------------------------
def true_range(df: pd.DataFrame) -> pd.Series:
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    pc = np.empty_like(c)
    pc[0] = np.nan
    pc[1:] = c[:-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    tr[0] = h[0] - l[0]
    return pd.Series(tr, index=df.index)


# ATR is requested by many analysis modules for the same frame; memoise the last few
# frames (identity + shape + last timestamp/close guard against stale hits).
_ATR_CACHE: Dict[tuple, tuple] = {}


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    n = len(df)
    key = (id(df), n, period, int(df.index[-1].value) if n else 0, float(df["close"].iloc[-1]) if n else 0.0)
    hit = _ATR_CACHE.get(key)
    if hit is not None and hit[0] is df:
        return hit[1]
    s = rma(true_range(df), period)
    if len(_ATR_CACHE) > 48:
        _ATR_CACHE.clear()
    _ATR_CACHE[key] = (df, s)
    return s


def rolling_percentile_rank(s: pd.Series, lookback: int, min_periods: int) -> pd.Series:
    """Fraction of the previous window values strictly below the current value.

    Exact vectorised equivalent of
    ``s.rolling(lookback, min_periods).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)``
    (NaN comparisons count as "not below", a window needs ``min_periods`` non-NaN
    observations, single-element windows are NaN).
    """
    from numpy.lib.stride_tricks import sliding_window_view
    x = s.to_numpy(dtype=float)
    n = len(x)
    out = np.full(n, np.nan)
    if n == 0:
        return pd.Series(out, index=s.index)
    w = min(lookback, n)
    if w >= 2:
        win = sliding_window_view(x, w)                    # windows ending at i = w-1 .. n-1
        cur = win[:, -1:]
        with np.errstate(invalid="ignore"):
            cnt = np.sum(win[:, :-1] < cur, axis=1)
        valid = np.sum(~np.isnan(win), axis=1)
        vals = cnt / (w - 1)
        vals[valid < min_periods] = np.nan
        out[w - 1:] = vals
    # partial windows at the start (fewer than `lookback` elements)
    for i in range(1, min(w - 1, n)):
        window = x[:i + 1]
        if np.sum(~np.isnan(window)) < min_periods:
            continue
        with np.errstate(invalid="ignore"):
            out[i] = float(np.mean(window[:-1] < window[-1]))
    return pd.Series(out, index=s.index)


def atr_percentile(df: pd.DataFrame, period: int = 14, lookback: int = 250) -> pd.Series:
    """Rolling percentile rank of ATR/price in [0,1] — the core volatility-regime feature.

    Normalising by price makes the percentile robust to long trends where raw
    ATR grows with the price level.
    """
    a = atr(df, period) / df["close"]
    return rolling_percentile_rank(a, lookback, max(20, lookback // 5))


def bollinger(s: pd.Series, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    mid = sma(s, period)
    dev = s.rolling(period).std(ddof=0)
    upper, lower = mid + std * dev, mid - std * dev
    width = (upper - lower) / mid
    pct_b = (s - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "bb_width": width, "bb_pct": pct_b})


def bb_squeeze(df: pd.DataFrame, bb_period: int = 20, kc_period: int = 20, kc_mult: float = 1.5) -> pd.Series:
    """TTM-style squeeze: Bollinger inside Keltner → True."""
    bb = bollinger(df["close"], bb_period)
    mid = ema(df["close"], kc_period)
    rng = atr(df, kc_period) * kc_mult
    return (bb["bb_upper"] < mid + rng) & (bb["bb_lower"] > mid - rng)


def keltner(df: pd.DataFrame, period: int = 20, mult: float = 1.5) -> pd.DataFrame:
    mid = ema(df["close"], period)
    rng = atr(df, period) * mult
    return pd.DataFrame({"kc_mid": mid, "kc_upper": mid + rng, "kc_lower": mid - rng})


def historical_volatility(s: pd.Series, period: int = 20, annualise: Optional[float] = None) -> pd.Series:
    r = np.log(s / s.shift(1))
    hv = r.rolling(period).std()
    return hv * np.sqrt(annualise) if annualise else hv


# --------------------------------------------------------------------------
# Momentum
# --------------------------------------------------------------------------
def rsi(s: pd.Series, period: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = rma(gain, period)
    avg_loss = rma(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    out = out.where(avg_loss != 0, 100.0)          # no losses at all → 100
    out = out.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)  # flat → neutral
    out[avg_gain.isna() | avg_loss.isna()] = np.nan
    return out


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(s, fast) - ema(s, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3, smooth: int = 3) -> pd.DataFrame:
    ll = df["low"].rolling(k).min()
    hh = df["high"].rolling(k).max()
    raw_k = 100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)
    k_s = raw_k.rolling(smooth).mean()
    d_s = k_s.rolling(d).mean()
    return pd.DataFrame({"stoch_k": k_s, "stoch_d": d_s})


def roc(s: pd.Series, period: int = 10) -> pd.Series:
    return 100 * (s / s.shift(period) - 1)


def momentum(s: pd.Series, period: int = 10) -> pd.Series:
    return s - s.shift(period)


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = sma(tp, period)
    md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


# --------------------------------------------------------------------------
# Trend strength
# --------------------------------------------------------------------------
def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr_s = rma(true_range(df), period)
    plus_di = 100 * rma(plus_dm, period) / tr_s.replace(0, np.nan)
    minus_di = 100 * rma(minus_dm, period) / tr_s.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_s = rma(dx, period)
    return pd.DataFrame({"adx": adx_s, "plus_di": plus_di, "minus_di": minus_di})


def efficiency_ratio(s: pd.Series, period: int = 20) -> pd.Series:
    """Kaufman ER: |net change| / sum(|bar changes|) ∈ [0,1]. 1 = perfect trend, 0 = noise."""
    change = (s - s.shift(period)).abs()
    vol = s.diff().abs().rolling(period).sum()
    return change / vol.replace(0, np.nan)


def ema_slope(s: pd.Series, period: int = 50, lookback: int = 10, atr_series: Optional[pd.Series] = None) -> pd.Series:
    """EMA slope normalised by ATR (so it is comparable across instruments)."""
    e = ema(s, period)
    slope = (e - e.shift(lookback)) / lookback
    if atr_series is not None:
        return slope / atr_series.replace(0, np.nan)
    return slope / e


def linear_regression_slope(s: pd.Series, period: int = 20) -> pd.Series:
    x = np.arange(period)
    x = x - x.mean()
    denom = (x ** 2).sum()
    return s.rolling(period).apply(lambda y: np.dot(x, y - y.mean()) / denom, raw=True)


def choppiness(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr_sum = true_range(df).rolling(period).sum()
    rng = df["high"].rolling(period).max() - df["low"].rolling(period).min()
    return 100 * np.log10(tr_sum / rng.replace(0, np.nan)) / np.log10(period)


# --------------------------------------------------------------------------
# Ichimoku
# --------------------------------------------------------------------------
def ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52, shift: int = 26) -> pd.DataFrame:
    def mid(p):
        return (df["high"].rolling(p).max() + df["low"].rolling(p).min()) / 2
    t, k = mid(tenkan), mid(kijun)
    span_a = ((t + k) / 2).shift(shift)     # plotted forward → compare price to span shifted
    span_b = mid(senkou_b).shift(shift)
    chikou = df["close"].shift(-shift)       # NOTE: forward-looking by construction; never use in signals
    return pd.DataFrame({"tenkan": t, "kijun": k, "span_a": span_a, "span_b": span_b, "chikou_DO_NOT_USE": chikou})


# --------------------------------------------------------------------------
# Volume
# --------------------------------------------------------------------------
def vwap(df: pd.DataFrame, anchor: str = "D") -> pd.Series:
    """Session-anchored VWAP (resets each `anchor` period; 'D' = daily)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pv = tp * df["volume"]
    grp = df.index.tz_convert(None).to_period(anchor) if df.index.tz is not None else df.index.to_period(anchor)
    cum_pv = pv.groupby(grp).cumsum()
    cum_v = df["volume"].groupby(grp).cumsum().replace(0, np.nan)
    return cum_pv / cum_v


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


def volume_zscore(df: pd.DataFrame, period: int = 20) -> pd.Series:
    v = df["volume"]
    return (v - v.rolling(period).mean()) / v.rolling(period).std(ddof=0).replace(0, np.nan)


def relative_volume(df: pd.DataFrame, period: int = 20) -> pd.Series:
    return df["volume"] / df["volume"].rolling(period).mean().replace(0, np.nan)


def volume_profile(df: pd.DataFrame, bins: int = 24, lookback: Optional[int] = None) -> Dict[str, object]:
    """Approximate volume-at-price profile from OHLCV (volume spread across bar range).

    Returns POC, value area (70%), HVN/LVN levels.
    """
    d = df.tail(lookback) if lookback else df
    lo, hi = d["low"].min(), d["high"].max()
    if hi <= lo:
        return {"poc": float(d["close"].iloc[-1]), "vah": hi, "val": lo, "hvn": [], "lvn": [], "bins": []}
    edges = np.linspace(lo, hi, bins + 1)
    vol_at = np.zeros(bins)
    for h, l, v in zip(d["high"].to_numpy(), d["low"].to_numpy(), d["volume"].to_numpy()):
        if v <= 0:
            v = 1.0  # tick-count fallback when volume is absent
        lo_i = np.searchsorted(edges, l, side="right") - 1
        hi_i = np.searchsorted(edges, h, side="left")
        lo_i, hi_i = max(0, lo_i), min(bins, max(hi_i, lo_i + 1))
        vol_at[lo_i:hi_i] += v / (hi_i - lo_i)
    centers = (edges[:-1] + edges[1:]) / 2
    poc_i = int(vol_at.argmax())
    total = vol_at.sum()
    # value area: expand around POC until 70% of volume
    inc = {poc_i}
    acc = vol_at[poc_i]
    lo_i, hi_i = poc_i, poc_i
    while acc < 0.7 * total and (lo_i > 0 or hi_i < bins - 1):
        left = vol_at[lo_i - 1] if lo_i > 0 else -1
        right = vol_at[hi_i + 1] if hi_i < bins - 1 else -1
        if right >= left:
            hi_i += 1
            acc += vol_at[hi_i]
        else:
            lo_i -= 1
            acc += vol_at[lo_i]
    mean_v = vol_at.mean()
    hvn = [float(c) for c, v in zip(centers, vol_at) if v > 1.5 * mean_v]
    lvn = [float(c) for c, v in zip(centers, vol_at) if v < 0.5 * mean_v]
    return {
        "poc": float(centers[poc_i]),
        "vah": float(edges[hi_i + 1]),
        "val": float(edges[lo_i]),
        "hvn": hvn,
        "lvn": lvn,
        "bins": [(float(c), float(v)) for c, v in zip(centers, vol_at)],
    }


# --------------------------------------------------------------------------
# Fibonacci
# --------------------------------------------------------------------------
FIB_RETRACEMENTS = (0.236, 0.382, 0.5, 0.618, 0.705, 0.786)
FIB_EXTENSIONS = (1.272, 1.618, 2.0, 2.618)


def fibonacci_levels(swing_low: float, swing_high: float, direction: str = "up") -> Dict[str, float]:
    """Retracement + extension levels for a swing. direction='up' means impulse low→high."""
    rng = swing_high - swing_low
    out: Dict[str, float] = {}
    if direction == "up":
        for r in FIB_RETRACEMENTS:
            out[f"ret_{r}"] = swing_high - rng * r
        for e in FIB_EXTENSIONS:
            out[f"ext_{e}"] = swing_low + rng * e
        out["ote_low"], out["ote_high"] = swing_high - rng * 0.786, swing_high - rng * 0.618
    else:
        for r in FIB_RETRACEMENTS:
            out[f"ret_{r}"] = swing_low + rng * r
        for e in FIB_EXTENSIONS:
            out[f"ext_{e}"] = swing_high - rng * e
        out["ote_low"], out["ote_high"] = swing_low + rng * 0.618, swing_low + rng * 0.786
    out["equilibrium"] = swing_low + rng * 0.5
    return out


# --------------------------------------------------------------------------
# Bundle
# --------------------------------------------------------------------------
def compute_indicator_frame(df: pd.DataFrame, atr_period: int = 14, adx_period: int = 14) -> pd.DataFrame:
    """Compute the standard feature set used across the system (one pass)."""
    out = df.copy()
    c = out["close"]
    out["ema20"] = ema(c, 20)
    out["ema50"] = ema(c, 50)
    out["ema200"] = ema(c, 200)
    out["sma20"] = sma(c, 20)
    out["atr"] = atr(out, atr_period)
    out["atr_pct"] = out["atr"] / c
    out["atr_percentile"] = atr_percentile(out, atr_period, lookback=min(250, max(40, len(out) // 2)))
    out["rsi"] = rsi(c, 14)
    m = macd(c)
    out[["macd", "macd_signal", "macd_hist"]] = m
    st = stochastic(out)
    out[["stoch_k", "stoch_d"]] = st
    a = adx(out, adx_period)
    out[["adx", "plus_di", "minus_di"]] = a
    b = bollinger(c, 20)
    out[["bb_mid", "bb_upper", "bb_lower", "bb_width", "bb_pct"]] = b
    out["bb_width_pct"] = rolling_percentile_rank(out["bb_width"], min(120, max(30, len(out) // 3)), 20)
    out["squeeze"] = bb_squeeze(out)
    out["er"] = efficiency_ratio(c, 20)
    out["ema50_slope"] = ema_slope(c, 50, 10, out["atr"])
    out["chop"] = choppiness(out, 14)
    out["vol_z"] = volume_zscore(out, 20)
    out["rvol"] = relative_volume(out, 20)
    out["vwap"] = vwap(out, "D") if out["volume"].sum() > 0 else np.nan
    out["obv"] = obv(out)
    return out


def latest(ind: pd.DataFrame) -> Dict[str, float]:
    row = ind.iloc[-1]
    return {k: (None if pd.isna(v) else (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)))
            for k, v in row.items() if k not in ("open", "high", "low", "close", "volume")}

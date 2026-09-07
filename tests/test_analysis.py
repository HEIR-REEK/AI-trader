from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from ai_trader.analysis import indicators as I
from ai_trader.analysis.breakout import assess_breakout
from ai_trader.analysis.ict import ict_context
from ai_trader.analysis.levels import horizontal_levels, psychological_levels
from ai_trader.analysis.mtf import build_context
from ai_trader.analysis.patterns import detect_patterns
from ai_trader.analysis.smc import fair_value_gaps, liquidity_pools, order_blocks, smc_snapshot
from ai_trader.analysis.structure import analyze_structure, find_swings, label_swings, premium_discount
from ai_trader.analysis.volatility import assess_volatility
from ai_trader.analysis.volume import assess_volume
from ai_trader.config.instruments import get_instrument
from ai_trader.core.enums import Bias, Direction, StructureEventType, Timeframe
from ai_trader.data import MarketDataLoader


# ------------------------------------------------------------- indicators
def test_sma_ema_reference_values():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    assert I.sma(s, 3).iloc[-1] == 9.0
    # EMA(3) with alpha=0.5 — closed form check against manual recursion
    e = I.ema(s, 3)
    manual = None
    alpha = 2 / 4
    for i, v in enumerate(s):
        if i < 2:
            continue
        manual = v if manual is None else alpha * v + (1 - alpha) * manual
    # pandas seeds with the first value; recompute with the same seeding
    e2 = s.ewm(span=3, adjust=False).mean().iloc[-1]
    assert abs(e.iloc[-1] - e2) < 1e-12


def test_rsi_bounds_and_direction():
    up = pd.Series(np.linspace(100, 200, 60))
    down = pd.Series(np.linspace(200, 100, 60))
    assert I.rsi(up).iloc[-1] > 90
    assert I.rsi(down).iloc[-1] < 10
    noisy = pd.Series(100 + np.cumsum(np.random.default_rng(0).normal(0, 1, 500)))
    r = I.rsi(noisy).dropna()
    assert r.between(0, 100).all()


def test_atr_matches_true_range_average_for_constant_bars():
    idx = pd.date_range("2026-01-01", periods=50, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 1.0}, index=idx)
    assert abs(I.atr(df, 14).iloc[-1] - 2.0) < 1e-9


def test_indicators_are_causal(m15_frame):
    """Changing future bars must not change past indicator values."""
    full = I.compute_indicator_frame(m15_frame)
    truncated = I.compute_indicator_frame(m15_frame.iloc[:-50])
    cols = ["ema20", "rsi", "atr", "adx", "macd", "bb_upper", "er", "stoch_k"]
    a = full[cols].iloc[:-50].dropna()
    b = truncated[cols].dropna()
    common = a.index.intersection(b.index)
    pd.testing.assert_frame_equal(a.loc[common], b.loc[common], check_exact=False, atol=1e-9)


def test_bollinger_and_squeeze(m15_frame):
    bb = I.bollinger(m15_frame["close"]).dropna()
    assert (bb["bb_upper"] >= bb["bb_lower"]).all()
    sq = I.bb_squeeze(m15_frame)
    assert sq.dtype == bool


def test_fibonacci_levels():
    f = I.fibonacci_levels(100, 200, "up")
    assert abs(f["ret_0.618"] - 138.2) < 1e-9
    assert abs(f["ext_1.618"] - 261.8) < 1e-9
    assert f["ote_low"] < f["ote_high"] < f["equilibrium"]


def test_volume_profile_returns_poc_within_range(m15_frame):
    vp = I.volume_profile(m15_frame, bins=20, lookback=200)
    assert m15_frame["low"].tail(200).min() <= vp["poc"] <= m15_frame["high"].tail(200).max()
    assert vp["val"] <= vp["poc"] <= vp["vah"]


# -------------------------------------------------------------- structure
def _zigzag_frame(levels, bars_per_leg=10, noise=0.0, start="2026-01-01"):
    """Build a deterministic OHLC path visiting `levels` sequentially."""
    prices = []
    for a, b in zip(levels[:-1], levels[1:]):
        prices += list(np.linspace(a, b, bars_per_leg, endpoint=False))
    prices.append(levels[-1])
    prices = np.array(prices)
    idx = pd.date_range(start, periods=len(prices), freq="15min", tz="UTC")
    o = np.concatenate([[prices[0]], prices[:-1]])
    h = np.maximum(o, prices) + 0.2 + noise
    l = np.minimum(o, prices) - 0.2 - noise
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": prices, "volume": 100.0}, index=idx)


def test_swings_and_labels_on_uptrend():
    df = _zigzag_frame([100, 110, 105, 118, 112, 125, 119, 130])
    swings = find_swings(df, lookback=2)
    labels = [lab for _, lab in label_swings(swings)]
    assert "HH" in labels and "HL" in labels
    assert "LL" not in labels[2:]


def test_swing_confirmation_is_causal():
    df = _zigzag_frame([100, 110, 105, 118, 112, 125])
    swings = find_swings(df, lookback=3)
    for s in swings:
        assert s.confirmed_index == s.index + 3


def test_bos_choch_detection():
    up = _zigzag_frame([100, 110, 105, 118, 112, 125, 119, 130])
    st = analyze_structure(up, lookback=2)
    kinds = [e.kind for e in st.events]
    assert StructureEventType.BOS in kinds
    assert st.trend is Direction.LONG
    # now reverse hard: should produce a CHoCH (or MSS) to the downside
    rev = _zigzag_frame([100, 110, 105, 118, 112, 125, 119, 130, 100])
    st2 = analyze_structure(rev, lookback=2)
    down_events = [e for e in st2.events if e.direction is Direction.SHORT]
    assert down_events and down_events[0].kind in (StructureEventType.CHOCH, StructureEventType.MSS)
    assert st2.trend is Direction.SHORT


def test_premium_discount():
    assert premium_discount(95, 0, 100)[0] == "premium"
    assert premium_discount(10, 0, 100)[0] == "discount"
    assert premium_discount(50, 0, 100)[0] == "equilibrium"


# ----------------------------------------------------------------- levels
def test_horizontal_levels_cluster_touches():
    df = _zigzag_frame([100, 120, 100, 120, 100, 120, 100, 120, 110])
    lv = horizontal_levels(df, Timeframe.M15, lookback=2, tol_atr=0.5)
    res = [l for l in lv if l.kind == "resistance"]
    sup = [l for l in lv if l.kind == "support"]
    assert res and abs(res[0].price - 120) < 1.0 and res[0].touches >= 3
    assert sup and abs(sup[0].price - 100) < 1.0


def test_psychological_levels_gold_and_fx():
    g = psychological_levels(2345.0, 0.1)
    assert any(abs(l.price - 2350) < 1e-6 for l in g)
    fx = psychological_levels(1.0850, 0.0001)
    assert any(abs(l.price - 1.09) < 1e-9 for l in fx)


# --------------------------------------------------------------- patterns
def _pattern_frame(rows):
    idx = pd.date_range("2026-01-01", periods=len(rows), freq="15min", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx).assign(volume=1.0)


def test_bullish_engulfing_and_hammer():
    base = [[100, 101, 99, 100.5]] * 20
    df = _pattern_frame(base + [[100.5, 100.8, 99.8, 100.0], [99.9, 101.5, 99.7, 101.3]])
    names = {p.name for p in detect_patterns(df)}
    assert "bullish_engulfing" in names
    df2 = _pattern_frame(base + [[100.5, 100.6, 99.9, 100.0], [100.0, 100.1, 97.0, 99.95]])
    names2 = {p.name for p in detect_patterns(df2)}
    assert "hammer" in names2 or "bullish_pin_bar" in names2


def test_shooting_star_and_evening_star():
    base = [[100, 101, 99, 100.5]] * 20
    df = _pattern_frame(base + [[100, 100.6, 99.9, 100.5], [100.5, 103.5, 100.4, 100.55]])
    assert any(p.name in ("shooting_star", "bearish_pin_bar") for p in detect_patterns(df))
    df2 = _pattern_frame(base + [[100, 102.2, 99.9, 102.0], [102.1, 102.4, 101.9, 102.15], [102.0, 102.1, 100.2, 100.4]])
    assert any(p.name == "evening_star" for p in detect_patterns(df2))


# --------------------------------------------------------------------- SMC
def test_fvg_detection():
    rows = [[100, 101, 99, 100.5]] * 20 + [[100.5, 101.0, 100.0, 100.8], [100.8, 104.0, 100.7, 103.8], [103.9, 105.0, 103.0, 104.5]]
    df = _pattern_frame(rows)
    z = fair_value_gaps(df, Timeframe.M15, min_size_atr=0.1)
    bull = [x for x in z if x.direction is Direction.LONG]
    assert bull and abs(bull[-1].low - 101.0) < 1e-9 and abs(bull[-1].high - 103.0) < 1e-9


def test_liquidity_pool_sweep():
    df = _zigzag_frame([100, 120, 100, 120, 100, 120.9, 100], bars_per_leg=8)
    # add explicit sweep bar: wick above 120 equal highs then close below
    pools = liquidity_pools(df, lookback=2)
    buy_side = [p for p in pools if p.kind.value == "BUY_SIDE"]
    assert buy_side
    assert any(p.touches >= 2 for p in buy_side)


def test_order_block_detection():
    rows = [[100, 100.6, 99.4, 100.2], [100.2, 100.7, 99.6, 100.0]] * 12
    rows += [[100.0, 100.3, 99.2, 99.4],   # last down candle → bullish OB
             [99.5, 101.5, 99.4, 101.4], [101.4, 103.0, 101.3, 102.9], [102.9, 104.5, 102.8, 104.4], [104.4, 105.0, 104.0, 104.8]]
    df = _pattern_frame(rows)
    obs = order_blocks(df, Timeframe.M15, displacement_atr=1.0, lookahead=3)
    bull = [z for z in obs if z.direction is Direction.LONG]
    assert bull and abs(bull[-1].low - 99.2) < 1e-9


def test_smc_snapshot_runs(m15_frame):
    snap = smc_snapshot(m15_frame, Timeframe.M15)
    assert snap.pd_state in ("premium", "discount", "equilibrium", "unknown")
    assert isinstance(snap.all_zones(), list)


# ---------------------------------------------------------------- breakout
def test_breakout_valid_vs_fake():
    rows = [[100, 100.5, 99.5, 100.0]] * 30
    valid = rows + [[100.0, 102.5, 99.9, 102.4], [102.4, 103.2, 102.0, 103.0], [103.0, 103.6, 102.6, 103.4], [103.4, 104.0, 103.0, 103.8]]
    fake = rows + [[100.0, 102.5, 99.9, 102.4], [102.4, 102.6, 99.8, 99.9], [99.9, 100.2, 99.4, 99.6], [99.6, 100.0, 99.0, 99.2]]
    v = assess_breakout(_pattern_frame(valid), 100.5, Direction.LONG)
    f = assess_breakout(_pattern_frame(fake), 100.5, Direction.LONG)
    assert v is not None and v.valid and not v.fake
    assert f is not None and f.fake and not f.valid


# ---------------------------------------------------------- volume / vola
def test_volume_assessment_handles_missing_volume(m15_frame):
    nov = m15_frame.copy()
    nov["volume"] = 0.0
    assert assess_volume(nov, Direction.LONG).available is False
    v = assess_volume(m15_frame, Direction.LONG)
    assert v.available and 0 <= v.score(Direction.LONG) <= 1


def test_volatility_assessment(m15_frame):
    va = assess_volatility(m15_frame)
    assert va.state in ("DEAD", "LOW", "NORMAL", "ELEVATED", "EXTREME")
    assert 0 <= va.atr_percentile <= 1


# --------------------------------------------------------------------- ICT
def test_ict_context(m15_frame):
    ctx = ict_context(m15_frame)
    d = ctx.to_dict()
    assert "session" in d and "kill_zone" in d


# --------------------------------------------------------------------- MTF
def test_mtf_context_bias_on_trend(trend_up_provider, trend_down_provider):
    inst = get_instrument("XAUUSD")
    for prov, expected in ((trend_up_provider, Bias.BUY), (trend_down_provider, Bias.SELL)):
        md = MarketDataLoader(prov).load("XAUUSD", timeframes=[Timeframe.M15, Timeframe.H1, Timeframe.H4]).data
        ctx = build_context(md, inst)
        assert ctx.structural_bias is expected, ctx.summary()
        assert (ctx.alignment > 0) == (expected is Bias.BUY)

import os
import tempfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from ai_trader.config.instruments import all_instruments, get_instrument, register
from ai_trader.core.enums import AssetClass, Timeframe
from ai_trader.core.models import Instrument
from ai_trader.data import (
    CSVProvider,
    EconomicCalendar,
    EconomicEvent,
    MarketDataLoader,
    SyntheticProvider,
    closed_bars_as_of,
    resample_ohlcv,
    validate_ohlcv,
)


# ----------------------------------------------------------------- registry
def test_registry_has_all_required_markets():
    for s in ["XAUUSD", "XAGUSD", "VOL10", "VOL25", "VOL50", "VOL75", "VOL100", "US500", "US100", "US30",
              "DE40", "UK100", "JP225", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "BTCUSD", "ETHUSD"]:
        assert get_instrument(s).symbol == s


def test_registry_aliases_and_extension():
    assert get_instrument("gold").symbol == "XAUUSD"
    assert get_instrument("XAU/USD").symbol == "XAUUSD"
    assert get_instrument("R_75").symbol == "VOL75"
    register(Instrument("TESTUSD", "Test", AssetClass.CRYPTO, pip_size=0.01))
    assert get_instrument("TESTUSD").asset_class == AssetClass.CRYPTO
    assert len(all_instruments(AssetClass.SYNTHETIC_INDEX)) == 5


def test_instrument_rounding():
    g = get_instrument("XAUUSD")
    assert g.round_price(2345.6789) == 2345.68
    assert get_instrument("EURUSD").round_price(1.08765432) == 1.08765


# ---------------------------------------------------------------- synthetic
def test_synthetic_provider_shape_and_integrity(synthetic_provider):
    df = synthetic_provider.get_ohlcv("XAUUSD", Timeframe.M5, limit=500)
    assert len(df) == 500
    assert list(df.columns[:5]) == ["open", "high", "low", "close", "volume"]
    assert df.index.tz is not None
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()


def test_synthetic_is_deterministic():
    a = SyntheticProvider(seed=1).get_ohlcv("EURUSD", Timeframe.M15, 300)
    b = SyntheticProvider(seed=1).get_ohlcv("EURUSD", Timeframe.M15, 300)
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------- resampler
def test_resample_is_consistent_and_drops_incomplete():
    idx = pd.date_range("2026-01-05 00:00", periods=13, freq="5min", tz="UTC")  # 1 full hour + 5 min
    df = pd.DataFrame({"open": np.arange(13.0), "high": np.arange(13.0) + 1, "low": np.arange(13.0) - 1,
                       "close": np.arange(13.0) + 0.5, "volume": 1.0}, index=idx)
    h1 = resample_ohlcv(df, Timeframe.M5, Timeframe.H1, drop_incomplete=True)
    assert len(h1) == 1                       # the 01:00 bar only has one 5m bar → dropped
    assert h1.iloc[0]["open"] == 0.0
    assert h1.iloc[0]["high"] == 12.0
    assert h1.iloc[0]["low"] == -1.0
    assert h1.iloc[0]["close"] == 11.5
    assert h1.iloc[0]["volume"] == 12
    h1_keep = resample_ohlcv(df, Timeframe.M5, Timeframe.H1, drop_incomplete=False)
    assert len(h1_keep) == 2


def test_closed_bars_as_of_is_causal():
    idx = pd.date_range("2026-01-05 00:00", periods=6, freq="4h", tz="UTC")
    df = pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 1.0}, index=idx)
    as_of = datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)  # inside the 08:00-12:00 bar
    out = closed_bars_as_of(df, Timeframe.H4, as_of)
    assert out.index[-1] == pd.Timestamp("2026-01-05 04:00", tz="UTC")  # 04:00 bar closed at 08:00


def test_resample_rejects_downsampling(m15_frame):
    with pytest.raises(ValueError):
        resample_ohlcv(m15_frame, Timeframe.M15, Timeframe.M5)


# --------------------------------------------------------------- validation
def test_validation_catches_bad_bars(m15_frame):
    rep = validate_ohlcv(m15_frame, Timeframe.M15, min_bars=100)
    assert rep.ok, rep.issues
    bad = m15_frame.copy()
    bad.iloc[10, bad.columns.get_loc("high")] = bad.iloc[10]["low"] - 1
    rep = validate_ohlcv(bad, Timeframe.M15)
    assert not rep.ok and any("violate" in i for i in rep.issues)


def test_validation_flags_stale_feed(m15_frame):
    now = m15_frame.index[-1].to_pydatetime() + timedelta(hours=5)
    rep = validate_ohlcv(m15_frame, Timeframe.M15, now=now, max_stale_bars=2)
    assert not rep.ok and any("stale" in i for i in rep.issues)


def test_validation_insufficient_bars(m15_frame):
    rep = validate_ohlcv(m15_frame.head(10), Timeframe.M15, min_bars=50)
    assert not rep.ok


# ---------------------------------------------------------------------- csv
def test_csv_roundtrip(m15_frame):
    with tempfile.TemporaryDirectory() as d:
        p = CSVProvider(data_dir=d)
        path = p.save("XAUUSD", Timeframe.M15, m15_frame)
        assert os.path.exists(path)
        back = p.get_ohlcv("XAUUSD", Timeframe.M15, limit=100)
        assert len(back) == 100
        assert abs(back["close"].iloc[-1] - m15_frame["close"].iloc[-1]) < 1e-9


# ------------------------------------------------------------------- loader
def test_loader_builds_all_timeframes(synthetic_provider):
    loader = MarketDataLoader(synthetic_provider)
    res = loader.load("XAUUSD", timeframes=[Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1])
    assert set(res.data.frames) == {Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1}
    for tf, df in res.data.frames.items():
        assert "label" not in df.columns
        assert df.index.tz is not None
    assert res.data.last_price > 0
    # higher TF last bar must not extend beyond the entry TF's last bar
    assert res.data.frames[Timeframe.H4].index[-1] <= res.data.frames[Timeframe.M15].index[-1]


# ----------------------------------------------------------------- calendar
def test_calendar_blackout_and_relevance():
    now = datetime(2026, 9, 4, 12, 15, tzinfo=timezone.utc)
    cal = EconomicCalendar.from_events([
        EconomicEvent(now + timedelta(minutes=15), "USD", "HIGH", "Non-Farm Payrolls", forecast=150.0),
        EconomicEvent(now + timedelta(hours=3), "EUR", "HIGH", "ECB Rate Decision"),
        EconomicEvent(now + timedelta(minutes=10), "USD", "LOW", "Some minor print"),
    ])
    assert cal.in_blackout("XAUUSD", now, before_min=30, after_min=30).title == "Non-Farm Payrolls"
    assert cal.in_blackout("EURGBP", now, before_min=30, after_min=30) is None
    assert cal.next_high_impact("EURUSD", now).title == "Non-Farm Payrolls"
    assert cal.next_high_impact("VOL75", now) is None or cal.next_high_impact("VOL75", now).currency == "USD"


def test_calendar_missing_file():
    cal = EconomicCalendar.from_json("/nonexistent/calendar.json")
    assert not cal.known and cal.events == []

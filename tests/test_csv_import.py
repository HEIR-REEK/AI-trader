"""CSV import: broker dialects, timestamp flavours, aliases and CLI handling."""
import argparse

import pandas as pd
import pytest

from ai_trader.cli import _parse_cli_dt, cmd_backtest
from ai_trader.core.enums import Timeframe
from ai_trader.core.exceptions import DataError, ProviderError
from ai_trader.data.providers import CSVProvider, normalise_ohlcv, read_csv_smart


def _ohlc(n=48, start="2026-03-02 00:00", freq="15min"):
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": 2000.0, "high": 2001.0, "low": 1999.0, "close": 2000.5, "volume": 10.0},
        index=idx,
    )


def _provider(tmp_path):
    return CSVProvider(data_dir=str(tmp_path))


# ------------------------------------------------------------ broker dialects
def test_mt5_tab_export_with_angle_brackets(tmp_path):
    df = _ohlc()
    lines = ["<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>"]
    for ts, r in df.iterrows():
        lines.append(f"{ts:%Y.%m.%d}\t{ts:%H:%M}\t{r['open']:.2f}\t{r['high']:.2f}\t"
                     f"{r['low']:.2f}\t{r['close']:.2f}\t100\t0\t20")
    (tmp_path / "XAUUSD_15m.csv").write_text("\n".join(lines) + "\n")
    out = _provider(tmp_path).get_ohlcv("XAUUSD", Timeframe.M15)
    assert len(out) == len(df)
    assert out.index[0] == df.index[0]
    assert out["close"].iloc[-1] == pytest.approx(2000.5)
    assert (out["volume"] == 100).all()  # TickVol preferred over Vol


@pytest.mark.parametrize("sep", [";", "|"])
def test_semicolon_and_pipe_delimiters(tmp_path, sep):
    df = _ohlc().reset_index(names="time")
    text = df.to_csv(index=False, sep=sep)
    (tmp_path / "EURUSD_1h.csv").write_text(text)
    out = _provider(tmp_path).get_ohlcv("EURUSD", Timeframe.H1)
    assert len(out) == len(df)
    assert out.index.tz is not None


def test_oanda_single_letter_ohlc(tmp_path):
    df = _ohlc().reset_index(names="time")
    oanda = pd.DataFrame({"time": df["time"].dt.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
                          "volume": df["volume"], "complete": True,
                          "o": df["open"], "h": df["high"], "l": df["low"], "c": df["close"]})
    (tmp_path / "EURUSD_15m.csv").write_text(oanda.to_csv(index=False))
    out = _provider(tmp_path).get_ohlcv("EURUSD", Timeframe.M15)
    assert out["open"].iloc[0] == pytest.approx(2000.0)
    assert out["close"].iloc[-1] == pytest.approx(2000.5)


def test_separate_date_and_time_columns_are_merged(tmp_path):
    df = _ohlc().reset_index(names="ts")
    split = pd.DataFrame({"Date": df["ts"].dt.strftime("%Y-%m-%d"), "Time": df["ts"].dt.strftime("%H:%M"),
                          "Open": df["open"], "High": df["high"], "Low": df["low"],
                          "Close": df["close"], "Volume": df["volume"]})
    (tmp_path / "GBPUSD_15m.csv").write_text(split.to_csv(index=False))
    out = _provider(tmp_path).get_ohlcv("GBPUSD", Timeframe.M15)
    assert len(out) == len(df)
    assert out.index[0] == df["ts"].iloc[0]


def test_bom_and_whitespace_tolerated(tmp_path):
    df = _ohlc(n=5).reset_index(names="ts")
    path = tmp_path / "XAUUSD_15m.csv"
    path.write_bytes(b"\xef\xbb\xbf" + " ts , open ,high,low , close,volume \n".encode()
                     + df.to_csv(index=False, header=False).encode())
    out = _provider(tmp_path).get_ohlcv("XAUUSD", Timeframe.M15)
    assert len(out) == 5


# ---------------------------------------------------------------- timestamps
@pytest.mark.parametrize("factor", [1, 1_000, 1_000_000, 1_000_000_000, 1.0],
                         ids=["s", "ms", "us", "ns", "s-float"])
def test_epoch_timestamps_by_magnitude(tmp_path, factor):
    df = _ohlc(n=10).reset_index(names="ts")
    secs = (df["ts"] - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s")
    df["ts"] = secs * factor
    (tmp_path / "BTCUSD_15m.csv").write_text(df.to_csv(index=False))
    out = _provider(tmp_path).get_ohlcv("BTCUSD", Timeframe.M15)
    assert out.index.year.unique().tolist() == [2026]
    assert (out.index == df["ts"].map(lambda v: pd.Timestamp(v / factor, unit="s", tz="UTC"))).all()


def test_yyyymmdd_integers(tmp_path):
    dates = pd.date_range("2026-03-02", periods=10, freq="D")
    df = pd.DataFrame({"date": dates.strftime("%Y%m%d").astype(int), "open": 2000.0,
                       "high": 2001.0, "low": 1999.0, "close": 2000.5, "volume": 5.0})
    (tmp_path / "XAUUSD_1d.csv").write_text(df.to_csv(index=False))
    out = _provider(tmp_path).get_ohlcv("XAUUSD", Timeframe.D1)
    assert len(out) == 10
    assert out.index[0] == pd.Timestamp("2026-03-02", tz="UTC")


def test_aware_timestamps_converted_to_utc(tmp_path):
    df = _ohlc(n=5).reset_index(names="ts")
    df["ts"] = df["ts"].dt.tz_convert("Europe/Berlin").dt.strftime("%Y-%m-%d %H:%M:%S%z")
    (tmp_path / "EURUSD_15m.csv").write_text(df.to_csv(index=False))
    out = _provider(tmp_path).get_ohlcv("EURUSD", Timeframe.M15)
    assert str(out.index.tz) == "UTC"
    assert out.index[0] == pd.Timestamp("2026-03-02 00:00", tz="UTC")


def test_open_time_preferred_over_close_time(tmp_path):
    df = _ohlc(n=5).reset_index(names="open_time")
    df["close_time"] = df["open_time"] + pd.Timedelta(minutes=15)
    (tmp_path / "XAUUSD_15m.csv").write_text(df.to_csv(index=False))
    out = _provider(tmp_path).get_ohlcv("XAUUSD", Timeframe.M15)
    assert (out.index == df["open_time"]).all()


def test_newest_first_rows_are_sorted(tmp_path):
    df = _ohlc().reset_index(names="ts").iloc[::-1]
    (tmp_path / "XAUUSD_15m.csv").write_text(df.to_csv(index=False))
    out = _provider(tmp_path).get_ohlcv("XAUUSD", Timeframe.M15)
    assert out.index.is_monotonic_increasing
    assert out.index[0] == pd.Timestamp("2026-03-02 00:00", tz="UTC")


def test_duplicate_timestamps_keep_last(tmp_path):
    df = _ohlc(n=5).reset_index(names="ts")
    dup = df.iloc[[2]].copy()
    dup["close"] = 1234.5
    df = pd.concat([df, dup], ignore_index=True)
    (tmp_path / "XAUUSD_15m.csv").write_text(df.to_csv(index=False))
    out = _provider(tmp_path).get_ohlcv("XAUUSD", Timeframe.M15)
    assert len(out) == 5
    assert out["close"].iloc[2] == pytest.approx(1234.5)


# ------------------------------------------------------------------- aliases
@pytest.mark.parametrize("vol_col", ["TickVol", "Tick Volume", "Vol", "tick_volume", "VOLUME"])
def test_volume_aliases(tmp_path, vol_col):
    df = _ohlc(n=5).reset_index(names="Time").rename(columns={"volume": vol_col})
    (tmp_path / "GBPUSD_15m.csv").write_text(df.to_csv(index=False))
    out = _provider(tmp_path).get_ohlcv("GBPUSD", Timeframe.M15)
    assert (out["volume"] == 10.0).all()


def test_missing_volume_defaults_to_zero():
    out = normalise_ohlcv(_ohlc(n=5).reset_index(names="ts").drop(columns=["volume"]))
    assert (out["volume"] == 0.0).all()


def test_datetime_index_without_ts_column_still_works():
    out = normalise_ohlcv(_ohlc(n=5))  # e.g. CSVProvider.save round-trip input
    assert len(out) == 5 and out.index.name == "ts"


# -------------------------------------------------------------------- errors
def test_missing_file_names_expected_filename(tmp_path):
    with pytest.raises(ProviderError, match=r"XAUUSD_15m\.csv"):
        _provider(tmp_path).get_ohlcv("XAUUSD", Timeframe.M15)


@pytest.mark.parametrize("body", ["", "   \n  "])
def test_empty_file_raises_data_error(tmp_path, body):
    (tmp_path / "XAUUSD_15m.csv").write_text(body)
    with pytest.raises(DataError, match="[Ee]mpty"):
        _provider(tmp_path).get_ohlcv("XAUUSD", Timeframe.M15)


def test_garbage_timestamps_raise_data_error(tmp_path):
    df = _ohlc(n=3).reset_index(names="ts", drop=True)
    df.insert(0, "ts", ["not-a-date", "still-not", "nope"])
    (tmp_path / "XAUUSD_15m.csv").write_text(df.to_csv(index=False))
    with pytest.raises(DataError, match="[Cc]annot parse timestamps"):
        _provider(tmp_path).get_ohlcv("XAUUSD", Timeframe.M15)


def test_non_numeric_ohlc_raises_data_error():
    df = _ohlc(n=3).reset_index(names="ts")
    df["open"] = ["a", "b", "c"]
    with pytest.raises(DataError, match="[Nn]on-numeric"):
        normalise_ohlcv(df)


def test_no_timestamp_column_lists_what_was_found():
    df = _ohlc(n=3).reset_index(drop=True)
    with pytest.raises(DataError, match="no timestamp column"):
        normalise_ohlcv(df)


def test_read_csv_without_delimiter_raises_data_error(tmp_path):
    p = tmp_path / "weird.csv"
    p.write_text("just some text\nmore text\n")
    with pytest.raises(DataError, match="[Nn]o delimiter"):
        read_csv_smart(str(p))


# ----------------------------------------------------------------------- cli
def test_parse_cli_dt_naive_assumes_utc():
    assert _parse_cli_dt("2025-01-01", "--start").isoformat() == "2025-01-01T00:00:00+00:00"
    assert _parse_cli_dt("2025-06-01 12:00", "--start").isoformat() == "2025-06-01T12:00:00+00:00"


def test_parse_cli_dt_aware_converts_to_utc():
    assert _parse_cli_dt("2025-06-01T12:00:00+02:00", "--start").isoformat() == "2025-06-01T10:00:00+00:00"
    assert _parse_cli_dt("2025-06-01T12:00:00Z", "--end").isoformat() == "2025-06-01T12:00:00+00:00"


def test_parse_cli_dt_rejects_garbage():
    with pytest.raises(ValueError, match="--start"):
        _parse_cli_dt("not-a-date", "--start")


def _backtest_ns(tmp_path, **over):
    kw = dict(symbol="XAUUSD", source="csv", data_dir=str(tmp_path), timeframes=None, start=None, end=None,
              split=None, bars=1500, warmup=400, every=50, seed=1, direction="long", resolve=None, spread=None,
              commission=0.0, ttl=8, max_hold=96, equity=None, label="", walk_forward=False, trades=False,
              progress=False, json=None, no_news_penalty=True)
    kw.update(over)
    return argparse.Namespace(**kw)


def test_backtest_csv_preflight_missing_entry_file(tmp_path, capsys):
    (tmp_path / "XAUUSD_1d.csv").write_text(_ohlc(n=70, freq="D").reset_index(names="ts").to_csv(index=False))
    rc = cmd_backtest(_backtest_ns(tmp_path))
    assert rc == 2
    assert "XAUUSD_15m.csv not found" in capsys.readouterr().out


def test_backtest_csv_end_to_end_over_saved_synthetic(tmp_path, synthetic_provider):
    from ai_trader.backtest import BacktestConfig, Backtester

    prov = _provider(tmp_path)
    for tf, n in [(Timeframe.M15, 2000), (Timeframe.H1, 600), (Timeframe.H4, 400), (Timeframe.D1, 200)]:
        prov.save("XAUUSD", tf, synthetic_provider.get_ohlcv("XAUUSD", tf, limit=n))
    cfg = BacktestConfig("XAUUSD", timeframes=[Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1],
                         warmup_bars=400, analyze_every=50, label="csv-e2e")
    res = Backtester(CSVProvider(data_dir=str(tmp_path)), cfg).run()
    assert res.data_source == "csv"
    assert res.decisions_made > 0
    assert res.candidates_frame() is not None

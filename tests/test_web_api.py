"""Tests for the browser backend (ai_trader.web) — no terminal needed."""
from __future__ import annotations

import time

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from ai_trader.web.app import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_index_serves_spa(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "AI-Trader" in r.text
    assert "view-analyze" in r.text


def test_static_assets(client):
    for asset, needle in (("app.js", "renderDecision"), ("chart.js", "CandleChart"), ("styles.css", ".sidebar")):
        r = client.get(f"/{asset}")
        assert r.status_code == 200, asset
        assert needle in r.text, asset


def test_health(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok"
    assert "15m" in h["timeframes"]


def test_instruments(client):
    d = client.get("/api/instruments").json()
    assert d["count"] > 20
    syms = {i["symbol"] for i in d["instruments"]}
    assert {"XAUUSD", "EURUSD", "BTCUSD"} <= syms


def test_scenarios_meta(client):
    d = client.get("/api/scenarios").json()
    assert {s["id"] for s in d["scenarios"]} == {"textbook_long", "textbook_short", "range_fade", "choppy"}


def test_settings_view(client):
    s = client.get("/api/settings").json()
    assert abs(sum(s["scoring_weights"].values()) - 100) < 1e-6
    assert s["thresholds"]["effective_min_score"] >= 70


def test_analyze_synthetic(client):
    r = client.post("/api/analyze", json={"symbol": "XAUUSD", "source": "synthetic", "seed": 1})
    assert r.status_code == 200, r.text
    d = r.json()["decision"]
    assert d["instrument"] == "XAUUSD"
    assert d["decision"] in ("TRADE", "NO_TRADE")
    assert "disclaimer" in d


def test_analyze_unknown_symbol_is_400(client):
    r = client.post("/api/analyze", json={"symbol": "NOPE"})
    assert r.status_code == 400


def test_analyze_csv_missing_is_400(client):
    r = client.post("/api/analyze", json={"symbol": "XAUUSD", "source": "csv", "data_dir": "data"})
    assert r.status_code == 400
    assert "15m.csv" in r.json()["detail"]


def test_scenario_trade_path(client):
    r = client.post("/api/scenario", json={"name": "textbook_long"})
    assert r.status_code == 200, r.text
    d = r.json()["decision"]
    assert d["decision"] == "TRADE"
    assert d["plan"]["direction"] == "LONG"
    assert d["plan"]["score"] >= 80


def test_scenario_no_trade_path(client):
    r = client.post("/api/scenario", json={"name": "choppy"})
    assert r.status_code == 200, r.text
    assert r.json()["decision"]["decision"] == "NO_TRADE"


def test_scenario_without_breaks_setup(client):
    full = client.post("/api/scenario", json={"name": "textbook_long"}).json()["decision"]
    broken = client.post(
        "/api/scenario", json={"name": "textbook_long", "without": ["displacement", "confirmation_candle"]}
    ).json()["decision"]
    assert full["decision"] == "TRADE"
    assert broken["decision"] == "NO_TRADE"


def test_candles_synthetic(client):
    d = client.get("/api/candles", params={"symbol": "XAUUSD", "timeframe": "15m", "limit": 50}).json()
    assert d["count"] == 50
    c = d["candles"][0]
    assert {"time", "open", "high", "low", "close"} <= set(c)
    assert c["high"] >= c["low"]


def test_candles_scenario(client):
    d = client.get("/api/candles", params={"source": "scenario", "scenario": "textbook_long", "limit": 50}).json()
    assert d["symbol"] == "XAUUSD"
    assert d["count"] == 50


def test_backtest_job_flow(client):
    started = client.post("/api/backtest", json={"symbol": "XAUUSD", "source": "scenario", "resolve": "win"})
    assert started.status_code == 202
    job_id = started.json()["job_id"]
    job = None
    for _ in range(60):
        job = client.get(f"/api/backtest/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(1)
    assert job["status"] == "done", job.get("error")
    res = job["result"]["result"]
    assert res["symbol"] == "XAUUSD"
    assert "metrics" in res and "equity_curve" in res
    # job appears in the listing (without the heavy payload)
    listing = client.get("/api/backtest/jobs").json()
    assert any(j["id"] == job_id and "result" not in j for j in listing["jobs"])


def test_backtest_unknown_job_404(client):
    assert client.get("/api/backtest/jobs/doesnotexist").status_code == 404


# ---------------------------------------------------------------------------
# CSV upload / datasets (browser path to REAL market history)
# ---------------------------------------------------------------------------

def _csv_bytes(symbol="EURUSD", timeframe="15m", bars=400, start="2026-01-05 08:00:00+00:00"):
    import numpy as np
    import pandas as pd
    freq = {"15m": "15min", "1h": "1h"}[timeframe]
    idx = pd.date_range(start, periods=bars, freq=freq, tz="UTC")
    close = np.linspace(1.10, 1.12, bars) + np.random.default_rng(3).normal(0, 0.001, bars)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 0.0008
    low = np.minimum(open_, close) - 0.0008
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 100.0}, index=idx)
    df.index.name = "ts"
    return df.to_csv().encode()


def test_upload_csv_roundtrip(client):
    import os
    path = os.path.join("data", "EURUSD_15m.csv")
    if os.path.exists(path):
        os.remove(path)
    try:
        r = client.post("/api/upload_csv", files={"file": ("EURUSD_15m.csv", _csv_bytes(), "text/csv")})
        assert r.status_code == 200, r.text
        info = r.json()
        assert info["symbol"] == "EURUSD" and info["timeframe"] == "15m"
        assert info["bars"] == 400 and os.path.exists(path)

        # now the csv source can serve the uploaded real data (chart + analysis entry)
        d = client.get("/api/candles", params={"symbol": "EURUSD", "timeframe": "15m",
                                               "limit": 100, "source": "csv"}).json()
        assert d["count"] == 100
        assert 1.0 < d["candles"][-1]["close"] < 1.3  # real EURUSD-ish level, not the ~2000 synthetic base
        # analyze on the uploaded file (entry timeframe present, HTFs resample)
        r2 = client.post("/api/analyze", json={"symbol": "EURUSD", "source": "csv", "seed": 1,
                                               "timeframes": ["15m", "1h", "4h"]})
        assert r2.status_code == 200, r2.text

        ds = client.get("/api/data/datasets").json()
        assert any(x["file"] == "EURUSD_15m.csv" for x in ds["datasets"])
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_upload_csv_rejects_bad_names_and_unknown_symbols(client):
    bad_name = client.post("/api/upload_csv", files={"file": ("random.txt", b"1,2,3", "text/plain")})
    assert bad_name.status_code == 400
    assert "SYMBOL_TIMEFRAME.csv" in bad_name.json()["detail"]

    unknown = client.post("/api/upload_csv", files={"file": ("DOGEUSD_15m.csv", _csv_bytes(), "text/csv")})
    assert unknown.status_code == 400
    assert "Unknown instrument" in unknown.json()["detail"]

    tiny = client.post("/api/upload_csv", files={"file": ("XAUUSD_15m.csv", _csv_bytes(bars=10), "text/csv")})
    assert tiny.status_code == 400
    assert "bars" in tiny.json()["detail"]

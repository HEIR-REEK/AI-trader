"""Orchestration between the HTTP layer and the analysis engine.

This module contains no HTTP code — it is the browser equivalent of
``ai_trader.cli`` (same pipeline, same outputs, JSON-friendly returns).
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from ai_trader.backtest import (
    BacktestConfig,
    Backtester,
    ExecutionCosts,
    ManagementRules,
    format_backtest,
    in_out_of_sample,
    overfit_report,
    result_to_dict,
    walk_forward_thresholds,
)
from ai_trader.config.instruments import all_instruments, get_instrument
from ai_trader.config.settings import get_settings
from ai_trader.core.enums import AssetClass, Timeframe
from ai_trader.core.exceptions import DataError, ProviderError
from ai_trader.data import (
    CSVProvider,
    EconomicCalendar,
    FrameProvider,
    ScenarioConfig,
    SyntheticProvider,
    TwelveDataProvider,
    choppy_no_edge,
    range_fade_setup,
    textbook_pullback_setup,
)
from ai_trader.data.resampler import resample_ohlcv
from ai_trader.decision import DecisionEngine, decision_to_dict, format_decision

DEFAULT_TFS = [Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1]

SCENARIO_TOGGLES = [
    "equal_lows",
    "sweep",
    "displacement",
    "retrace_into_fvg",
    "confirmation_candle",
    "volume_confirms",
]

SCENARIOS: List[Dict[str, Any]] = [
    {
        "id": "textbook_long",
        "title": "Textbook Long",
        "symbol": "XAUUSD",
        "expected": "TRADE (BUY)",
        "description": "Impulsive uptrend → deep pullback into discount → equal lows sweep → "
                       "bullish displacement with FVG → 50% retrace → engulfing confirmation.",
        "toggles": SCENARIO_TOGGLES,
    },
    {
        "id": "textbook_short",
        "title": "Textbook Short",
        "symbol": "XAUUSD",
        "expected": "TRADE (SELL)",
        "description": "Mirror of the textbook long: downtrend → pullback into premium → equal "
                       "highs sweep → bearish displacement → retrace → confirmation.",
        "toggles": SCENARIO_TOGGLES,
    },
    {
        "id": "range_fade",
        "title": "Range Fade",
        "symbol": "EURUSD",
        "expected": "TRADE (fade the edge)",
        "description": "Well-respected horizontal box, price in the outer 20% with stretched "
                       "oscillators and a rejection candle at the edge.",
        "toggles": ["rejection"],
    },
    {
        "id": "choppy",
        "title": "Choppy / No Edge",
        "symbol": "EURUSD",
        "expected": "NO TRADE",
        "description": "Featureless low-conviction chop. The engine should refuse to trade — "
                       "this scenario proves the NO TRADE path works.",
        "toggles": [],
    },
]


# ------------------------------------------------------------------ helpers

def parse_timeframes(values: Optional[List[str]]) -> List[Timeframe]:
    if not values:
        return list(DEFAULT_TFS)
    out = []
    for v in values:
        try:
            out.append(Timeframe(v))
        except ValueError:
            raise ValueError(f"Unknown timeframe {v!r}. Valid: {[t.value for t in Timeframe]}")
    return out


def known_empty_calendar() -> EconomicCalendar:
    cal = EconomicCalendar.from_events([])
    cal.source = "explicit"  # checked: nothing scheduled (no missing-calendar penalty)
    return cal


def build_provider(source: str, seed: int = 1, data_dir: str = "data"):
    src = (source or "synthetic").lower()
    if src == "synthetic":
        return SyntheticProvider(seed=seed)
    if src == "csv":
        prov = CSVProvider(data_dir)
        if not prov.available():
            raise ValueError(f"CSV directory {data_dir!r} not found.")
        return prov
    if src == "twelvedata":
        prov = TwelveDataProvider()
        if not prov.available():
            raise ValueError(
                "TwelveData API key missing. Set AITRADER_TWELVEDATA_API_KEY in the environment, "
                "or use the synthetic / csv source."
            )
        return prov
    raise ValueError(f"Unknown data source {source!r}. Use synthetic | csv | twelvedata.")


def scenario_frame(name: str, seed: int = 1, without: Optional[List[str]] = None):
    """Return (dataframe, symbol) for a scripted scenario."""
    off = set(without or [])
    if name in ("textbook_long", "textbook_short"):
        cfg = ScenarioConfig(direction="long" if name == "textbook_long" else "short", seed=seed)
        for flag in off:
            if not hasattr(cfg, flag):
                raise ValueError(f"Unknown scenario component {flag!r}. Valid: {', '.join(SCENARIO_TOGGLES)}")
            setattr(cfg, flag, False)
        return textbook_pullback_setup(cfg), "XAUUSD"
    if name == "range_fade":
        return range_fade_setup(seed=seed, rejection="rejection" not in off), "EURUSD"
    if name == "choppy":
        return choppy_no_edge(seed=seed), "EURUSD"
    raise ValueError(f"Unknown scenario {name!r}. Use textbook_long | textbook_short | range_fade | choppy.")


def _parse_dt(value: Optional[str], flag: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{flag}: cannot parse {value!r} (expected ISO date/datetime, e.g. 2025-01-01)")
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _safe_float(x: Any) -> Optional[float]:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):  # NaN / inf → null (valid JSON)
        return None
    return f


# ------------------------------------------------------------------ analyze

def analyze_symbol(symbol: str, source: str = "synthetic", seed: int = 1,
                   timeframes: Optional[List[str]] = None, no_news_penalty: bool = False,
                   data_dir: str = "data") -> Dict[str, Any]:
    tfs = parse_timeframes(timeframes)
    prov = build_provider(source, seed, data_dir)
    symbol = symbol.upper().replace("/", "").replace(" ", "")
    if isinstance(prov, CSVProvider):
        lowest = min(tfs, key=lambda t: t.minutes)
        if lowest not in prov.existing(symbol, tfs):
            expected = ", ".join(f"{symbol}_{tf.value}.csv" for tf in tfs)
            raise ValueError(
                f"Entry timeframe file {symbol}_{lowest.value}.csv not found in {data_dir!r} "
                f"(expected one of: {expected})."
            )
    eng = DecisionEngine(prov, calendar=known_empty_calendar() if no_news_penalty else None)
    try:
        decision = eng.analyze(symbol, timeframes=tfs)
    except KeyError as e:
        raise ValueError(str(e))
    except (ProviderError, DataError) as e:
        raise ValueError(str(e))
    return {
        "decision": decision_to_dict(decision),
        "text": format_decision(decision),
        "meta": {"symbol": symbol, "source": source, "seed": seed,
                 "timeframes": [t.value for t in tfs]},
    }


def run_scenario(name: str, seed: int = 1, without: Optional[List[str]] = None) -> Dict[str, Any]:
    df, symbol = scenario_frame(name, seed, without)
    eng = DecisionEngine(FrameProvider({symbol: {Timeframe.M15: df}}), calendar=known_empty_calendar())
    decision = eng.analyze(symbol, timeframes=list(DEFAULT_TFS))
    meta = next((s for s in SCENARIOS if s["id"] == name), {"id": name})
    return {
        "decision": decision_to_dict(decision),
        "text": format_decision(decision),
        "meta": {"scenario": name, "symbol": symbol, "seed": seed, "without": without or [],
                 "expected": meta.get("expected")},
    }


# ------------------------------------------------------------------ candles (charting)

def get_candles(symbol: str, timeframe: str = "15m", limit: int = 300,
                source: str = "synthetic", seed: int = 1,
                scenario: Optional[str] = None, data_dir: str = "data") -> Dict[str, Any]:
    try:
        tf = Timeframe(timeframe)
    except ValueError:
        raise ValueError(f"Unknown timeframe {timeframe!r}.")
    limit = max(10, min(int(limit), 2000))
    symbol = symbol.upper().replace("/", "").replace(" ", "")

    actual_tf = tf
    note = None
    if source == "scenario" or scenario:
        df, sym = scenario_frame(scenario or "textbook_long", seed)
        symbol = sym
        if tf == Timeframe.M15:
            frame = df
        elif tf.minutes > Timeframe.M15.minutes:
            frame = resample_ohlcv(df, Timeframe.M15, tf)
        else:
            frame = df
            actual_tf = Timeframe.M15
            note = f"Scenario data is M15; showing M15 instead of {tf.value}."
    else:
        prov = build_provider(source, seed, data_dir)
        try:
            frame = prov.get_ohlcv(symbol, tf, limit)
        except (ProviderError, DataError) as e:
            raise ValueError(str(e))
    frame = frame.tail(limit)
    candles = []
    for ts, row in frame.iterrows():
        t = pd.Timestamp(ts)
        t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
        candles.append({
            "time": int(t.timestamp()),
            "iso": t.isoformat(),
            "open": _safe_float(row.get("open")),
            "high": _safe_float(row.get("high")),
            "low": _safe_float(row.get("low")),
            "close": _safe_float(row.get("close")),
            "volume": _safe_float(row.get("volume", 0)) or 0,
        })
    return {"symbol": symbol, "timeframe": actual_tf.value,
            "requested_timeframe": tf.value, "source": source, "seed": seed,
            "count": len(candles), "note": note, "candles": candles}


# ------------------------------------------------------------------ local data files (browser CSV upload)
#
# The engine treats the data/ directory as its history store: {SYMBOL}_{tf}.csv.
# These helpers let the browser UI upload real broker/MT4/export history into it
# so users can run REAL analysis without touching the server filesystem.

_CSV_NAME_RE = re.compile(r"(?i)^([a-z0-9]{1,16})_(1m|5m|15m|30m|1h|4h|1d|1w|1M)\.csv$")
_MAX_UPLOAD_BYTES = 80 * 1024 * 1024


def _parse_csv_name(filename: str) -> Optional[tuple]:
    """Return (symbol, Timeframe) for a valid upload name like ``EURUSD_15m.csv``."""
    m = _CSV_NAME_RE.match(os.path.basename(filename or "").strip())
    if not m:
        return None
    tf_part = m.group(2)
    tf = Timeframe.MN1 if tf_part.upper() == "1M" else Timeframe(tf_part.lower())
    return m.group(1).upper(), tf


def list_datasets(data_dir: str = "data") -> Dict[str, Any]:
    """History files currently on disk that the ``csv`` source can serve."""
    items: List[Dict[str, Any]] = []
    if os.path.isdir(data_dir):
        for fn in sorted(os.listdir(data_dir)):
            parsed = _parse_csv_name(fn)
            if not parsed:
                continue
            path = os.path.join(data_dir, fn)
            try:
                size = os.path.getsize(path)
                mod = os.path.getmtime(path)
            except OSError:
                continue
            items.append({"file": fn, "symbol": parsed[0], "timeframe": parsed[1].value,
                          "size_kb": round(size / 1024, 1),
                          "modified": datetime.fromtimestamp(mod, timezone.utc).isoformat()})
    return {"count": len(items), "datasets": items}


def save_uploaded_csv(filename: str, content: bytes, data_dir: str = "data") -> Dict[str, Any]:
    """Validate + store an uploaded OHLCV CSV so the ``csv`` source can serve it.

    Accepts any dialect the engine already reads (comma/tab/semicolon/pipe,
    MT4/5 headers, split DATE+TIME, epoch, ...). Returns a summary incl. the
    number of bars and the date range actually imported.
    """
    if not content:
        raise ValueError("The uploaded file is empty.")
    if len(content) > _MAX_UPLOAD_BYTES:
        raise ValueError(f"File too large ({len(content) / 1048576:.1f} MB, max {_MAX_UPLOAD_BYTES // 1048576} MB).")
    parsed = _parse_csv_name(filename)
    if parsed is None:
        raise ValueError("Filename must look like SYMBOL_TIMEFRAME.csv — e.g. EURUSD_15m.csv or XAUUSD_1h.csv "
                         "(timeframes: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M).")
    symbol, tf = parsed
    try:
        get_instrument(symbol)
    except KeyError as e:
        from ai_trader.config.instruments import symbols as known_symbols
        raise ValueError(f"Unknown instrument {symbol!r} — the engine can only analyse its registry "
                         f"(see the Instruments page): {', '.join(known_symbols())}.") from e

    from ai_trader.data.providers import CSVProvider, normalise_ohlcv, read_csv_smart
    os.makedirs(data_dir, exist_ok=True)
    dest = os.path.join(data_dir, f"{symbol}_{tf.value}.csv")
    tmp = dest + ".uploading"
    try:
        with open(tmp, "wb") as fh:
            fh.write(content)
        df = normalise_ohlcv(read_csv_smart(tmp))
    except DataError as e:
        raise ValueError(str(e)) from e
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    if len(df) < 60:
        raise ValueError(f"Only {len(df)} bars in the file — at least 60 are needed; "
                         "several hundred are recommended for reliable multi-timeframe context.")
    df.to_csv(tmp, index_label="ts")
    os.replace(tmp, dest)  # atomic: never leave a half-written dataset behind
    first = pd.Timestamp(df.index[0]).isoformat()
    last = pd.Timestamp(df.index[-1]).isoformat()
    return {"file": os.path.basename(dest), "symbol": symbol, "timeframe": tf.value,
            "bars": int(len(df)), "first": first, "last": last, "path": dest}


# ------------------------------------------------------------------ backtest

def _equity_curve(trades: List[Dict[str, Any]], start_equity: float) -> List[Dict[str, Any]]:
    pts = [{"t": None, "equity": round(start_equity, 2)}]
    eq = start_equity
    for t in sorted(trades, key=lambda r: str(r.get("exit_time") or "")):
        try:
            eq += float(t.get("pnl") or 0)
        except (TypeError, ValueError):
            continue
        pts.append({"t": t.get("exit_time"), "equity": round(eq, 2),
                    "pnl": t.get("pnl"), "r": t.get("r")})
    return pts


def run_backtest(params: Dict[str, Any],
                 progress: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
    symbol = str(params.get("symbol", "XAUUSD")).upper().replace("/", "").replace(" ", "")
    source = str(params.get("source", "scenario")).lower()
    seed = int(params.get("seed", 1))
    tfs = parse_timeframes(params.get("timeframes"))
    data_dir = str(params.get("data_dir", "data"))
    warmup = int(params.get("warmup", 400))

    start = end = None
    if source == "scenario":
        cfg = ScenarioConfig(direction=str(params.get("direction", "long")),
                             seed=seed, resolve=params.get("resolve") or "win")
        df = textbook_pullback_setup(cfg)
        warmup = int(df.attrs["scenario"]["setup_end"]) - 40
        prov, symbol = FrameProvider({"XAUUSD": {Timeframe.M15: df}}), "XAUUSD"
    elif source == "synthetic":
        bars = max(50, min(int(params.get("bars", 600)), 20000))
        prov = SyntheticProvider(seed=seed, base_bars=max(20_000, bars * 3 * 4))
        end = datetime.now(timezone.utc)
        full = prov.get_ohlcv(symbol, min(tfs, key=lambda t: t.minutes), 10 ** 7, end=end)
        if len(full) < bars:
            raise ValueError(f"Synthetic provider returned {len(full)} bars, need {bars}.")
        start = full.index[-bars].to_pydatetime()
    elif source == "csv":
        prov = CSVProvider(data_dir)
        if not prov.available():
            raise ValueError(f"CSV directory {data_dir!r} not found.")
        lowest = min(tfs, key=lambda t: t.minutes)
        if lowest not in prov.existing(symbol, tfs):
            raise ValueError(f"Entry timeframe file {symbol}_{lowest.value}.csv not found in {data_dir!r}.")
        start = _parse_dt(params.get("start"), "start")
        end = _parse_dt(params.get("end"), "end")
    else:
        raise ValueError(f"Unknown backtest source {source!r}. Use scenario | synthetic | csv.")

    spread = params.get("spread")
    costs = ExecutionCosts(spread_points=float(spread) if spread is not None else None,
                           commission_per_lot_side=float(params.get("commission", 0.0)))
    rules = ManagementRules(order_ttl_bars=int(params.get("ttl", 8)),
                            max_hold_bars=int(params.get("max_hold", 96)))
    equity = params.get("equity")
    cfg = BacktestConfig(symbol, timeframes=tfs, start=start, end=end, warmup_bars=warmup,
                         costs=costs, rules=rules,
                         analyze_every=max(1, int(params.get("every", 1))),
                         start_equity=float(equity) if equity else None,
                         label=str(params.get("label") or source))
    cal = known_empty_calendar() if params.get("no_news_penalty") else None
    walk_forward = bool(params.get("walk_forward"))

    start_equity = cfg.start_equity if cfg.start_equity is not None else get_settings().risk.account_equity

    split = _parse_dt(params.get("split"), "split")
    if split:
        sr = in_out_of_sample(prov, cfg, split, calendar=cal)
        parts = []
        for res in (sr.in_sample, sr.out_of_sample):
            wf = walk_forward_thresholds(res.candidates_frame()) if walk_forward else None
            ov = overfit_report(res, wf)
            d = result_to_dict(res, wf, ov)
            d["equity_curve"] = _equity_curve(d.get("trades", []), start_equity)
            parts.append({"result": d, "text": format_backtest(res, wf, ov)})
        return {"mode": "split", "in_sample": parts[0], "out_of_sample": parts[1],
                "degradation": sr.degradation, "notes": sr.notes}

    bt = Backtester(prov, cfg, calendar=cal, progress=progress)
    res = bt.run()
    wf = walk_forward_thresholds(res.candidates_frame()) if walk_forward else None
    ov = overfit_report(res, wf)
    d = result_to_dict(res, wf, ov)
    d["equity_curve"] = _equity_curve(d.get("trades", []), start_equity)
    return {"mode": "single", "result": d, "text": format_backtest(res, wf, ov)}


# ------------------------------------------------------------------ reference data

def list_instruments(asset_class: Optional[str] = None) -> List[Dict[str, Any]]:
    ac = None
    if asset_class:
        try:
            ac = AssetClass(asset_class)
        except ValueError:
            raise ValueError(f"Unknown asset class {asset_class!r}. "
                             f"Valid: {[a.value for a in AssetClass]}")
    out = []
    for inst in all_instruments(ac):
        out.append({
            "symbol": inst.symbol, "name": inst.name,
            "asset_class": inst.asset_class.value,
            "pip_size": inst.pip_size, "digits": inst.digits,
            "lot_size": inst.lot_size, "min_lot": inst.min_lot,
            "spread_points": inst.spread_points,
            "sessions": [s.value if hasattr(s, "value") else str(s) for s in inst.sessions],
            "is_synthetic": inst.is_synthetic,
            "aliases": list(inst.aliases or ()),
        })
    return out


def settings_view() -> Dict[str, Any]:
    s = get_settings()
    return {
        "environment": s.environment,
        "execution_mode": s.execution_mode,
        "default_provider": s.default_provider,
        "data_dir": s.data_dir,
        "scoring_weights": s.scoring.as_dict(),
        "thresholds": {**s.thresholds.model_dump(), "effective_min_score": s.thresholds.effective_min_score},
        "risk": s.risk.model_dump(),
        "news": s.news.model_dump(),
        "regime_map": {r.value if hasattr(r, "value") else str(r):
                       [f.value if hasattr(f, "value") else str(f) for f in fams]
                       for r, fams in s.regime_map.mapping.items()},
    }

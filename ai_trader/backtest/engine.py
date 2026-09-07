"""
Event-driven backtester that replays the *production* DecisionEngine bar by bar.

Why replay the real engine instead of a separate "backtest strategy"?
  * zero logic drift: what is tested is exactly what would trade,
  * every causal safeguard of the data layer (closed bars only, confirmed
    swings, drop-incomplete resampling) applies automatically,
  * the RiskManager is part of the loop, so daily-loss kill-switches,
    cool-downs and size reduction after losses are all *inside* the results.

Anti-look-ahead: before every decision the loaded frames are checked — the
last bar of every timeframe must have CLOSED at or before the decision time,
otherwise the run aborts with ``LookaheadError``.

Besides the portfolio simulation the backtester records EVERY strategy
candidate the engine produced (including the rejected ones, with score, major
/ minor conflicts and geometry) and evaluates its hypothetical outcome with an
isolated simulator. This is what turns "min score 80" from an opinion into a
number that can be checked on out-of-sample data (see ``validation.py``).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ai_trader.analysis.indicators import atr as _atr
from ai_trader.config.instruments import get_instrument
from ai_trader.config.settings import Settings, get_settings
from ai_trader.core.enums import AssetClass, DecisionType, Direction, EntryType, Timeframe
from ai_trader.core.models import Decision
from ai_trader.data.calendar import EconomicCalendar
from ai_trader.data.providers import DataProvider, FrameProvider
from ai_trader.data.resampler import bar_close_time
from ai_trader.decision.engine import DecisionEngine
from ai_trader.risk.manager import RiskManager

from .metrics import Metrics, MonteCarloResult, compute_metrics, metrics_by, monte_carlo
from .simulator import BrokerSimulator, ClosedTrade, ExecutionCosts, ManagementRules


class LookaheadError(RuntimeError):
    pass


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def snapshot_provider(provider: DataProvider, symbol: str, timeframes: List[Timeframe]) -> FrameProvider:
    """Pull the full history ONCE and serve the replay from memory.

    Live providers regenerate/refetch per ``end`` (the synthetic generator would rebuild its
    whole path on every bar); a frozen snapshot makes the replay fast *and* guarantees that
    every decision sees the same underlying series. Natively available higher timeframes are
    kept (real D1/H4 files usually reach further back than intraday data); missing ones are
    causally resampled from the lowest timeframe by ``FrameProvider``."""
    frames: Dict[Timeframe, pd.DataFrame] = {}
    for tf in sorted(timeframes, key=lambda t: t.minutes):
        try:
            df = provider.get_ohlcv(symbol, tf, limit=10_000_000)
        except Exception:
            continue
        if df is None or len(df) == 0:
            continue
        df = df.drop(columns=[c for c in df.columns if c == "label"]).copy()
        df.attrs = {}
        frames[tf] = df
    if not frames:
        raise ValueError(f"{symbol}: provider returned no data for {[t.value for t in timeframes]}")
    return FrameProvider({symbol.upper(): frames})


@dataclass
class BacktestConfig:
    symbol: str
    timeframes: List[Timeframe] = field(default_factory=lambda: [Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1])
    start: Optional[datetime] = None          # first decision time (inclusive); None → after warm-up
    end: Optional[datetime] = None            # last bar to process
    warmup_bars: int = 400                    # entry-TF bars skipped before the first decision
    analyze_in_position: bool = False         # re-run the engine while a position/pending order exists
    analyze_every: int = 1                    # decide on every k-th entry bar (1 = every bar)
    limits: Optional[Dict[Timeframe, int]] = None   # bars per timeframe handed to the engine
    costs: ExecutionCosts = field(default_factory=ExecutionCosts)
    rules: ManagementRules = field(default_factory=ManagementRules)
    record_candidates: bool = True
    candidate_min_score: float = 40.0
    setup_cluster_bars: int = 12              # same strategy/direction/stop within this many bars → same setup_id
    start_equity: Optional[float] = None
    label: str = ""
    strict_causality: bool = True


@dataclass
class CandidateRecord:
    """One strategy candidate as the engine saw it on one bar. Consecutive bars usually
    re-propose the same setup with an evolving score; such records share a ``setup_id`` so
    validation can count each setup once ("the first bar it would have been taken")."""
    bar: int
    ts: datetime
    setup_id: int
    strategy: str
    family: str
    direction: str
    score: float
    grade: str
    n_major: int
    n_minor: int
    rr_tp2: float
    regime: str
    regime_confidence: float
    passed: bool
    rejected_because: str
    entry_type: str
    entry_low: float
    entry_high: float
    stop: float
    targets: List[float]
    conflicts: List[str] = field(default_factory=list)
    # hypothetical outcome (isolated simulation, 1 unit)
    filled: bool = False
    r_multiple: Optional[float] = None
    exit_reason: str = ""
    bars_held: int = 0
    mae_r: float = 0.0
    mfe_r: float = 0.0

    def to_dict(self) -> Dict:
        d = dict(self.__dict__)
        d["ts"] = self.ts.isoformat()
        return d


@dataclass
class BacktestResult:
    config: BacktestConfig
    trades: List[ClosedTrade]
    candidates: List[CandidateRecord]
    metrics: Metrics
    by_regime: Dict[str, Metrics]
    by_strategy: Dict[str, Metrics]
    by_grade: Dict[str, Metrics]
    by_direction: Dict[str, Metrics]
    monte_carlo: Optional[MonteCarloResult]
    equity_curve: pd.Series
    decision_counts: Dict[str, int]
    block_reasons: Dict[str, int]
    regime_counts: Dict[str, int]
    bars_processed: int
    decisions_made: int
    first_ts: Optional[datetime]
    last_ts: Optional[datetime]
    runtime_s: float
    data_source: str
    warnings: List[str] = field(default_factory=list)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    def candidates_frame(self) -> pd.DataFrame:
        if not self.candidates:
            return pd.DataFrame()
        df = pd.DataFrame([c.to_dict() for c in self.candidates])
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df

    def trades_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        rows = []
        for t in self.trades:
            rows.append({"id": t.id, "entry_time": t.entry_time, "exit_time": t.exit_time, "direction": t.direction, "strategy": t.strategy,
                         "regime": t.regime, "score": t.score, "grade": t.grade, "entry": t.entry_price, "exit": t.exit_price,
                         "stop": t.initial_stop, "tp1": t.targets[0] if t.targets else None, "tp2": t.targets[1] if len(t.targets) > 1 else None,
                         "tp3": t.targets[2] if len(t.targets) > 2 else None, "lots": t.lots, "risk": t.risk_amount, "pnl": t.pnl,
                         "costs": t.costs, "r": t.r_multiple, "bars": t.bars_held, "exit_reason": t.exit_reason, "tp_hit": t.tp_hit,
                         "mae_r": t.mae_r, "mfe_r": t.mfe_r})
        return pd.DataFrame(rows)


class Backtester:
    def __init__(self, provider: DataProvider, config: BacktestConfig, settings: Optional[Settings] = None,
                 calendar: Optional[EconomicCalendar] = None, macro_series: Optional[Dict[str, pd.Series]] = None,
                 progress: Optional[Callable[[int, int], None]] = None):
        self.source = provider
        self.source_name = getattr(provider, "name", type(provider).__name__)
        self.provider = provider if isinstance(provider, FrameProvider) else snapshot_provider(provider, config.symbol, config.timeframes)
        self.cfg = config
        self.settings = settings or get_settings()
        if config.start_equity is not None:
            self.settings = self.settings.model_copy(deep=True)
            self.settings.risk.account_equity = config.start_equity
        self.calendar = calendar
        self.macro_series = macro_series
        self.progress = progress
        self.inst = get_instrument(config.symbol)
        self.risk = RiskManager(self.settings.risk)
        self.engine = DecisionEngine(self.provider, settings=self.settings, calendar=calendar, risk=self.risk,
                                     macro_series=macro_series, cache_analysis=True)

    # ------------------------------------------------------------------ run
    def run(self) -> BacktestResult:
        t0 = time.time()
        cfg = self.cfg
        inst = self.inst
        entry_tf = min(cfg.timeframes, key=lambda t: t.minutes)
        frame = self.provider.get_ohlcv(inst.symbol, entry_tf, limit=10_000_000)
        frame = frame.drop(columns=[c for c in frame.columns if c == "label"])
        if cfg.end is not None:
            frame = frame[frame.index <= _utc(cfg.end)]
        n = len(frame)
        idx = frame.index
        o, h, l, c = (frame[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
        atr_arr = _atr(frame, self.settings.analysis.atr_period).bfill().to_numpy(dtype=float)
        tf_minutes = entry_tf.minutes
        close_ts = [ts.to_pydatetime() + pd.Timedelta(minutes=tf_minutes) for ts in idx]
        i0 = cfg.warmup_bars
        if cfg.start is not None:
            i0 = max(i0, int(np.searchsorted(idx.values, _utc(cfg.start).to_datetime64())))
        i0 = min(i0, n)
        sim = BrokerSimulator(inst, cfg.costs, cfg.rules)
        sessions_24_7 = inst.is_synthetic or inst.asset_class is AssetClass.CRYPTO
        trades: List[ClosedTrade] = []
        cands: List[CandidateRecord] = []
        recent_cands: Dict[tuple, List[CandidateRecord]] = {}
        self._next_setup_id = 1
        decision_counts: Dict[str, int] = {"TRADE": 0, "NO_TRADE": 0, "WAIT": 0, "SKIPPED_IN_POSITION": 0, "SKIPPED_STEP": 0}
        block_reasons: Dict[str, int] = {}
        regime_counts: Dict[str, int] = {}
        warnings: List[str] = []
        decisions = 0
        first_ts = last_ts = None
        open_ids: set = set()

        for i in range(i0, n):
            ts = close_ts[i]
            # 1) market moves through bar i → fills / exits of orders created before bar i
            closed_now = sim.on_bar(i, ts, o[i], h[i], l[i], c[i], atr_arr[i])
            for pos in sim.positions:
                if pos.id not in open_ids:
                    open_ids.add(pos.id)
                    self.risk.register_open(inst.symbol, pos.direction, pos.risk_amount, pos.units, now=ts)
            for ct in closed_now:
                if ct.id not in open_ids:          # filled and closed within the same bar
                    self.risk.register_open(inst.symbol, Direction(ct.direction), ct.risk_amount, ct.units, now=ts)
                open_ids.discard(ct.id)
                self.risk.register_close(inst.symbol, ct.pnl, now=ts)
                trades.append(ct)
            # 2) decision on the close of bar i
            if sim.has_open and not cfg.analyze_in_position:
                decision_counts["SKIPPED_IN_POSITION"] += 1
                continue
            if cfg.analyze_every > 1 and (i - i0) % cfg.analyze_every:
                decision_counts["SKIPPED_STEP"] += 1
                continue
            load = self.engine.loader.load(inst.symbol, timeframes=cfg.timeframes, as_of=ts, limits=cfg.limits, sessions_24_7=sessions_24_7)
            if cfg.strict_causality:
                for tf, df in load.data.frames.items():
                    if len(df) and bar_close_time(df.index[-1], tf) > pd.Timestamp(ts):
                        raise LookaheadError(f"{tf.value} bar opened {df.index[-1]} has not closed at decision time {ts}")
            d = self.engine.decide(load, inst, ts)
            decisions += 1
            first_ts = first_ts or ts
            last_ts = ts
            self._log_decision(d, decision_counts, block_reasons, regime_counts)
            if cfg.record_candidates:
                self._record_candidates(d, i, ts, cands, recent_cands)
            if d.is_trade and d.plan is not None:
                p = d.plan
                sim.cancel_all_pending("replaced by newer plan")
                sim.submit(p.direction, p.entry_type, p.entry_low, p.entry_high, p.stop, [p.tp1, p.tp2, p.tp3],
                           p.risk.units, p.risk.lots, p.risk.risk_amount, i, ts,
                           meta={"strategy": p.strategy, "regime": p.regime.primary.value, "score": p.score, "grade": p.grade.value,
                                 "regime_confidence": round(p.regime.confidence, 3), "factors": len(p.factors),
                                 "minor_conflicts": len(p.confluence.minor_conflicts)})
            if self.progress and (i - i0) % 50 == 0:
                self.progress(i - i0, n - i0)
        # end of data: flatten
        if n:
            for ct in sim.force_close_all(n - 1, close_ts[-1], c[-1]):
                self.risk.register_close(inst.symbol, ct.pnl, now=close_ts[-1])
                trades.append(ct)
        # 3) hypothetical outcome of every recorded candidate
        if cands:
            self._evaluate_candidates(cands, o, h, l, c, atr_arr, close_ts)
        start_eq = self.settings.risk.account_equity
        bars_per_year = (365 if sessions_24_7 else 252) * (1440 / tf_minutes)
        metrics = compute_metrics(trades, start_eq, bars_per_year)
        mc = monte_carlo(trades) if len(trades) >= 5 else None
        curve = _equity_curve(trades, start_eq)
        src = self.source_name
        if src in ("synthetic", "frames"):
            warnings.append("DATA IS SYNTHETIC/SCRIPTED — results validate mechanics only and say nothing about real-market edge")
        if len(trades) < 30:
            warnings.append(f"{len(trades)} trades: far too few for statistical conclusions (need ≥ 30, ideally ≥ 100)")
        return BacktestResult(cfg, trades, cands, metrics, metrics_by(trades, "regime", start_eq), metrics_by(trades, "strategy", start_eq),
                              metrics_by(trades, "grade", start_eq), metrics_by(trades, "direction", start_eq), mc, curve,
                              decision_counts, dict(sorted(block_reasons.items(), key=lambda kv: -kv[1])),
                              dict(sorted(regime_counts.items(), key=lambda kv: -kv[1])), max(0, n - i0), decisions, first_ts, last_ts,
                              time.time() - t0, src, warnings)

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _log_decision(d: Decision, counts: Dict[str, int], blocks: Dict[str, int], regimes: Dict[str, int]) -> None:
        from ai_trader import WAIT_MESSAGE
        if d.is_trade:
            counts["TRADE"] += 1
        elif d.message == WAIT_MESSAGE:
            counts["WAIT"] += 1
        else:
            counts["NO_TRADE"] += 1
        for r in d.reasons:
            key = r.split(":")[0].strip()
            if key.startswith("no strategy in"):
                key = "no strategy produced a candidate"
            elif key.startswith("no candidate reached"):
                key = "no candidate reached threshold without major conflicts"
            elif key.startswith("passing candidates disagree"):
                key = "direction conflict among passing candidates"
            blocks[key] = blocks.get(key, 0) + 1
        reg = (d.context or {}).get("regime", {}).get("primary")
        if reg:
            regimes[reg] = regimes.get(reg, 0) + 1

    def _record_candidates(self, d: Decision, bar: int, ts: datetime, out: List[CandidateRecord],
                           recent: Dict[tuple, List[CandidateRecord]]) -> None:
        reg = (d.context or {}).get("regime", {})
        win = self.cfg.setup_cluster_bars
        for cnd in d.candidates:
            if "stop" not in cnd or cnd.get("score", 0) < self.cfg.candidate_min_score:
                continue
            key = (cnd["strategy"], cnd["direction"])
            prev = [p for p in recent.get(key, []) if bar - p.bar <= win]
            risk_ref = abs(cnd["entry_high"] - cnd["stop"]) or 1e-9
            same = next((p for p in reversed(prev) if abs(p.stop - cnd["stop"]) <= 0.25 * risk_ref), None)
            setup_id = same.setup_id if same is not None else self._next_setup_id
            if same is None:
                self._next_setup_id += 1
            rec = CandidateRecord(bar, ts, setup_id, cnd["strategy"], cnd.get("family", "?"), cnd["direction"], float(cnd["score"]), cnd.get("grade", "?"),
                                  int(cnd.get("n_major", 0)), int(cnd.get("n_minor", 0)), float(cnd.get("rr_tp2", 0.0)),
                                  reg.get("primary", "?"), float(reg.get("confidence", 0.0)), cnd.get("rejected_because") == "passed",
                                  cnd.get("rejected_because", ""), cnd.get("entry_type", "Limit"), float(cnd["entry_low"]), float(cnd["entry_high"]),
                                  float(cnd["stop"]), [float(t) for t in cnd["targets"]], list(cnd.get("conflicts", [])))
            out.append(rec)
            recent[key] = prev + [rec]

    def _evaluate_candidates(self, cands: List[CandidateRecord], o, h, l, c, atr_arr, close_ts) -> None:
        n = len(o)
        rules = self.cfg.rules
        horizon = rules.order_ttl_bars + rules.max_hold_bars + 2
        for rec in cands:
            sim = BrokerSimulator(self.inst, self.cfg.costs, rules)
            direction = Direction(rec.direction)
            mid = (rec.entry_low + rec.entry_high) / 2
            risk_amount = abs(mid - rec.stop) * self.inst.point_value      # 1 unit, no commission (lots=0)
            sim.submit(direction, EntryType(rec.entry_type) if rec.entry_type in [e.value for e in EntryType] else EntryType.LIMIT,
                       rec.entry_low, rec.entry_high, rec.stop, rec.targets, 1.0, 0.0, risk_amount, rec.bar, rec.ts,
                       meta={"strategy": rec.strategy})
            closed: List[ClosedTrade] = []
            for j in range(rec.bar + 1, min(n, rec.bar + horizon)):
                closed = sim.on_bar(j, close_ts[j], o[j], h[j], l[j], c[j], atr_arr[j])
                if closed or (not sim.has_open):
                    break
            if not closed and sim.positions:
                closed = sim.force_close_all(n - 1, close_ts[-1], c[-1], "end of data")
            if closed:
                ct = closed[0]
                rec.filled = True
                rec.r_multiple = float(ct.r_multiple)
                rec.exit_reason = ct.exit_reason
                rec.bars_held = ct.bars_held
                rec.mae_r, rec.mfe_r = ct.mae_r, ct.mfe_r


def _equity_curve(trades: List[ClosedTrade], start: float) -> pd.Series:
    if not trades:
        return pd.Series(dtype=float)
    ts = sorted(trades, key=lambda t: t.exit_time)
    return pd.Series(start + np.cumsum([t.pnl for t in ts]), index=pd.DatetimeIndex([t.exit_time for t in ts]))

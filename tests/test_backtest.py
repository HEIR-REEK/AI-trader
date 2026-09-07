"""Phase 7 — backtesting: broker simulator, metrics, Monte Carlo, validation, end-to-end replay."""
from __future__ import annotations

import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from ai_trader.backtest import (
    BacktestConfig,
    Backtester,
    BrokerSimulator,
    ClosedTrade,
    ExecutionCosts,
    LookaheadError,
    ManagementRules,
    compute_metrics,
    monte_carlo,
    overfit_report,
    score_bucket_table,
    select_threshold,
    threshold_table,
    walk_forward_thresholds,
)
from ai_trader.backtest.report import format_backtest, result_to_dict
from ai_trader.config.instruments import get_instrument
from ai_trader.core.enums import Direction, EntryType, Timeframe
from ai_trader.data.calendar import EconomicCalendar
from ai_trader.data.providers import FrameProvider
from ai_trader.data.scenarios import ScenarioConfig, textbook_pullback_setup

warnings.filterwarnings("ignore")
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
XAU = get_instrument("XAUUSD")
NO_SLIP = ExecutionCosts(slippage_atr=0.0, stop_slippage_atr=0.0)
TFS = [Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1]


def _run(bars, direction=Direction.LONG, entry=(1999.5, 2000.5), stop=1995.0, tps=(2010.0, 2020.0, 2030.0),
         etype=EntryType.LIMIT, costs=NO_SLIP, rules=None):
    sim = BrokerSimulator(XAU, costs, rules or ManagementRules(order_ttl_bars=5, max_hold_bars=20))
    sim.submit(direction, etype, entry[0], entry[1], stop, list(tps), 10.0, 0.1, 50.0, 0, T0, meta={"strategy": "t", "regime": "R", "score": 85, "grade": "A"})
    closed = []
    for i, (o, h, l, c) in enumerate(bars, start=1):
        closed += sim.on_bar(i, T0 + timedelta(minutes=15 * i), o, h, l, c, 5.0)
    return sim, closed


# ---------------------------------------------------------------- simulator
def test_no_same_bar_fill_and_limit_fills_on_touch():
    sim = BrokerSimulator(XAU, NO_SLIP, ManagementRules(order_ttl_bars=5))
    sim.submit(Direction.LONG, EntryType.LIMIT, 1999.5, 2000.5, 1995.0, [2010, 2020, 2030], 10, 0.1, 50, 0, T0)
    # order created on bar 0 → bar 0 is never processed by the caller; first processed bar is 1
    sim.on_bar(1, T0, 2003, 2004, 2001, 2002, 5.0)          # does not touch the zone
    assert sim.pending and not sim.positions
    sim.on_bar(2, T0, 2002, 2003, 2000.4, 2001, 5.0)        # touches zone top → filled at zone top + half spread
    assert sim.positions and not sim.pending
    assert sim.positions[0].entry_price == pytest.approx(2000.5 + XAU.spread_points / 2)


def test_full_target_ladder_and_partial_exits():
    sim, cl = _run([(2003, 2004, 1999, 2001), (2001, 2011, 2000, 2010.5), (2010.5, 2021, 2009, 2020.5), (2020.5, 2031, 2019, 2030.5)])
    t = cl[0]
    assert t.exit_reason == "TP3" and t.tp_hit == 3
    # 1/3 at each target → VWAP exit ≈ 2020 minus half-spread; R multiple ≈ (20 - spread)/risk
    assert 3.5 < t.r_multiple < 4.0
    assert t.exit_price == pytest.approx(2020 - XAU.spread_points / 2)
    assert t.pnl > 0 and t.bars_held == 3                       # filled on bar 1, final exit on bar 4


def test_stop_first_when_stop_and_target_touch_in_same_bar():
    _, cl = _run([(2003, 2004, 1999, 2001), (2001, 2012, 1990, 2005)])
    assert cl[0].exit_reason == "stop loss"
    assert cl[0].r_multiple < -1.0                    # spread makes a stop-out slightly worse than -1R


def test_gap_beyond_target_fills_at_open():
    _, cl = _run([(2003, 2004, 1999, 2001), (2012, 2013, 1990, 1991)])   # opens above TP1, then collapses
    t = cl[0]
    assert t.tp_hit >= 1                                       # TP1 filled at the open before the stop
    assert t.exit_reason.startswith("break-even")


def test_break_even_after_tp1_protects_capital():
    _, cl = _run([(2003, 2004, 1999, 2001), (2001, 2011, 2000, 2010.5), (2010.5, 2011, 1998, 1999)])
    t = cl[0]
    assert t.tp_hit == 1 and t.exit_reason == "break-even stop (after TP1)" and t.pnl > 0


def test_pending_expiry_and_invalidation():
    sim, cl = _run([(2003, 2005, 2002, 2004)] * 7)             # never touches → expires after ttl
    assert cl == [] and not sim.pending and not sim.positions
    sim, cl = _run([(2003, 2004, 2002, 2003), (2002, 2003, 1994, 1996)])   # stop trades *and* zone → fill then stop
    assert cl and cl[0].exit_reason == "stop loss"
    sim2 = BrokerSimulator(XAU, NO_SLIP, ManagementRules())
    sim2.submit(Direction.LONG, EntryType.LIMIT, 1999.5, 2000.5, 1995.0, [2010, 2020, 2030], 10, 0.1, 50, 0, T0)
    sim2.on_bar(1, T0, 1990, 1992, 1985, 1991, 5.0)             # gaps through zone AND stop
    assert not sim2.pending and sim2.closed and sim2.closed[0].exit_reason == "gapped through stop at fill"
    assert -0.2 < sim2.closed[0].r_multiple <= 0


def test_market_order_costs_and_slippage():
    costs = ExecutionCosts(slippage_atr=0.05, stop_slippage_atr=0.08)
    _, cl = _run([(2003, 2004, 2002, 2003.5), (2003.5, 2004, 1990, 1991)], etype=EntryType.MARKET, costs=costs)
    t = cl[0]
    assert t.entry_price == pytest.approx(2003 + XAU.spread_points / 2 + 0.05 * 5.0)
    assert t.exit_price == pytest.approx(1995 - 0.08 * 5.0 - XAU.spread_points / 2)
    assert t.costs == 0.0
    _, cl2 = _run([(2003, 2004, 2002, 2003.5), (2003.5, 2004, 1990, 1991)], etype=EntryType.MARKET,
                  costs=ExecutionCosts(slippage_atr=0.05, stop_slippage_atr=0.08, commission_per_lot_side=7.0))
    assert cl2[0].costs == pytest.approx(1.4)                 # 0.1 lot × 7 × 2 sides
    assert cl2[0].pnl == pytest.approx(cl[0].pnl - 1.4)


def test_short_side_mirror_and_time_stop():
    _, cl = _run([(1997, 2001, 1996, 1999), (1999, 2000, 1989, 1989.5), (1989.5, 1991, 1979, 1979.5), (1979.5, 1981, 1969, 1969.5)],
                 direction=Direction.SHORT, stop=2005.0, tps=(1990.0, 1980.0, 1970.0))
    assert cl[0].exit_reason == "TP3" and cl[0].r_multiple > 3
    _, cl = _run([(2003, 2004, 1999, 2001)] + [(2001, 2002, 2000, 2001)] * 25)
    assert cl[0].exit_reason == "time stop" and cl[0].bars_held == 20


def test_mae_mfe_recorded_in_r():
    _, cl = _run([(2003, 2004, 1999, 2001), (2001, 2008, 1998, 2007), (2007, 2011, 2005, 2010.5), (2010.5, 2011, 1998, 1999)])
    t = cl[0]
    assert t.mfe_r >= (2011 - t.entry_price) / (t.entry_price - 1995) - 1e-9
    assert t.mae_r >= (t.entry_price - 1998) / (t.entry_price - 1995) - 1e-9


# ------------------------------------------------------------------ metrics
def _trade(r: float, i: int, strategy="s", regime="R", bars=5) -> ClosedTrade:
    risk = 100.0
    return ClosedTrade(i, "XAUUSD", "LONG", strategy, regime, 85.0, "A", T0 + timedelta(hours=i), T0 + timedelta(hours=i, minutes=15 * bars),
                       2000.0, 2000.0 + r, 1990.0, [2010, 2020, 2030], 10, 0.1, risk, r * risk, 0.0, r, bars,
                       "TP3" if r > 0 else "stop loss", 3 if r > 0 else 0, 0.3, max(r, 0.2))


def test_metrics_basic_numbers():
    rs = [2.0, -1.0, 3.0, -1.0, -1.0, 1.5]
    m = compute_metrics([_trade(r, i) for i, r in enumerate(rs)], 10_000.0, bars_per_year=252 * 96)
    assert m.trades == 6 and m.wins == 3 and m.losses == 3
    assert m.win_rate == pytest.approx(0.5)
    assert m.profit_factor == pytest.approx(6.5 / 3.0)
    assert m.expectancy_r == pytest.approx(np.mean(rs))
    assert m.max_consecutive_losses == 2
    assert m.net_pnl == pytest.approx(sum(rs) * 100)
    assert m.max_drawdown_r == pytest.approx(2.0)              # peak +4 after trade 3, trough +2 after trade 5
    assert m.notes                                             # warns about the tiny sample


def test_monte_carlo_is_deterministic_and_sane():
    rng = np.random.default_rng(3)
    rs = [float(x) for x in np.where(rng.random(60) < 0.45, 2.0, -1.0)]
    trades = [_trade(r, i) for i, r in enumerate(rs)]
    a = monte_carlo(trades, runs=500, seed=1)
    b = monte_carlo(trades, runs=500, seed=1)
    assert a.to_dict() == b.to_dict()
    assert a.median_final_r == pytest.approx(sum(rs), abs=1e-6)     # shuffling preserves the sum
    assert a.p95_max_dd_r >= a.median_max_dd_r >= 0
    assert 0.0 <= a.prob_negative <= 1.0
    c = monte_carlo(trades, runs=300, seed=1, method="bootstrap")
    assert c.p05_final_r < c.median_final_r < c.p95_final_r


# --------------------------------------------------------------- validation
def _cand_frame() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    n = 400
    score = rng.uniform(40, 95, n)
    # edge that grows with score: p(win) = 0.25 + 0.5 * (score-40)/55
    win = rng.random(n) < 0.25 + 0.5 * (score - 40) / 55
    df = pd.DataFrame({
        "bar": np.arange(n) * 10, "ts": pd.date_range("2025-01-01", periods=n, freq="6h", tz="UTC"), "setup_id": np.arange(n),
        "score": score, "n_major": (rng.random(n) < 0.2).astype(int), "rr_tp2": 2.5, "filled": True,
        "r_multiple": np.where(win, 2.0, -1.0),
    })
    return df


def test_threshold_tables_and_selection():
    df = _cand_frame()
    tab = threshold_table(df, thresholds=(50, 60, 70, 80, 90))
    assert list(tab["min_score"]) == [50, 60, 70, 80, 90]
    assert tab["n"].is_monotonic_decreasing
    buckets = score_bucket_table(df)
    assert buckets["n"].sum() == len(df)
    thr, st = select_threshold(df, thresholds=(50, 60, 70, 80, 90), min_trades=20)
    assert thr in (70, 80, 90) and st["n"] >= 20
    # falls back to the default when no threshold has enough samples
    thr2, st2 = select_threshold(df.head(10), min_trades=20, fallback=80.0)
    assert thr2 == 80.0 and st2["n"] == 0


def test_walk_forward_is_out_of_sample():
    df = _cand_frame()
    wf = walk_forward_thresholds(df, n_folds=4, thresholds=(50, 60, 70, 80, 90), min_trades=15)
    assert len(wf.folds) == 4
    for f in wf.folds:
        assert f.train_end <= f.test_start                     # never trained on the test window
    assert wf.oos_trades > 0 and wf.oos_trades == sum(f.test_stats["n"] for f in wf.folds)
    assert wf.oos_expectancy_r > 0                              # planted edge survives out-of-sample


def test_first_bar_per_setup_counts_a_setup_once():
    df = _cand_frame().head(3).copy()
    df["setup_id"] = 1                                          # same setup re-proposed on 3 bars
    df["score"] = [70.0, 85.0, 90.0]
    df["n_major"] = 0
    tab = threshold_table(df, thresholds=(80,))
    assert int(tab.loc[0, "n"]) == 1                            # counted once (first bar that qualified)


# --------------------------------------------------------------- end-to-end
def _scenario_backtest(resolve: str, direction: str = "long", **cfg_kwargs):
    df = textbook_pullback_setup(ScenarioConfig(direction=direction, resolve=resolve))
    setup_end = df.attrs["scenario"]["setup_end"]
    prov = FrameProvider({"XAUUSD": {Timeframe.M15: df}})
    cfg = BacktestConfig("XAUUSD", timeframes=TFS, warmup_bars=setup_end - 30, label=f"{direction} {resolve}", **cfg_kwargs)
    return Backtester(prov, cfg, calendar=EconomicCalendar.from_events([])).run()


@pytest.mark.slow
def test_end_to_end_textbook_win_resolves_to_targets():
    res = _scenario_backtest("win")
    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.direction == "LONG" and t.score >= 80
    assert t.exit_reason == "TP3" and t.tp_hit == 3 and t.r_multiple > 2.5
    assert res.metrics.win_rate == 1.0 and res.metrics.net_pnl > 0
    assert res.decision_counts["TRADE"] == 1 and res.decision_counts["SKIPPED_IN_POSITION"] >= 1
    assert res.by_regime and res.by_strategy
    # candidate log includes rejected candidates and the taken one
    cf = res.candidates_frame()
    assert (cf["passed"]).sum() >= 1 and (~cf["passed"]).sum() >= 1
    assert cf["filled"].any()
    txt = format_backtest(res)
    assert "PORTFOLIO" in txt and "SYNTHETIC/SCRIPTED" in txt
    d = result_to_dict(res)
    assert d["metrics"]["trades"] == 1 and d["trades"][0]["exit_reason"] == "TP3"


@pytest.mark.slow
def test_end_to_end_textbook_loss_is_stopped_and_risk_manager_books_it():
    res = _scenario_backtest("loss")
    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.exit_reason == "stop loss" and -1.35 < t.r_multiple < -1.0
    # 1 % risk per trade on 10k → loss ≈ 100 plus spread/slippage
    assert -140 < t.pnl < -100
    assert res.metrics.end_equity == pytest.approx(10_000 + t.pnl)


@pytest.mark.slow
def test_end_to_end_short_mirror():
    res = _scenario_backtest("win", direction="short")
    assert len(res.trades) == 1 and res.trades[0].direction == "SHORT" and res.trades[0].exit_reason == "TP3"


def test_lookahead_guard_rejects_unclosed_bars(monkeypatch):
    from ai_trader.backtest import engine as eng_mod
    res_frames = {}
    df = textbook_pullback_setup(ScenarioConfig(direction="long", resolve="flat"))
    setup_end = df.attrs["scenario"]["setup_end"]
    prov = FrameProvider({"XAUUSD": {Timeframe.M15: df}})
    cfg = BacktestConfig("XAUUSD", timeframes=TFS, warmup_bars=setup_end, end=df.index[setup_end + 2])
    bt = Backtester(prov, cfg, calendar=EconomicCalendar.from_events([]))
    # sabotage the loader so it hands back one bar that has not closed yet → the guard must fire
    real_load = bt.engine.loader.load

    def leaky_load(symbol, timeframes=None, as_of=None, **kw):
        lr = real_load(symbol, timeframes=timeframes, as_of=as_of, **kw)
        full = prov.get_ohlcv(symbol, Timeframe.M15, 10**6)
        lr.data.frames[Timeframe.M15] = full[full.index <= pd.Timestamp(as_of)]      # includes the bar that opens AT as_of
        return lr

    bt.engine.loader.load = leaky_load
    with pytest.raises(LookaheadError):
        bt.run()


def test_causal_replay_sees_only_closed_bars():
    df = textbook_pullback_setup(ScenarioConfig(direction="long", resolve="flat"))
    setup_end = df.attrs["scenario"]["setup_end"]
    prov = FrameProvider({"XAUUSD": {Timeframe.M15: df}})
    seen = []
    cfg = BacktestConfig("XAUUSD", timeframes=TFS, warmup_bars=setup_end, end=df.index[setup_end + 3], record_candidates=False)
    bt = Backtester(prov, cfg, calendar=EconomicCalendar.from_events([]))
    real_decide = bt.engine.decide

    def spy(load, inst, as_of):
        for tf, f in load.data.frames.items():
            seen.append((tf, f.index[-1], pd.Timestamp(as_of)))
        return real_decide(load, inst, as_of)

    bt.engine.decide = spy
    bt.run()
    assert seen
    for tf, last_open, as_of in seen:
        assert last_open + pd.Timedelta(minutes=tf.minutes) <= as_of


def test_overfit_report_flags_thin_synthetic_evidence():
    res = _scenario_backtest("flat", record_candidates=False)
    rep = overfit_report(res)
    assert rep.verdict == "NOT VALIDATED"
    assert any("synthetic" in c for c in rep.checks) and any("sample size" in c for c in rep.checks)

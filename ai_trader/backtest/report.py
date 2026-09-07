"""Plain-text / dict reporting for backtest results."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .engine import BacktestResult
from .metrics import Metrics
from .validation import OverfitReport, WalkForwardResult, conflict_table, regime_table, score_bucket_table, threshold_table


def _pf(x: float) -> str:
    return "inf" if not np.isfinite(x) else f"{x:.2f}"


def metrics_lines(m: Metrics, title: str) -> List[str]:
    out = [title, "-" * len(title)]
    if m.trades == 0:
        out.append("no closed trades")
        return out
    out += [
        f"trades {m.trades}  |  win rate {m.win_rate:.1%}  |  profit factor {_pf(m.profit_factor)}  |  expectancy {m.expectancy_r:+.2f} R ({m.expectancy_ccy:+.2f} per trade)",
        f"avg win {m.avg_win_r:+.2f} R  |  avg loss {m.avg_loss_r:+.2f} R  |  realised payoff {m.avg_rr_realised:.2f}  |  break-even win rate {m.payoff_needed_win_rate:.0%}",
        f"net P&L {m.net_pnl:+.2f} ({m.return_pct:+.2f}%)  |  costs {m.total_costs:.2f}  |  max DD {m.max_drawdown_pct:.2f}% / {m.max_drawdown_r:.1f} R",
        f"sharpe/trade {m.sharpe_per_trade:.2f}" + (f"  |  sharpe (annualised) {m.sharpe_annualised:.2f}" if m.sharpe_annualised is not None else "")
        + f"  |  sortino/trade {m.sortino_per_trade:.2f}  |  t-stat {m.t_stat:.2f}",
        f"max consecutive losses {m.max_consecutive_losses}  |  max consecutive wins {m.max_consecutive_wins}  |  avg bars held {m.avg_bars_held:.1f}",
        f"avg MAE {m.avg_mae_r:.2f} R  |  avg MFE {m.avg_mfe_r:.2f} R  |  exits {m.exit_reasons}",
    ]
    out += [f"! {n}" for n in m.notes]
    return out


def _table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> List[str]:
    if df is None or df.empty:
        return ["(empty)"]
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "inf" if isinstance(v, float) and not np.isfinite(v) else floatfmt.format(v))
    return d.to_string(index=False).splitlines()


def format_backtest(res: BacktestResult, wf: Optional[WalkForwardResult] = None, overfit: Optional[OverfitReport] = None,
                    thresholds_table: bool = True) -> str:
    cfg = res.config
    L: List[str] = []
    L.append("=" * 78)
    L.append(f"BACKTEST  {cfg.symbol}  {cfg.label}".rstrip())
    L.append(f"data: {res.data_source}  |  entry TF {min(cfg.timeframes, key=lambda t: t.minutes).value}  |  "
             f"{res.first_ts:%Y-%m-%d} → {res.last_ts:%Y-%m-%d}" if res.first_ts else f"data: {res.data_source}")
    L.append(f"bars {res.bars_processed}  |  decisions {res.decisions_made}  |  runtime {res.runtime_s:.1f}s")
    L.append(f"decisions: {res.decision_counts}")
    L.append("regimes seen: " + ", ".join(f"{k} {v}" for k, v in list(res.regime_counts.items())[:8]))
    L.append("why no trade: " + ", ".join(f"{k} {v}" for k, v in list(res.block_reasons.items())[:8]))
    L.append("")
    L += metrics_lines(res.metrics, "PORTFOLIO (risk manager in the loop)")
    L.append("")
    if res.by_regime:
        L.append("BY REGIME")
        L += _table(regime_table(res))
        L.append("")
    if res.by_strategy:
        L.append("BY STRATEGY")
        rows = [{"strategy": k, "trades": m.trades, "win_rate": m.win_rate, "expectancy_r": m.expectancy_r, "profit_factor": m.profit_factor,
                 "max_dd_r": m.max_drawdown_r} for k, m in res.by_strategy.items()]
        L += _table(pd.DataFrame(rows))
        L.append("")
    if res.by_grade:
        L.append("BY GRADE")
        rows = [{"grade": k, "trades": m.trades, "win_rate": m.win_rate, "expectancy_r": m.expectancy_r, "profit_factor": m.profit_factor}
                for k, m in res.by_grade.items()]
        L += _table(pd.DataFrame(rows))
        L.append("")
    if res.monte_carlo:
        mc = res.monte_carlo
        L.append(f"MONTE CARLO ({mc.method}, {mc.runs} runs of {mc.trades_per_run} trades)")
        L.append(f"final R: median {mc.median_final_r:+.1f}  p05 {mc.p05_final_r:+.1f}  p95 {mc.p95_final_r:+.1f}  |  "
                 f"max DD R: median {mc.median_max_dd_r:.1f}  p95 {mc.p95_max_dd_r:.1f}")
        L.append(f"P(negative) {mc.prob_negative:.1%}  |  P(ruin −20R) {mc.prob_ruin:.1%}  |  consecutive losses median {mc.median_max_consecutive_losses:.0f} p95 {mc.p95_max_consecutive_losses:.0f}")
        L += [f"! {n}" for n in mc.notes]
        L.append("")
    cf = res.candidates_frame()
    if thresholds_table and not cf.empty:
        L.append(f"CANDIDATE ANALYSIS ({len(cf)} candidates incl. rejected, {int(cf['filled'].sum())} would have filled)")
        L.append("score buckets (all candidates):")
        L += _table(score_bucket_table(cf))
        L.append("minimum-score sweep (conflict-free candidates, R:R ≥ 2):")
        L += _table(threshold_table(cf))
        L.append("major-conflict veto check (score ≥ 70):")
        L += _table(conflict_table(cf))
        L.append("")
    if wf is not None:
        L.append(f"WALK-FORWARD THRESHOLD VALIDATION ({len(wf.folds)} folds)")
        for f in wf.folds:
            L.append(f"fold {f.fold}: train {f.train_start:%Y-%m-%d}→{f.train_end:%Y-%m-%d} chose {f.chosen_threshold:.0f} "
                     f"(n {f.train_stats.get('n', 0):.0f}, {f.train_stats.get('expectancy_r', 0):+.2f}R) | test {f.test_start:%Y-%m-%d}→{f.test_end:%Y-%m-%d}: "
                     f"n {f.test_stats['n']} {f.test_stats['expectancy_r']:+.2f}R  (default {wf.default_threshold:.0f}: n {f.test_stats_default['n']} {f.test_stats_default['expectancy_r']:+.2f}R)")
        L.append(f"OOS aggregate: n {wf.oos_trades}  win rate {wf.oos_win_rate:.1%}  expectancy {wf.oos_expectancy_r:+.2f}R  PF {_pf(wf.oos_profit_factor)}  |  "
                 f"IS expectancy {wf.is_expectancy_r:+.2f}R  |  default {wf.default_threshold:.0f} OOS {wf.default_oos_expectancy_r:+.2f}R")
        L += [f"! {n}" for n in wf.notes]
        L.append("")
    if overfit is not None:
        L.append(f"ANTI-OVERFITTING CHECKS → {overfit.verdict}")
        L += [f"  {c}" for c in overfit.checks]
        L.append("")
    for w in res.warnings:
        L.append(f"WARNING: {w}")
    L.append("=" * 78)
    return "\n".join(L)


def result_to_dict(res: BacktestResult, wf: Optional[WalkForwardResult] = None, overfit: Optional[OverfitReport] = None) -> Dict:
    out = {
        "symbol": res.config.symbol, "label": res.config.label, "data_source": res.data_source,
        "first_ts": res.first_ts.isoformat() if res.first_ts else None, "last_ts": res.last_ts.isoformat() if res.last_ts else None,
        "bars_processed": res.bars_processed, "decisions": res.decisions_made, "decision_counts": res.decision_counts,
        "block_reasons": res.block_reasons, "regime_counts": res.regime_counts, "metrics": res.metrics.to_dict(),
        "by_regime": {k: v.to_dict() for k, v in res.by_regime.items()}, "by_strategy": {k: v.to_dict() for k, v in res.by_strategy.items()},
        "by_grade": {k: v.to_dict() for k, v in res.by_grade.items()},
        "monte_carlo": res.monte_carlo.to_dict() if res.monte_carlo else None,
        "trades": res.trades_frame().assign(entry_time=lambda d: d["entry_time"].astype(str), exit_time=lambda d: d["exit_time"].astype(str)).to_dict("records") if res.trades else [],
        "candidates": len(res.candidates), "warnings": res.warnings, "runtime_s": round(res.runtime_s, 2),
    }
    if wf is not None:
        out["walk_forward"] = wf.to_dict()
    if overfit is not None:
        out["overfit"] = overfit.to_dict()
    return out

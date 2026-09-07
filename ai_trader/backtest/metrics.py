"""
Performance metrics, Monte-Carlo resampling and statistical sanity checks.

Nothing here optimises for win rate: the headline numbers are expectancy (R),
profit factor, max drawdown and their Monte-Carlo distributions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .simulator import ClosedTrade


@dataclass
class Metrics:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy_r: float = 0.0             # mean R per trade
    expectancy_ccy: float = 0.0           # mean P&L per trade (account currency)
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0
    avg_rr_realised: float = 0.0          # avg_win_r / |avg_loss_r|
    payoff_needed_win_rate: float = 0.0   # break-even win rate for the realised payoff
    net_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    total_costs: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_ccy: float = 0.0
    max_drawdown_r: float = 0.0
    sharpe_per_trade: float = 0.0         # mean(R)/std(R) — trade-based, not annualised
    sharpe_annualised: Optional[float] = None
    sortino_per_trade: float = 0.0
    max_consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    avg_bars_held: float = 0.0
    avg_mae_r: float = 0.0
    avg_mfe_r: float = 0.0
    exit_reasons: Dict[str, int] = field(default_factory=dict)
    start_equity: float = 0.0
    end_equity: float = 0.0
    return_pct: float = 0.0
    t_stat: float = 0.0                   # mean(R)/(std(R)/sqrt(n)) — is expectancy distinguishable from 0?
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def _streaks(flags: Sequence[bool]) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def equity_curve(trades: Iterable[ClosedTrade], start_equity: float) -> pd.Series:
    ts = sorted(trades, key=lambda t: t.exit_time)
    eq = [start_equity]
    idx = [None]
    for t in ts:
        eq.append(eq[-1] + t.pnl)
        idx.append(t.exit_time)
    return pd.Series(eq[1:], index=pd.DatetimeIndex(idx[1:])) if len(eq) > 1 else pd.Series(dtype=float)


def max_drawdown(curve: Sequence[float], start: float) -> tuple[float, float]:
    """(max drawdown in currency, max drawdown in % of the running peak)."""
    peak = start
    dd_ccy = dd_pct = 0.0
    for v in curve:
        peak = max(peak, v)
        d = peak - v
        dd_ccy = max(dd_ccy, d)
        dd_pct = max(dd_pct, d / peak * 100 if peak > 0 else 0.0)
    return dd_ccy, dd_pct


def compute_metrics(trades: List[ClosedTrade], start_equity: float, bars_per_year: Optional[float] = None) -> Metrics:
    m = Metrics(start_equity=start_equity, end_equity=start_equity)
    if not trades:
        m.notes.append("no closed trades")
        return m
    ts = sorted(trades, key=lambda t: t.exit_time)
    r = np.array([t.r_multiple for t in ts], dtype=float)
    pnl = np.array([t.pnl for t in ts], dtype=float)
    wins = pnl > 0
    m.trades = len(ts)
    m.wins = int(wins.sum())
    m.losses = int((~wins).sum())
    m.win_rate = m.wins / m.trades
    m.gross_profit = float(pnl[wins].sum())
    m.gross_loss = float(-pnl[~wins].sum())
    m.profit_factor = float(m.gross_profit / m.gross_loss) if m.gross_loss > 0 else float("inf") if m.gross_profit > 0 else 0.0
    m.expectancy_r = float(r.mean())
    m.expectancy_ccy = float(pnl.mean())
    m.avg_win_r = float(r[wins].mean()) if wins.any() else 0.0
    m.avg_loss_r = float(r[~wins].mean()) if (~wins).any() else 0.0
    m.avg_rr_realised = float(m.avg_win_r / abs(m.avg_loss_r)) if m.avg_loss_r < 0 else 0.0
    m.payoff_needed_win_rate = float(1 / (1 + m.avg_rr_realised)) if m.avg_rr_realised > 0 else 1.0
    m.net_pnl = float(pnl.sum())
    m.total_costs = float(sum(t.costs for t in ts))
    curve = start_equity + np.cumsum(pnl)
    m.max_drawdown_ccy, m.max_drawdown_pct = max_drawdown(curve, start_equity)
    m.max_drawdown_r, _ = max_drawdown(np.cumsum(r), 0.0)
    sd = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    m.sharpe_per_trade = float(r.mean() / sd) if sd > 0 else 0.0
    downside = r[r < 0]
    dsd = float(np.sqrt(np.mean(downside ** 2))) if len(downside) else 0.0
    m.sortino_per_trade = float(r.mean() / dsd) if dsd > 0 else 0.0
    m.t_stat = float(r.mean() / (sd / np.sqrt(len(r)))) if sd > 0 else 0.0
    if bars_per_year and ts[-1].exit_time > ts[0].entry_time:
        years = (ts[-1].exit_time - ts[0].entry_time).total_seconds() / (365.25 * 86400)
        if years > 0 and sd > 0:
            trades_per_year = len(ts) / years
            m.sharpe_annualised = float(m.sharpe_per_trade * np.sqrt(trades_per_year))
    m.max_consecutive_losses = _streaks([not w for w in wins])
    m.max_consecutive_wins = _streaks(list(wins))
    m.avg_bars_held = float(np.mean([t.bars_held for t in ts]))
    m.avg_mae_r = float(np.mean([t.mae_r for t in ts]))
    m.avg_mfe_r = float(np.mean([t.mfe_r for t in ts]))
    reasons: Dict[str, int] = {}
    for t in ts:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    m.exit_reasons = dict(sorted(reasons.items(), key=lambda kv: -kv[1]))
    m.end_equity = float(start_equity + pnl.sum())
    m.return_pct = float((m.end_equity / start_equity - 1) * 100) if start_equity else 0.0
    if m.trades < 30:
        m.notes.append(f"only {m.trades} trades — statistics are NOT reliable (need ≥ 30, preferably ≥ 100)")
    if abs(m.t_stat) < 2 and m.trades >= 30:
        m.notes.append(f"t-stat {m.t_stat:.2f}: expectancy not statistically distinguishable from zero")
    return m


def metrics_by(trades: List[ClosedTrade], key: str, start_equity: float) -> Dict[str, Metrics]:
    groups: Dict[str, List[ClosedTrade]] = {}
    for t in trades:
        k = str(getattr(t, key, None) if hasattr(t, key) else t.meta.get(key))
        groups.setdefault(k, []).append(t)
    return {k: compute_metrics(v, start_equity) for k, v in sorted(groups.items())}


# --------------------------------------------------------------------------
# Monte Carlo
# --------------------------------------------------------------------------
@dataclass
class MonteCarloResult:
    runs: int
    trades_per_run: int
    median_final_r: float
    p05_final_r: float
    p95_final_r: float
    median_max_dd_r: float
    p95_max_dd_r: float
    prob_ruin: float                # probability the R-curve breaches -ruin_r at any point
    prob_negative: float            # probability the run ends below 0 R
    median_max_consecutive_losses: float
    p95_max_consecutive_losses: float
    method: str
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def monte_carlo(trades: List[ClosedTrade], runs: int = 2000, seed: int = 7, ruin_r: float = 20.0,
                method: str = "shuffle", block: int = 5) -> MonteCarloResult:
    """Resample the *order* (shuffle), or bootstrap trades with replacement, or
    block-bootstrap (keeps short-range clustering of wins/losses)."""
    r = np.array([t.r_multiple for t in trades], dtype=float)
    n = len(r)
    if n == 0:
        return MonteCarloResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, method, ["no trades"])
    rng = np.random.default_rng(seed)
    finals = np.empty(runs)
    dds = np.empty(runs)
    ruins = np.zeros(runs, dtype=bool)
    streaks = np.empty(runs)
    for i in range(runs):
        if method == "shuffle":
            seq = rng.permutation(r)
        elif method == "bootstrap":
            seq = rng.choice(r, size=n, replace=True)
        else:  # block bootstrap
            starts = rng.integers(0, max(1, n - block + 1), size=int(np.ceil(n / block)))
            seq = np.concatenate([r[s:s + block] for s in starts])[:n]
        curve = np.cumsum(seq)
        peak = np.maximum.accumulate(np.concatenate([[0.0], curve]))[1:]
        dd = float(np.max(peak - curve)) if n else 0.0
        finals[i] = curve[-1]
        dds[i] = dd
        ruins[i] = bool(np.any(curve <= -ruin_r))
        streaks[i] = _streaks(list(seq < 0))
    notes = []
    if n < 30:
        notes.append("fewer than 30 trades — Monte Carlo widths are dominated by sampling error")
    return MonteCarloResult(runs, n, float(np.median(finals)), float(np.percentile(finals, 5)), float(np.percentile(finals, 95)),
                            float(np.median(dds)), float(np.percentile(dds, 95)), float(ruins.mean()), float((finals < 0).mean()),
                            float(np.median(streaks)), float(np.percentile(streaks, 95)), method, notes)

"""
Validation: score-threshold analysis, walk-forward / out-of-sample evaluation,
per-regime breakdown and anti-overfitting diagnostics.

Two levels of evidence are produced:

1. **Candidate level** — every setup the engine generated (accepted *or*
   rejected) with its isolated hypothetical outcome. Because these outcomes do
   not depend on the threshold, the same records can be re-cut at any minimum
   score, which is how the 70/80/90 cut-offs are validated (in-sample choice,
   out-of-sample check, fold by fold).
2. **Portfolio level** — full replays with the RiskManager in the loop, run
   separately on the in-sample and out-of-sample windows, so kill-switches,
   size reduction and sequencing effects are included.

Nothing here optimises for win rate. Selection uses expectancy (R per trade)
subject to a minimum trade count, and the selected value is then judged only
on data it was not chosen on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .engine import Backtester, BacktestConfig, BacktestResult
from .metrics import Metrics, compute_metrics

DEFAULT_THRESHOLDS: Tuple[float, ...] = (50, 55, 60, 65, 70, 75, 80, 85, 90)


# --------------------------------------------------------------------------
# Candidate-level statistics
# --------------------------------------------------------------------------
def _stats(r: np.ndarray) -> Dict[str, float]:
    if len(r) == 0:
        return {"n": 0, "win_rate": 0.0, "expectancy_r": 0.0, "profit_factor": 0.0, "sum_r": 0.0, "t_stat": 0.0}
    wins = r[r > 0]
    losses = r[r <= 0]
    pf = float(wins.sum() / -losses.sum()) if losses.sum() < 0 else (float("inf") if wins.sum() > 0 else 0.0)
    sd = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    return {"n": int(len(r)), "win_rate": float((r > 0).mean()), "expectancy_r": float(r.mean()), "profit_factor": pf,
            "sum_r": float(r.sum()), "t_stat": float(r.mean() / (sd / np.sqrt(len(r)))) if sd > 0 else 0.0}


def first_per_setup(df: pd.DataFrame, mask: pd.Series) -> pd.Series:
    """Restrict ``mask`` to the FIRST bar of every setup (``setup_id``) that satisfies it.
    A setup that is re-proposed on consecutive bars is therefore counted once, with the
    geometry and score it had on the bar a trader following the rule would have acted."""
    if "setup_id" not in df.columns:
        return mask
    sel = df[mask].sort_values("bar")
    keep = sel.groupby("setup_id", sort=False).head(1).index
    out = pd.Series(False, index=df.index)
    out.loc[keep] = True
    return out


def tradeable_mask(df: pd.DataFrame, min_score: float, max_major: int = 0, require_rr: bool = True, min_rr: float = 2.0) -> pd.Series:
    """Candidates that would have been *taken* under a given rule set (one entry per setup:
    the first bar on which the setup satisfied the rule)."""
    m = (df["score"] >= min_score) & (df["n_major"] <= max_major)
    if require_rr:
        m &= df["rr_tp2"] >= min_rr
    m = first_per_setup(df, m)
    return m & df["filled"] & df["r_multiple"].notna()


def score_bucket_table(df: pd.DataFrame, edges: Sequence[float] = (0, 60, 70, 80, 90, 101), max_major: Optional[int] = None) -> pd.DataFrame:
    """Outcome statistics per confluence-score bucket (filled candidates only).
    ``max_major=None`` keeps every candidate; ``0`` keeps only conflict-free ones."""
    if df.empty:
        return pd.DataFrame()
    base = pd.Series(True, index=df.index) if max_major is None else (df["n_major"] <= max_major)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = first_per_setup(df, base & (df["score"] >= lo) & (df["score"] < hi)) & df["filled"] & df["r_multiple"].notna()
        st = _stats(df.loc[m, "r_multiple"].to_numpy(dtype=float))
        rows.append({"bucket": f"{lo:.0f}–{hi - 1:.0f}" if hi < 101 else f"{lo:.0f}+", **st})
    return pd.DataFrame(rows)


def threshold_table(df: pd.DataFrame, thresholds: Sequence[float] = DEFAULT_THRESHOLDS, max_major: int = 0) -> pd.DataFrame:
    """For each candidate minimum score: what the *taken* set would have produced."""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for t in thresholds:
        m = tradeable_mask(df, t, max_major)
        st = _stats(df.loc[m, "r_multiple"].to_numpy(dtype=float))
        rows.append({"min_score": t, **st})
    return pd.DataFrame(rows)


def conflict_table(df: pd.DataFrame, min_score: float = 70) -> pd.DataFrame:
    """Does the MAJOR-conflict veto earn its keep? Outcomes of high-score candidates with vs without major conflicts."""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for label, cond in (("no major conflict", df["n_major"] == 0), ("≥1 major conflict", df["n_major"] >= 1)):
        m = first_per_setup(df, cond & (df["score"] >= min_score)) & df["filled"] & df["r_multiple"].notna()
        rows.append({"group": label, **_stats(df.loc[m, "r_multiple"].to_numpy(dtype=float))})
    return pd.DataFrame(rows)


def select_threshold(df: pd.DataFrame, thresholds: Sequence[float] = DEFAULT_THRESHOLDS, min_trades: int = 20,
                     max_major: int = 0, fallback: float = 80.0) -> Tuple[float, Dict[str, float]]:
    """In-sample choice: the threshold with the highest expectancy among those with
    at least ``min_trades`` samples. Falls back to the configured default when no
    threshold has enough trades — never invents a number from thin data."""
    tab = threshold_table(df, thresholds, max_major)
    if tab.empty:
        return fallback, {"n": 0}
    ok = tab[tab["n"] >= min_trades]
    if ok.empty:
        return fallback, {"n": 0, "reason": f"no threshold had ≥ {min_trades} trades in-sample"}
    best = ok.sort_values(["expectancy_r", "min_score"], ascending=[False, True]).iloc[0]
    return float(best["min_score"]), {k: float(best[k]) for k in ("n", "win_rate", "expectancy_r", "profit_factor")}


# --------------------------------------------------------------------------
# Walk-forward on candidate records
# --------------------------------------------------------------------------
@dataclass
class WalkForwardFold:
    fold: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    chosen_threshold: float
    train_stats: Dict[str, float]
    test_stats: Dict[str, float]
    test_stats_default: Dict[str, float]     # same test window with the configured default threshold

    def to_dict(self) -> Dict:
        d = dict(self.__dict__)
        for k in ("train_start", "train_end", "test_start", "test_end"):
            d[k] = getattr(self, k).isoformat()
        return d


@dataclass
class WalkForwardResult:
    folds: List[WalkForwardFold]
    oos_expectancy_r: float
    oos_trades: int
    oos_win_rate: float
    oos_profit_factor: float
    is_expectancy_r: float
    degradation: Optional[float]            # OOS expectancy / IS expectancy (1 = none, <0.5 = suspicious)
    positive_folds: int
    default_threshold: float
    default_oos_expectancy_r: float
    chosen_thresholds: List[float]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        d = {k: v for k, v in self.__dict__.items() if k != "folds"}
        d["folds"] = [f.to_dict() for f in self.folds]
        return d


def walk_forward_thresholds(df: pd.DataFrame, n_folds: int = 4, thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
                            min_trades: int = 20, max_major: int = 0, default_threshold: float = 80.0,
                            anchored: bool = True) -> WalkForwardResult:
    """Rolling (or anchored) walk-forward over the candidate records: the minimum
    score is chosen on the training window and *only* evaluated on the following
    test window. Aggregated OOS statistics are what the threshold is judged by."""
    if df.empty or df["ts"].nunique() < n_folds + 1:
        return WalkForwardResult([], 0.0, 0, 0.0, 0.0, 0.0, None, 0, default_threshold, 0.0, [], ["not enough candidates for walk-forward"])
    d = df.sort_values("ts").reset_index(drop=True)
    edges = np.linspace(0, len(d), n_folds + 2, dtype=int)     # n_folds+1 chunks: first = initial train
    folds: List[WalkForwardFold] = []
    oos_r: List[float] = []
    oos_default_r: List[float] = []
    is_r: List[float] = []
    for k in range(1, n_folds + 1):
        tr_lo = 0 if anchored else edges[k - 1]
        train = d.iloc[tr_lo:edges[k]]
        test = d.iloc[edges[k]:edges[k + 1]]
        if train.empty or test.empty:
            continue
        thr, tr_stats = select_threshold(train, thresholds, min_trades, max_major, default_threshold)
        m_test = tradeable_mask(test, thr, max_major)
        m_def = tradeable_mask(test, default_threshold, max_major)
        te = _stats(test.loc[m_test, "r_multiple"].to_numpy(dtype=float))
        te_def = _stats(test.loc[m_def, "r_multiple"].to_numpy(dtype=float))
        folds.append(WalkForwardFold(k, train["ts"].iloc[0].to_pydatetime(), train["ts"].iloc[-1].to_pydatetime(),
                                     test["ts"].iloc[0].to_pydatetime(), test["ts"].iloc[-1].to_pydatetime(), thr, tr_stats, te, te_def))
        oos_r += list(test.loc[m_test, "r_multiple"].to_numpy(dtype=float))
        oos_default_r += list(test.loc[m_def, "r_multiple"].to_numpy(dtype=float))
        is_r += list(train.loc[tradeable_mask(train, thr, max_major), "r_multiple"].to_numpy(dtype=float))
    oos = _stats(np.array(oos_r))
    ins = _stats(np.array(is_r))
    dflt = _stats(np.array(oos_default_r))
    degradation = (oos["expectancy_r"] / ins["expectancy_r"]) if ins["expectancy_r"] > 0 else None
    notes = []
    if oos["n"] < 30:
        notes.append(f"only {oos['n']} out-of-sample trades — treat every number here as anecdotal")
    if degradation is not None and degradation < 0.5:
        notes.append(f"OOS expectancy is {degradation:.0%} of in-sample → threshold selection is over-fitting")
    if folds and len({f.chosen_threshold for f in folds}) > 2:
        notes.append("chosen threshold is unstable across folds → the score edge is not robust yet")
    positive = sum(1 for f in folds if f.test_stats["expectancy_r"] > 0 and f.test_stats["n"] > 0)
    return WalkForwardResult(folds, oos["expectancy_r"], oos["n"], oos["win_rate"], oos["profit_factor"], ins["expectancy_r"],
                             degradation, positive, default_threshold, dflt["expectancy_r"], [f.chosen_threshold for f in folds], notes)


# --------------------------------------------------------------------------
# Portfolio-level in-sample / out-of-sample split
# --------------------------------------------------------------------------
@dataclass
class SplitResult:
    in_sample: BacktestResult
    out_of_sample: BacktestResult
    split_at: datetime
    degradation: Optional[float]
    notes: List[str] = field(default_factory=list)


def in_out_of_sample(provider, cfg: BacktestConfig, split_at: datetime, settings=None, calendar=None,
                     macro_series=None) -> SplitResult:
    """Two independent replays (fresh RiskManager each) either side of ``split_at``.
    Any parameter tuning must be done on ``in_sample`` only; ``out_of_sample`` is
    looked at once, at the end."""
    from dataclasses import replace
    is_cfg = replace(cfg, end=split_at, label=(cfg.label + " IS").strip())
    oos_cfg = replace(cfg, start=split_at, label=(cfg.label + " OOS").strip())
    is_res = Backtester(provider, is_cfg, settings, calendar, macro_series).run()
    oos_res = Backtester(provider, oos_cfg, settings, calendar, macro_series).run()
    deg = None
    if is_res.metrics.expectancy_r > 0 and oos_res.metrics.trades:
        deg = oos_res.metrics.expectancy_r / is_res.metrics.expectancy_r
    notes = []
    if oos_res.metrics.trades < 30:
        notes.append("out-of-sample trade count < 30 — not conclusive")
    if deg is not None and deg < 0.5:
        notes.append("large in-sample → out-of-sample degradation: suspect over-fitting")
    return SplitResult(is_res, oos_res, split_at, deg, notes)


# --------------------------------------------------------------------------
# Regime report
# --------------------------------------------------------------------------
def regime_table(result: BacktestResult) -> pd.DataFrame:
    rows = []
    for reg, m in result.by_regime.items():
        rows.append({"regime": reg, "trades": m.trades, "win_rate": round(m.win_rate, 3), "expectancy_r": round(m.expectancy_r, 3),
                     "profit_factor": round(m.profit_factor, 2) if np.isfinite(m.profit_factor) else float("inf"),
                     "max_dd_r": round(m.max_drawdown_r, 2), "max_consec_losses": m.max_consecutive_losses})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Anti-overfitting diagnostics
# --------------------------------------------------------------------------
FREE_PARAMETERS = {
    "confluence weights (8)": 8, "decision thresholds (5)": 5, "regime detector cut-offs (~12)": 12,
    "strategy parameters (~25)": 25, "risk / management rules (~8)": 8,
}


@dataclass
class OverfitReport:
    trades: int
    free_parameters: int
    trades_per_parameter: float
    is_expectancy_r: Optional[float]
    oos_expectancy_r: Optional[float]
    degradation: Optional[float]
    positive_folds: Optional[str]
    threshold_stability: Optional[str]
    monte_carlo_p05_final_r: Optional[float]
    verdict: str
    checks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return dict(self.__dict__)


def overfit_report(result: BacktestResult, wf: Optional[WalkForwardResult] = None, split: Optional[SplitResult] = None) -> OverfitReport:
    n_par = sum(FREE_PARAMETERS.values())
    n = result.metrics.trades
    tpp = n / n_par if n_par else 0.0
    checks: List[str] = []
    fails = 0
    if tpp < 5:
        checks.append(f"FAIL sample size: {n} trades for ~{n_par} free parameters ({tpp:.1f} per parameter; want ≥ 5, ideally ≥ 10)")
        fails += 1
    else:
        checks.append(f"ok  sample size: {tpp:.1f} trades per free parameter")
    if result.metrics.trades >= 30:
        if abs(result.metrics.t_stat) < 2:
            checks.append(f"FAIL significance: t-stat {result.metrics.t_stat:.2f} — expectancy indistinguishable from zero")
            fails += 1
        else:
            checks.append(f"ok  significance: t-stat {result.metrics.t_stat:.2f}")
    else:
        checks.append("n/a significance: fewer than 30 trades")
    deg = None
    is_e = oos_e = None
    if wf is not None and wf.folds:
        deg = wf.degradation
        is_e, oos_e = wf.is_expectancy_r, wf.oos_expectancy_r
        if wf.oos_trades < 30:
            checks.append(f"WARN walk-forward: {wf.oos_trades} OOS trades — inconclusive")
        elif deg is None or deg < 0.5:
            checks.append(f"FAIL walk-forward: OOS expectancy {oos_e:+.2f}R vs IS {is_e:+.2f}R")
            fails += 1
        else:
            checks.append(f"ok  walk-forward: OOS keeps {deg:.0%} of in-sample expectancy")
        pos = f"{wf.positive_folds}/{len(wf.folds)}"
        if wf.positive_folds < len(wf.folds) / 2:
            checks.append(f"FAIL fold stability: only {pos} folds positive out-of-sample")
            fails += 1
        else:
            checks.append(f"ok  fold stability: {pos} folds positive out-of-sample")
        stab = ", ".join(f"{t:.0f}" for t in wf.chosen_thresholds)
    else:
        pos = None
        stab = None
        checks.append("n/a walk-forward: not run")
    if split is not None:
        checks.append(f"{'ok ' if (split.degradation or 0) >= 0.5 else 'FAIL'} portfolio IS/OOS: IS {split.in_sample.metrics.expectancy_r:+.2f}R "
                      f"({split.in_sample.metrics.trades} trades) → OOS {split.out_of_sample.metrics.expectancy_r:+.2f}R ({split.out_of_sample.metrics.trades} trades)")
        if (split.degradation or 0) < 0.5:
            fails += 1
    mc05 = result.monte_carlo.p05_final_r if result.monte_carlo else None
    if mc05 is not None:
        if mc05 < 0:
            checks.append(f"WARN Monte Carlo: 5th percentile of final R is {mc05:+.1f} — a losing sequence is well within chance")
        else:
            checks.append(f"ok  Monte Carlo: 5th percentile of final R {mc05:+.1f}")
    if result.data_source in ("synthetic", "frames"):
        checks.append("FAIL data: synthetic/scripted — no statement about real-market edge is possible")
        fails += 1
    checks.append("ok  look-ahead: strict causality assertion active on every decision (closed bars only)" if result.config.strict_causality
                  else "WARN look-ahead: strict causality check disabled")
    checks.append("note survivorship: instrument list is fixed by the user (no delisting bias for FX/metals/indices; crypto lists are survivorship-prone)")
    verdict = "NOT VALIDATED" if fails else ("INCONCLUSIVE" if n < 100 else "PASSES BASIC CHECKS")
    return OverfitReport(n, n_par, tpp, is_e, oos_e, deg, pos, stab, mc05, verdict, checks)

"""
Command-line entry point.

    python -m ai_trader.cli analyze XAUUSD                     # random synthetic path (mechanics demo)
    python -m ai_trader.cli analyze XAUUSD --seed 7 --json
    python -m ai_trader.cli scenario textbook_long              # scripted textbook setup → TRADE
    python -m ai_trader.cli scenario textbook_short
    python -m ai_trader.cli scenario range_fade
    python -m ai_trader.cli scenario choppy                     # featureless → NO TRADE
    python -m ai_trader.cli scenario textbook_long --without displacement confirmation_candle
    python -m ai_trader.cli instruments
    python -m ai_trader.cli backtest XAUUSD --source scenario --resolve win     # scripted setup → 1 trade → TP3
    python -m ai_trader.cli backtest XAUUSD --source synthetic --bars 1500      # random path → (almost) never trades
    python -m ai_trader.cli backtest XAUUSD --source csv --data-dir data --walk-forward --json out.json
    python -m ai_trader.cli backtest XAUUSD --source csv --split 2025-01-01     # in-sample / out-of-sample portfolio replays

Synthetic data proves the *plumbing*, never the edge. Plug a CSV/API provider
into ``MarketDataLoader`` for real analysis.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List

from ai_trader.config.instruments import all_instruments
from ai_trader.core.enums import Timeframe
from ai_trader.data import (
    EconomicCalendar,
    FrameProvider,
    ScenarioConfig,
    SyntheticProvider,
    choppy_no_edge,
    range_fade_setup,
    textbook_pullback_setup,
)
from ai_trader.decision import DecisionEngine, decision_to_dict, format_decision

DEFAULT_TFS = [Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1]


def _known_empty_calendar() -> EconomicCalendar:
    cal = EconomicCalendar.from_events([])
    cal.source = "explicit"          # "we checked: nothing scheduled" (no missing-calendar penalty)
    return cal


def cmd_analyze(args: argparse.Namespace) -> int:
    prov = SyntheticProvider(seed=args.seed)
    eng = DecisionEngine(prov, calendar=_known_empty_calendar() if args.no_news_penalty else None)
    d = eng.analyze(args.symbol.upper(), timeframes=DEFAULT_TFS)
    print(json.dumps(decision_to_dict(d), indent=2, default=str) if args.json else format_decision(d))
    if args.candidates and not args.json:
        _print_candidates(d)
    return 0


def cmd_scenario(args: argparse.Namespace) -> int:
    off = set(args.without or [])
    if args.name in ("textbook_long", "textbook_short"):
        cfg = ScenarioConfig(direction="long" if args.name == "textbook_long" else "short", seed=args.seed)
        for flag in off:
            if not hasattr(cfg, flag):
                print(f"unknown scenario flag {flag!r}; valid: equal_lows sweep displacement retrace_into_fvg confirmation_candle volume_confirms")
                return 2
            setattr(cfg, flag, False)
        df, symbol = textbook_pullback_setup(cfg), "XAUUSD"
    elif args.name == "range_fade":
        df, symbol = range_fade_setup(seed=args.seed, rejection="rejection" not in off), "EURUSD"
    elif args.name == "choppy":
        df, symbol = choppy_no_edge(seed=args.seed), "EURUSD"
    else:
        print("unknown scenario; choose textbook_long | textbook_short | range_fade | choppy")
        return 2
    eng = DecisionEngine(FrameProvider({symbol: {Timeframe.M15: df}}), calendar=_known_empty_calendar())
    d = eng.analyze(symbol, timeframes=DEFAULT_TFS)
    print(json.dumps(decision_to_dict(d), indent=2, default=str) if args.json else format_decision(d))
    if args.candidates and not args.json:
        _print_candidates(d)
    return 0


def _print_candidates(d) -> None:
    print("\nCANDIDATES (all strategies that produced a signal):")
    if not d.candidates:
        print("  none")
    for c in d.candidates:
        print(f"  {c['strategy']:28s} {c['direction']:5s} score {c['score']:5.1f}  R:R(TP2) 1:{c['rr_tp2']}  → {c['rejected_because']}")
        for k in c.get("conflicts", []):
            print(f"      - {k}")


def cmd_backtest(args: argparse.Namespace) -> int:
    from datetime import datetime, timezone

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
    from ai_trader.data import CSVProvider

    symbol = args.symbol.upper()
    tfs = [Timeframe(t) for t in args.timeframes] if args.timeframes else DEFAULT_TFS
    start = end = None
    warmup = args.warmup
    if args.source == "scenario":
        cfg = ScenarioConfig(direction=args.direction, seed=args.seed, resolve=args.resolve or "win")
        df = textbook_pullback_setup(cfg)
        warmup = df.attrs["scenario"]["setup_end"] - 40           # start deciding shortly before the scripted setup
        prov, symbol = FrameProvider({"XAUUSD": {Timeframe.M15: df}}), "XAUUSD"
    elif args.source == "synthetic":
        prov = SyntheticProvider(seed=args.seed, base_bars=max(20_000, args.bars * 3 * 4))
        end = datetime.now(timezone.utc)
        full = prov.get_ohlcv(symbol, min(tfs, key=lambda t: t.minutes), 10**7, end=end)
        start = full.index[-args.bars].to_pydatetime()
    else:
        prov = CSVProvider(args.data_dir)
        if not prov.available():
            print(f"CSV directory {args.data_dir!r} not found; expected files like {args.data_dir}/{symbol}_15m.csv")
            return 2
        start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc) if args.start else None
        end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc) if args.end else None
    costs = ExecutionCosts(spread_points=args.spread, commission_per_lot_side=args.commission)
    rules = ManagementRules(order_ttl_bars=args.ttl, max_hold_bars=args.max_hold)
    cfg = BacktestConfig(symbol, timeframes=tfs, start=start, end=end, warmup_bars=warmup, costs=costs, rules=rules,
                         analyze_every=args.every, start_equity=args.equity, label=args.label or args.source)
    cal = _known_empty_calendar() if args.no_news_penalty else None
    if args.split:
        split_at = datetime.fromisoformat(args.split).replace(tzinfo=timezone.utc)
        sr = in_out_of_sample(prov, cfg, split_at, calendar=cal)
        for res in (sr.in_sample, sr.out_of_sample):
            wf = walk_forward_thresholds(res.candidates_frame()) if args.walk_forward else None
            print(format_backtest(res, wf, overfit_report(res, wf)))
        print(f"IS → OOS expectancy degradation: {sr.degradation if sr.degradation is not None else 'n/a'}; notes: {sr.notes}")
        return 0
    res = Backtester(prov, cfg, calendar=cal, progress=(lambda i, n: print(f"  {i}/{n} bars", end="\r", flush=True)) if args.progress else None).run()
    wf = walk_forward_thresholds(res.candidates_frame()) if args.walk_forward else None
    ov = overfit_report(res, wf)
    print(format_backtest(res, wf, ov))
    if args.trades:
        tf_ = res.trades_frame()
        print(tf_.to_string(index=False) if len(tf_) else "no trades")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(result_to_dict(res, wf, ov), fh, indent=2, default=str)
        print(f"wrote {args.json}")
    return 0


def cmd_instruments(_: argparse.Namespace) -> int:
    for inst in all_instruments():
        print(f"{inst.symbol:8s} {inst.asset_class.value:10s} {inst.name}")
    return 0


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ai_trader", description="AI trading market analysis — decision support, never guarantees")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="analyse a symbol on a random synthetic path")
    a.add_argument("symbol")
    a.add_argument("--seed", type=int, default=1)
    a.add_argument("--json", action="store_true")
    a.add_argument("--candidates", action="store_true", help="print every strategy candidate and why it was rejected")
    a.add_argument("--no-news-penalty", action="store_true", help="treat the calendar as checked-and-empty")
    a.set_defaults(fn=cmd_analyze)
    s = sub.add_parser("scenario", help="run a scripted textbook scenario")
    s.add_argument("name")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--without", nargs="*", help="switch scenario components off (e.g. displacement confirmation_candle)")
    s.add_argument("--json", action="store_true")
    s.add_argument("--candidates", action="store_true")
    s.set_defaults(fn=cmd_scenario)
    i = sub.add_parser("instruments", help="list the instrument registry")
    i.set_defaults(fn=cmd_instruments)
    b = sub.add_parser("backtest", help="replay the production decision engine bar by bar")
    b.add_argument("symbol")
    b.add_argument("--source", choices=["scenario", "synthetic", "csv"], default="scenario")
    b.add_argument("--data-dir", default="data", help="CSV directory: {SYMBOL}_{tf}.csv (e.g. XAUUSD_15m.csv)")
    b.add_argument("--timeframes", nargs="*", help="e.g. 15m 1h 4h 1d (lowest = entry timeframe)")
    b.add_argument("--start"); b.add_argument("--end"); b.add_argument("--split", help="ISO date: in-sample before, out-of-sample after")
    b.add_argument("--bars", type=int, default=1500, help="synthetic: number of entry bars to replay")
    b.add_argument("--warmup", type=int, default=400)
    b.add_argument("--every", type=int, default=1, help="decide on every k-th bar (speed)")
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--direction", choices=["long", "short"], default="long")
    b.add_argument("--resolve", choices=["win", "loss", "flat"], help="scenario: how the scripted setup plays out")
    b.add_argument("--spread", type=float, help="override spread (price units)")
    b.add_argument("--commission", type=float, default=0.0, help="per lot per side, account currency")
    b.add_argument("--ttl", type=int, default=8, help="pending order lifetime in entry bars")
    b.add_argument("--max-hold", type=int, default=96, help="time stop in entry bars")
    b.add_argument("--equity", type=float)
    b.add_argument("--label", default="")
    b.add_argument("--walk-forward", action="store_true", help="walk-forward validation of the minimum confluence score")
    b.add_argument("--trades", action="store_true", help="print the trade list")
    b.add_argument("--progress", action="store_true")
    b.add_argument("--json", help="write the full result to this file")
    b.add_argument("--no-news-penalty", action="store_true", help="treat the calendar as checked-and-empty")
    b.set_defaults(fn=cmd_backtest)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

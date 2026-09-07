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
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

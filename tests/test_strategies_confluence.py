"""Strategy generation + confluence scoring tests on the scripted scenarios."""
import pytest

from ai_trader.analysis import build_context
from ai_trader.config.instruments import get_instrument
from ai_trader.config.settings import get_settings
from ai_trader.core.enums import Direction, EntryType, Regime, StrategyFamily, Timeframe
from ai_trader.core.models import StrategySignal
from ai_trader.data import FrameProvider, MarketDataLoader, ScenarioConfig, choppy_no_edge, textbook_pullback_setup
from ai_trader.decision.confluence import ConfluenceScorer
from ai_trader.regime import RegimeDetector
from ai_trader.strategies import StrategyRegistry, default_strategies
from ai_trader.strategies.base import Strategy

TFS = [Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1]


def _pipeline(df, symbol="XAUUSD"):
    inst = get_instrument(symbol)
    loader = MarketDataLoader(FrameProvider({symbol: {Timeframe.M15: df}}))
    load = loader.load(symbol, timeframes=TFS)
    assert load.ok, load.problems
    ctx = build_context(load.data)
    regime = RegimeDetector().detect(ctx.structural)
    return inst, ctx, regime


@pytest.fixture(scope="module")
def textbook():
    return _pipeline(textbook_pullback_setup(ScenarioConfig()))


def test_registry_filters_by_regime_and_instrument(textbook):
    inst, ctx, regime = textbook
    reg = StrategyRegistry(default_strategies())
    assert len(reg.all()) == 6
    allowed = reg.for_regime(regime, inst)
    assert allowed, regime.explanation
    assert all(s.family in regime.allowed_families for s in allowed)
    # synthetic indices never get the momentum family
    syn = reg.for_regime(regime, get_instrument("VOL75"))
    assert all(s.family is not StrategyFamily.MOMENTUM for s in syn)


def test_strategies_generate_long_signals_on_textbook_setup(textbook):
    inst, ctx, regime = textbook
    assert regime.primary is Regime.STRONG_BULL, regime.explanation
    sigs = []
    for s in StrategyRegistry(default_strategies()).for_regime(regime, inst):
        sigs += s.generate(ctx, regime, inst)
    names = {s.strategy for s in sigs}
    assert {"trend_pullback_smc", "momentum_bos_continuation", "liquidity_sweep_mss"} <= names, names
    for s in sigs:
        assert s.direction is Direction.LONG
        assert s.stop < s.entry_low <= s.entry_high
        assert len(s.targets) == 3 and s.targets[0] < s.targets[1] < s.targets[2]
        assert s.rr(1) >= 1.5
        assert s.reasons and s.invalidation
        assert isinstance(s.entry_type, EntryType)


def test_signals_are_sparse_in_chop():
    inst, ctx, regime = _pipeline(choppy_no_edge(), "EURUSD")
    sigs = []
    for s in StrategyRegistry(default_strategies()).for_regime(regime, inst):
        sigs += s.generate(ctx, regime, inst)
    # a random walk must not produce a flood of setups
    assert len(sigs) <= 2


def test_targets_from_levels_monotonic_and_in_front_of_liquidity():
    t = Strategy.targets_from_levels(entry=100.0, stop=98.0, direction=Direction.LONG, candidates=[100.5, 103.2, 103.3, 108.0])
    assert t[0] < t[1] < t[2]
    assert t[0] <= 103.2 and t[0] >= 103.0          # shaded 5% of risk in front of 103.2
    assert t[2] >= 100.0 + 2.0 * 4.0 - 0.1 * 2.0     # >= required 4R minus shading
    s = Strategy.targets_from_levels(entry=100.0, stop=102.0, direction=Direction.SHORT, candidates=[97.0, 95.0])
    assert s[0] > s[1] > s[2]


def test_confluence_scores_textbook_high_and_reacts_to_missing_factors(textbook):
    inst, ctx, regime = textbook
    scorer = ConfluenceScorer()
    reg = StrategyRegistry(default_strategies())
    sigs = [x for s in reg.for_regime(regime, inst) for x in s.generate(ctx, regime, inst)]
    best = max((scorer.score(s, ctx, regime) for s in sigs), key=lambda r: r.total)
    assert best.total >= 80
    assert best.n_major == 0
    comp = best.breakdown
    assert best.total <= sum(comp.values()) + 0.2          # minor conflicts only ever subtract
    weights = get_settings().scoring.as_dict()
    for k, v in comp.items():
        assert 0 <= v <= weights[k] + 1e-9

    # the same setup without volume confirmation must score lower on the volume component
    inst2, ctx2, regime2 = _pipeline(textbook_pullback_setup(ScenarioConfig(volume_confirms=False)))
    sigs2 = [x for s in reg.for_regime(regime2, inst2) for x in s.generate(ctx2, regime2, inst2)]
    res2 = {s.strategy: scorer.score(s, ctx2, regime2) for s in sigs2}
    assert res2["momentum_bos_continuation"].breakdown["volume"] < best.breakdown["volume"]


def test_confluence_flags_major_conflict_for_counter_trend_entry(textbook):
    inst, ctx, regime = textbook
    price = float(ctx.entry.df["close"].iloc[-1])
    atr = ctx.entry.atr
    bad = StrategySignal(strategy="manual_short", family=StrategyFamily.TREND_FOLLOWING, direction=Direction.SHORT,
                         entry_low=price - 0.1 * atr, entry_high=price, entry_type=EntryType.MARKET,
                         stop=price + 1.0 * atr, targets=[price - 1.5 * atr, price - 2.5 * atr, price - 4 * atr],
                         timeframe=Timeframe.M15, reasons=["test"], invalidation="test")
    res = ConfluenceScorer().score(bad, ctx, regime)
    assert res.n_major >= 1
    assert any("MAJOR" in c for c in res.conflicts)
    assert res.total < 80


def test_confluence_flags_bad_stop_and_rr(textbook):
    inst, ctx, regime = textbook
    price = float(ctx.entry.df["close"].iloc[-1])
    atr = ctx.entry.atr
    tight = StrategySignal(strategy="tight", family=StrategyFamily.TREND_FOLLOWING, direction=Direction.LONG,
                           entry_low=price, entry_high=price, entry_type=EntryType.MARKET,
                           stop=price - 0.1 * atr, targets=[price + 0.1 * atr, price + 0.15 * atr, price + 0.2 * atr],
                           timeframe=Timeframe.M15)
    res = ConfluenceScorer().score(tight, ctx, regime)
    joined = " ".join(res.conflicts)
    assert "stop" in joined and "R:R" in joined
    assert res.n_major >= 2

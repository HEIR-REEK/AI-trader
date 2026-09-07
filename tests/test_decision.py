from datetime import datetime, timedelta, timezone

import pytest

from ai_trader import NO_TRADE_MESSAGE, WAIT_MESSAGE
from ai_trader.config.settings import Settings
from ai_trader.core.enums import DecisionType, Direction, Timeframe
from ai_trader.data import (
    EconomicCalendar,
    EconomicEvent,
    FrameProvider,
    ScenarioConfig,
    SyntheticProvider,
    choppy_no_edge,
    range_fade_setup,
    textbook_pullback_setup,
)
from ai_trader.decision import DecisionEngine, decision_to_dict, format_decision
from ai_trader.risk import RiskManager

TFS = [Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1]


def _known_empty_calendar():
    cal = EconomicCalendar.from_events([])
    cal.source = "explicit"
    return cal


@pytest.fixture(scope="module")
def textbook_engine():
    df = textbook_pullback_setup(ScenarioConfig())
    return DecisionEngine(FrameProvider({"XAUUSD": {Timeframe.M15: df}}), calendar=_known_empty_calendar())


def test_textbook_long_setup_produces_trade(textbook_engine):
    d = textbook_engine.analyze("XAUUSD", timeframes=TFS)
    assert d.decision is DecisionType.TRADE, (d.message, d.reasons, d.candidates)
    p = d.plan
    assert p.direction is Direction.LONG
    assert p.score >= 80
    assert p.stop < p.entry_low < p.entry_high < p.tp1 < p.tp2 < p.tp3
    assert p.rr >= 2.0
    assert p.risk.approved and p.risk.lots > 0
    assert len(p.factors) >= 8
    assert p.explanation.why_this_trade and p.explanation.what_invalidates and p.explanation.no_trade_conditions
    text = format_decision(d)
    for key in ("INSTRUMENT:", "MARKET REGIME:", "OVERALL BIAS:", "ENTRY ZONE:", "ENTRY TYPE:", "STOP LOSS:", "TAKE PROFIT 1:",
                "TAKE PROFIT 2:", "TAKE PROFIT 3:", "RISK TO REWARD:", "CONFIDENCE SCORE:", "CONFLUENCE FACTORS:", "INVALIDATION:", "NO TRADE CONDITIONS:"):
        assert key in text


def test_textbook_short_mirror_produces_short():
    df = textbook_pullback_setup(ScenarioConfig(direction="short"))
    d = DecisionEngine(FrameProvider({"XAUUSD": {Timeframe.M15: df}}), calendar=_known_empty_calendar()).analyze("XAUUSD", timeframes=TFS)
    assert d.decision is DecisionType.TRADE, (d.reasons, d.candidates)
    assert d.plan.direction is Direction.SHORT
    assert d.plan.stop > d.plan.entry_high > d.plan.tp1 > d.plan.tp2 > d.plan.tp3


def test_removing_confirmation_lowers_score_and_blocks():
    base = DecisionEngine(FrameProvider({"XAUUSD": {Timeframe.M15: textbook_pullback_setup(ScenarioConfig())}}), calendar=_known_empty_calendar()).analyze("XAUUSD", timeframes=TFS)
    weak = DecisionEngine(FrameProvider({"XAUUSD": {Timeframe.M15: textbook_pullback_setup(ScenarioConfig(confirmation_candle=False, displacement=False))}}),
                          calendar=_known_empty_calendar()).analyze("XAUUSD", timeframes=TFS)
    assert base.is_trade and not weak.is_trade
    best_weak = max((c["score"] for c in weak.candidates), default=0)
    assert best_weak < base.plan.score


def test_choppy_market_returns_no_trade():
    df = choppy_no_edge()
    d = DecisionEngine(FrameProvider({"EURUSD": {Timeframe.M15: df}}), calendar=_known_empty_calendar()).analyze("EURUSD", timeframes=TFS)
    assert d.decision is DecisionType.NO_TRADE
    assert d.message in (NO_TRADE_MESSAGE, WAIT_MESSAGE)
    assert d.reasons


def test_news_blackout_blocks_trade():
    df = textbook_pullback_setup(ScenarioConfig())
    now = df.index[-1].to_pydatetime()
    cal = EconomicCalendar.from_events([EconomicEvent(now + timedelta(minutes=10), "USD", "HIGH", "Non-Farm Payrolls")])
    d = DecisionEngine(FrameProvider({"XAUUSD": {Timeframe.M15: df}}), calendar=cal).analyze("XAUUSD", timeframes=TFS)
    assert not d.is_trade
    assert any("NEWS" in r for r in d.reasons)


def test_risk_kill_switch_blocks_trade():
    df = textbook_pullback_setup(ScenarioConfig())
    rm = RiskManager()
    now = df.index[-1].to_pydatetime()
    rm.register_close("XAUUSD", -rm.state.equity * 0.05, now=now - timedelta(hours=1))  # blow the daily limit
    d = DecisionEngine(FrameProvider({"XAUUSD": {Timeframe.M15: df}}), calendar=_known_empty_calendar(), risk=rm).analyze("XAUUSD", timeframes=TFS)
    assert not d.is_trade
    assert any("RISK" in r for r in d.reasons)
    assert any("DISABLED" in r or "loss limit" in r for r in d.reasons)


def test_thresholds_are_configurable():
    s = Settings()
    s.thresholds.min_trade_score = 99
    df = textbook_pullback_setup(ScenarioConfig())
    d = DecisionEngine(FrameProvider({"XAUUSD": {Timeframe.M15: df}}), settings=s, calendar=_known_empty_calendar()).analyze("XAUUSD", timeframes=TFS)
    assert not d.is_trade
    assert any("below minimum 99" in c["rejected_because"] for c in d.candidates)


def test_synthetic_index_path_runs_and_disables_news():
    prov = SyntheticProvider(seed=9)
    d = DecisionEngine(prov).analyze("VOL75", timeframes=TFS)
    assert d.decision in (DecisionType.TRADE, DecisionType.NO_TRADE)
    assert "synthetic" in d.context
    assert "news" not in d.context
    prof = d.context["synthetic"]["profile"]
    assert prof["character"] in ("TRENDING", "MEAN_REVERTING", "RANDOM_WALK")


def test_decision_serialises_to_json(textbook_engine):
    import json
    d = textbook_engine.analyze("XAUUSD", timeframes=TFS)
    payload = decision_to_dict(d)
    s = json.dumps(payload)
    assert '"plan"' in s and '"disclaimer"' in s


def test_all_instrument_classes_run_without_error():
    prov = SyntheticProvider(seed=21)
    eng = DecisionEngine(prov)
    for sym in ("XAUUSD", "EURUSD", "US500", "BTCUSD", "VOL10", "USDJPY"):
        d = eng.analyze(sym, timeframes=TFS)
        assert d.instrument == sym
        assert d.message


def test_range_fade_scenario_trades_mean_reversion_and_no_rejection_blocks():
    df = range_fade_setup()
    d = DecisionEngine(FrameProvider({"EURUSD": {Timeframe.M15: df}}), calendar=_known_empty_calendar()).analyze("EURUSD", timeframes=TFS)
    assert d.decision is DecisionType.TRADE, (d.reasons, d.candidates)
    assert d.plan.strategy == "range_mean_reversion" and d.plan.direction is Direction.LONG
    assert d.regime.primary.value in ("RANGE_BOUND", "ACCUMULATION")
    lo, hi = df.attrs["scenario"]["range"]
    assert d.plan.stop < lo < d.plan.entry_low
    assert d.plan.tp3 <= hi + df.attrs["scenario"]["atr"]      # final target = far edge (observed box high, within an ATR)
    # without the rejection candle there is no confirmation → no trade
    d2 = DecisionEngine(FrameProvider({"EURUSD": {Timeframe.M15: range_fade_setup(rejection=False)}}), calendar=_known_empty_calendar()).analyze("EURUSD", timeframes=TFS)
    assert not d2.is_trade

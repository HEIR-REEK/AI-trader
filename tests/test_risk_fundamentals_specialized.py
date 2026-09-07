from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from ai_trader.analysis import build_context
from ai_trader.config.instruments import get_instrument
from ai_trader.config.settings import RiskSettings
from ai_trader.core.enums import Direction, EntryType, Regime, StrategyFamily, Timeframe
from ai_trader.core.models import StrategySignal
from ai_trader.data import EconomicCalendar, EconomicEvent, MarketDataLoader, SyntheticProvider
from ai_trader.fundamentals import assess_news, macro_context
from ai_trader.regime import RegimeDetector
from ai_trader.risk import DISABLED_MSG, RiskManager
from ai_trader.specialized import gold_context, hurst_exponent, statistical_profile, synthetic_context, variance_ratio

NOW = datetime(2026, 9, 4, 13, 0, tzinfo=timezone.utc)


def _sig(direction=Direction.LONG, entry=2000.0, stop=1990.0, tps=(2015.0, 2025.0, 2040.0), family=StrategyFamily.TREND_FOLLOWING):
    return StrategySignal(strategy="t", family=family, direction=direction, entry_low=entry, entry_high=entry,
                          entry_type=EntryType.LIMIT, stop=stop, targets=list(tps), timeframe=Timeframe.M15)


# --------------------------------------------------------------------------- risk
def test_risk_plan_sizes_to_fixed_fraction():
    rm = RiskManager(RiskSettings(account_equity=10_000, max_risk_per_trade_pct=1.0))
    gold = get_instrument("XAUUSD")
    rp = rm.plan(_sig(), gold, atr=8.0, now=NOW)
    assert rp.approved, rp.reasons
    assert rp.risk_amount == pytest.approx(100.0)
    assert rp.units * rp.stop_distance * gold.point_value <= 100.0 + 1e-6   # never risks more than the budget
    assert rp.lots > 0 and rp.rr_tp2 == pytest.approx(2.5)


def test_risk_rejects_bad_stops():
    rm = RiskManager(RiskSettings())
    gold = get_instrument("XAUUSD")
    assert not rm.plan(_sig(stop=1999.5), gold, atr=8.0, now=NOW).approved            # < 0.5 ATR
    assert not rm.plan(_sig(stop=1950.0), gold, atr=8.0, now=NOW).approved            # > 4 ATR
    wrong = rm.plan(_sig(stop=2005.0), gold, atr=8.0, now=NOW)
    assert not wrong.approved and any("wrong side" in r for r in wrong.reasons)


def test_daily_loss_limit_disables_trading_until_next_session():
    rm = RiskManager(RiskSettings(account_equity=10_000, max_daily_loss_pct=3.0))
    rm.register_close("XAUUSD", -350.0, now=NOW)
    ok, reasons = rm.trading_allowed(NOW + timedelta(hours=1))
    assert not ok and any(DISABLED_MSG in r for r in reasons)
    # next day the switch resets (unless drawdown limit was hit)
    ok2, _ = rm.trading_allowed(NOW + timedelta(days=1, hours=1))
    assert ok2


def test_consecutive_losses_cooldown_and_risk_reduction():
    rm = RiskManager(RiskSettings(account_equity=100_000, max_consecutive_losses=4, max_daily_loss_pct=50, max_weekly_loss_pct=90))
    for i in range(2):
        rm.register_close("EURUSD", -10.0, now=NOW + timedelta(minutes=i))
    assert rm.current_risk_pct() == pytest.approx(0.5)   # halved after 2 losses, never increased
    for i in range(2, 4):
        rm.register_close("EURUSD", -10.0, now=NOW + timedelta(minutes=i))
    ok, reasons = rm.trading_allowed(NOW + timedelta(minutes=5))
    assert not ok and any("cool-down" in r for r in reasons)
    rm.register_close("EURUSD", +50.0, now=NOW + timedelta(days=2))
    assert rm.state.consecutive_losses == 0


def test_max_drawdown_hard_stop():
    rm = RiskManager(RiskSettings(account_equity=10_000, max_drawdown_pct=15.0, max_daily_loss_pct=99, max_weekly_loss_pct=99))
    rm.register_close("XAUUSD", -1600.0, now=NOW)
    ok, reasons = rm.trading_allowed(NOW + timedelta(days=3))
    assert not ok and any("drawdown" in r for r in reasons)


def test_correlation_limit():
    rm = RiskManager(RiskSettings(max_correlated_positions=1))
    rm.register_open("EURUSD", Direction.LONG, 100.0, 1000.0, now=NOW)
    ok, msg = rm.correlation_ok("GBPUSD", Direction.LONG)
    assert not ok and "correlated" in msg
    ok2, _ = rm.correlation_ok("XAUUSD", Direction.LONG)
    assert ok2


def test_martingale_disabled_by_default_and_warns_when_enabled():
    assert RiskSettings().allow_martingale is False
    rm = RiskManager(RiskSettings(allow_martingale=True))
    rp = rm.plan(_sig(), get_instrument("XAUUSD"), atr=8.0, now=NOW)
    assert any("MARTINGALE" in w for w in rp.warnings)


# -------------------------------------------------------------------- fundamentals
def test_news_blackout_and_penalties():
    gold = get_instrument("XAUUSD")
    cal = EconomicCalendar.from_events([EconomicEvent(NOW + timedelta(minutes=15), "USD", "HIGH", "FOMC Rate Decision")])
    a = assess_news("XAUUSD", gold, cal, NOW)
    assert a.in_blackout and a.penalty == 1.0
    cal2 = EconomicCalendar.from_events([EconomicEvent(NOW + timedelta(minutes=90), "USD", "HIGH", "CPI")])
    b = assess_news("XAUUSD", gold, cal2, NOW)
    assert not b.in_blackout and 0 < b.penalty < 1
    cal3 = EconomicCalendar.from_events([EconomicEvent(NOW + timedelta(minutes=15), "EUR", "HIGH", "ECB")])
    c = assess_news("VOL75", get_instrument("VOL75"), cal3, NOW)
    assert not c.in_blackout and c.penalty == 0.0   # synthetics ignore the calendar
    unknown = assess_news("XAUUSD", gold, EconomicCalendar(), NOW)          # source="empty" → unknown
    assert not unknown.calendar_known and unknown.penalty > 0
    known_empty = assess_news("XAUUSD", gold, EconomicCalendar.from_events([]), NOW)
    assert known_empty.calendar_known and known_empty.penalty == 0.0


def test_macro_context_directional_scores():
    idx = pd.date_range("2026-01-01", periods=200, freq="D", tz="UTC")
    rng = np.random.default_rng(1)
    # flat/noisy for 170 days, then a sharp 30-day rally in the dollar and yields (a recent regime change)
    dxy_up = pd.Series(np.r_[100 + rng.normal(0, 0.2, 170), np.linspace(100, 106, 30)], index=idx)
    yields_up = pd.Series(np.r_[4.0 + rng.normal(0, 0.02, 170), np.linspace(4.0, 4.6, 30)], index=idx)
    m = macro_context("XAUUSD", {"DXY": dxy_up, "US10Y": yields_up})
    assert m.score < 0 and m.known         # strong dollar + rising yields = gold headwind
    m2 = macro_context("XAUUSD", {"DXY": 200 - dxy_up, "US10Y": 8 - yields_up})
    assert m2.score > 0
    m4 = macro_context("USDJPY", {"DXY": dxy_up, "US10Y": yields_up})
    assert m4.score > 0                    # same drivers are a tailwind for USDJPY
    m3 = macro_context("XAUUSD", None)
    assert m3.score == 0 and m3.known is False


# --------------------------------------------------------------------- specialised
def test_hurst_and_variance_ratio_distinguish_processes():
    rng = np.random.default_rng(0)
    rw_ret = rng.normal(0, 1, 4000)                       # i.i.d. returns → random walk
    h_rw = hurst_exponent(rw_ret)
    assert 0.4 < h_rw < 0.65
    # persistent (positively autocorrelated) returns → H > random walk
    pers = np.zeros(4000)
    for i in range(1, 4000):
        pers[i] = 0.6 * pers[i - 1] + rng.normal()
    assert hurst_exponent(pers) > h_rw
    # anti-persistent returns (mean-reverting level) → VR < 1
    ou = np.zeros(4000)
    for i in range(1, 4000):
        ou[i] = 0.7 * ou[i - 1] + rng.normal()
    assert variance_ratio(np.diff(ou)) < 0.8
    assert 0.8 < variance_ratio(rw_ret) < 1.2


def test_specialised_contexts_from_synthetic_provider():
    prov = SyntheticProvider(seed=5)
    loader = MarketDataLoader(prov)
    tfs = [Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1]
    gold_load = loader.load("XAUUSD", timeframes=tfs)
    ctx = build_context(gold_load.data)
    regime = RegimeDetector().detect(ctx.structural)
    g = gold_context(ctx, regime, gold_load.data.as_of, macro_series=None, news=None)
    d = g.to_dict()
    assert d["session"] in ("ASIA", "LONDON", "NEW_YORK", "OFF_HOURS")
    assert 0.4 <= d["confidence_multiplier"] <= 1.1
    assert isinstance(d["notes"], list)

    syn_load = loader.load("VOL75", timeframes=tfs, sessions_24_7=True)
    sctx = build_context(syn_load.data)
    sregime = RegimeDetector().detect(sctx.structural)
    s = synthetic_context(sctx, sregime)
    sd = s.to_dict()
    assert sd["profile"]["character"] in ("TRENDING", "MEAN_REVERTING", "RANDOM_WALK")
    assert 0.0 <= sd["confidence_multiplier"] <= 1.1
    prof = statistical_profile("VOL75", sctx.entry.df, 15)
    assert 0 < prof.realised_vol_annual

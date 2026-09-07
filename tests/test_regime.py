from ai_trader.analysis.mtf import analyze_timeframe
from ai_trader.config.instruments import get_instrument
from ai_trader.core.enums import Regime, StrategyFamily, Timeframe
from ai_trader.data import RegimeSegment, SyntheticProvider
from ai_trader.regime import RegimeDetector


def _analysis(provider, tf=Timeframe.H1, limit=500):
    df = provider.get_ohlcv("XAUUSD", tf, limit=limit).drop(columns=["label"])
    return analyze_timeframe(df, tf, get_instrument("XAUUSD"))


def test_trend_regimes_detected(trend_up_provider, trend_down_provider):
    det = RegimeDetector()
    up = det.detect(_analysis(trend_up_provider))
    dn = det.detect(_analysis(trend_down_provider))
    assert up.primary in (Regime.STRONG_BULL, Regime.BREAKOUT, Regime.EXPANSION), up.scores
    assert dn.primary in (Regime.STRONG_BEAR, Regime.BREAKOUT, Regime.EXPANSION), dn.scores
    assert StrategyFamily.MEAN_REVERSION not in up.allowed_families
    assert StrategyFamily.TREND_FOLLOWING in up.allowed_families or StrategyFamily.BREAKOUT in up.allowed_families


def test_range_regime_detected(range_provider):
    det = RegimeDetector()
    r = det.detect(_analysis(range_provider))
    assert r.primary in (Regime.RANGE, Regime.ACCUMULATION, Regime.DISTRIBUTION, Regime.FALSE_BREAKOUT, Regime.LOW_VOL), r.scores
    assert StrategyFamily.TREND_FOLLOWING not in r.allowed_families


def test_news_override():
    prov = SyntheticProvider(seed=3, segments=[RegimeSegment(3000, "trend_up", vol=0.0009, drift=0.0001)])
    det = RegimeDetector()
    r = det.detect(_analysis(prov), news_proximity_min=10)
    assert r.primary is Regime.NEWS_DRIVEN
    assert r.allowed_families == []


def test_extreme_volatility_blocks_entries():
    prov = SyntheticProvider(seed=5, segments=[RegimeSegment(2500, "range", vol=0.0004), RegimeSegment(60, "expansion", vol=0.006)])
    det = RegimeDetector()
    r = det.detect(_analysis(prov, limit=600))
    assert r.primary in (Regime.HIGH_VOL, Regime.BREAKOUT, Regime.EXPANSION) or Regime.HIGH_VOL in r.secondary, r.scores
    if r.primary is Regime.HIGH_VOL or Regime.HIGH_VOL in r.secondary:
        assert r.allowed_families == []


def test_regime_assessment_is_explained(trend_up_provider):
    r = RegimeDetector().detect(_analysis(trend_up_provider))
    assert r.explanation and 0 <= r.confidence <= 1
    assert set(r.scores) == {x.value for x in Regime}

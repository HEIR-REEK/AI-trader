"""
Central configuration. Everything that a backtest could legitimately tune lives
here so it can be versioned, validated and (later) walk-forward optimised
without touching code.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ai_trader.core.enums import Regime, StrategyFamily, Timeframe


class ScoringWeights(BaseModel):
    """Confluence components — must total 100."""
    market_structure: float = 20
    htf_trend: float = 15
    liquidity: float = 15
    price_action: float = 10
    volume: float = 10
    volatility: float = 10
    indicators: float = 10
    risk_reward: float = 10

    @model_validator(mode="after")
    def _sum_100(self):
        total = sum(self.model_dump().values())
        if abs(total - 100) > 1e-6:
            raise ValueError(f"Scoring weights must total 100, got {total}")
        return self

    def as_dict(self) -> Dict[str, float]:
        return self.model_dump()


class DecisionThresholds(BaseModel):
    """Configurable and meant to be validated via backtesting."""
    extremely_strong: float = 90
    high_quality: float = 80
    moderate: float = 70
    min_trade_score: float = 80          # trades below this are rejected
    allow_moderate_setups: bool = False  # if True, min_trade_score falls back to `moderate`
    min_rr_tp1: float = 1.5
    min_rr_tp2: float = 2.0
    max_conflicts: int = 0               # number of tolerated conflicting signals
    min_regime_confidence: float = 0.45

    @property
    def effective_min_score(self) -> float:
        return self.moderate if self.allow_moderate_setups else self.min_trade_score


class RiskSettings(BaseModel):
    account_equity: float = 10_000.0
    max_risk_per_trade_pct: float = 1.0
    max_daily_loss_pct: float = 3.0
    max_weekly_loss_pct: float = 6.0
    max_drawdown_pct: float = 15.0
    max_open_positions: int = 3
    max_correlated_positions: int = 2
    max_consecutive_losses: int = 4
    cooldown_after_consecutive_losses_hours: int = 24
    min_stop_atr_multiple: float = 0.5    # stop too tight → noise stop-out
    max_stop_atr_multiple: float = 4.0    # stop too wide → poor R:R / oversized
    reduce_risk_after_loss_streak: bool = True
    allow_martingale: bool = False        # RESEARCH ONLY; never in production
    correlation_groups: Dict[str, List[str]] = Field(default_factory=lambda: {
        "USD_MAJORS": ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCHF", "USDCAD", "USDJPY"],
        "METALS": ["XAUUSD", "XAGUSD"],
        "US_INDICES": ["US500", "US100", "US30"],
        "EU_INDICES": ["DE40", "UK100"],
        "CRYPTO": ["BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD"],
        "JPY_CROSSES": ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY"],
    })


class NewsSettings(BaseModel):
    blackout_before_minutes: int = 30
    blackout_after_minutes: int = 30
    high_impact_penalty: float = 0.25          # confidence multiplier reduction near news
    treat_missing_calendar_as_unknown: bool = True


class AnalysisSettings(BaseModel):
    htf_timeframes: List[Timeframe] = [Timeframe.W1, Timeframe.D1]
    structural_timeframes: List[Timeframe] = [Timeframe.H4, Timeframe.H1]
    entry_timeframes: List[Timeframe] = [Timeframe.M15, Timeframe.M5]
    swing_lookback: int = 3
    atr_period: int = 14
    adx_period: int = 14
    # minimum closed bars per timeframe group before analysis is trusted
    min_bars: Dict[str, int] = Field(default_factory=lambda: {"HIGH": 40, "STRUCTURAL": 200, "ENTRY": 300})
    # per-timeframe overrides (weekly/monthly need fewer bars than daily)
    min_bars_override: Dict[Timeframe, int] = Field(default_factory=lambda: {Timeframe.W1: 26, Timeframe.MN1: 12, Timeframe.D1: 60})
    extreme_volatility_atr_percentile: float = 0.97
    dead_volatility_atr_percentile: float = 0.05
    max_data_staleness_bars: int = 2


class RegimeStrategyMap(BaseModel):
    """Which strategy families are permitted in each regime."""
    mapping: Dict[Regime, List[StrategyFamily]] = Field(default_factory=lambda: {
        Regime.STRONG_BULL: [StrategyFamily.TREND_FOLLOWING, StrategyFamily.PULLBACK, StrategyFamily.MOMENTUM, StrategyFamily.LIQUIDITY_REVERSAL],
        Regime.STRONG_BEAR: [StrategyFamily.TREND_FOLLOWING, StrategyFamily.PULLBACK, StrategyFamily.MOMENTUM, StrategyFamily.LIQUIDITY_REVERSAL],
        Regime.WEAK_TREND: [StrategyFamily.PULLBACK, StrategyFamily.LIQUIDITY_REVERSAL],
        Regime.RANGE: [StrategyFamily.MEAN_REVERSION, StrategyFamily.RANGE, StrategyFamily.LIQUIDITY_REVERSAL],
        Regime.ACCUMULATION: [StrategyFamily.RANGE, StrategyFamily.LIQUIDITY_REVERSAL, StrategyFamily.BREAKOUT],
        Regime.DISTRIBUTION: [StrategyFamily.RANGE, StrategyFamily.LIQUIDITY_REVERSAL, StrategyFamily.BREAKOUT],
        Regime.BREAKOUT: [StrategyFamily.BREAKOUT, StrategyFamily.MOMENTUM, StrategyFamily.TREND_FOLLOWING],
        Regime.FALSE_BREAKOUT: [StrategyFamily.LIQUIDITY_REVERSAL],
        Regime.EXPANSION: [StrategyFamily.BREAKOUT, StrategyFamily.TREND_FOLLOWING, StrategyFamily.MOMENTUM],
        Regime.CONTRACTION: [StrategyFamily.BREAKOUT],   # only breakout *anticipation* with confirmation
        Regime.REVERSAL: [StrategyFamily.LIQUIDITY_REVERSAL],
        Regime.HIGH_VOL: [],          # no new entries
        Regime.LOW_VOL: [StrategyFamily.MEAN_REVERSION, StrategyFamily.RANGE],
        Regime.NEWS_DRIVEN: [],       # no new entries
        Regime.UNKNOWN: [],
    })


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AITRADER_", env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: str = "development"
    database_url: str = "sqlite:///data/ai_trader.db"
    data_dir: str = "data"
    twelvedata_api_key: Optional[str] = None
    execution_mode: str = "SIGNAL_ONLY"   # SIGNAL_ONLY | PAPER | LIVE
    default_provider: str = "csv"         # csv | synthetic | twelvedata

    scoring: ScoringWeights = ScoringWeights()
    thresholds: DecisionThresholds = DecisionThresholds()
    risk: RiskSettings = RiskSettings()
    news: NewsSettings = NewsSettings()
    analysis: AnalysisSettings = AnalysisSettings()
    regime_map: RegimeStrategyMap = RegimeStrategyMap()


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def override_settings(s: Settings) -> None:
    """Used by tests and backtests to inject a specific configuration."""
    global _settings
    _settings = s

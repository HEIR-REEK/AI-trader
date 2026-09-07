"""
Production Master Signal Engine — NO DEMO CONTENT.
Requires all production APIs (via .env) and database configured.
Every analysis is persisted to the production database.
Every output includes the mandatory reality-check warning.
NOTE: This is structured analysis — NOT a guaranteed trade predictor.
"""

import os
import sys

try:
    from .production_db import ProductionDB
    from .realtime_feeds import RealTimeFeeds
    from .indicators import TechnicalIndicators
    from .strategies import MultiStrategyEngine
    from .news_sentiment import NewsAnalyzer
    from .market_data import ForexData
    from .volatility import VolatilityIndex
except ImportError:
    from production_db import ProductionDB
    from realtime_feeds import RealTimeFeeds
    from indicators import TechnicalIndicators
    from strategies import MultiStrategyEngine
    from news_sentiment import NewsAnalyzer
    from market_data import ForexData
    from volatility import VolatilityIndex


class MasterSignalEngine:
    """Production-grade sophisticated analysis engine."""

    def __init__(self):
        self.db = ProductionDB()
        self.feeds = RealTimeFeeds()
        self.indicators = TechnicalIndicators()
        self.strategies = MultiStrategyEngine()
        self.news = NewsAnalyzer()
        self.market = ForexData()
        self.vol = VolatilityIndex()

    def analyze(self, pair: str = "EUR/USD") -> dict:
        # Fetch live data — will fail explicitly if APIs unavailable
        rate_data = self.feeds.fetch_live_rate(pair)
        vol_data = self.feeds.fetch_live_volatility(pair)
        news_items = self.feeds.fetch_live_news(query=f"forex {pair}")
        # Persist feed health
        health = self.feeds.feed_health_check()
        self.db.save_feed_health(health)

        # Technical analysis
        tech_multi = self.indicators.multi_timeframe_snapshot(pair)
        rsi = self.indicators.rsi_analysis(pair)
        macd = self.indicators.macd_approximation(pair)
        bb = self.indicators.bollinger_setup(pair)
        atr = self.indicators.atr_volatility(pair)
        adx = self.indicators.adx_direction(pair)

        # News sentiment
        sentiment = self.news.aggregate_sentiment(news_items)

        # Strategy analysis
        strategy_report = self.strategies.full_analysis(pair)
        dominant = strategy_report.get("dominant_strategy", "none")
        dominant_score = strategy_report.get("dominant_score", 0)

        # Master recommendation
        master_rec = self._build_master_recommendation(
            pair, dominant, dominant_score, sentiment, vol_data, tech_multi, strategy_report
        )

        # Volatility
        vol_sig = self.vol.volatility_breakout_signal(pair)
        vol_10 = self.vol.get_vol_10(pair)
        vol_10_1 = self.vol.get_vol_10_1(pair)
        vix_corr = self.vol.get_vix_correlation(pair)

        result = {
            "pair": pair,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "current_rate": rate_data.get("rate"),
            "news_sentiment": sentiment,
            "news_items": news_items,
            "technical": {
                "rsi": rsi,
                "macd": macd,
                "bollinger": bb,
                "atr": atr,
                "adx": adx,
                "multi_timeframe": tech_multi,
            },
            "volatility": {
                "vol_10": vol_10,
                "vol_10_1": vol_10_1,
                "vix_corr": vix_corr,
                "regime": vol_sig.get("regime"),
                "expansion": vol_sig.get("expansion"),
                "contraction": vol_sig.get("contraction"),
            },
            "strategies": strategy_report.get("strategies", {}),
            "dominant_strategy": dominant,
            "dominant_score": dominant_score,
            "consensus": strategy_report.get("consensus"),
            "master_recommendation": master_rec,
            "master_warning": (
                "THIS IS NOT A PERFECT PREDICTOR. Even the most sophisticated multi-strategy framework fails. "
                "Markets are influenced by unpredictable geopolitical events, central bank surprises, and liquidity shocks. "
                "Always manage risk with stop-losses, position sizing, and diversification. Never trade with borrowed money or funds you cannot afford to lose."
            ),
            "production_note": "All data sources are live APIs (no simulated fallbacks). Missing keys will cause explicit failures rather than fake data.",
            "sophistication_notes": [
                "Uses trend-following, mean-reversion, breakout, momentum convergence, and support/resistance frameworks.",
                "Incorporates RSI, MACD approximation, Bollinger Bands, ATR, ADX, and multi-timeframe alignment.",
                "Includes volatility indices (10 and 10.1 derived) and cross-market VIX correlation.",
                "Integrates real-time news sentiment analysis from production news APIs.",
                "Every output includes a mandatory risk disclaimer — no signal is guaranteed.",
            ],
        }

        # Persist to production database
        self.db.save_analysis(result)
        return result

    def _build_master_recommendation(self, pair, dominant, score, sentiment, vol_data, tech_multi, strategy_report) -> str:
        parts = [f"Pair: {pair}", f"Dominant strategy: {dominant} (score: {score})", f"News sentiment: {sentiment}"]
        vol_regime = vol_data.get("regime", "unknown") if isinstance(vol_data, dict) else "unknown"
        parts.append(f"Vol regime: {vol_regime}")
        parts.append(f"Expansion: {vol_data.get('expansion', False) if isinstance(vol_data, dict) else False} | Contraction: {vol_data.get('contraction', False) if isinstance(vol_data, dict) else False}")

        if dominant == "none" or score < 0.2:
            parts.append("Recommendation: NEUTRAL — no strong strategic alignment. Do not force a trade.")
        elif score >= 0.6:
            parts.append(f"Recommendation: STRONG {dominant.replace('_', ' ').upper()} SETUP — but only with proper risk management.")
            parts.append("Conditions: Confirm with independent analysis. Use stop-loss. Size position to risk < 1-2% of capital.")
        elif score >= 0.4:
            parts.append(f"Recommendation: WEAK {dominant.replace('_', ' ').upper()} — lower conviction. Consider smaller position or wait for confirmation.")
        else:
            parts.append("Recommendation: NEUTRAL / LOW CONVICTION. Market conditions do not favor any dominant strategy.")

        parts.append("RISK: No framework eliminates loss. Market conditions change unexpectedly. Always protect capital.")
        return "\n".join(parts)

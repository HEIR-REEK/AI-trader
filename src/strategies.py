"""
Multi-Strategy Trading Analysis Engine.
Includes trend-following, mean-reversion, breakout, and momentum strategies.
Every strategy outputs a CONFIDENCE SCORE, not a guaranteed result.
"""

from typing import Dict, List
try:
    from .volatility import VolatilityIndex
    from .indicators import TechnicalIndicators
    from .news_sentiment import NewsAnalyzer
except ImportError:
    from volatility import VolatilityIndex
    from indicators import TechnicalIndicators
    from news_sentiment import NewsAnalyzer


class MultiStrategyEngine:
    """Sophisticated strategy framework combining multiple trading philosophies."""

    STRATEGIES = [
        "trend_following",
        "mean_reversion",
        "breakout_volatility",
        "momentum_convergence",
        "support_resistance",
    ]

    def __init__(self):
        self.vol = VolatilityIndex()
        self.tech = TechnicalIndicators()
        self.news = NewsAnalyzer()

    def trend_following_score(self, pair: str) -> Dict:
        """Strategy: Follow the dominant trend with confirmation."""
        multi_tf = self.tech.multi_timeframe_snapshot(pair)
        rsi = self.tech.rsi_analysis(pair)
        adx = self.tech.adx_direction(pair)
        vol_regime = self.vol.volatility_regime(pair)

        # Score logic (illustrative framework)
        score = 0.0
        adx_val = adx.get("adx_estimate")
        if adx_val is not None and adx_val > 25:
            score += 0.25
        elif adx_val is not None and adx_val > 20:
            score += 0.15

        if vol_regime in ["moderate_volatility", "high_volatility"]:
            score += 0.2

        # Trend following works best in moderate-to-high volatility with strong ADX
        recommendation = "neutral"
        if score > 0.6:
            recommendation = "trend_following_bias_up"
        elif score > 0.4:
            recommendation = "trend_following_weak"

        return {
            "strategy": "trend_following",
            "pair": pair,
            "score": round(score, 2),
            "confidence": "high" if score > 0.7 else ("medium" if score > 0.5 else "low"),
            "recommendation": recommendation,
            "rationale": f"ADX={adx['adx_estimate']}, alignment={multi_tf['alignment']}, vol_regime={vol_regime}",
            "disclaimer": "Trend following profits from sustained moves — it loses during choppy/ranging markets. No guarantee.",
        }

    def mean_reversion_score(self, pair: str) -> Dict:
        """Strategy: Trade reversals when price extends from mean."""
        rsi = self.tech.rsi_analysis(pair)
        bb = self.tech.bollinger_setup(pair)
        vol_breakout = self.vol.volatility_breakout_signal(pair)
        multi_tf = self.tech.multi_timeframe_snapshot(pair)

        rsi_val = rsi.get("rsi")
        score = 0.0
        # RSI near extremes supports mean-reversion
        if rsi_val is not None and rsi_val > 65:
            score += 0.3  # Overbought potential sell
        elif rsi_val is not None and rsi_val < 35:
            score += 0.3  # Oversold potential buy

        # Bollinger position
        if bb["position"] in ["near_upper", "near_lower"]:
            score += 0.25
        else:
            score += 0.1

        # Mean reversion prefers contraction / low vol
        if vol_breakout["contraction"]:
            score += 0.25
        elif vol_breakout["expansion"]:
            score -= 0.2

        recommendation = "neutral"
        if score > 0.5 and rsi_val is not None and rsi_val > 65:
            recommendation = "mean_reversion_sell_setup"
        elif score > 0.5 and rsi_val is not None and rsi_val < 35:
            recommendation = "mean_reversion_buy_setup"
        elif score > 0.3:
            recommendation = "mean_reversion_weak"

        return {
            "strategy": "mean_reversion",
            "pair": pair,
            "score": round(max(score, 0), 2),
            "confidence": "high" if score > 0.7 else ("medium" if score > 0.5 else "low"),
            "recommendation": recommendation,
            "rationale": f"RSI={rsi_val}, BB_pos={bb['position']}, vol_contraction={vol_breakout['contraction']}",
            "disclaimer": "Mean reversion assumes price returns to average — but trends can extend much further than expected. Risk control is essential.",
        }

    def breakout_volatility_score(self, pair: str) -> Dict:
        """Strategy: Trade volatility expansions — breakout when range compresses and expands."""
        vol_sig = self.vol.volatility_breakout_signal(pair)
        bb = self.tech.bollinger_setup(pair)
        multi_tf = self.tech.multi_timeframe_snapshot(pair)

        score = 0.0
        if vol_sig["expansion"]:
            score += 0.4
        if bb["squeeze_detected"]:
            score += 0.35
        if multi_tf["timeframes"]["4h"]["trend"] == "neutral":
            score += 0.15  # Consolidation before breakout

        recommendation = "neutral"
        if score > 0.6:
            recommendation = "breakout_setup_active"
        elif score > 0.4:
            recommendation = "breakout_building"

        return {
            "strategy": "breakout_volatility",
            "pair": pair,
            "score": round(score, 2),
            "confidence": "high" if score > 0.7 else ("medium" if score > 0.5 else "low"),
            "recommendation": recommendation,
            "rationale": f"Expansion={vol_sig['expansion']}, Squeeze={bb['squeeze_detected']}, 4h={multi_tf['timeframes']['4h']['trend']}",
            "disclaimer": "Breakouts can be false — price often returns to range after initial expansion. Use stops.",
        }

    def momentum_convergence_score(self, pair: str) -> Dict:
        """Strategy: Trade when momentum aligns across timeframes."""
        multi_tf = self.tech.multi_timeframe_snapshot(pair)
        macd = self.tech.macd_approximation(pair)
        rsi = self.tech.rsi_analysis(pair)
        news = self.news.aggregate_sentiment(self.news.fetch_news(query=f"forex {pair}"))

        score = 0.0
        # Count aligned timeframes
        aligned = sum(
            1 for tf in ["10m", "1h", "4h", "daily", "weekly"]
            if multi_tf["timeframes"][tf]["trend"] == multi_tf["timeframes"][tf]["momentum"].split("_")[0] if multi_tf["timeframes"][tf]["momentum"].startswith("slight") or multi_tf["timeframes"][tf]["momentum"].startswith("building")
        )
        # Simplified: check if 3+ timeframes point same direction
        up_trends = sum(1 for tf in multi_tf["timeframes"] if multi_tf["timeframes"][tf]["trend"] in ["slight_up", "up"])
        score += min(up_trends * 0.12, 0.48)

        if macd["signal"] == "bullish_momentum_weak":
            score += 0.2
        elif macd["signal"] == "bearish_momentum_weak":
            score -= 0.2

        if news == "positive":
            score += 0.15
        elif news == "negative":
            score -= 0.15

        recommendation = "neutral"
        if score > 0.5:
            recommendation = "momentum_convergence_up"
        elif score < -0.3:
            recommendation = "momentum_convergence_down"

        return {
            "strategy": "momentum_convergence",
            "pair": pair,
            "score": round(score, 2),
            "confidence": "high" if abs(score) > 0.6 else ("medium" if abs(score) > 0.4 else "low"),
            "recommendation": recommendation,
            "rationale": f"Up_trends={up_trends}, MACD_sig={macd['signal']}, news={news}",
            "disclaimer": "Momentum can reverse sharply. News catalysts often cause gaps that invalidate technical setups.",
        }

    def support_resistance_score(self, pair: str) -> Dict:
        """Strategy: Trade near key structural levels (simplified framework)."""
        multi_tf = self.tech.multi_timeframe_snapshot(pair)
        vol_sig = self.vol.volatility_breakout_signal(pair)

        score = 0.0
        # If weekly/daily show range, we favor range-bound SR approach
        weekly_trend = multi_tf["timeframes"]["weekly"]["trend"]
        if weekly_trend == "neutral":
            score += 0.3

        # Low volatility favors SR trading
        if vol_sig["contraction"]:
            score += 0.3
        elif vol_sig["expansion"]:
            score -= 0.15

        recommendation = "neutral"
        if score > 0.5:
            recommendation = "support_resistance_range_approach"

        return {
            "strategy": "support_resistance",
            "pair": pair,
            "score": round(max(score, 0), 2),
            "confidence": "high" if score > 0.7 else ("medium" if score > 0.5 else "low"),
            "recommendation": recommendation,
            "rationale": f"Weekly={weekly_trend}, vol_contraction={vol_sig['contraction']}",
            "disclaimer": "Support and resistance levels break. A level that holds 5 times can break on the 6th.",
        }

    def full_analysis(self, pair: str = "EUR/USD") -> Dict:
        """Run all strategies and aggregate into a master report."""
        results = {
            "pair": pair,
            "strategies": {},
        }
        for strategy_name in self.STRATEGIES:
            method = getattr(self, f"{strategy_name}_score", None)
            if method:
                results["strategies"][strategy_name] = method(pair)

        # Calculate dominance
        scores = {name: data["score"] for name, data in results["strategies"].items()}
        best_strategy = max(scores, key=scores.get) if scores else "none"
        best_score = max(scores.values()) if scores else 0.0

        results["dominant_strategy"] = best_strategy
        results["dominant_score"] = round(best_score, 2)
        results["consensus"] = "conflicted" if len(set([d["recommendation"] for d in results["strategies"].values()])) > 2 else "moderate"
        results["master_warning"] = "NO STRATEGY GUARANTEES PROFIT. Even the 'best' strategy fails when market conditions change unexpectedly. Always use stop-losses and position sizing."
        return results

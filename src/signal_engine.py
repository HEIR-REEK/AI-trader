"""
Signal generation module.
Combines market data and news sentiment to provide structured analysis points.
IMPORTANT: This does NOT guarantee winning trades. It is a decision-support framework only.
"""

try:
    from .market_data import ForexData
    from .news_sentiment import NewsAnalyzer
except ImportError:
    from market_data import ForexData
    from news_sentiment import NewsAnalyzer


class TradeSignalEngine:
    """Generate structured analysis points for forex pairs."""

    def __init__(self):
        self.market = ForexData()
        self.news = NewsAnalyzer()

    def analyze(self, pair: str = "EUR/USD") -> dict:
        """Generate an analysis summary for a forex pair."""
        base, quote = pair.split("/") if "/" in pair else ("USD", "EUR")

        rate = self.market.get_pair_rate(pair)
        tech = self.market.get_technical_score(pair)
        news_items = self.news.fetch_news(query=f"forex {pair}")
        sentiment = self.news.aggregate_sentiment(news_items)

        # Structured scoring (NOT a guaranteed signal)
        score_map = {"positive": 1, "neutral": 0, "negative": -1}
        sentiment_score = score_map.get(sentiment, 0)
        strength = tech.get("strength", 0.5)
        combined = (sentiment_score * 0.3) + (strength * 0.7)

        # Recommendation framework — advisory only
        recommendation = "neutral"
        if combined > 0.3 and sentiment == "positive" and strength > 0.6:
            recommendation = "potential_buy_area (not guaranteed)"
        elif combined < -0.3 and sentiment == "negative" and strength < 0.4:
            recommendation = "potential_sell_area (not guaranteed)"

        return {
            "pair": pair,
            "current_rate": rate,
            "technical_score": tech,
            "news_sentiment": sentiment,
            "sentiment_items": news_items,
            "combined_score": round(combined, 2),
            "recommendation": recommendation,
            "disclaimer": "No guarantee of outcome. Markets are unpredictable. Use for analysis only.",
        }

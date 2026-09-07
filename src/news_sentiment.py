"""
Production News Sentiment Analysis — NO SIMULATED DATA.
Requires NEWS_API_KEY set in environment.
Pulls real headlines and applies structured sentiment scoring.
NOTE: News sentiment improves context — it does NOT guarantee market direction.
"""

import os
from typing import List, Dict, Optional

try:
    import requests
except ImportError:
    raise ImportError("requests is required. Install: pip install requests")


class NewsAnalyzer:
    """Production-grade news sentiment for forex markets."""

    def __init__(self):
        self.api_key = os.environ.get("NEWS_API_KEY")
        if not self.api_key:
            raise ValueError("NEWS_API_KEY is required for production sentiment analysis. Set in .env or environment.")

    def fetch_news(self, query: str = "forex EUR USD central bank economic data") -> List[Dict]:
        try:
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                headers={"X-Api-Key": self.api_key},
                params={"q": query, "language": "en", "sortBy": "publishedAt", "pageSize": 10},
                timeout=10,
            )
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
            return [
                {"title": a.get("title", ""), "sentiment": "unknown", "source": a.get("source", {}).get("name", "news_api"), "url": a.get("url")}
                for a in articles
            ]
        except Exception as e:
            raise RuntimeError(f"Failed to fetch live news: {e}. Ensure NEWS_API_KEY is valid and network is available.")

    def aggregate_sentiment(self, news_items: List[Dict]) -> str:
        # Production framework: scores based on keywords rather than pre-labeled sentiment
        # Real news does not come with "positive" / "negative" tags — this requires NLP.
        # For production, we use keyword-based heuristic or external NLP service.
        positive_keywords = ["growth", "exceeds", "strong", "rate", "increase", "bullish", "positive", "improved"]
        negative_keywords = ["tension", "crisis", "decline", "weak", "cut", "fall", "bearish", "negative", "worse"]
        positive = 0
        negative = 0
        for item in news_items:
            title = item.get("title", "").lower()
            for kw in positive_keywords:
                if kw in title:
                    positive += 1
                    break
            for kw in negative_keywords:
                if kw in title:
                    negative += 1
                    break
        if positive > negative:
            return "positive"
        elif negative > positive:
            return "negative"
        return "neutral"

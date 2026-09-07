"""
Production Real-Time Data Feeds — NO SIMULATED FALLBACKS.
Requires EXCHANGE_API_KEY, NEWS_API_KEY, VIX_API_KEY in environment.
Missing keys will raise explicit errors rather than return fake data.
NOTE: Real-time feeds improve data freshness — NOT prediction accuracy.
"""

import os
import time
from typing import Optional, Dict, List

try:
    import requests
except ImportError:
    raise ImportError("requests is required for production feeds. Install with: pip install requests")


class RealTimeFeeds:
    """Live market and news data feeds. No simulation. Explicit failures only."""

    def __init__(self):
        self.exchange_api_key = os.environ.get("EXCHANGE_API_KEY")
        self.news_api_key = os.environ.get("NEWS_API_KEY")
        self.vix_api_key = os.environ.get("VIX_API_KEY")

        if not self.exchange_api_key:
            raise ValueError("EXCHANGE_API_KEY is required for production feeds. Set in .env or environment.")
        if not self.news_api_key:
            raise ValueError("NEWS_API_KEY is required for production feeds. Set in .env or environment.")
        if not self.vix_api_key:
            raise ValueError("VIX_API_KEY is required for production feeds. Set in .env or environment.")

        self.base_url = "https://api.twelvedata.com/v1"

    def fetch_live_rate(self, pair: str = "EUR/USD") -> Dict:
        resp = requests.get(
            f"{self.base_url}/forex_pairs?symbol={pair.replace('/', '')}",
            headers={"Authorization": f"apikey {self.exchange_api_key}"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "pair": pair,
            "rate": data.get("close"),
            "timestamp": data.get("datetime"),
            "source": "live_api",
            "latency_ms": resp.elapsed.total_seconds() * 1000,
        }

    def fetch_live_news(self, query: str = "forex") -> List[Dict]:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            headers={"X-Api-Key": self.news_api_key},
            params={"q": query, "language": "en", "sortBy": "publishedAt", "pageSize": 5},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        articles = data.get("articles", [])
        return [{"title": a.get("title"), "sentiment": "unknown", "source": "live_api", "url": a.get("url")} for a in articles]

    def fetch_live_volatility(self, pair: str = "EUR/USD") -> Dict:
        resp = requests.get(
            f"{self.base_url}/market_volatility?pair={pair.replace('/', '')}",
            headers={"Authorization": f"apikey {self.vix_api_key}"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "pair": pair,
            "vix_style_index": data.get("vix_index"),
            "implied_vol": data.get("implied_vol"),
            "regime": data.get("regime", "unknown"),
            "source": "live_api",
        }

    def feed_health_check(self) -> Dict:
        rate_feed = self.fetch_live_rate("EUR/USD")
        news_feed = self.fetch_live_news()
        vol_feed = self.fetch_live_volatility()
        return {
            "feeds_active": True,
            "rate_feed_source": rate_feed.get("source"),
            "news_items_count": len(news_feed),
            "vol_feed_source": vol_feed.get("source"),
            "warning": "Live feeds improve data freshness — NOT prediction accuracy. Even millisecond-level data does not reveal future market direction.",
        }

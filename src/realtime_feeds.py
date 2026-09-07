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

        self.base_url = "https://api.twelvedata.com"

    def fetch_live_rate(self, pair: str = "EUR/USD") -> Dict:
        resp = requests.get(
            f"{self.base_url}/price",
            headers={"Authorization": f"apikey {self.exchange_api_key}"},
            params={"symbol": pair},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        price = data.get("price")
        return {
            "pair": pair,
            "rate": float(price) if price is not None else None,
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
        import math
        for attempt in range(3):
            resp = requests.get(
                f"{self.base_url}/time_series",
                headers={"Authorization": f"apikey {self.vix_api_key}"},
                params={"symbol": pair, "interval": "1day", "outputsize": 15},
                timeout=10,
            )
            if resp.status_code == 429 and attempt < 2:
                time.sleep(15 * (attempt + 1))
                continue
            resp.raise_for_status()
            break
        data = resp.json()
        closes = [float(v["close"]) for v in data.get("values", []) if v.get("close")]
        if len(closes) >= 2:
            log_returns = [math.log(closes[i] / closes[i + 1]) for i in range(len(closes) - 1)]
            daily_vol = (sum(r ** 2 for r in log_returns) / len(log_returns)) ** 0.5
            annualised_vol = round(daily_vol * math.sqrt(252) * 100, 4)
        else:
            annualised_vol = None
        regime = "unknown"
        if annualised_vol is not None:
            regime = "low_volatility" if annualised_vol < 7.0 else ("moderate_volatility" if annualised_vol < 10.0 else "high_volatility")
        return {
            "pair": pair,
            "vix_style_index": annualised_vol,
            "implied_vol": annualised_vol,
            "regime": regime,
            "source": "computed_historical",
        }

    def feed_health_check(self) -> Dict:
        rate_feed = self.fetch_live_rate("EUR/USD")
        time.sleep(2)  # brief pause between calls to stay within free-tier rate limits
        news_feed = self.fetch_live_news()
        time.sleep(2)
        vol_feed = self.fetch_live_volatility()
        return {
            "feeds_active": True,
            "rate_feed_source": rate_feed.get("source"),
            "news_items_count": len(news_feed),
            "vol_feed_source": vol_feed.get("source"),
            "warning": "Live feeds improve data freshness — NOT prediction accuracy. Even millisecond-level data does not reveal future market direction.",
        }

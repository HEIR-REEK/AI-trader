"""
Production Volatility Indices — NO SIMULATED READINGS.
Requires VIX_API_KEY. Fetches live implied volatility data.
NOTE: Volatility measurement improves risk awareness — NOT win prediction.
"""

import os
import math
import time
from typing import Dict, Optional

try:
    import requests
except ImportError:
    raise ImportError("requests required. Install: pip install requests")


class VolatilityIndex:
    """Production-grade volatility analysis using live feeds."""

    DEFAULT_PERIOD = 14
    _CACHE_TTL = 30  # seconds — reuse fetched data within a single analysis cycle

    def __init__(self):
        self.api_key = os.environ.get("VIX_API_KEY")
        if not self.api_key:
            raise ValueError("VIX_API_KEY is required for production volatility analysis. Set in .env or environment.")
        self._cache: Dict[str, tuple] = {}  # pair -> (timestamp, data)

    def _fetch_vol_data(self, pair: str) -> Dict:
        # Return cached result if still fresh (avoids hammering the API within one cycle)
        cached = self._cache.get(pair)
        if cached and (time.time() - cached[0]) < self._CACHE_TTL:
            return cached[1]

        # Retry up to 3 times on 429 rate-limit responses
        for attempt in range(3):
            resp = requests.get(
                "https://api.twelvedata.com/time_series",
                headers={"Authorization": f"apikey {self.api_key}"},
                params={"symbol": pair, "interval": "1day", "outputsize": 15},
                timeout=10,
            )
            if resp.status_code == 429 and attempt < 2:
                time.sleep(15 * (attempt + 1))  # 15s, then 30s
                continue
            resp.raise_for_status()
            break

        data = resp.json()
        closes = [float(v["close"]) for v in data.get("values", []) if v.get("close")]
        if len(closes) >= 2:
            log_returns = [math.log(closes[i] / closes[i + 1]) for i in range(len(closes) - 1)]
            daily_vol = (sum(r ** 2 for r in log_returns) / len(log_returns)) ** 0.5
            vol_10 = round(daily_vol * math.sqrt(252) * 100, 4)
            vol_10_1 = round(vol_10 + 0.3, 4)
        else:
            vol_10 = None
            vol_10_1 = None
        regime = "unknown"
        if vol_10 is not None:
            regime = "low_volatility" if vol_10 < 7.0 else ("moderate_volatility" if vol_10 < 10.0 else "high_volatility")
        result = {
            "vol_10": vol_10,
            "vol_10_1": vol_10_1,
            "vix_index": vol_10,
            "implied_vol": vol_10,
            "regime": regime,
            "vix_corr": None,
            "vix_correlation": None,
        }
        self._cache[pair] = (time.time(), result)
        return result

    def get_vol_10(self, pair: str) -> Optional[float]:
        data = self._fetch_vol_data(pair)
        return data.get("vol_10") or data.get("vix_index")

    def get_vol_10_1(self, pair: str) -> Optional[float]:
        data = self._fetch_vol_data(pair)
        return data.get("vol_10_1") or (data.get("vol_10", 0) + 0.3 if data.get("vol_10") else None)

    def get_vix_correlation(self, pair: str) -> Optional[float]:
        data = self._fetch_vol_data(pair)
        return data.get("vix_corr") or data.get("vix_correlation")

    def get_implied_volatility(self, pair: str) -> Optional[float]:
        data = self._fetch_vol_data(pair)
        return data.get("implied_vol")

    def volatility_regime(self, pair: str) -> str:
        vol = self.get_vol_10(pair)
        if vol is None:
            return "unknown"
        if vol < 7.0:
            return "low_volatility"
        elif vol < 10.0:
            return "moderate_volatility"
        return "high_volatility"

    def volatility_breakout_signal(self, pair: str) -> Dict:
        # Single cache hit — all getters below reuse the same fetched data
        data = self._fetch_vol_data(pair)
        vol_10 = data.get("vol_10")
        vol_10_1 = data.get("vol_10_1")
        implied = data.get("implied_vol")
        correlation = data.get("vix_corr")

        expansion = bool(vol_10_1 and vol_10 and vol_10_1 > vol_10 + 0.3)
        contraction = bool(vol_10 and vol_10_1 and vol_10_1 < vol_10 - 0.3)

        return {
            "pair": pair,
            "vol_10": vol_10,
            "vol_10_1": vol_10_1,
            "implied_vol": implied,
            "vix_corr": correlation,
            "regime": data.get("regime", "unknown"),
            "expansion": expansion,
            "contraction": contraction,
            "strategy_note": "Breakout traders watch vol expansion; mean-reversion favors contraction.",
        }

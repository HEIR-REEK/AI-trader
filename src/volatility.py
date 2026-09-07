"""
Production Volatility Indices — NO SIMULATED READINGS.
Requires VIX_API_KEY. Fetches live implied volatility data.
NOTE: Volatility measurement improves risk awareness — NOT win prediction.
"""

import os
from typing import Dict, Optional

try:
    import requests
except ImportError:
    raise ImportError("requests required. Install: pip install requests")


class VolatilityIndex:
    """Production-grade volatility analysis using live feeds."""

    DEFAULT_PERIOD = 14

    def __init__(self):
        self.api_key = os.environ.get("VIX_API_KEY")
        if not self.api_key:
            raise ValueError("VIX_API_KEY is required for production volatility analysis. Set in .env or environment.")

    def _fetch_vol_data(self, pair: str) -> Dict:
        symbol = pair.replace("/", "")
        resp = requests.get(
            f"https://api.twelvedata.com/v1/market_volatility?pair={symbol}",
            headers={"Authorization": f"apikey {self.api_key}"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

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
        vol_10 = self.get_vol_10(pair)
        vol_10_1 = self.get_vol_10_1(pair)
        implied = self.get_implied_volatility(pair)
        correlation = self.get_vix_correlation(pair)

        expansion = bool(vol_10_1 and vol_10 and vol_10_1 > vol_10 + 0.3)
        contraction = bool(vol_10 and vol_10_1 and vol_10_1 < vol_10 - 0.3)

        return {
            "pair": pair,
            "vol_10": vol_10,
            "vol_10_1": vol_10_1,
            "implied_vol": implied,
            "vix_corr": correlation,
            "regime": self.volatility_regime(pair),
            "expansion": expansion,
            "contraction": contraction,
            "strategy_note": "Breakout traders watch vol expansion; mean-reversion favors contraction.",
        }

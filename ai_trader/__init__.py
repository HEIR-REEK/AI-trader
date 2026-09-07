"""
AI-Trader — Advanced AI-powered market analysis & decision-support system.

Design principles
-----------------
1. Trading is probabilistic. Nothing in this package claims or implies a
   guaranteed outcome. Every output is a *probability-weighted opinion* with
   an explicit explanation and an explicit invalidation.
2. Regime first. No strategy is evaluated before the market regime is known.
3. Confluence over conviction. A trade is only proposed when several
   independent lenses (structure, HTF trend, liquidity, price action, volume,
   volatility, indicators, risk/reward) agree.
4. NO TRADE is a first-class output. When conditions are unclear the system
   returns the canonical NO TRADE message instead of forcing a setup.
5. Survival first. The risk layer can veto any signal regardless of score.
6. No look-ahead. All analysis is causal; swing points, zones and higher
   timeframe bars are only used once they are confirmed/closed.
"""

__version__ = "1.0.0"

NO_TRADE_MESSAGE = "NO TRADE — MARKET CONDITIONS DO NOT MEET HIGH-PROBABILITY REQUIREMENTS."
WAIT_MESSAGE = "NO TRADE — WAIT FOR BETTER MARKET CONFIRMATION."

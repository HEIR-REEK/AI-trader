# AI-Trader — Production System

A structured forex market analysis system with production-grade infrastructure.

**IMPORTANT: This is a structured decision-support framework — NOT a guaranteed trade predictor.**

## Production Status

- **No simulated fallbacks**: All data sources require real API keys (`EXCHANGE_API_KEY`, `NEWS_API_KEY`, `VIX_API_KEY`). Missing keys cause explicit failures, not fake data.
- **Persistent database**: All analyses saved to `data/trader.db` via SQLite.
- **Continuous service**: `production_server.py` runs analysis cycles at configured intervals.
- **Audit logging**: All operations logged to `logs/production.log`.

## Architecture

- `src/realtime_feeds.py` — Live market/news/volatility feeds (production APIs only)
- `src/indicators.py` — Technical indicator framework (requires OHLC feeds)
- `src/strategies.py` — 5 strategy frameworks (trend, mean-reversion, breakout, momentum, SR)
- `src/volatility.py` — Volatility indices (10 / 10.1 derived) with live feeds
- `src/news_sentiment.py` — Real-time sentiment analysis
- `src/master_engine.py` — Master signal engine combining all layers
- `src/production_db.py` — Persistent SQLite storage
- `src/production_server.py` — Continuous production service
- `run_production.py` — Production entry point
- `dashboard/index.html` — Structured analysis dashboard

## Requirements

```bash
# Set real API keys
export EXCHANGE_API_KEY="your_key"
export NEWS_API_KEY="your_key"
export VIX_API_KEY="your_key"

# Install dependencies
pip install -r requirements.txt

# Run production service
PYTHONPATH=src python run_production.py
```

## Reality Check

Even with full production infrastructure — live feeds, persistent storage, continuous monitoring — this system provides **structured analysis points**, not guaranteed winning trades. Markets are driven by unpredictable events, liquidity gaps, and sentiment shifts that no framework can fully model.

Always manage risk independently. Never trade based solely on automated analysis.

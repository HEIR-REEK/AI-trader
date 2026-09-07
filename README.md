# AI-Trader — Browser Trading System

A structured forex / metals / crypto / indices market analysis system with a
**browser interface**. No terminal needed — start the server, open the website,
click to analyze.

**IMPORTANT: This is a structured decision-support framework — NOT a guaranteed
trade predictor.**

## Quickstart (browser)

```bash
pip install -r requirements.txt
python run_web.py
```

Then open **http://localhost:8000** in your browser.

| Page | What it does |
|------|--------------|
| **Analyze** | Pick a symbol + data source → full engine verdict (TRADE / NO TRADE), price chart with entry/SL/TP, score breakdown, candidates, explanation |
| **Scenarios** | Scripted textbook markets (long / short / range fade / choppy) with switchable setup components |
| **Backtest** | Replay history as background jobs with live progress, equity curve, trades, walk-forward & overfit reports |
| **Instruments** | Searchable list of all 28 supported instruments |
| **Settings** | Live engine config: thresholds, scoring weights, risk limits, regime map |

API docs (optional, for integrations): http://localhost:8000/api/docs

## Data sources

| Source | Needs | Notes |
|--------|-------|-------|
| `synthetic` | nothing | Random regime-based price paths. Proves the *plumbing*, never the edge. Default. |
| `csv` | `data/{SYMBOL}_{tf}.csv` files | Real history. Only the entry timeframe file is required (e.g. `XAUUSD_15m.csv`); higher timeframes resample automatically. |
| `twelvedata` | `AITRADER_TWELVEDATA_API_KEY` | Live market data via the TwelveData API. |

Set keys in the environment or a `.env` file (see `.env.example`).

## Project layout

- `run_web.py` — **start here**: launches the browser system (same as `python -m ai_trader.web`)
- `ai_trader/web/` — browser backend (FastAPI) + self-contained UI (no build step, no CDN needed)
  - `app.py` — HTTP routes, `service.py` — engine orchestration,
    `jobs.py` — background backtests, `static/` — the website (HTML/CSS/JS)
- `ai_trader/` — analysis engine: data, indicators, structure/SMC/ICT, regime,
  strategies, confluence scoring, risk, backtester
- `ai_trader/cli.py` — terminal interface (advanced / scripting use; the browser
  covers the same commands: analyze, scenario, backtest, instruments)
- `src/` — legacy prototype modules (kept for reference)
- `tests/` — 144 tests including `test_web_api.py` for the browser backend

## Reality check

Even with full infrastructure — multi-strategy frameworks, volatility analysis,
multi-timeframe confluence, backtests — this system provides **structured
analysis points**, not guaranteed winning trades. Markets are driven by
unpredictable events, liquidity gaps, and sentiment shifts that no framework can
fully model.

Always manage risk independently. Never trade based solely on automated analysis.

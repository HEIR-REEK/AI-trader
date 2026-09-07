# AI-Trader — System Architecture

> **Principle zero:** this is a probabilistic decision-support system. It never
> claims a trade will win. It looks for *confluence*, rejects weak setups, and
> returns `NO TRADE — MARKET CONDITIONS DO NOT MEET HIGH-PROBABILITY REQUIREMENTS.`
> whenever conditions are unclear, conflicting, manipulated, excessively volatile,
> or below the configured confluence threshold.

---

## 1. Overall architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                               CLIENTS                                        │
│   React dashboard (frontend/)   ·   CLI (ai_trader.cli)   ·   REST / WS      │
└───────────────▲─────────────────────────▲─────────────────────────▲──────────┘
                │ HTTP (JSON)             │                         │ WebSocket
┌───────────────┴─────────────────────────┴─────────────────────────┴──────────┐
│  API LAYER  (ai_trader/api)  FastAPI · pydantic schemas · WebSocket hub       │
│   /instruments  /analyze  /regime  /signals  /backtest  /risk  /paper  /ws    │
└───────────────▲──────────────────────────────────────────────────────────────┘
                │
┌───────────────┴──────────────────────────────────────────────────────────────┐
│  DECISION LAYER  (ai_trader/decision)                                        │
│                                                                              │
│   DecisionEngine ─► HardFilters ─► StrategyRegistry(regime) ─► Confluence    │
│        │                                                       Scorer        │
│        │                                                          │          │
│        └──────────── Explainer  ◄─────── RiskManager.veto ◄───────┘          │
│                          │                                                   │
│                     TradePlan  |  NoTradeDecision (+ reasons)                │
└───────▲──────────────────────────────────────────▲───────────────────────────┘
        │                                          │
┌───────┴───────────────────┐        ┌─────────────┴────────────────────────────┐
│  AI LAYER (ai_trader/ml,  │        │  RISK LAYER (ai_trader/risk)             │
│  ai_trader/regime)        │        │  position sizing · SL validation         │
│  regime rules + RF model  │        │  daily/weekly loss · drawdown            │
│  setup meta-labeller      │        │  consecutive-loss guard · correlation    │
│  volatility forecaster    │        │  kill-switch: TRADING DISABLED           │
└───────▲───────────────────┘        └─────────────▲────────────────────────────┘
        │                                          │
┌───────┴──────────────────────────────────────────┴───────────────────────────┐
│  ANALYSIS LAYER (ai_trader/analysis, ai_trader/strategies,                   │
│                  ai_trader/fundamentals, ai_trader/specialized)               │
│   indicators · market structure (BOS/CHoCH/MSS) · S/R & psychological levels │
│   candlesticks · breakout/fakeout · SMC (liquidity, OB, FVG, PD arrays, OTE) │
│   ICT (kill zones, Judas) · volume/VWAP/profile · volatility regimes         │
│   strategies (trend, pullback, breakout, mean-reversion, sweep reversal,     │
│   momentum) · macro/news calendar · Gold module · Synthetic-index module     │
└───────▲──────────────────────────────────────────────────────────────────────┘
        │
┌───────┴──────────────────────────────────────────────────────────────────────┐
│  DATA LAYER (ai_trader/data)                                                 │
│   DataProvider protocol → CSV · Synthetic(GBM/OU, labelled) · TwelveData     │
│   causal resampler (LTF→HTF, closed bars only) · validation · store          │
│   economic calendar (JSON/ICS loader) · macro series (DXY, yields)           │
└───────▲──────────────────────────────────────────────────────────────────────┘
        │
┌───────┴──────────────────────────────────────────────────────────────────────┐
│  PERSISTENCE (ai_trader/db)  SQLAlchemy → SQLite (dev) / PostgreSQL (prod)   │
│  EXECUTION (ai_trader/execution)  PaperBroker · Broker protocol · signals    │
│  BACKTEST (ai_trader/backtest)  event-driven engine · metrics · walk-forward │
│                                 · Monte Carlo · anti-overfitting audits      │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Layer responsibilities (single direction of dependency: top depends on bottom)

| Layer | Owns | Never does |
|---|---|---|
| Data | OHLCV retrieval, resampling, integrity checks, calendars | interpret price |
| Analysis | pure, causal functions `DataFrame -> features/events` | make trade decisions |
| AI | regime classification, setup probability, vol forecast | bypass risk rules |
| Decision | filters → strategies → confluence → explanation | size positions |
| Risk | sizing, limits, veto, kill-switch | generate signals |
| Execution | paper/live order routing, fills, journaling | analysis |
| API | transport & schemas | business logic |

---

## 2. Technology stack

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.11 | ecosystem, numeric libs |
| API | FastAPI + uvicorn | async, typed, OpenAPI, WebSockets |
| Numerics | NumPy, pandas, SciPy | vectorised, well-tested |
| Indicators | in-house vectorised implementations (`analysis/indicators.py`), unit-tested against reference formulas | no C-build deps, fully auditable, no hidden look-ahead |
| ML | scikit-learn (RandomForest, GradientBoosting), optional XGBoost, optional PyTorch | purged CV, feature importance |
| Validation | pydantic v2 / pydantic-settings | typed configs & API schemas |
| Persistence | SQLAlchemy 2 → SQLite (dev) / PostgreSQL (prod) | same models both envs |
| Real-time | WebSockets (FastAPI) | signal streaming to UI |
| Frontend | React (Vite) | modern SPA, proxied to API |
| Container | Docker + docker-compose (api, db, frontend) | reproducible deploy |
| Tests | pytest | every module has tests |

---

## 3. Folder structure

```
AI-trader/
├── ai_trader/                    # the system (installable package)
│   ├── config/                   # settings, instrument registry, scoring config
│   ├── core/                     # enums, dataclasses, timeframe utils, exceptions
│   ├── data/                     # providers, resampler, validation, calendar, store
│   ├── analysis/                 # indicators, structure, levels, patterns, breakout,
│   │                             # smc, ict, volume, volatility, mtf context
│   ├── regime/                   # rule-based regime detector (+ ML hook)
│   ├── strategies/               # strategy protocol + implementations + registry
│   ├── fundamentals/             # economic calendar, macro context, gold macro
│   ├── specialized/              # XAUUSD module, synthetic/volatility-index module
│   ├── decision/                 # filters, confluence scorer, engine, explainer, formatter
│   ├── risk/                     # sizing, limits, validator, RiskManager
│   ├── backtest/                 # engine, metrics, walk-forward, monte carlo, overfitting
│   ├── ml/                       # features, purged CV, regime model, setup classifier
│   ├── execution/                # broker protocol, paper broker, signal bus
│   ├── db/                       # SQLAlchemy models, session, repositories
│   ├── api/                      # FastAPI app, routers, schemas, websocket hub
│   └── cli.py                    # `python -m ai_trader.cli analyze XAUUSD`
├── frontend/                     # React dashboard (Vite)
├── tests/                        # pytest suite, mirrors package layout
├── docs/                         # this file + module notes
├── data/                         # local csv / sqlite (git-ignored except samples)
├── legacy/ (src/)                # previous prototype, superseded
├── docker-compose.yml, Dockerfile
├── pyproject.toml, requirements.txt
└── README.md
```

---

## 4. Database design

```
instruments            candles                         analyses
─────────────          ──────────────────────          ─────────────────────────
id PK                  id PK                           id PK
symbol UNIQUE          instrument_id FK                instrument_id FK
asset_class            timeframe                       created_at
pip_size               ts (UTC)                        entry_timeframe
point_value            open high low close volume      regime, regime_confidence
sessions (json)        UNIQUE(instrument_id,tf,ts)     bias
                                                       confluence_score
signals                trades                          decision  (TRADE|NO_TRADE)
─────────────          ─────────────────────           payload (json: full report)
id PK                  id PK
analysis_id FK         signal_id FK          risk_events              backtest_runs
instrument_id FK       instrument_id FK      ───────────────          ──────────────
created_at             direction             id PK                    id PK
direction              entry_price           created_at               created_at
entry_low/high         stop_price            kind (DAILY_LIMIT, ...)  strategy
entry_type             tp1 tp2 tp3           details (json)           instrument_id FK
stop, tp1, tp2, tp3    size_units                                     params (json)
rr, score              opened_at closed_at   economic_events          metrics (json)
factors (json)         exit_price, pnl       ──────────────           is_oos (bool)
invalidation           r_multiple, status    id PK, ts, currency,     walk_forward_id
                       mode (PAPER|LIVE)     impact, title, actual,
                                             forecast, previous
```

Indexes: `candles(instrument_id, timeframe, ts)`, `analyses(instrument_id, created_at)`,
`trades(status)`, `economic_events(ts, impact)`.

---

## 5. API architecture

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | liveness, provider status |
| GET | `/api/instruments` | registry (symbol, class, pip, sessions) |
| GET | `/api/instruments/{symbol}/candles?tf=1h&limit=500` | OHLCV |
| POST | `/api/analyze` | full pipeline → TradePlan or NO TRADE (+ explanation) |
| GET | `/api/regime/{symbol}?tf=4h` | regime assessment only |
| GET | `/api/signals?limit=50` | recent decisions |
| POST | `/api/backtest` | run strategy backtest (metrics + equity curve) |
| POST | `/api/backtest/walk-forward` | rolling IS/OOS validation |
| GET | `/api/risk/status` | limits, drawdown, kill-switch state |
| POST | `/api/paper/orders` | submit paper order from a signal |
| GET | `/api/paper/positions` | open paper positions |
| WS | `/ws/signals` | streaming decisions / paper fills |

All responses carry `disclaimer` and, for decisions, `explanation` blocks.

---

## 6. Data flow

```
provider ─► validate ─► store ─► resample (closed bars only) ─► MarketData{tf: df}
                                                   │
                     ┌─────────────────────────────┘
                     ▼
        per-timeframe analysis (indicators · structure · levels · smc · volume · vol)
                     │
                     ▼
              MultiTimeframeContext  (HTF bias → structural → entry)
                     │
        ┌────────────┼──────────────┐
        ▼            ▼              ▼
   RegimeDetector  Calendar/Macro  Specialized (gold / synthetic)
        │            │              │
        └────────────┴──────┬───────┘
                            ▼
                 DecisionEngine.decide()
                  1. hard filters (news blackout, extreme vol, data quality, risk kill-switch)
                  2. strategies allowed for regime → candidate StrategySignals
                  3. confluence scoring (8 components, 100 pts)
                  4. threshold + conflict check
                  5. risk validation (SL distance, R:R, sizing, limits)
                  6. explanation (why / why now / invalidation / failure modes)
                            │
                ┌───────────┴────────────┐
                ▼                        ▼
            TradePlan                 NoTrade(reasons)
                │                        │
                └────► persist → API/WS → dashboard / paper broker
```

---

## 7. Trading analysis pipeline (per instrument)

1. **Data integrity** – gaps, duplicates, stale feed → abort with NO TRADE if bad.
2. **Regime** – ADX/EMA-slope/efficiency-ratio/ATR-percentile/BB-width/structure votes →
   `StrongBull | StrongBear | WeakTrend | Range | Accumulation | Distribution |
   Breakout | FalseBreakout | HighVol | LowVol | Expansion | Contraction | Reversal | NewsDriven`.
3. **HTF bias** – Monthly/Weekly/Daily structure + EMA stack → BUY/SELL/NEUTRAL bias.
4. **Structural TF** – 4H/1H: BOS/CHoCH/MSS, S/R, supply/demand, order blocks, FVGs,
   liquidity pools, premium/discount of the active dealing range.
5. **Entry TF** – 30m/15m/5m: sweep + MSS confirmation, candlestick confirmation,
   breakout/retest, momentum alignment, volume confirmation.
6. **Strategy candidates** – only strategies whitelisted for the regime are evaluated.
7. **Confluence score** – Structure 20 · HTF trend 15 · Liquidity 15 · Price action 10 ·
   Volume 10 · Volatility 10 · Indicators 10 · R:R 10 = 100 (configurable).
8. **Decision** – `score ≥ high_quality` AND no hard-filter breach AND risk validation
   pass → TradePlan; otherwise NO TRADE with explicit reasons.
9. **Explain** – every plan lists confluence factors, invalidation, no-trade conditions,
   and failure modes. Every NO TRADE lists what was missing.
10. **Journal & evaluate** – decisions persisted; realised outcomes feed the
    meta-labeller and performance monitors.

See `docs/MODULES.md` for per-module notes (what/how/test/limitations/next).

# Module guide — what is built, how it works, how to test it, what it cannot do

This document accompanies `docs/ARCHITECTURE.md`. Each section covers one implemented
module: purpose, mechanics, entry points, tests, limitations. Everything lives in the
`ai_trader/` package (the legacy `src/` prototype is untouched and not used).

> The system is **probabilistic decision support**. It never guarantees a trade. When
> conditions are unclear it returns exactly
> `NO TRADE — MARKET CONDITIONS DO NOT MEET HIGH-PROBABILITY REQUIREMENTS.` or
> `NO TRADE — WAIT FOR BETTER MARKET CONFIRMATION.`

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q                       # 73 tests
python -m ai_trader.cli scenario textbook_long --candidates
python -m ai_trader.cli scenario range_fade
python -m ai_trader.cli scenario choppy
python -m ai_trader.cli analyze XAUUSD --seed 3 --candidates
```

---

## Phase 1 — Core, configuration and data layer

**Files:** `core/enums.py`, `core/models.py`, `config/instruments.py`, `config/settings.py`,
`data/providers.py`, `data/resampler.py`, `data/validation.py`, `data/loader.py`,
`data/calendar.py`, `data/scenarios.py`

* **Enums / models** – one vocabulary for the whole system (`Timeframe` with
  `.minutes`/`.group`, `Regime` (15 values), `StrategyFamily`, `Direction`, `Bias`,
  `StructureEventType`, `ZoneType`, …) and typed dataclasses for every artefact
  (`StrategySignal`, `ConfluenceResult`, `RiskPlan`, `TradePlan`, `Decision`).
* **Instrument registry** – 28 instruments (metals, 5 Volatility indices, 6 equity
  indices, FX majors + minors, crypto) with pip size, point value, lot size, spread,
  sessions, `macro_sensitive`, `is_synthetic`. Adding one is a `register(Instrument(...))`.
* **Settings** – pydantic, env-overridable (`AITRADER_*`): `ScoringWeights` (must sum to
  100), `DecisionThresholds` (90/80/70, `min_trade_score=80`, `max_conflicts=0`,
  `min_rr_tp1=1.5`, `min_rr_tp2=2.0`, `min_regime_confidence=0.45`), `RiskSettings`,
  `NewsSettings`, `AnalysisSettings` (minimum bars per timeframe), `RegimeStrategyMap`.
* **Providers** – `SyntheticProvider` (one canonical M5 path per symbol, every timeframe
  resampled from it, regime segments scriptable), `FrameProvider` (in-memory frames,
  missing timeframes resampled causally), CSV provider. All return closed bars only.
* **Resampler / validation** – causal resampling (`drop_incomplete=True`), OHLC sanity,
  gap detection, duplicate/NaN checks → `ValidationReport`.
* **Loader** – loads the requested timeframe set, enforces minimum bars, returns
  `LoadResult` with problems (a data-quality problem yields NO TRADE, never a guess).
* **Economic calendar** – `EconomicCalendar` with blackout windows, next high-impact
  event, recent surprises; `source="empty"|"missing"` means *unknown* (penalised),
  `from_events([])` means *checked and empty*.
* **Scenarios** – deterministic scripted markets used to exercise the TRADE path:
  `textbook_pullback_setup` (impulsive HH/HL trend → deep pullback → equal lows →
  sweep → displacement MSS with FVG → 50 % retrace → engulfing confirmation, mirrorable
  to short, every component switchable), `range_fade_setup` (respected box + hammer at
  the edge), `choppy_no_edge` (featureless).

**Tests:** `tests/test_data_layer.py` (15). **Limitations:** no live broker/API
provider yet (interface is ready); synthetic data validates mechanics, never edge.

---

## Phase 2 — Technical analysis engine

**Files:** `analysis/indicators.py`, `structure.py`, `levels.py`, `patterns.py`,
`breakout.py`, `smc.py`, `ict.py`, `volume.py`, `volatility.py`, `mtf.py`

* **Indicators** – RSI, MACD, ATR, ADX/DI, Bollinger (+ width percentile, squeeze),
  EMA/SMA stacks & slopes, VWAP (session-anchored), Stochastic, Ichimoku, Fibonacci,
  Volume Profile (POC/VAH/VAL), efficiency ratio, choppiness, OBV, relative volume.
  All vectorised, all causal.
* **Market structure** – fractal swings with a *confirmation index* (a swing is only
  known `k` bars later), HH/HL/LH/LL labelling, BOS / CHoCH / MSS events with
  displacement and swept-liquidity flags, structure bias score.
* **Levels / patterns / breakouts** – S/R clusters, round numbers, PDH/PDL/PWH/PWL;
  engulfing, pin bars/hammers, inside bars, doji, morning/evening star with strength;
  breakout assessment (close beyond, displacement, volume, acceptance, retest) and FAKE
  classification when price closes back inside within 3 bars.
* **SMC / ICT** – liquidity pools (equal highs/lows, swing pools, key levels) with sweep
  detection; FVGs, order blocks, breaker/mitigation blocks, supply/demand; dealing range
  (current leg) → premium/discount/OTE; sessions and kill zones, Asia range, Judas swing
  hints. All outputs are *scored hypotheses* that the confluence layer weighs, never
  entry triggers on their own.
* **Volume / volatility** – volume confirmation score (rvol, direction agreement, VWAP
  side, OBV trend, climax detection; markets without volume are neutral and the weight
  is redistributed); volatility state DEAD/LOW/NORMAL/ELEVATED/EXTREME (EXTREME requires
  top percentile **and** ≥1.8× median ATR), phase EXPANSION/CONTRACTION/STABLE,
  `tradeable` flag.
* **Multi-timeframe context** – `build_context(load)` analyses every timeframe, assigns
  HIGH / STRUCTURAL / ENTRY roles, computes HTF bias & strength, structural bias,
  log2(minutes)-weighted alignment, and flags HTF-vs-structural conflicts.

**Tests:** `tests/test_analysis.py` (24). **Limitations:** pattern strengths are
heuristic; volume profile is bar-based (no tick data); no order-flow/DOM.

---

## Phase 3 — Regime detection

**Files:** `regime/detector.py`

Rule-based voting over ~30 causal features (ADX/DI, efficiency ratio, EMA slope, structure
bias, ATR/BB percentiles, spike ratio, choppiness, range width/position, breakout score,
fake-breakout flag, sweeps, CHoCH, news proximity). Primary regime ∈ {STRONG_BULL,
STRONG_BEAR, WEAK_TREND, RANGE, ACCUMULATION, DISTRIBUTION, BREAKOUT, FALSE_BREAKOUT,
REVERSAL}, overlays ∈ {HIGH_VOL, LOW_VOL, EXPANSION, CONTRACTION, NEWS_DRIVEN}.
Confidence = 0.5·top + 0.5·min(1, 2·(top − second)); < 0.30 → UNKNOWN (no entries).
The regime maps to *allowed strategy families* (`RegimeStrategyMap`); LOW_VOL disables
momentum, HIGH_VOL / NEWS_DRIVEN / UNKNOWN allow nothing. An optional ML model can be
blended 0.6/0.4 as an advisor (Phase 8).

**Tests:** `tests/test_regime.py` (5) + regime assertions in the scenario tests.
**Limitations:** thresholds are hand-set; they must be re-validated per instrument class
in backtests (Phase 7).

---

## Phase 4 — Strategies

**Files:** `strategies/base.py`, `trend.py`, `reversal.py`, `registry.py`

| strategy | family | trigger (all conditions) |
|---|---|---|
| `trend_pullback_smc` | PULLBACK | trend regime, pullback into discount/premium of the current leg **and** a same-direction zone (OB/FVG/demand/EMA), fresh entry-TF confirmation |
| `ema_trend_continuation` | TREND_FOLLOWING | EMA20/50 stack, pullback to the EMA band, reclaim candle, HTF agreement |
| `momentum_bos_continuation` | MOMENTUM | fresh entry-TF BOS with displacement + volume, shallow retest, RSI/MACD aligned (disabled for synthetics and LOW_VOL) |
| `liquidity_sweep_mss` | LIQUIDITY_REVERSAL | sweep of a pool (wick through, close back) followed by an MSS/CHoCH in the opposite direction, entry at the origin zone |
| `range_mean_reversion` | MEAN_REVERSION | RANGE/ACC/DIST regime, box ≥ 3 ATR, price in outer 20 %, RSI+Stoch+BB stretched, rejection candle, no accepted breakout |
| `breakout_retest` | BREAKOUT | validated breakout (close, displacement, volume, acceptance) of a box/key level, then a successful retest |

Every signal carries entry zone, stop (beyond structure/sweep + ATR buffer), three
targets chosen from *structural* levels in front of liquidity (`targets_from_levels`),
reasons, invalidation and evidence for the scorer. `StrategyRegistry.for_regime` gates
strategies by allowed family and instrument class.

**Tests:** `tests/test_strategies_confluence.py`. **Limitations:** no parameter
optimisation yet — every parameter is a documented default to be validated in Phase 7.

---

## Phase 5 — Confluence scoring & decision engine

**Files:** `decision/confluence.py`, `filters.py`, `engine.py`, `explainer.py`, `formatter.py`

Score 0–100 with configurable weights (structure 20, HTF trend 15, liquidity 15, price
action 10, volume 10, volatility 10, indicators 10, R:R 10). Non-applicable components
(no volume) redistribute their weight pro-rata. Conflicts are split:

* **MAJOR** (any one vetoes with `max_conflicts=0`): structure against a non-counter-trend
  trade, opposing structure event ≤ 3 bars, value entry in the wrong premium/discount
  zone, against a strong HTF trend, significant opposing liquidity < 0.5 R, strong
  opposing last candle, EXTREME/DEAD volatility, stop < 0.5 or > 4 ATR, R:R below
  minimums, regime confidence below minimum.
* **minor** (−2 points each, always reported): counter-trend notes, weak HTF headwind,
  liquidity 0.5–1.0 R, stop just beyond a pool, older opposing pattern, volume against,
  climax volume, RSI exhaustion, MACD against continuation, wide entry zone, news penalty.

Family awareness matters: a continuation entry at the leg's premium is acceptable, a
range fade *wants* non-trending higher timeframes and treats internal-range liquidity as
the path, not a veto.

**Engine pipeline:** data quality → MTF context → news → regime → specialised context
(gold / synthetic) → hard filters → strategies for regime → score → reject on score,
major conflicts or R:R → direction disagreement → risk plan → explanation → `Decision`.
Every strategy that produced a candidate is kept in `decision.candidates` with its
breakdown, conflicts and the rejection reason, so a NO TRADE is always auditable.

**Output** (`format_decision`): INSTRUMENT, MARKET REGIME, OVERALL BIAS, ENTRY ZONE,
ENTRY TYPE, STOP LOSS, TP1/TP2/TP3, RISK TO REWARD, CONFIDENCE SCORE (+ breakdown),
CONFLUENCE FACTORS, STRATEGY, POSITION SIZE, WHY THIS TRADE / WHY NOW / INVALIDATION /
WHAT COULD MAKE IT FAIL / NO TRADE CONDITIONS, disclaimer.

**Tests:** `tests/test_decision.py` (11) — TRADE on textbook long/short and range-fade
scenarios; NO TRADE on chop, missing confirmation, news blackout, risk kill-switch,
raised threshold; JSON serialisation; all instrument classes run.
**Limitations:** thresholds are *configurable but not yet backtest-validated* (Phase 7).

---

## Phase 6 — Risk engine, fundamentals, specialised modules

**Files:** `risk/manager.py`, `fundamentals/macro.py`, `specialized/gold.py`,
`specialized/synthetic.py`

* **RiskManager** – fixed-fractional sizing (`max_risk_per_trade_pct`, default 1 %),
  automatic halving after two consecutive losses and again in deep drawdown (never
  increased), stop validation (≥ 3× spread, 0.5–4 ATR, correct side, three monotone
  targets), daily / weekly loss limits (→ `TRADING DISABLED UNTIL NEXT TRADING SESSION.`),
  max drawdown hard stop (manual review), consecutive-loss cool-down, correlation groups,
  max open positions, broker min-lot check, optional JSON state persistence,
  `allow_martingale=False` (research flag with loud warnings).
* **Fundamentals** – `assess_news` (blackout ⇒ penalty 1.0, event ≤ 2 h ⇒ partial
  penalty, recent surprise ⇒ penalty, unknown calendar ⇒ half penalty and a note;
  synthetics ignore news); `macro_context` (DXY, yields, VIX, oil, copper… z-scored
  20-day changes weighted by per-instrument sensitivities; unknown ≠ neutral).
* **Gold** – session quality (London/NY 1.0, Asia 0.7, off-hours 0.5), kill zones,
  Asia-range sweep, daily/weekly structure, key liquidity, macro correlation notes,
  vetoes on blackout / EXTREME volatility.
* **Synthetic indices** – Hurst exponent, variance ratio, autocorrelation, realised vs
  nominal volatility (deviation > 40 % ⇒ veto), character TRENDING / MEAN_REVERTING /
  RANDOM_WALK → recommended families; momentum disabled; the calendar is ignored.

**Tests:** `tests/test_risk_fundamentals_specialized.py` (12).
**Limitations:** macro series must be supplied by the caller (no data feed yet); gold
correlation is a context note, not a signal; synthetic profile needs independent
backtesting (Phase 7) before any parameter is trusted.

---

## Roadmap (not yet implemented)

Phase 7 backtesting (historical, walk-forward, out-of-sample, Monte Carlo, per-regime
metrics, threshold validation) → Phase 8 ML advisors (regime / setup classifiers with
purged CV, feature importance, explainability) → Phase 9 paper trading + persistence +
FastAPI/WebSocket + React dashboard → Phase 10 broker adapters → Phase 11 live with
kill-switches.

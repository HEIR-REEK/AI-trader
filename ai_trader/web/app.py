"""FastAPI application — the browser backend for AI-Trader.

Serves the single-page web UI (no build step, no terminal needed) plus a JSON
API that mirrors every ``ai_trader.cli`` command:

* ``POST /api/analyze``   ← ``ai_trader analyze``
* ``POST /api/scenario``  ← ``ai_trader scenario``
* ``POST /api/backtest``  ← ``ai_trader backtest`` (background job + polling)
* ``GET  /api/instruments`` ← ``ai_trader instruments``
* ``GET  /api/candles``   ← OHLC data for the browser charts
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from ai_trader.core.enums import AssetClass, Timeframe

from . import jobs
from .schemas import AnalyzeRequest, BacktestRequest, ScenarioRequest
from .service import (
    SCENARIOS,
    analyze_symbol,
    get_candles,
    list_instruments,
    run_backtest,
    run_scenario,
    settings_view,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
STARTED_AT = time.time()

app = FastAPI(
    title="AI-Trader Web",
    version="1.0.0",
    description="Browser interface for AI-Trader — decision support, never guarantees.",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

# Local analysis tool: allow any origin (LAN access, preview proxies, file:// dev).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _err(e: Exception, status: int = 400) -> HTTPException:
    return HTTPException(status_code=status, detail=str(e))


# ------------------------------------------------------------------ pages

@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> JSONResponse:
    return JSONResponse(status_code=204, content=None)


@app.get("/{asset}", include_in_schema=False)
def static_asset(asset: str):
    """Serve the SPA's own JS/CSS (only these exact files, no directory walk)."""
    allowed = {"app.js", "styles.css", "chart.js"}
    if asset not in allowed:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")
    media = "text/javascript" if asset.endswith(".js") else "text/css"
    return FileResponse(STATIC_DIR / asset, media_type=media)


# ------------------------------------------------------------------ meta

@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "ai-trader-web",
        "uptime_s": round(time.time() - STARTED_AT, 1),
        "timeframes": [t.value for t in Timeframe],
        "asset_classes": [a.value for a in AssetClass],
        "disclaimer": "Probabilistic decision support only. No trade is guaranteed.",
    }


@app.get("/api/instruments")
def instruments(asset_class: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    try:
        items = list_instruments(asset_class)
    except ValueError as e:
        raise _err(e)
    return {"count": len(items), "instruments": items}


@app.get("/api/scenarios")
def scenarios() -> Dict[str, Any]:
    return {"count": len(SCENARIOS), "scenarios": SCENARIOS}


@app.get("/api/settings")
def settings() -> Dict[str, Any]:
    return settings_view()


# ------------------------------------------------------------------ analysis

@app.post("/api/analyze")
def analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    try:
        return analyze_symbol(req.symbol, req.source, req.seed, req.timeframes,
                              req.no_news_penalty, req.data_dir)
    except ValueError as e:
        raise _err(e)
    except Exception as e:  # noqa: BLE001
        raise _err(e, 500)


@app.post("/api/scenario")
def scenario(req: ScenarioRequest) -> Dict[str, Any]:
    try:
        return run_scenario(req.name, req.seed, req.without)
    except ValueError as e:
        raise _err(e)
    except Exception as e:  # noqa: BLE001
        raise _err(e, 500)


@app.get("/api/candles")
def candles(
    symbol: str = Query(default="XAUUSD"),
    timeframe: str = Query(default="15m"),
    limit: int = Query(default=300, ge=10, le=2000),
    source: str = Query(default="synthetic"),
    seed: int = Query(default=1),
    scenario: Optional[str] = Query(default=None),
    data_dir: str = Query(default="data"),
) -> Dict[str, Any]:
    try:
        return get_candles(symbol, timeframe, limit, source, seed, scenario, data_dir)
    except ValueError as e:
        raise _err(e)
    except Exception as e:  # noqa: BLE001
        raise _err(e, 500)


# ------------------------------------------------------------------ backtests (background jobs)

@app.post("/api/backtest", status_code=202)
def backtest_start(req: BacktestRequest) -> Dict[str, Any]:
    params = req.model_dump()
    label = f"{req.symbol.upper()} · {req.source}" + (f" · {req.label}" if req.label else "")
    job_id = jobs.submit("backtest", label, lambda progress=None: run_backtest(params, progress))
    return {"job_id": job_id, "status": "queued", "label": label,
            "poll": f"/api/backtest/jobs/{job_id}"}


@app.get("/api/backtest/jobs")
def backtest_jobs() -> Dict[str, Any]:
    items = jobs.recent()
    return {"count": len(items), "jobs": items}


@app.get("/api/backtest/jobs/{job_id}")
def backtest_job(job_id: str) -> Dict[str, Any]:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job {job_id!r}")
    return job

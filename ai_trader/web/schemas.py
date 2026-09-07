"""Pydantic request schemas for the web API."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    symbol: str = Field(default="XAUUSD", description="Instrument symbol, e.g. XAUUSD, EURUSD")
    source: str = Field(default="synthetic", description="synthetic | csv | twelvedata")
    seed: int = Field(default=1, ge=0, le=10_000_000)
    timeframes: List[str] = Field(default_factory=lambda: ["15m", "1h", "4h", "1d"])
    no_news_penalty: bool = Field(default=False, description="Treat calendar as checked-and-empty")
    data_dir: str = Field(default="data")


class ScenarioRequest(BaseModel):
    name: str = Field(default="textbook_long", description="textbook_long | textbook_short | range_fade | choppy")
    seed: int = Field(default=1, ge=0, le=10_000_000)
    without: List[str] = Field(default_factory=list, description="Scenario components to switch off")


class BacktestRequest(BaseModel):
    symbol: str = Field(default="XAUUSD")
    source: str = Field(default="scenario", description="scenario | synthetic | csv")
    # scenario source
    direction: str = Field(default="long", description="long | short (scenario source only)")
    resolve: Optional[str] = Field(default="win", description="win | loss | flat (scenario source only)")
    # synthetic source
    bars: int = Field(default=600, ge=50, le=20000)
    # csv source
    data_dir: str = Field(default="data")
    start: Optional[str] = Field(default=None, description="ISO date/datetime, e.g. 2025-01-01")
    end: Optional[str] = Field(default=None)
    split: Optional[str] = Field(default=None, description="ISO date/datetime for in-sample / out-of-sample split")
    # common
    seed: int = Field(default=1, ge=0, le=10_000_000)
    timeframes: List[str] = Field(default_factory=lambda: ["15m", "1h", "4h", "1d"])
    warmup: int = Field(default=400, ge=0, le=20000)
    spread: Optional[float] = Field(default=None, description="Spread override (price units). None = instrument default")
    commission: float = Field(default=0.0, ge=0.0)
    ttl: int = Field(default=8, ge=1, le=500)
    max_hold: int = Field(default=96, ge=1, le=5000)
    every: int = Field(default=1, ge=1, le=500, description="Decide on every k-th entry bar")
    equity: Optional[float] = Field(default=None, ge=100.0)
    label: str = Field(default="")
    walk_forward: bool = Field(default=False)
    no_news_penalty: bool = Field(default=False)

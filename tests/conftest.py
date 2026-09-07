import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ai_trader.core.enums import Timeframe  # noqa: E402
from ai_trader.data.providers import RegimeSegment, SyntheticProvider  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_provider() -> SyntheticProvider:
    return SyntheticProvider(seed=7, start_price=2000.0)


@pytest.fixture(scope="session")
def trend_up_provider() -> SyntheticProvider:
    return SyntheticProvider(seed=11, start_price=2000.0,
                             segments=[RegimeSegment(4000, "trend_up", vol=0.0009, drift=0.00010)])


@pytest.fixture(scope="session")
def trend_down_provider() -> SyntheticProvider:
    return SyntheticProvider(seed=12, start_price=2000.0,
                             segments=[RegimeSegment(4000, "trend_down", vol=0.0009, drift=-0.00010)])


@pytest.fixture(scope="session")
def range_provider() -> SyntheticProvider:
    return SyntheticProvider(seed=13, start_price=2000.0,
                             segments=[RegimeSegment(4000, "range", vol=0.0006, mean_revert=0.08)])


@pytest.fixture
def m15_frame(synthetic_provider):
    return synthetic_provider.get_ohlcv("XAUUSD", Timeframe.M15, limit=800).drop(columns=["label"])


@pytest.fixture
def h1_frame(synthetic_provider):
    return synthetic_provider.get_ohlcv("XAUUSD", Timeframe.H1, limit=600).drop(columns=["label"])

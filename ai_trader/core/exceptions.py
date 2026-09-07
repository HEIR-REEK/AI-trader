"""Typed exceptions so callers can distinguish data problems from logic problems."""


class AITraderError(Exception):
    """Base class."""


class DataError(AITraderError):
    """Missing / corrupt / stale market data."""


class InsufficientDataError(DataError):
    """Not enough bars for the requested analysis."""


class ConfigError(AITraderError):
    """Invalid configuration."""


class RiskLimitBreached(AITraderError):
    """A hard risk rule blocks trading."""


class ProviderError(DataError):
    """Upstream data provider failure."""

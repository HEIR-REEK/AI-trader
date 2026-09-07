"""
Instrument registry. Adding an instrument = adding one entry (or calling
``register``). Everything else (analysis, risk, API) reads from here.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from ai_trader.core.enums import AssetClass, Session
from ai_trader.core.models import Instrument

FX = (Session.ASIA, Session.LONDON, Session.NEW_YORK)

_REGISTRY: Dict[str, Instrument] = {}
_ALIASES: Dict[str, str] = {}


def register(inst: Instrument) -> Instrument:
    _REGISTRY[inst.symbol] = inst
    for a in inst.aliases:
        _ALIASES[a.upper()] = inst.symbol
    return inst


def get_instrument(symbol: str) -> Instrument:
    key = symbol.upper().replace("/", "").replace(" ", "")
    if key in _REGISTRY:
        return _REGISTRY[key]
    if key in _ALIASES:
        return _REGISTRY[_ALIASES[key]]
    raise KeyError(f"Unknown instrument '{symbol}'. Known: {sorted(_REGISTRY)}")


def all_instruments(asset_class: Optional[AssetClass] = None) -> List[Instrument]:
    items = list(_REGISTRY.values())
    if asset_class:
        items = [i for i in items if i.asset_class == asset_class]
    return items


def symbols(asset_class: Optional[AssetClass] = None) -> List[str]:
    return [i.symbol for i in all_instruments(asset_class)]


def bulk_register(items: Iterable[Instrument]) -> None:
    for i in items:
        register(i)


# --------------------------------------------------------------------------
# PRECIOUS METALS
# --------------------------------------------------------------------------
bulk_register([
    Instrument("XAUUSD", "Gold vs USD", AssetClass.METAL, pip_size=0.1, point_value=1.0,
               lot_size=100, sessions=FX, macro_sensitive=True, spread_points=0.25, aliases=("GOLD", "XAU")),
    Instrument("XAGUSD", "Silver vs USD", AssetClass.METAL, pip_size=0.01, point_value=1.0,
               lot_size=5000, sessions=FX, macro_sensitive=True, spread_points=0.02, aliases=("SILVER", "XAG")),
])

# --------------------------------------------------------------------------
# SYNTHETIC VOLATILITY INDICES (24/7, no macro sensitivity, statistically
# engineered — analysed by the specialised synthetic module)
# --------------------------------------------------------------------------
bulk_register([
    Instrument(f"VOL{n}", f"Volatility {n} Index", AssetClass.SYNTHETIC_INDEX, pip_size=0.01,
               point_value=1.0, lot_size=1, sessions=(Session.ALWAYS,), is_synthetic=True,
               aliases=(f"V{n}", f"VOLATILITY{n}", f"R_{n}", f"1HZ{n}V"))
    for n in (10, 25, 50, 75, 100)
])

# --------------------------------------------------------------------------
# MAJOR STOCK INDICES (CFD-style quoting)
# --------------------------------------------------------------------------
bulk_register([
    Instrument("US500", "S&P 500", AssetClass.STOCK_INDEX, pip_size=0.1, lot_size=1,
               sessions=(Session.NEW_YORK,), macro_sensitive=True, spread_points=0.4, aliases=("SPX", "SP500", "SPX500")),
    Instrument("US100", "NASDAQ 100", AssetClass.STOCK_INDEX, pip_size=0.1, lot_size=1,
               sessions=(Session.NEW_YORK,), macro_sensitive=True, spread_points=1.0, aliases=("NAS100", "NDX", "USTEC")),
    Instrument("US30", "Dow Jones 30", AssetClass.STOCK_INDEX, pip_size=1.0, lot_size=1,
               sessions=(Session.NEW_YORK,), macro_sensitive=True, spread_points=2.0, aliases=("DJI", "DOW", "WS30")),
    Instrument("DE40", "DAX 40", AssetClass.STOCK_INDEX, pip_size=0.1, lot_size=1,
               sessions=(Session.LONDON,), macro_sensitive=True, spread_points=1.0, aliases=("DAX", "GER40", "DE30")),
    Instrument("UK100", "FTSE 100", AssetClass.STOCK_INDEX, pip_size=0.1, lot_size=1,
               sessions=(Session.LONDON,), macro_sensitive=True, spread_points=1.0, aliases=("FTSE", "FTSE100")),
    Instrument("JP225", "Nikkei 225", AssetClass.STOCK_INDEX, pip_size=1.0, lot_size=1,
               sessions=(Session.ASIA,), macro_sensitive=True, spread_points=7.0, aliases=("NIKKEI", "NI225", "JPN225")),
])

# --------------------------------------------------------------------------
# FOREX MAJORS + selected minors
# --------------------------------------------------------------------------
_fx = {
    "EURUSD": ("Euro vs USD", 0.0001, "USD"),
    "GBPUSD": ("Pound vs USD", 0.0001, "USD"),
    "USDJPY": ("USD vs Yen", 0.01, "JPY"),
    "AUDUSD": ("Aussie vs USD", 0.0001, "USD"),
    "USDCAD": ("USD vs CAD", 0.0001, "CAD"),
    "USDCHF": ("USD vs CHF", 0.0001, "CHF"),
    "NZDUSD": ("Kiwi vs USD", 0.0001, "USD"),
    "EURGBP": ("Euro vs Pound", 0.0001, "GBP"),
    "EURJPY": ("Euro vs Yen", 0.01, "JPY"),
    "GBPJPY": ("Pound vs Yen", 0.01, "JPY"),
    "AUDJPY": ("Aussie vs Yen", 0.01, "JPY"),
}
bulk_register([
    Instrument(sym, name, AssetClass.FOREX, pip_size=pip, lot_size=100_000, sessions=FX,
               macro_sensitive=True, quote_currency=q, spread_points=pip * 1.2)
    for sym, (name, pip, q) in _fx.items()
])

# --------------------------------------------------------------------------
# CRYPTO (24/7)
# --------------------------------------------------------------------------
bulk_register([
    Instrument("BTCUSD", "Bitcoin vs USD", AssetClass.CRYPTO, pip_size=1.0, lot_size=1,
               sessions=(Session.ALWAYS,), spread_points=15.0, aliases=("BTC", "BTCUSDT")),
    Instrument("ETHUSD", "Ether vs USD", AssetClass.CRYPTO, pip_size=0.1, lot_size=1,
               sessions=(Session.ALWAYS,), spread_points=1.0, aliases=("ETH", "ETHUSDT")),
    Instrument("SOLUSD", "Solana vs USD", AssetClass.CRYPTO, pip_size=0.01, lot_size=1,
               sessions=(Session.ALWAYS,), spread_points=0.05, aliases=("SOL",)),
    Instrument("XRPUSD", "XRP vs USD", AssetClass.CRYPTO, pip_size=0.0001, lot_size=1,
               sessions=(Session.ALWAYS,), spread_points=0.0005, aliases=("XRP",)),
])

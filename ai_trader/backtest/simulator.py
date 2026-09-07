"""
Broker simulator for backtests — fills, stops, targets, costs.

Design rules (all chosen to *understate* performance rather than flatter it):

* Orders are created from a decision made on the CLOSE of bar *t*; nothing can
  fill before bar *t+1* opens (no same-bar fills).
* Limit entries fill only if the next bars trade *through* the zone; market
  entries fill at the next open plus half the spread plus slippage.
* Inside one bar the stop is assumed to be hit BEFORE any target when both are
  touched (worst case), unless the bar's open is already beyond the target.
* Costs: spread on entry (buy at ask / sell at bid), slippage on market entries
  and on stop exits (stops are momentum fills), commission per lot per side.
* Position management is fixed and documented: 1/3 out at TP1, stop moved to
  break-even (+ costs); 1/3 out at TP2, stop trailed to TP1; remainder at TP3
  or stop. A time stop closes what is left after ``max_hold_bars``.
* Pending orders expire after ``order_ttl_bars`` and are cancelled if the stop
  level trades before the entry fills (the setup is invalidated, not entered).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ai_trader.core.enums import Direction, EntryType
from ai_trader.core.models import Instrument


@dataclass
class ExecutionCosts:
    spread_points: Optional[float] = None    # None → instrument default
    slippage_atr: float = 0.05               # market / stop fills slip 5 % of ATR against us
    commission_per_lot_side: float = 0.0     # e.g. 3.5 for FX ECN accounts (in account currency)
    stop_slippage_atr: float = 0.08          # stops slip a little more (momentum fills)


@dataclass
class ManagementRules:
    tp1_fraction: float = 1 / 3
    tp2_fraction: float = 1 / 3
    move_to_breakeven_after_tp1: bool = True
    trail_to_tp1_after_tp2: bool = True
    order_ttl_bars: int = 8                  # pending limit order lifetime (entry-TF bars)
    max_hold_bars: int = 96                  # time stop (entry-TF bars) after entry
    fill_limit_on_touch: bool = True         # limit fills when the bar's range touches the zone


@dataclass
class PendingOrder:
    id: int
    symbol: str
    direction: Direction
    entry_type: EntryType
    entry_low: float
    entry_high: float
    stop: float
    targets: List[float]
    units: float
    lots: float
    risk_amount: float
    created_bar: int
    created_at: datetime
    ttl_bars: int
    meta: Dict = field(default_factory=dict)


@dataclass
class Position:
    id: int
    symbol: str
    direction: Direction
    entry_price: float
    entry_bar: int
    entry_time: datetime
    initial_stop: float
    stop: float
    targets: List[float]
    units: float
    units_open: float
    lots: float
    risk_amount: float
    risk_per_unit: float
    point_value: float
    tp_hit: int = 0
    realised: float = 0.0
    costs: float = 0.0
    fills: List[Dict] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)
    mae_r: float = 0.0     # max adverse excursion in R
    mfe_r: float = 0.0     # max favourable excursion in R


@dataclass
class ClosedTrade:
    id: int
    symbol: str
    direction: str
    strategy: str
    regime: str
    score: float
    grade: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float           # volume-weighted average exit
    initial_stop: float
    targets: List[float]
    units: float
    lots: float
    risk_amount: float
    pnl: float                  # net of costs, account currency
    costs: float
    r_multiple: float           # pnl / risk_amount
    bars_held: int
    exit_reason: str
    tp_hit: int
    mae_r: float
    mfe_r: float
    meta: Dict = field(default_factory=dict)

    @property
    def win(self) -> bool:
        return self.pnl > 0


class BrokerSimulator:
    def __init__(self, instrument: Instrument, costs: Optional[ExecutionCosts] = None,
                 rules: Optional[ManagementRules] = None):
        self.inst = instrument
        self.costs = costs or ExecutionCosts()
        self.rules = rules or ManagementRules()
        self.spread = self.costs.spread_points if self.costs.spread_points is not None else instrument.spread_points
        self.pending: List[PendingOrder] = []
        self.positions: List[Position] = []
        self.closed: List[ClosedTrade] = []
        self._next_id = 1

    # ------------------------------------------------------------- orders
    def submit(self, direction: Direction, entry_type: EntryType, entry_low: float, entry_high: float, stop: float,
               targets: List[float], units: float, lots: float, risk_amount: float, bar: int, ts: datetime,
               meta: Optional[Dict] = None) -> PendingOrder:
        o = PendingOrder(self._next_id, self.inst.symbol, direction, entry_type, float(entry_low), float(entry_high), float(stop),
                         [float(t) for t in targets], float(units), float(lots), float(risk_amount), bar, ts,
                         self.rules.order_ttl_bars, meta or {})
        self._next_id += 1
        self.pending.append(o)
        return o

    def cancel_all_pending(self, reason: str = "cancelled") -> None:
        for o in self.pending:
            o.meta["cancel_reason"] = reason
        self.pending.clear()

    @property
    def has_open(self) -> bool:
        return bool(self.positions) or bool(self.pending)

    # --------------------------------------------------------------- step
    def on_bar(self, bar: int, ts: datetime, o: float, h: float, l: float, c: float, atr: float) -> List[ClosedTrade]:
        """Process one *new* entry-timeframe bar (bar index `bar`, OHLC of that bar)."""
        closed_now: List[ClosedTrade] = []
        # 1) pending orders: invalidation → expiry → fill
        still: List[PendingOrder] = []
        for od in self.pending:
            if bar <= od.created_bar:          # created on the close of created_bar; nothing before next bar
                still.append(od)
                continue
            long = od.direction is Direction.LONG
            stop_traded = (l <= od.stop) if long else (h >= od.stop)
            fill = self._try_fill(od, o, h, l, c, atr)
            if fill is None:
                if stop_traded:                      # stop level traded before entry → setup invalidated, cancel
                    od.meta["cancel_reason"] = "invalidated before fill"
                    continue
                if bar - od.created_bar >= od.ttl_bars:
                    od.meta["cancel_reason"] = "expired"
                    continue
                still.append(od)
                continue
            pos = self._open(od, fill, bar, ts)
            opened_beyond_stop = (o <= od.stop) if long else (o >= od.stop)
            if opened_beyond_stop:
                # gapped through the zone AND the stop: filled at the open, stopped at the open (spread + slippage loss)
                px = o - (self.costs.stop_slippage_atr * atr + self.spread / 2) if long else o + (self.costs.stop_slippage_atr * atr + self.spread / 2)
                closed_now.append(self._close(pos, px, bar, ts, "gapped through stop at fill"))
                continue
            # same-bar management with the remainder of the bar (conservative: a touched stop comes first)
            res = self._manage(pos, bar, ts, o, h, l, c, atr, entered_this_bar=True)
            if res is not None:
                closed_now.append(res)
        self.pending = still
        # 2) open positions
        remaining: List[Position] = []
        for pos in self.positions:
            if pos.entry_bar == bar:
                remaining.append(pos)
                continue
            res = self._manage(pos, bar, ts, o, h, l, c, atr, entered_this_bar=False)
            if res is None:
                remaining.append(pos)
            else:
                closed_now.append(res)
        self.positions = [p for p in remaining if p.units_open > 0]
        return closed_now

    # ------------------------------------------------------------ helpers
    def _try_fill(self, od: PendingOrder, o: float, h: float, l: float, c: float, atr: float) -> Optional[float]:
        half = self.spread / 2
        slip = self.costs.slippage_atr * atr
        if od.entry_type is EntryType.MARKET:
            return o + half + slip if od.direction is Direction.LONG else o - half - slip
        # LIMIT / zone: fill at the zone edge nearest to price; if the bar OPENS inside or beyond
        # the zone the order fills at the open (a limit order never fills worse than its price)
        if od.direction is Direction.LONG:
            if o <= od.entry_high:
                return o + half
            if l <= od.entry_high:
                return od.entry_high + half
            return None
        else:
            if o >= od.entry_low:
                return o - half
            if h >= od.entry_low:
                return od.entry_low - half
            return None

    def _open(self, od: PendingOrder, price: float, bar: int, ts: datetime) -> Position:
        risk_per_unit = abs(price - od.stop)
        pos = Position(od.id, od.symbol, od.direction, price, bar, ts, od.stop, od.stop, list(od.targets), od.units, od.units,
                       od.lots, od.risk_amount, risk_per_unit, self.inst.point_value, meta=dict(od.meta))
        pos.costs += self.costs.commission_per_lot_side * od.lots
        pos.fills.append({"kind": "entry", "price": price, "units": od.units, "bar": bar})
        self.positions.append(pos)
        return pos

    def _stop_fill(self, pos: Position, atr: float) -> float:
        slip = self.costs.stop_slippage_atr * atr + self.spread / 2
        return pos.stop - slip if pos.direction is Direction.LONG else pos.stop + slip

    def _manage(self, pos: Position, bar: int, ts: datetime, o: float, h: float, l: float, c: float, atr: float,
                entered_this_bar: bool) -> Optional[ClosedTrade]:
        sgn = pos.direction.sign
        r = pos.risk_per_unit or 1e-12
        # excursions (in R of the *filled* risk)
        adverse = (pos.entry_price - l) / r if pos.direction is Direction.LONG else (h - pos.entry_price) / r
        favourable = (h - pos.entry_price) / r if pos.direction is Direction.LONG else (pos.entry_price - l) / r
        pos.mae_r = max(pos.mae_r, adverse)
        pos.mfe_r = max(pos.mfe_r, favourable)
        stop_hit = (l <= pos.stop) if pos.direction is Direction.LONG else (h >= pos.stop)
        # targets in order
        while pos.tp_hit < len(pos.targets):
            tp = pos.targets[pos.tp_hit]
            tp_hit = (h >= tp) if pos.direction is Direction.LONG else (l <= tp)
            if not tp_hit:
                break
            # if the stop was also touched in this bar assume the stop came first (worst case), unless the bar
            # OPENED beyond the target (gap through the target → target fills at the open)
            opened_beyond = (o >= tp) if pos.direction is Direction.LONG else (o <= tp)
            if stop_hit and not opened_beyond:
                break
            fill_px = o if opened_beyond else tp
            self._take_profit(pos, fill_px, bar)
            if pos.units_open <= 1e-12:
                return self._finalise(pos, bar, ts, "TP3")
        if stop_hit:
            return self._close(pos, self._stop_fill(pos, atr), bar, ts, self._stop_reason(pos))
        if not entered_this_bar and bar - pos.entry_bar >= self.rules.max_hold_bars:
            return self._close(pos, c - sgn * self.spread / 2, bar, ts, "time stop")
        return None

    def _take_profit(self, pos: Position, price: float, bar: int) -> None:
        if pos.tp_hit == 0:
            units = pos.units * self.rules.tp1_fraction
        elif pos.tp_hit == 1:
            units = pos.units * self.rules.tp2_fraction
        else:
            units = pos.units_open
        units = min(units, pos.units_open)
        px = price - pos.direction.sign * self.spread / 2          # exit crosses the spread
        pos.realised += (px - pos.entry_price) * pos.direction.sign * units * pos.point_value
        pos.costs += self.costs.commission_per_lot_side * pos.lots * (units / pos.units)
        pos.units_open -= units
        pos.fills.append({"kind": f"TP{pos.tp_hit + 1}", "price": px, "units": units, "bar": bar})
        pos.tp_hit += 1
        if pos.tp_hit == 1 and self.rules.move_to_breakeven_after_tp1:
            be = pos.entry_price + pos.direction.sign * (self.spread + self.costs.commission_per_lot_side * pos.lots / max(pos.units, 1e-12) / pos.point_value)
            pos.stop = max(pos.stop, be) if pos.direction is Direction.LONG else min(pos.stop, be)
        if pos.tp_hit == 2 and self.rules.trail_to_tp1_after_tp2:
            t1 = pos.targets[0]
            pos.stop = max(pos.stop, t1) if pos.direction is Direction.LONG else min(pos.stop, t1)

    def _stop_reason(self, pos: Position) -> str:
        if pos.tp_hit == 0:
            return "stop loss"
        if pos.tp_hit == 1:
            return "break-even stop (after TP1)"
        return "trailing stop (after TP2)"

    def _close(self, pos: Position, price: float, bar: int, ts: datetime, reason: str) -> ClosedTrade:
        units = pos.units_open
        pos.realised += (price - pos.entry_price) * pos.direction.sign * units * pos.point_value
        pos.costs += self.costs.commission_per_lot_side * pos.lots * (units / max(pos.units, 1e-12))
        pos.fills.append({"kind": "exit", "price": price, "units": units, "bar": bar, "reason": reason})
        pos.units_open = 0.0
        return self._finalise(pos, bar, ts, reason)

    def _finalise(self, pos: Position, bar: int, ts: datetime, reason: str) -> ClosedTrade:
        exits = [f for f in pos.fills if f["kind"] != "entry"]
        w = sum(f["units"] for f in exits) or 1.0
        vwap_exit = sum(f["price"] * f["units"] for f in exits) / w
        pnl = pos.realised - pos.costs
        ct = ClosedTrade(pos.id, pos.symbol, pos.direction.value, pos.meta.get("strategy", "?"), pos.meta.get("regime", "?"),
                         float(pos.meta.get("score", 0.0)), pos.meta.get("grade", "?"), pos.entry_time, ts, pos.entry_price, vwap_exit,
                         pos.initial_stop, pos.targets, pos.units, pos.lots, pos.risk_amount, pnl, pos.costs,
                         pnl / pos.risk_amount if pos.risk_amount else 0.0, bar - pos.entry_bar, reason, pos.tp_hit,
                         pos.mae_r, pos.mfe_r, dict(pos.meta))
        self.closed.append(ct)
        if pos in self.positions:
            self.positions.remove(pos)
        return ct

    # ------------------------------------------------------------ marks
    def unrealised(self, price: float) -> float:
        return sum((price - p.entry_price) * p.direction.sign * p.units_open * p.point_value for p in self.positions)

    def force_close_all(self, bar: int, ts: datetime, price: float, reason: str = "end of data") -> List[ClosedTrade]:
        out = [self._close(p, price, bar, ts, reason) for p in list(self.positions)]
        self.positions.clear()
        self.cancel_all_pending("end of data")
        return out

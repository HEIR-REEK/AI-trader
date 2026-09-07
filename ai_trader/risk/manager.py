"""
Risk management engine — survival first.

Responsibilities
* position sizing from a fixed fractional risk (never martingale unless the
  research flag is on, and then only with loud warnings),
* stop-loss validation (ATR band, not inside spread, not beyond a pool),
* risk/reward validation,
* daily / weekly loss limits, max drawdown, consecutive-loss cool-down,
* correlation exposure limits,
* the kill-switch: ``TRADING DISABLED UNTIL NEXT TRADING SESSION``.

The manager is *stateful*: it tracks realised P&L per day/week and open
positions. State can be persisted by the DB layer; here it is in-memory with
a simple JSON snapshot for the paper trader.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np

from ai_trader.config.settings import RiskSettings, get_settings
from ai_trader.core.enums import Direction
from ai_trader.core.models import Instrument, RiskPlan, StrategySignal

DISABLED_MSG = "TRADING DISABLED UNTIL NEXT TRADING SESSION."


@dataclass
class OpenPosition:
    symbol: str
    direction: Direction
    risk_amount: float
    units: float
    opened_at: datetime
    group: Optional[str] = None


@dataclass
class RiskState:
    equity: float
    peak_equity: float
    day: Optional[str] = None
    week: Optional[str] = None
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    consecutive_losses: int = 0
    cooldown_until: Optional[datetime] = None
    disabled_until: Optional[datetime] = None
    disabled_reason: str = ""
    open_positions: List[OpenPosition] = field(default_factory=list)
    trades_today: int = 0
    events: List[Dict] = field(default_factory=list)

    @property
    def drawdown_pct(self) -> float:
        return 0.0 if self.peak_equity <= 0 else (self.peak_equity - self.equity) / self.peak_equity * 100


class RiskManager:
    def __init__(self, settings: Optional[RiskSettings] = None, state_path: Optional[str] = None):
        self.s = settings or get_settings().risk
        self.state = RiskState(equity=self.s.account_equity, peak_equity=self.s.account_equity)
        self.state_path = state_path
        if state_path and os.path.exists(state_path):
            self._load()

    # ---------------------------------------------------------------- time
    def _roll(self, now: datetime) -> None:
        day = now.strftime("%Y-%m-%d")
        week = now.strftime("%G-W%V")
        if self.state.day != day:
            self.state.day, self.state.daily_pnl, self.state.trades_today = day, 0.0, 0
            if self.state.disabled_until and now >= self.state.disabled_until:
                self.state.disabled_until, self.state.disabled_reason = None, ""
        if self.state.week != week:
            self.state.week, self.state.weekly_pnl = week, 0.0

    # ------------------------------------------------------------- queries
    def trading_allowed(self, now: Optional[datetime] = None) -> tuple[bool, List[str]]:
        now = now or datetime.now(timezone.utc)
        self._roll(now)
        st, s = self.state, self.s
        reasons: List[str] = []
        if st.disabled_until and now < st.disabled_until:
            reasons.append(f"{DISABLED_MSG} ({st.disabled_reason})")
        if st.daily_pnl <= -(s.max_daily_loss_pct / 100) * self._day_base():
            reasons.append(f"daily loss limit {s.max_daily_loss_pct}% reached → {DISABLED_MSG}")
        if st.weekly_pnl <= -(s.max_weekly_loss_pct / 100) * self._day_base():
            reasons.append(f"weekly loss limit {s.max_weekly_loss_pct}% reached → trading disabled until next week")
        if st.drawdown_pct >= s.max_drawdown_pct:
            reasons.append(f"max drawdown {s.max_drawdown_pct}% breached ({st.drawdown_pct:.1f}%) → trading disabled, manual review required")
        if st.cooldown_until and now < st.cooldown_until:
            reasons.append(f"cool-down after {s.max_consecutive_losses} consecutive losses until {st.cooldown_until:%Y-%m-%d %H:%M} UTC")
        if len(st.open_positions) >= s.max_open_positions:
            reasons.append(f"max open positions ({s.max_open_positions}) reached")
        return (len(reasons) == 0), reasons

    def _day_base(self) -> float:
        return max(self.state.equity - self.state.daily_pnl, 1e-9)

    def group_of(self, symbol: str) -> Optional[str]:
        for g, syms in self.s.correlation_groups.items():
            if symbol in syms:
                return g
        return None

    def correlation_ok(self, symbol: str, direction: Direction) -> tuple[bool, str]:
        g = self.group_of(symbol)
        if g is None:
            return True, ""
        same = [p for p in self.state.open_positions if p.group == g]
        if len(same) >= self.s.max_correlated_positions:
            return False, f"already {len(same)} open positions in correlated group {g}"
        if any(p.symbol == symbol for p in same):
            return False, f"already in a position on {symbol}"
        return True, ""

    # --------------------------------------------------------------- sizing
    def current_risk_pct(self) -> float:
        pct = self.s.max_risk_per_trade_pct
        if self.s.reduce_risk_after_loss_streak and self.state.consecutive_losses >= 2:
            pct *= 0.5  # halve risk after 2 losses in a row (anti-tilt); never increase
        if self.state.drawdown_pct > self.s.max_drawdown_pct * 0.5:
            pct *= 0.5  # halve again in deep drawdown
        return pct

    def plan(self, sig: StrategySignal, instrument: Instrument, atr: float, now: Optional[datetime] = None) -> RiskPlan:
        now = now or datetime.now(timezone.utc)
        allowed, reasons = self.trading_allowed(now)
        warnings: List[str] = []
        s = self.s
        equity = self.state.equity
        risk_pct = self.current_risk_pct()
        if risk_pct < s.max_risk_per_trade_pct:
            warnings.append(f"risk reduced to {risk_pct:.2f}% (loss streak / drawdown protection)")
        risk_amount = equity * risk_pct / 100
        entry = sig.entry_mid
        stop_dist = abs(entry - sig.stop)
        stop_atr = stop_dist / atr if atr > 0 else 0.0
        ok = allowed
        if stop_dist <= 0:
            ok = False
            reasons.append("stop distance is zero")
        if stop_dist <= instrument.spread_points * 3:
            ok = False
            reasons.append(f"stop {stop_dist:.5g} within 3× spread ({instrument.spread_points:.5g}) — will be stopped by noise")
        if stop_atr < s.min_stop_atr_multiple:
            ok = False
            reasons.append(f"stop {stop_atr:.2f} ATR < minimum {s.min_stop_atr_multiple} ATR")
        if stop_atr > s.max_stop_atr_multiple:
            ok = False
            reasons.append(f"stop {stop_atr:.2f} ATR > maximum {s.max_stop_atr_multiple} ATR")
        # direction sanity
        if (sig.direction is Direction.LONG and sig.stop >= sig.entry_low) or (sig.direction is Direction.SHORT and sig.stop <= sig.entry_high):
            ok = False
            reasons.append("stop is on the wrong side of the entry zone")
        rr = [sig.rr(i) for i in range(3)]
        if len(sig.targets) < 3 or any(x <= 0 for x in rr):
            ok = False
            reasons.append("three monotonic targets are required")
        corr_ok, corr_msg = self.correlation_ok(instrument.symbol, sig.direction)
        if not corr_ok:
            ok = False
            reasons.append(corr_msg)
        # units so that stop_dist × units × point_value = risk_amount
        units = risk_amount / (stop_dist * instrument.point_value) if stop_dist > 0 else 0.0
        lots = units / instrument.lot_size if instrument.lot_size else units
        if lots < instrument.min_lot:
            warnings.append(f"computed size {lots:.4f} lots below min lot {instrument.min_lot} — trade would exceed risk budget; skip or increase equity")
            ok = False
            reasons.append("position size below broker minimum at the configured risk")
        lots = float(np.floor(lots / instrument.min_lot) * instrument.min_lot) if lots >= instrument.min_lot else 0.0
        units = lots * instrument.lot_size
        if self.s.allow_martingale:
            warnings.append("MARTINGALE FLAG ENABLED — research only; this multiplies ruin probability and must never be used live")
        return RiskPlan(equity, risk_pct, risk_amount, units, lots, stop_dist, stop_atr, rr[0], rr[1], rr[2], ok, reasons, warnings)

    # ------------------------------------------------------- state updates
    def register_open(self, symbol: str, direction: Direction, risk_amount: float, units: float, now: Optional[datetime] = None) -> None:
        now = now or datetime.now(timezone.utc)
        self._roll(now)
        self.state.open_positions.append(OpenPosition(symbol, direction, risk_amount, units, now, self.group_of(symbol)))
        self.state.trades_today += 1
        self._save()

    def register_close(self, symbol: str, pnl: float, now: Optional[datetime] = None) -> None:
        now = now or datetime.now(timezone.utc)
        self._roll(now)
        st, s = self.state, self.s
        st.open_positions = [p for p in st.open_positions if p.symbol != symbol]
        st.equity += pnl
        st.peak_equity = max(st.peak_equity, st.equity)
        st.daily_pnl += pnl
        st.weekly_pnl += pnl
        if pnl < 0:
            st.consecutive_losses += 1
            if st.consecutive_losses >= s.max_consecutive_losses:
                st.cooldown_until = now + timedelta(hours=s.cooldown_after_consecutive_losses_hours)
                self._event(now, "CONSECUTIVE_LOSS_COOLDOWN", {"losses": st.consecutive_losses, "until": st.cooldown_until.isoformat()})
        else:
            st.consecutive_losses = 0
        if st.daily_pnl <= -(s.max_daily_loss_pct / 100) * self._day_base():
            st.disabled_until = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            st.disabled_reason = "daily loss limit"
            self._event(now, "DAILY_LIMIT", {"daily_pnl": st.daily_pnl})
        if st.drawdown_pct >= s.max_drawdown_pct:
            st.disabled_until = now + timedelta(days=365)
            st.disabled_reason = "max drawdown — manual review required"
            self._event(now, "MAX_DRAWDOWN", {"drawdown_pct": st.drawdown_pct})
        self._save()

    def _event(self, now: datetime, kind: str, details: Dict) -> None:
        self.state.events.append({"ts": now.isoformat(), "kind": kind, **details})

    def status(self, now: Optional[datetime] = None) -> Dict:
        now = now or datetime.now(timezone.utc)
        allowed, reasons = self.trading_allowed(now)
        st = self.state
        return {
            "trading_allowed": allowed,
            "block_reasons": reasons,
            "equity": round(st.equity, 2),
            "peak_equity": round(st.peak_equity, 2),
            "drawdown_pct": round(st.drawdown_pct, 2),
            "daily_pnl": round(st.daily_pnl, 2),
            "weekly_pnl": round(st.weekly_pnl, 2),
            "consecutive_losses": st.consecutive_losses,
            "risk_per_trade_pct": self.current_risk_pct(),
            "open_positions": [{"symbol": p.symbol, "direction": p.direction.value, "risk": p.risk_amount, "group": p.group} for p in st.open_positions],
            "limits": {"max_risk_per_trade_pct": self.s.max_risk_per_trade_pct, "max_daily_loss_pct": self.s.max_daily_loss_pct,
                       "max_weekly_loss_pct": self.s.max_weekly_loss_pct, "max_drawdown_pct": self.s.max_drawdown_pct,
                       "max_open_positions": self.s.max_open_positions, "max_consecutive_losses": self.s.max_consecutive_losses},
            "events": st.events[-10:],
        }

    # ------------------------------------------------------------ persist
    def _save(self) -> None:
        if not self.state_path:
            return
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        st = self.state
        payload = {
            "equity": st.equity, "peak_equity": st.peak_equity, "day": st.day, "week": st.week,
            "daily_pnl": st.daily_pnl, "weekly_pnl": st.weekly_pnl, "consecutive_losses": st.consecutive_losses,
            "cooldown_until": st.cooldown_until.isoformat() if st.cooldown_until else None,
            "disabled_until": st.disabled_until.isoformat() if st.disabled_until else None,
            "disabled_reason": st.disabled_reason, "trades_today": st.trades_today, "events": st.events[-100:],
            "open_positions": [{"symbol": p.symbol, "direction": p.direction.value, "risk_amount": p.risk_amount,
                                "units": p.units, "opened_at": p.opened_at.isoformat(), "group": p.group} for p in st.open_positions],
        }
        with open(self.state_path, "w") as f:
            json.dump(payload, f, indent=2)

    def _load(self) -> None:
        with open(self.state_path) as f:
            p = json.load(f)
        st = self.state
        st.equity, st.peak_equity = p["equity"], p["peak_equity"]
        st.day, st.week = p.get("day"), p.get("week")
        st.daily_pnl, st.weekly_pnl = p.get("daily_pnl", 0.0), p.get("weekly_pnl", 0.0)
        st.consecutive_losses = p.get("consecutive_losses", 0)
        st.cooldown_until = datetime.fromisoformat(p["cooldown_until"]) if p.get("cooldown_until") else None
        st.disabled_until = datetime.fromisoformat(p["disabled_until"]) if p.get("disabled_until") else None
        st.disabled_reason = p.get("disabled_reason", "")
        st.trades_today = p.get("trades_today", 0)
        st.events = p.get("events", [])
        st.open_positions = [OpenPosition(o["symbol"], Direction(o["direction"]), o["risk_amount"], o["units"],
                                          datetime.fromisoformat(o["opened_at"]), o.get("group")) for o in p.get("open_positions", [])]

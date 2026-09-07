"""
Trend-following & pullback strategies.

TrendPullbackStrategy
    HTF/structural trend + pullback into a *zone of value* (OB / FVG / demand
    / OTE / EMA20-50 / prior structure) in a discount (longs) or premium
    (shorts), confirmed on the entry timeframe by a structure shift or
    reversal candle. This is the workhorse for XAUUSD and FX in trends.

EMATrendStrategy
    Classic EMA stack (20>50>200) + price reclaim of EMA20 after a pullback +
    ADX > 20 + RSI not exhausted. Simpler, indicator-led continuation.

MomentumContinuationStrategy
    After a BOS with displacement and volume, enter on the first shallow
    pullback (≤ 0.5 ATR from the breakout level) with MACD/RSI momentum
    aligned. Used in BREAKOUT / EXPANSION / STRONG trend regimes only.
"""
from __future__ import annotations

from typing import List

from ai_trader.analysis.patterns import best_pattern
from ai_trader.core.enums import Bias, Direction, EntryType, Regime, StrategyFamily, StructureEventType
from ai_trader.core.models import Instrument, RegimeAssessment, StrategySignal

from .base import Strategy


class TrendPullbackStrategy(Strategy):
    name = "trend_pullback_smc"
    family = StrategyFamily.PULLBACK
    description = "HTF trend + pullback into OB/FVG/demand/OTE in discount/premium + entry-TF confirmation"

    def generate(self, ctx, regime: RegimeAssessment, instrument: Instrument) -> List[StrategySignal]:
        bias = ctx.dominant_bias()
        direction = self.dir_from_bias(bias)
        if direction is None:
            return []
        st = ctx.structural
        en = ctx.entry
        price = en.price
        atr_s = st.atr
        atr_e = en.atr
        # 1) price must be pulling back: in discount for longs / premium for shorts on the structural TF
        pd_state = st.smc.pd_state
        if direction is Direction.LONG and pd_state == "premium":
            return []
        if direction is Direction.SHORT and pd_state == "discount":
            return []
        # 2) zone of value on structural or entry TF
        zones = st.smc.all_zones() + en.smc.all_zones()
        zone = self.zone_near_price(zones, price, direction, atr_s, max_atr=0.75)
        ema_touch = False
        l = st.last
        if l.get("ema20") and l.get("ema50"):
            lo, hi = sorted([l["ema20"], l["ema50"]])
            ema_touch = (lo - 0.3 * atr_s) <= price <= (hi + 0.3 * atr_s)
        if zone is None and not ema_touch:
            return []
        # 3) entry-TF confirmation: recent structure event in direction OR strong reversal candle
        ev = en.structure.last_event
        bars_since = (len(en.df) - 1 - ev.index) if ev else 999
        struct_conf = ev is not None and ev.direction is direction and bars_since <= 6
        pat = best_pattern(en.patterns, direction)
        pat_conf = pat is not None and pat.strength >= 0.5
        if not (struct_conf or pat_conf):
            return []
        # 4) build levels
        if zone is not None:
            entry_low, entry_high = (zone.low, min(zone.high, price)) if direction is Direction.LONG else (max(zone.low, price), zone.high)
            if entry_high <= entry_low:
                entry_low, entry_high = (price - 0.15 * atr_e, price) if direction is Direction.LONG else (price, price + 0.15 * atr_e)
            struct_level = zone.low if direction is Direction.LONG else zone.high
        else:
            entry_low, entry_high = (price - 0.2 * atr_e, price + 0.05 * atr_e) if direction is Direction.LONG else (price - 0.05 * atr_e, price + 0.2 * atr_e)
            struct_level = l["ema50"] - (0.5 * atr_s if direction is Direction.LONG else -0.5 * atr_s)
        swing = en.structure.last_low if direction is Direction.LONG else en.structure.last_high
        if swing is not None:
            struct_level = min(struct_level, swing.price) if direction is Direction.LONG else max(struct_level, swing.price)
        stop = self.stop_beyond(struct_level, direction, atr_e, 0.3)
        entry_mid = (entry_low + entry_high) / 2
        targets = self.targets_from_levels(entry_mid, stop, direction, self.candidate_target_levels(ctx, direction))
        entry_type = EntryType.LIMIT if zone is not None and zone.distance(price) > 0 else EntryType.MARKET
        reasons = [f"HTF/structural bias {bias.value} (alignment {ctx.alignment:+.2f})",
                   f"price in {pd_state} of structural dealing range"]
        if zone is not None:
            reasons.append(f"pullback into {zone.timeframe.value} {zone.note} {zone.low:.5g}–{zone.high:.5g}")
        if ema_touch:
            reasons.append("price at EMA20/50 dynamic support/resistance")
        if struct_conf:
            reasons.append(f"entry-TF {ev.kind.value} {ev.direction.value} {bars_since} bars ago" + (" after liquidity sweep" if ev.swept_liquidity else ""))
        if pat_conf:
            reasons.append(f"entry-TF {pat.name} (strength {pat.strength:.2f})")
        sweep = en.smc.recent_sweep or st.smc.recent_sweep
        if sweep and ((direction is Direction.LONG and sweep.kind.value == "SELL_SIDE") or (direction is Direction.SHORT and sweep.kind.value == "BUY_SIDE")):
            reasons.append(f"{sweep.kind.value.lower().replace('_', '-')} liquidity swept at {sweep.level:.5g} before the move")
        conf = 0.55 + (0.15 if zone is not None else 0) + (0.15 if struct_conf else 0) + (0.1 if pat_conf else 0)
        return [StrategySignal(self.name, self.family, direction, float(entry_low), float(entry_high), entry_type, float(stop),
                               targets, en.timeframe, min(1.0, conf), reasons,
                               invalidation=f"close beyond {stop:.5g} on {en.timeframe.value} (below zone/swing {'low' if direction is Direction.LONG else 'high'}) invalidates the pullback thesis",
                               evidence={"zone": zone, "structure_event": ev if struct_conf else None, "pattern": pat if pat_conf else None,
                                         "sweep": sweep, "pd_state": pd_state, "ema_touch": ema_touch})]


class EMATrendStrategy(Strategy):
    name = "ema_trend_continuation"
    family = StrategyFamily.TREND_FOLLOWING
    description = "EMA20>50>200 stack + reclaim of EMA20 + ADX>20, RSI not exhausted"

    def generate(self, ctx, regime: RegimeAssessment, instrument: Instrument) -> List[StrategySignal]:
        st = ctx.structural
        en = ctx.entry
        l = st.last
        le = en.last
        if None in (l.get("ema20"), l.get("ema50"), le.get("ema20"), le.get("rsi"), l.get("adx")):
            return []
        e200 = l.get("ema200")
        if l["ema20"] > l["ema50"] and (e200 is None or l["ema50"] > e200):
            direction = Direction.LONG
        elif l["ema20"] < l["ema50"] and (e200 is None or l["ema50"] < e200):
            direction = Direction.SHORT
        else:
            return []
        if ctx.htf_bias is not Bias.NEUTRAL and ctx.htf_bias is not direction.bias:
            return []
        if l["adx"] < 20:
            return []
        rsi = le["rsi"]
        if (direction is Direction.LONG and rsi > 72) or (direction is Direction.SHORT and rsi < 28):
            return []  # exhausted
        # reclaim: previous entry bar closed beyond EMA20 against trend or touched it, current closes with trend
        c_prev, c_now = float(en.df["close"].iloc[-2]), float(en.df["close"].iloc[-1])
        e20_prev, e20_now = float(en.ind["ema20"].iloc[-2]), float(en.ind["ema20"].iloc[-1])
        lo_prev, hi_prev = float(en.df["low"].iloc[-2]), float(en.df["high"].iloc[-2])
        touched = (lo_prev <= e20_prev <= hi_prev) or (c_prev - e20_prev) * direction.sign <= 0
        reclaimed = (c_now - e20_now) * direction.sign > 0
        if not (touched and reclaimed):
            return []
        atr_e = en.atr
        swing = en.structure.last_low if direction is Direction.LONG else en.structure.last_high
        base_stop = swing.price if swing else (e20_now - direction.sign * 1.0 * atr_e)
        stop = self.stop_beyond(base_stop, direction, atr_e, 0.3)
        if abs(c_now - stop) > 3.5 * atr_e:
            stop = c_now - direction.sign * 2.0 * atr_e
        entry_low, entry_high = (c_now - 0.15 * atr_e, c_now + 0.05 * atr_e) if direction is Direction.LONG else (c_now - 0.05 * atr_e, c_now + 0.15 * atr_e)
        targets = self.targets_from_levels(c_now, stop, direction, self.candidate_target_levels(ctx, direction))
        reasons = [f"structural EMA stack {'bullish' if direction is Direction.LONG else 'bearish'} (ADX {l['adx']:.0f})",
                   f"entry-TF pullback to EMA20 and reclaim (RSI {rsi:.0f}, not exhausted)"]
        return [StrategySignal(self.name, self.family, direction, float(entry_low), float(entry_high), EntryType.MARKET, float(stop),
                               targets, en.timeframe, 0.6, reasons,
                               invalidation=f"close back beyond EMA20 against the trend and below/above {stop:.5g}",
                               evidence={"ema_reclaim": True, "adx": l["adx"], "rsi": rsi})]


class MomentumContinuationStrategy(Strategy):
    name = "momentum_bos_continuation"
    family = StrategyFamily.MOMENTUM
    description = "BOS with displacement + volume, entry on first shallow pullback with MACD/RSI aligned"

    def generate(self, ctx, regime: RegimeAssessment, instrument: Instrument) -> List[StrategySignal]:
        en = ctx.entry
        st = ctx.structural
        ev = en.structure.last_event
        if ev is None or ev.kind is not StructureEventType.BOS or not ev.displacement:
            return []
        bars_since = len(en.df) - 1 - ev.index
        if bars_since > 8 or bars_since < 1:
            return []
        direction = ev.direction
        if ctx.dominant_bias() is not Bias.NEUTRAL and ctx.dominant_bias() is not direction.bias:
            return []
        le = en.last
        if le.get("macd_hist") is None or le.get("rsi") is None:
            return []
        if (le["macd_hist"] * direction.sign) <= 0:
            return []
        if (direction is Direction.LONG and le["rsi"] < 50) or (direction is Direction.SHORT and le["rsi"] > 50):
            return []
        price = en.price
        atr_e = en.atr
        dist = (price - ev.level) * direction.sign
        if dist < -0.2 * atr_e or dist > 1.0 * atr_e:
            return []  # not a shallow pullback to the broken level
        stop = self.stop_beyond(ev.level, direction, atr_e, 0.8)
        entry_low, entry_high = (ev.level - 0.1 * atr_e, price) if direction is Direction.LONG else (price, ev.level + 0.1 * atr_e)
        targets = self.targets_from_levels((entry_low + entry_high) / 2, stop, direction, self.candidate_target_levels(ctx, direction))
        vol_ok = en.volume.available and (en.volume.rvol or 0) >= 1.2
        reasons = [f"{en.timeframe.value} BOS {direction.value} with displacement at {ev.level:.5g} ({bars_since} bars ago)",
                   f"MACD histogram and RSI ({le['rsi']:.0f}) aligned with break",
                   "shallow pullback to broken level (retest)"]
        if vol_ok:
            reasons.append(f"relative volume {en.volume.rvol:.1f}× on the break")
        return [StrategySignal(self.name, self.family, direction, float(entry_low), float(entry_high), EntryType.LIMIT, float(stop),
                               targets, en.timeframe, 0.55 + (0.1 if vol_ok else 0), reasons,
                               invalidation=f"close back through the broken level with acceptance (beyond {stop:.5g}) = failed breakout",
                               evidence={"structure_event": ev, "volume_confirms": vol_ok})]

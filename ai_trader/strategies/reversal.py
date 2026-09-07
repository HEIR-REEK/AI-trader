"""
Liquidity-reversal (sweep + MSS), range mean-reversion and breakout strategies.

LiquiditySweepReversalStrategy  (SMC/ICT)
    A liquidity pool (equal highs/lows, PDH/PDL, swing) is swept — wick through,
    close back inside — followed by a CHoCH/MSS on the entry timeframe with
    displacement, ideally leaving an FVG / order block to enter from. Works in
    RANGE, REVERSAL, FALSE_BREAKOUT and, as a *pullback* form, in trends
    (sweep of a sell-side pool inside an uptrend → long).

RangeMeanReversionStrategy
    In RANGE/LOW_VOL: fade the range extremes when RSI/Stochastic/BB%b are
    stretched AND a rejection candle prints AND the range is wide enough
    (≥ 3 ATR) for positive expectancy after costs.

BreakoutRetestStrategy
    In BREAKOUT/EXPANSION/CONTRACTION regimes: a validated breakout (close,
    displacement, volume, acceptance) of a range box / key level followed by a
    retest that holds. Never enters on the breakout bar itself.
"""
from __future__ import annotations

from typing import List

from ai_trader.analysis.breakout import assess_breakout, range_box
from ai_trader.analysis.patterns import best_pattern
from ai_trader.core.enums import Bias, Direction, EntryType, LiquidityType, Regime, StrategyFamily, StructureEventType, ZoneType
from ai_trader.core.models import Instrument, RegimeAssessment, StrategySignal

from .base import Strategy


class LiquiditySweepReversalStrategy(Strategy):
    name = "liquidity_sweep_mss"
    family = StrategyFamily.LIQUIDITY_REVERSAL
    description = "Liquidity sweep + CHoCH/MSS with displacement, entry from FVG/OB left by the shift"

    def generate(self, ctx, regime: RegimeAssessment, instrument: Instrument) -> List[StrategySignal]:
        en = ctx.entry
        st = ctx.structural
        # sweep on entry TF within the last ~30 bars, else on the structural TF
        from ai_trader.analysis.smc import recent_sweep as _recent_sweep
        sweep = _recent_sweep(en.smc.pools, len(en.df), within=30) or st.smc.recent_sweep
        if sweep is None:
            return []
        direction = Direction.LONG if sweep.kind is LiquidityType.SELL_SIDE else Direction.SHORT
        # in a strong trend only take sweeps *with* the trend (pullback form)
        if regime.primary in (Regime.STRONG_BULL, Regime.STRONG_BEAR):
            if (regime.primary is Regime.STRONG_BULL and direction is Direction.SHORT) or (regime.primary is Regime.STRONG_BEAR and direction is Direction.LONG):
                return []
        elif ctx.htf_bias is not Bias.NEUTRAL and ctx.htf_bias is not direction.bias and ctx.htf_strength > 0.7:
            return []  # do not fade a strong HTF trend
        # confirmation: CHoCH/MSS/BOS in direction after the sweep
        ev = en.structure.last_event
        if ev is None or ev.direction is not direction:
            return []
        n = len(en.df)
        on_entry_tf = sweep in en.smc.pools
        sweep_idx = sweep.swept_index if (on_entry_tf and sweep.swept_index is not None) else max(0, n - 30)
        # the confirming shift must come AFTER the sweep and be recent
        if ev.index < sweep_idx or n - 1 - ev.index > 12:
            return []
        price = en.price
        atr_e = en.atr
        # entry zone: FVG or OB created by the displacement leg, else 50% of the shift candle range
        zones = [z for z in en.smc.fvgs + en.smc.order_blocks if z.direction is direction and z.index >= sweep_idx - 2 and not z.mitigated]
        zone = max(zones, key=lambda z: z.strength) if zones else None
        if zone is not None and zone.distance(price) <= 1.0 * atr_e:
            entry_low, entry_high = zone.low, zone.high
            entry_type = EntryType.LIMIT if zone.distance(price) > 0 else EntryType.MARKET
        else:
            leg_lo = float(en.df["low"].iloc[ev.index:].min())
            leg_hi = float(en.df["high"].iloc[ev.index:].max())
            mid = (leg_lo + leg_hi) / 2
            entry_low, entry_high = (min(mid, price), max(mid, price)) if direction is Direction.LONG else (min(mid, price), max(mid, price))
            if entry_high - entry_low > 1.0 * atr_e:
                return []
            entry_type = EntryType.LIMIT
        extreme = float(en.df["low"].iloc[max(0, sweep_idx - 1):].min()) if direction is Direction.LONG else float(en.df["high"].iloc[max(0, sweep_idx - 1):].max())
        stop = self.stop_beyond(extreme, direction, atr_e, 0.3)
        entry_mid = (entry_low + entry_high) / 2
        # targets: opposing liquidity first
        targets = self.targets_from_levels(entry_mid, stop, direction, self.candidate_target_levels(ctx, direction))
        pat = best_pattern(en.patterns, direction)
        reasons = [f"{sweep.kind.value.lower().replace('_', '-')} liquidity swept at {sweep.level:.5g} ({sweep.note})",
                   f"{en.timeframe.value} {ev.kind.value} {direction.value}" + (" with displacement" if ev.displacement else "") + f" at {ev.level:.5g}"]
        if zone is not None:
            reasons.append(f"entry from {zone.note} {zone.low:.5g}–{zone.high:.5g} left by the shift")
        if pat:
            reasons.append(f"{pat.name} confirmation")
        pd_ok = (direction is Direction.LONG and st.smc.pd_state == "discount") or (direction is Direction.SHORT and st.smc.pd_state == "premium")
        if pd_ok:
            reasons.append(f"structural TF in {st.smc.pd_state}")
        conf = 0.5 + (0.2 if ev.kind is StructureEventType.MSS else 0.1 if ev.kind is StructureEventType.CHOCH else 0.05) + (0.1 if zone else 0) + (0.1 if pd_ok else 0)
        return [StrategySignal(self.name, self.family, direction, float(entry_low), float(entry_high), entry_type, float(stop), targets,
                               en.timeframe, min(1.0, conf), reasons,
                               invalidation=f"a close beyond the sweep extreme ({extreme:.5g}) means the sweep was a genuine breakout, not a stop hunt",
                               evidence={"sweep": sweep, "structure_event": ev, "zone": zone, "pattern": pat, "pd_ok": pd_ok})]


class RangeMeanReversionStrategy(Strategy):
    name = "range_mean_reversion"
    family = StrategyFamily.MEAN_REVERSION
    description = "Fade range extremes with RSI/Stoch/BB stretch + rejection candle; range ≥ 3 ATR"

    def generate(self, ctx, regime: RegimeAssessment, instrument: Instrument) -> List[StrategySignal]:
        if regime.primary not in (Regime.RANGE, Regime.ACCUMULATION, Regime.DISTRIBUTION, Regime.LOW_VOL, Regime.FALSE_BREAKOUT):
            return []
        st = ctx.structural
        en = ctx.entry
        hi, lo = range_box(st.df, 40)
        atr_s = st.atr
        atr_e = en.atr
        width = hi - lo
        if width < 3.0 * atr_s:
            return []
        price = en.price
        pos = (price - lo) / width
        le = en.last
        rsi, k, pctb = le.get("rsi"), le.get("stoch_k"), le.get("bb_pct")
        if None in (rsi, k, pctb):
            return []
        # stretch: lower/upper 20% of the range AND oversold/overbought on RSI + Stoch AND
        # outside/at the Bollinger band (the rejection candle usually closes back to %b ≈ 0.1–0.2)
        if pos <= 0.2 and rsi <= 35 and k <= 25 and pctb <= 0.2:
            direction = Direction.LONG
        elif pos >= 0.8 and rsi >= 65 and k >= 75 and pctb >= 0.8:
            direction = Direction.SHORT
        else:
            return []
        pat = best_pattern(en.patterns, direction)
        if pat is None or pat.strength < 0.45:
            return []
        # do not fade if the range edge was just broken with acceptance (that is a breakout, not a range)
        edge = lo if direction is Direction.LONG else hi
        br = assess_breakout(en.df, edge, Direction.SHORT if direction is Direction.LONG else Direction.LONG)
        if br is not None and br.valid:
            return []
        extreme = float(en.df["low"].tail(5).min()) if direction is Direction.LONG else float(en.df["high"].tail(5).max())
        stop = self.stop_beyond(min(extreme, lo) if direction is Direction.LONG else max(extreme, hi), direction, atr_e, 0.5)
        entry_low, entry_high = (price - 0.2 * atr_e, price + 0.1 * atr_e) if direction is Direction.LONG else (price - 0.1 * atr_e, price + 0.2 * atr_e)
        mid = lo + width / 2
        far = hi if direction is Direction.LONG else lo
        risk = abs(price - stop)
        targets = [float(mid), float(far - direction.sign * 0.5 * atr_s), float(far)]
        # ensure monotone and at least 1R
        targets = [t if (t - price) * direction.sign >= risk else price + direction.sign * risk * m for t, m in zip(targets, (1.0, 1.8, 2.5))]
        reasons = [f"{regime.primary.value} regime; range {lo:.5g}–{hi:.5g} ({width / atr_s:.1f} ATR wide)",
                   f"price at {pos:.0%} of range with RSI {rsi:.0f}, Stoch {k:.0f}, BB%b {pctb:.2f}",
                   f"{pat.name} rejection at range edge"]
        return [StrategySignal(self.name, self.family, direction, float(entry_low), float(entry_high), EntryType.MARKET, float(stop), targets,
                               en.timeframe, 0.55, reasons,
                               invalidation=f"acceptance (2 closes) beyond the range edge {edge:.5g} converts the range into a breakout — exit",
                               evidence={"pattern": pat, "range": (lo, hi), "range_pos": pos})]


class BreakoutRetestStrategy(Strategy):
    name = "breakout_retest"
    family = StrategyFamily.BREAKOUT
    description = "Validated breakout (close+displacement+volume+acceptance) of range/key level + successful retest"

    def generate(self, ctx, regime: RegimeAssessment, instrument: Instrument) -> List[StrategySignal]:
        if regime.primary not in (Regime.BREAKOUT, Regime.EXPANSION, Regime.CONTRACTION, Regime.STRONG_BULL, Regime.STRONG_BEAR, Regime.ACCUMULATION, Regime.DISTRIBUTION):
            return []
        st = ctx.structural
        en = ctx.entry
        price = en.price
        atr_s = st.atr
        atr_e = en.atr
        # candidate levels: prior structural range box edges (excluding most recent bars) + key levels
        box_hi, box_lo = range_box(st.df.iloc[:-3], 30)
        cands = [(box_hi, Direction.LONG), (box_lo, Direction.SHORT)]
        for k in ("pdh", "pwh"):
            if k in ctx.key_levels:
                cands.append((ctx.key_levels[k], Direction.LONG))
        for k in ("pdl", "pwl"):
            if k in ctx.key_levels:
                cands.append((ctx.key_levels[k], Direction.SHORT))
        out: List[StrategySignal] = []
        for level, direction in cands:
            if ctx.htf_bias is not Bias.NEUTRAL and ctx.htf_bias is not direction.bias and ctx.htf_strength > 0.6:
                continue
            if (price - level) * direction.sign < 0:
                continue  # not beyond the level
            if (price - level) * direction.sign > 1.5 * atr_s:
                continue  # too extended past the level
            br = assess_breakout(st.df, level, direction)
            if br is None or not br.valid:
                continue
            # retest on entry TF: a bar touched the level after the break and closed back on the breakout side
            recent = en.df.tail(12)
            touched = ((recent["low"] <= level + 0.2 * atr_e) & (recent["high"] >= level - 0.2 * atr_e))
            held = (recent["close"] - level) * direction.sign > 0
            if not (touched & held).any():
                continue
            if not held.iloc[-1]:
                continue
            stop = self.stop_beyond(level, direction, atr_e, 1.0)
            entry_low, entry_high = (level - 0.1 * atr_e, level + 0.4 * atr_e) if direction is Direction.LONG else (level - 0.4 * atr_e, level + 0.1 * atr_e)
            mid = (entry_low + entry_high) / 2
            targets = self.targets_from_levels(mid, stop, direction, self.candidate_target_levels(ctx, direction))
            pat = best_pattern(en.patterns, direction)
            reasons = [f"validated {direction.value} breakout of {level:.5g} (score {br.score:.2f}: {', '.join(br.reasons[:3])})",
                       "retest of the level held on the entry timeframe"]
            if pat:
                reasons.append(f"{pat.name} at retest")
            out.append(StrategySignal(self.name, self.family, direction, float(entry_low), float(entry_high), EntryType.LIMIT, float(stop), targets,
                                      en.timeframe, min(1.0, 0.45 + 0.4 * br.score), reasons,
                                      invalidation=f"close back inside the range beyond {stop:.5g} = failed breakout (fade candidates take over)",
                                      evidence={"breakout": br, "pattern": pat, "level": level}))
        return out

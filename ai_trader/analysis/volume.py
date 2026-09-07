"""
Volume analysis: confirmation, spikes, VWAP relationship, profile levels.

Many CFD/FX feeds only provide tick volume; synthetic indices have none.
``VolumeAssessment.available`` tells the confluence layer whether to score
volume or to *redistribute* its weight (so volume-less markets are not
penalised for a missing feature they cannot have).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ai_trader.core.enums import Direction

from .indicators import obv, relative_volume, volume_profile, volume_zscore, vwap


@dataclass
class VolumeAssessment:
    available: bool
    rvol: Optional[float] = None
    vol_z: Optional[float] = None
    spike: bool = False
    confirms_direction: Optional[bool] = None
    vwap: Optional[float] = None
    price_vs_vwap: Optional[str] = None
    obv_trend: Optional[str] = None
    poc: Optional[float] = None
    vah: Optional[float] = None
    val: Optional[float] = None
    climax: bool = False
    notes: List[str] = field(default_factory=list)

    def score(self, direction: Direction) -> float:
        """0..1 volume confirmation score for the direction."""
        if not self.available:
            return 0.5  # neutral; weight gets redistributed
        s = 0.3
        if self.rvol is not None:
            if self.rvol >= 1.5:
                s += 0.25
            elif self.rvol >= 1.1:
                s += 0.12
            elif self.rvol < 0.6:
                s -= 0.15
        if self.confirms_direction:
            s += 0.2
        elif self.confirms_direction is False:
            s -= 0.15
        if self.price_vs_vwap == ("above" if direction is Direction.LONG else "below"):
            s += 0.1
        if self.obv_trend == ("up" if direction is Direction.LONG else "down"):
            s += 0.15
        if self.climax:
            s -= 0.2  # climactic volume against continuation
        return float(np.clip(s, 0, 1))

    def to_dict(self) -> Dict:
        return {k: (v if not isinstance(v, float) else round(v, 5)) for k, v in self.__dict__.items()}


def assess_volume(df: pd.DataFrame, direction: Optional[Direction] = None, bars: int = 3) -> VolumeAssessment:
    if "volume" not in df or df["volume"].tail(50).sum() <= 0 or df["volume"].tail(50).nunique() <= 2:
        return VolumeAssessment(available=False, notes=["no usable volume feed — weight redistributed"])
    rv = relative_volume(df)
    vz = volume_zscore(df)
    last_rv = float(rv.iloc[-1]) if not np.isnan(rv.iloc[-1]) else None
    last_vz = float(vz.iloc[-1]) if not np.isnan(vz.iloc[-1]) else None
    spike = bool(last_vz is not None and last_vz >= 2.0)
    # up-volume vs down-volume over recent bars
    recent = df.tail(bars)
    up_v = float(recent.loc[recent["close"] >= recent["open"], "volume"].sum())
    dn_v = float(recent.loc[recent["close"] < recent["open"], "volume"].sum())
    confirms = None
    if direction is not None and (up_v + dn_v) > 0:
        confirms = (up_v > dn_v * 1.2) if direction is Direction.LONG else (dn_v > up_v * 1.2)
    v = vwap(df, "D")
    last_vwap = float(v.iloc[-1]) if not np.isnan(v.iloc[-1]) else None
    price = float(df["close"].iloc[-1])
    pv = None if last_vwap is None else ("above" if price > last_vwap else "below")
    o = obv(df)
    obv_trend = None
    if len(o) > 20:
        slope = o.iloc[-1] - o.iloc[-20]
        obv_trend = "up" if slope > 0 else ("down" if slope < 0 else "flat")
    prof = volume_profile(df, bins=24, lookback=min(len(df), 200))
    # climax: extreme volume with a long-wick bar against the move
    last = df.iloc[-1]
    rng = float(last["high"] - last["low"]) or 1e-12
    body = abs(float(last["close"] - last["open"]))
    climax = spike and body / rng < 0.35
    notes = []
    if last_rv is not None:
        notes.append(f"relative volume {last_rv:.2f}×")
    if spike:
        notes.append("volume spike (z≥2)")
    if confirms is not None:
        notes.append("volume confirms direction" if confirms else "volume does NOT confirm direction")
    if pv:
        notes.append(f"price {pv} session VWAP")
    if climax:
        notes.append("possible climax / exhaustion volume")
    return VolumeAssessment(True, last_rv, last_vz, spike, confirms, last_vwap, pv, obv_trend,
                            prof["poc"], prof["vah"], prof["val"], climax, notes)

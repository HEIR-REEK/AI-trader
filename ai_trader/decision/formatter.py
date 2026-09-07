"""Human-readable and JSON renderings of a Decision (the spec's TRADE OUTPUT FORMAT)."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from ai_trader.core.enums import DecisionType
from ai_trader.core.models import Decision, TradePlan

DISCLAIMER = ("Probabilistic decision support only. No trade is guaranteed. Position size so that the stated stop loss "
              "is an acceptable loss, and never widen a stop.")


def _regime_line(d: Decision) -> str:
    if d.regime is None:
        return "UNKNOWN"
    sec = f" (+{', '.join(r.value for r in d.regime.secondary)})" if d.regime.secondary else ""
    return f"{d.regime.primary.value}{sec} — confidence {d.regime.confidence:.0%}"


def format_decision(d: Decision) -> str:
    lines = ["=" * 72, f"INSTRUMENT:\n{d.instrument}", "", f"MARKET REGIME:\n{_regime_line(d)}", "", f"OVERALL BIAS:\n{d.bias.value}", ""]
    if d.decision is DecisionType.NO_TRADE or d.plan is None:
        lines += [d.message, "", "REASONS:"]
        lines += [f"• {r}" for r in d.reasons]
        if d.candidates:
            lines += ["", "CANDIDATES EVALUATED:"]
            for c in d.candidates[:6]:
                lines.append(f"• {c['strategy']} {c['direction']}: {c['score']}/100 — {c['rejected_because']}")
        lines += ["", "AS OF: " + d.created_at.strftime("%Y-%m-%d %H:%M UTC"), DISCLAIMER, "=" * 72]
        return "\n".join(lines)
    p: TradePlan = d.plan
    lines += [
        f"ENTRY ZONE:\n{p.entry_low} – {p.entry_high}", "",
        f"ENTRY TYPE:\n{p.entry_type.value}", "",
        f"STOP LOSS:\n{p.stop}", "",
        f"TAKE PROFIT 1:\n{p.tp1}   (1:{p.risk.rr_tp1:.1f})", "",
        f"TAKE PROFIT 2:\n{p.tp2}   (1:{p.risk.rr_tp2:.1f})", "",
        f"TAKE PROFIT 3:\n{p.tp3}   (1:{p.risk.rr_tp3:.1f})", "",
        f"RISK TO REWARD:\n{p.rr_text} (TP2)", "",
        f"CONFIDENCE SCORE:\n{p.score:.0f}/100 — {p.grade.value}", "",
        "SCORE BREAKDOWN:",
    ]
    lines += [f"• {c.name.replace('_', ' ').title():<18} {c.points:5.1f} / {c.max_points:4.1f}" for c in p.confluence.components]
    lines += ["", "CONFLUENCE FACTORS:"] + [f"• {f}" for f in p.factors]
    lines += ["", "STRATEGY:", f"{p.strategy} on {p.timeframe.value}", "",
              "POSITION SIZE:", f"{p.risk.lots:.2f} lots ({p.risk.units:,.0f} units) risking {p.risk.risk_pct:.2f}% = {p.risk.risk_amount:,.2f} "
                                f"| stop {p.risk.stop_distance:.5g} ({p.risk.stop_distance_atr:.2f} ATR)", "",
              "WHY THIS TRADE?"] + [f"• {x}" for x in p.explanation.why_this_trade]
    lines += ["", "WHY NOW?"] + [f"• {x}" for x in p.explanation.why_now]
    lines += ["", "INVALIDATION:"] + [f"• {x}" for x in p.explanation.what_invalidates]
    lines += ["", "WHAT COULD MAKE IT FAIL?"] + [f"• {x}" for x in p.explanation.what_could_make_it_fail]
    lines += ["", "NO TRADE CONDITIONS:"] + [f"• {x}" for x in p.explanation.no_trade_conditions]
    if p.warnings:
        lines += ["", "WARNINGS:"] + [f"• {w}" for w in p.warnings]
    lines += ["", "AS OF: " + p.created_at.strftime("%Y-%m-%d %H:%M UTC"), DISCLAIMER, "=" * 72]
    return "\n".join(lines)


def decision_to_dict(d: Decision) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "instrument": d.instrument, "decision": d.decision.value, "message": d.message, "bias": d.bias.value,
        "created_at": d.created_at.isoformat(), "reasons": d.reasons, "candidates": d.candidates,
        "regime": None if d.regime is None else {"primary": d.regime.primary.value, "secondary": [r.value for r in d.regime.secondary],
                                                  "confidence": round(d.regime.confidence, 3), "explanation": d.regime.explanation,
                                                  "allowed_families": [f.value for f in d.regime.allowed_families]},
        "context": _jsonable(d.context), "disclaimer": DISCLAIMER,
    }
    if d.plan is not None:
        p = d.plan
        out["plan"] = {
            "direction": p.direction.value, "entry_low": p.entry_low, "entry_high": p.entry_high, "entry_type": p.entry_type.value,
            "stop": p.stop, "tp1": p.tp1, "tp2": p.tp2, "tp3": p.tp3, "rr": p.rr, "score": p.score, "grade": p.grade.value,
            "strategy": p.strategy, "timeframe": p.timeframe.value, "factors": p.factors, "invalidation": p.invalidation,
            "breakdown": p.confluence.breakdown, "conflicts": p.confluence.conflicts,
            "risk": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in asdict(p.risk).items()},
            "explanation": asdict(p.explanation), "warnings": p.warnings, "context": _jsonable(p.context),
        }
    return out


def _jsonable(x: Any) -> Any:
    import numpy as np
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating,)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, float):
        return None if x != x else x
    if hasattr(x, "isoformat"):
        return x.isoformat()
    if hasattr(x, "value") and not isinstance(x, (int, float, str)):
        return x.value
    if hasattr(x, "__dict__") and not isinstance(x, type):
        return _jsonable(vars(x))
    return x

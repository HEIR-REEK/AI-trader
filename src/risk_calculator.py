"""
Risk Management & Position Sizing Calculator.
Calculates trade size, stop-loss distance, risk/reward ratios, and portfolio exposure.
This does NOT make trades successful — it limits how much you can lose.
"""

from typing import Dict, Optional


class RiskCalculator:
    """Professional risk management framework for trade sizing and exposure control."""

    def __init__(self, account_size: float = 10000.0, max_risk_per_trade_pct: float = 2.0):
        self.account_size = account_size
        self.max_risk_per_trade_pct = max_risk_per_trade_pct  # Never risk more than 2% per trade
        self.max_portfolio_risk_pct = 20.0  # Never risk more than 20% of account across all open trades

    def calculate_position_size(
        self,
        pair: str = "EUR/USD",
        entry_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        risk_amount: Optional[float] = None,
        pip_value_per_lot: float = 10.0,
    ) -> Dict:
        """Calculate how many lots/units to trade based on risk parameters."""
        # Fallback rate if no live feed
        rate = entry_price or 1.0842
        stop = stop_loss_price or (rate * 0.995)  # 0.5% default stop distance

        if risk_amount is None:
            risk_amount = self.account_size * (self.max_risk_per_trade_pct / 100.0)

        # Distance from entry to stop (in price terms)
        distance = abs(rate - stop)
        if distance < 0.0001:
            distance = rate * 0.005  # Default 0.5% distance

        # Position size in base currency units
        # Size = Risk Amount / (Distance / Rate) roughly — simplified for framework
        # More precisely: units = risk_amount / (distance / rate) for direct pairs
        units = risk_amount / distance if distance > 0 else 0

        # Convert to lots (standard lot = 100,000 units for forex)
        lots = units / 100000.0

        # Risk/reward estimation (requires target price)
        # We'll return a framework for setting targets
        return {
            "pair": pair,
            "entry_price": round(rate, 4),
            "stop_loss_price": round(stop, 4),
            "stop_distance_pct": round((distance / rate) * 100, 2),
            "max_risk_amount": round(risk_amount, 2),
            "max_risk_pct": self.max_risk_per_trade_pct,
            "calculated_units": round(units, 0),
            "calculated_lots": round(lots, 4),
            "pip_value_per_lot": pip_value_per_lot,
            "estimated_pip_risk": round(distance * 10000, 1),  # Approximate pip distance
            "note": "Position sizing limits losses — it does NOT make trades win. Even a perfectly sized losing trade is still a loss.",
        }

    def calculate_risk_reward(self, entry: float, stop: float, target: float) -> Dict:
        """Calculate risk/reward ratio for a potential trade setup."""
        risk_distance = abs(entry - stop)
        reward_distance = abs(entry - target)
        rr_ratio = reward_distance / risk_distance if risk_distance > 0 else 0.0

        # Probability-adjusted expectation (simplified framework)
        # Even with 2:1 R:R, if win rate is only 40%, expectation is negative
        win_rate_estimate = 0.45  # Conservative real-world estimate
        expectation = (win_rate_estimate * reward_distance) - ((1 - win_rate_estimate) * risk_distance)

        return {
            "entry": entry,
            "stop": stop,
            "target": target,
            "risk_distance": round(risk_distance, 4),
            "reward_distance": round(reward_distance, 4),
            "risk_reward_ratio": f"1:{round(rr_ratio, 2)}",
            "win_rate_assumed": f"{win_rate_estimate * 100:.0f}%",
            "expected_value_per_unit_risk": round(expectation, 4),
            "assessment": (
                "Positive expectation only if win rate > 1 / (1 + R:R ratio). "
                "Even 3:1 setups fail often in real markets due to slippage, gaps, and unexpected reversals."
            ),
        }

    def portfolio_exposure_check(self, open_trades: int = 2) -> Dict:
        """Check if current open trades exceed portfolio risk limits."""
        total_risked = open_trades * (self.account_size * self.max_risk_per_trade_pct / 100)
        exposure_pct = (total_risked / self.account_size) * 100
        within_limit = exposure_pct <= self.max_portfolio_risk_pct

        return {
            "open_trades": open_trades,
            "total_risked": round(total_risked, 2),
            "exposure_pct": round(exposure_pct, 1),
            "max_portfolio_risk_pct": self.max_portfolio_risk_pct,
            "within_limit": within_limit,
            "recommendation": (
                "Reduce exposure if approaching 20% total portfolio risk. "
                "Diversification does not eliminate risk — it spreads it. "
                "Even well-diversified portfolios can lose significantly during systemic events."
            ),
        }

    def full_risk_report(self, pair: str = "EUR/USD", entry: float = 1.0842) -> Dict:
        """Generate a complete risk management report for a potential trade."""
        stop = entry * 0.995
        target_2r = entry + ((entry - stop) * 2)  # 2:1 target
        target_3r = entry + ((entry - stop) * 3)  # 3:1 target

        position = self.calculate_position_size(pair, entry, stop)
        rr_2 = self.calculate_risk_reward(entry, stop, target_2r)
        rr_3 = self.calculate_risk_reward(entry, stop, target_3r)
        exposure = self.portfolio_exposure_check(open_trades=1)

        return {
            "pair": pair,
            "entry": entry,
            "recommended_stop": round(stop, 4),
            "position_size": position,
            "risk_reward_2_1": rr_2,
            "risk_reward_3_1": rr_3,
            "portfolio_exposure": exposure,
            "master_disclaimer": (
                "RISK MANAGEMENT IS NOT A PREDICTION TOOL. Proper position sizing ensures you survive losing streaks — "
                "it does not turn losing trades into winning ones. A well-sized losing trade remains a loss. "
                "Always combine risk management with independent analysis and never over-leverage."
            ),
        }

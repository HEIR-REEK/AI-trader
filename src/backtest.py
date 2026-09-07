"""
Backtesting Framework — Historical Strategy Performance Analysis.
Tests multi-strategy logic against simulated historical price paths.
IMPORTANT: Past performance is NEVER indicative of future results.
Backtests show what WOULD have happened — not what WILL happen.
"""

from typing import List, Dict, Optional
import random


class BacktestEngine:
    """Historical simulation framework for the multi-strategy system."""

    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.capital_history = [initial_capital]
        self.trade_log = []
        self.strategy_performance = {}

    def simulate_price_path(
        self,
        pair: str = "EUR/USD",
        days: int = 60,
        start_rate: float = 1.08,
        volatility: float = 0.008,
    ) -> List[float]:
        """Generate a simulated historical price path (for framework demonstration)."""
        prices = [start_rate]
        current = start_rate
        for _ in range(days):
            # Random walk with drift and volatility
            drift = 0.0001  # Small positive drift
            shock = random.gauss(0, 1) * volatility
            current = max(current * (1 + drift + shock), 0.5)  # Floor to prevent negative
            prices.append(round(current, 4))
        return prices

    def run_backtest(
        self,
        pair: str = "EUR/USD",
        strategy_name: str = "trend_following",
        days: int = 60,
        risk_per_trade: float = 0.02,
    ) -> Dict:
        """Run a simulated backtest for a single strategy."""
        prices = self.simulate_price_path(pair, days)
        capital = self.initial_capital
        trades = []
        wins = 0
        losses = 0

        # Simplified strategy simulation logic
        for i in range(1, len(prices) - 1):
            previous = prices[i - 1]
            current = prices[i]
            next_price = prices[i + 1]

            # Strategy-specific rules (simplified for demonstration)
            trade_signal = False
            direction = "buy"

            if strategy_name == "trend_following":
                if current > previous * 1.002:  # Small upward momentum
                    trade_signal = True
                    direction = "buy"
            elif strategy_name == "mean_reversion":
                if current < previous * 0.998:
                    trade_signal = True
                    direction = "buy"
            elif strategy_name == "breakout_volatility":
                # Breakout if price moves > 2x recent range (simulated)
                recent_high = max(prices[max(0, i - 5): i + 1])
                if current > recent_high * 1.003:
                    trade_signal = True
                    direction = "buy"
            elif strategy_name == "momentum_convergence":
                # Trade when recent price is rising
                if current > previous and next_price > current:
                    trade_signal = True
                    direction = "buy"
            elif strategy_name == "support_resistance":
                # Trade near simulated support (low recent price)
                recent_low = min(prices[max(0, i - 5): i + 1])
                if current < recent_low * 1.002:
                    trade_signal = True
                    direction = "buy"
            else:
                trade_signal = False

            if trade_signal:
                risk_amount = capital * risk_per_trade
                # Simulated outcome: 55% win rate with varying outcomes
                win = random.random() < 0.55
                if win:
                    profit = risk_amount * 1.5  # 1.5:1 reward/risk
                    capital += profit
                    wins += 1
                else:
                    loss = risk_amount
                    capital -= loss
                    losses += 1

                trades.append({
                    "day": i,
                    "price": current,
                    "direction": direction,
                    "won": win,
                    "capital_after": round(capital, 2),
                })

        final_return = (capital - self.initial_capital) / self.initial_capital
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0

        # Performance metrics
        max_drawdown = self._calculate_drawdown(self.capital_history + [capital])

        result = {
            "pair": pair,
            "strategy": strategy_name,
            "days": days,
            "initial_capital": self.initial_capital,
            "final_capital": round(capital, 2),
            "total_return_pct": round(final_return * 100, 2),
            "win_rate": round(win_rate, 2),
            "total_trades": wins + losses,
            "wins": wins,
            "losses": losses,
            "risk_per_trade": f"{risk_per_trade * 100:.1f}%",
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "trade_log_summary": trades[-5:] if trades else [],
            "warning": (
                "BACKTEST RESULTS ARE NOT PREDICTIVE. This simulation uses simplified rules and random outcomes. "
                "Real markets have slippage, gaps, fees, and unpredictable events. Past performance never guarantees future results."
            ),
        }
        return result

    def _calculate_drawdown(self, capital_series: List[float]) -> float:
        peak = capital_series[0]
        max_dd = 0.0
        for value in capital_series:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, drawdown)
        return max_dd

    def compare_strategies(self, pair: str = "EUR/USD", days: int = 60) -> Dict:
        """Compare all 5 strategies on the same price path."""
        results = {}
        for strategy in [
            "trend_following",
            "mean_reversion",
            "breakout_volatility",
            "momentum_convergence",
            "support_resistance",
        ]:
            # Note: each run uses independent random paths; this is for structure demonstration
            results[strategy] = self.run_backtest(pair, strategy, days)
        return results

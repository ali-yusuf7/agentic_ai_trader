"""
Portfolio Manager - Higher-level portfolio management and risk control
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from data.collector import DataCollector
from trading.paper_trader import PaperTrader

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Manages portfolio allocation, risk, and rebalancing."""

    def __init__(self, trader: PaperTrader = None):
        self.trader = trader or PaperTrader()
        self.collector = DataCollector()
        self.target_allocations: Dict[str, float] = {}
        self.max_positions = 10
        self.rebalance_threshold = 0.05  # 5% deviation triggers rebalance

    def get_current_allocations(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate current portfolio allocations.

        Args:
            current_prices: Current prices for holdings

        Returns:
            Dictionary of symbol -> allocation percentage
        """
        total_value = self.trader.calculate_portfolio_value(current_prices)

        if total_value == 0:
            return {}

        allocations = {"cash": self.trader.cash / total_value}

        for symbol, position in self.trader.positions.items():
            price = current_prices.get(symbol, position["avg_price"])
            value = position["quantity"] * price
            allocations[symbol] = value / total_value

        return allocations

    def set_target_allocation(self, symbol: str, target_percent: float):
        """Set target allocation for a symbol."""
        self.target_allocations[symbol] = target_percent

    def calculate_rebalance_trades(
        self,
        current_prices: Dict[str, float]
    ) -> List[Dict]:
        """
        Calculate trades needed to rebalance portfolio.

        Args:
            current_prices: Current stock prices

        Returns:
            List of trades to execute
        """
        if not self.target_allocations:
            return []

        current = self.get_current_allocations(current_prices)
        total_value = self.trader.calculate_portfolio_value(current_prices)
        trades = []

        for symbol, target in self.target_allocations.items():
            current_alloc = current.get(symbol, 0)
            diff = target - current_alloc

            # Only rebalance if deviation exceeds threshold
            if abs(diff) < self.rebalance_threshold:
                continue

            target_value = total_value * target
            current_value = total_value * current_alloc
            value_diff = target_value - current_value

            price = current_prices.get(symbol)
            if not price:
                continue

            shares = int(abs(value_diff) / price)

            if shares > 0:
                action = "buy" if value_diff > 0 else "sell"
                trades.append({
                    "symbol": symbol,
                    "action": action,
                    "quantity": shares,
                    "price": price,
                    "reason": f"Rebalance: {current_alloc:.1%} -> {target:.1%}"
                })

        return trades

    def check_risk_limits(self, current_prices: Dict[str, float]) -> Dict:
        """
        Check if portfolio is within risk limits.

        Args:
            current_prices: Current stock prices

        Returns:
            Risk check results
        """
        allocations = self.get_current_allocations(current_prices)
        violations = []

        # Check single position concentration
        for symbol, alloc in allocations.items():
            if symbol != "cash" and alloc > config.MAX_POSITION_SIZE:
                violations.append({
                    "type": "concentration",
                    "symbol": symbol,
                    "current": alloc,
                    "limit": config.MAX_POSITION_SIZE,
                    "message": f"{symbol} is {alloc:.1%} of portfolio (limit: {config.MAX_POSITION_SIZE:.1%})"
                })

        # Check minimum cash buffer
        cash_ratio = allocations.get("cash", 0)
        min_cash = 0.05  # 5% minimum cash

        if cash_ratio < min_cash:
            violations.append({
                "type": "low_cash",
                "current": cash_ratio,
                "limit": min_cash,
                "message": f"Cash is only {cash_ratio:.1%} (minimum: {min_cash:.1%})"
            })

        # Check number of positions
        num_positions = len(self.trader.positions)
        if num_positions > self.max_positions:
            violations.append({
                "type": "too_many_positions",
                "current": num_positions,
                "limit": self.max_positions,
                "message": f"Too many positions: {num_positions} (limit: {self.max_positions})"
            })

        return {
            "within_limits": len(violations) == 0,
            "violations": violations,
            "allocations": allocations
        }

    def calculate_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        horizon_days: int = 1
    ) -> float:
        """
        Calculate Value at Risk (VaR).

        Args:
            returns: Historical returns series
            confidence: Confidence level (e.g., 0.95 for 95%)
            horizon_days: Time horizon in days

        Returns:
            VaR as a percentage
        """
        if len(returns) < 20:
            return 0

        # Historical VaR
        var = np.percentile(returns, (1 - confidence) * 100)

        # Scale to horizon
        var = var * np.sqrt(horizon_days)

        return abs(var)

    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.05
    ) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate

        Returns:
            Sharpe ratio
        """
        if len(returns) < 20 or returns.std() == 0:
            return 0

        # Annualize
        daily_rf = risk_free_rate / 252
        excess_returns = returns - daily_rf

        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)

        return sharpe

    def calculate_max_drawdown(self, values: pd.Series) -> Tuple[float, datetime, datetime]:
        """
        Calculate maximum drawdown.

        Args:
            values: Series of portfolio values

        Returns:
            Tuple of (max_drawdown, peak_date, trough_date)
        """
        if len(values) < 2:
            return 0, None, None

        # Calculate running maximum
        rolling_max = values.expanding().max()

        # Calculate drawdown
        drawdown = (values - rolling_max) / rolling_max

        # Find maximum drawdown
        max_dd = drawdown.min()
        trough_idx = drawdown.idxmin()

        # Find peak before trough
        peak_idx = values[:trough_idx].idxmax()

        return abs(max_dd), peak_idx, trough_idx

    def get_portfolio_metrics(self, current_prices: Dict[str, float]) -> Dict:
        """
        Calculate comprehensive portfolio metrics.

        Args:
            current_prices: Current stock prices

        Returns:
            Dictionary of portfolio metrics
        """
        summary = self.trader.get_summary(current_prices)

        # Get portfolio history for advanced metrics
        history = self.trader.storage.get_portfolio_history(days=config.PERFORMANCE_WINDOW)

        metrics = {
            **summary,
            "risk_check": self.check_risk_limits(current_prices)
        }

        if not history.empty and len(history) > 1:
            values = history["total_value"]
            returns = values.pct_change().dropna()

            if len(returns) > 0:
                metrics["daily_returns_mean"] = returns.mean()
                metrics["daily_returns_std"] = returns.std()
                metrics["sharpe_ratio"] = self.calculate_sharpe_ratio(returns)
                metrics["var_95"] = self.calculate_var(returns, 0.95)

                max_dd, peak, trough = self.calculate_max_drawdown(values)
                metrics["max_drawdown"] = max_dd
                metrics["max_drawdown_peak"] = str(peak) if peak else None
                metrics["max_drawdown_trough"] = str(trough) if trough else None

        return metrics

    def get_position_recommendation(
        self,
        symbol: str,
        signal: str,
        confidence: float,
        current_price: float
    ) -> Dict:
        """
        Get recommendation for position sizing.

        Args:
            symbol: Stock symbol
            signal: Trading signal (buy/sell/hold)
            confidence: Model confidence
            current_price: Current stock price

        Returns:
            Position recommendation
        """
        current_position = self.trader.get_position(symbol)

        if signal == "hold":
            return {
                "action": "hold",
                "symbol": symbol,
                "reason": "Signal is hold"
            }

        if signal == "buy":
            # Scale position size by confidence
            base_risk = config.RISK_PER_TRADE
            adjusted_risk = base_risk * confidence

            shares = self.trader.calculate_shares_to_buy(
                symbol, current_price, adjusted_risk
            )

            if shares == 0:
                return {
                    "action": "hold",
                    "symbol": symbol,
                    "reason": "Cannot afford any shares"
                }

            return {
                "action": "buy",
                "symbol": symbol,
                "quantity": shares,
                "price": current_price,
                "estimated_cost": shares * current_price,
                "confidence": confidence,
                "risk_percent": adjusted_risk
            }

        elif signal == "sell":
            if not current_position:
                return {
                    "action": "hold",
                    "symbol": symbol,
                    "reason": "No position to sell"
                }

            # Sell proportion based on confidence
            # High confidence = sell more
            sell_ratio = min(confidence, 1.0)
            shares_to_sell = int(current_position["quantity"] * sell_ratio)

            if shares_to_sell == 0:
                shares_to_sell = current_position["quantity"]  # Sell all if very small

            return {
                "action": "sell",
                "symbol": symbol,
                "quantity": shares_to_sell,
                "price": current_price,
                "estimated_proceeds": shares_to_sell * current_price,
                "confidence": confidence,
                "remaining_shares": current_position["quantity"] - shares_to_sell
            }

        return {
            "action": "hold",
            "symbol": symbol,
            "reason": "Unknown signal"
        }

    def execute_recommendation(self, recommendation: Dict, reasoning: str = None) -> Dict:
        """
        Execute a position recommendation.

        Args:
            recommendation: Recommendation from get_position_recommendation
            reasoning: Additional reasoning for the trade

        Returns:
            Execution result
        """
        action = recommendation.get("action")

        if action == "hold":
            return {"success": True, "action": "hold", "reason": recommendation.get("reason")}

        if action == "buy":
            return self.trader.buy(
                symbol=recommendation["symbol"],
                quantity=recommendation["quantity"],
                price=recommendation["price"],
                reasoning=reasoning,
                confidence=recommendation.get("confidence")
            )

        if action == "sell":
            return self.trader.sell(
                symbol=recommendation["symbol"],
                quantity=recommendation["quantity"],
                price=recommendation["price"],
                reasoning=reasoning,
                confidence=recommendation.get("confidence")
            )

        return {"success": False, "reason": "Invalid action"}

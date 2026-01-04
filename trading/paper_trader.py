"""
Paper Trader - Simulates trading without real money
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import config
from data.storage import DataStorage

logger = logging.getLogger(__name__)


class PaperTrader:
    """Simulates stock trading with paper money."""

    def __init__(self, initial_capital: float = config.INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Dict] = {}  # symbol -> {quantity, avg_price, total_cost}
        self.trade_history: List[Dict] = []
        self.storage = DataStorage()

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get current position for a symbol."""
        return self.positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Dict]:
        """Get all current positions."""
        return self.positions.copy()

    def calculate_position_value(self, symbol: str, current_price: float) -> float:
        """Calculate current value of a position."""
        position = self.positions.get(symbol)
        if not position:
            return 0
        return position["quantity"] * current_price

    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value.

        Args:
            current_prices: Dictionary of symbol -> current price

        Returns:
            Total portfolio value (cash + positions)
        """
        total = self.cash

        for symbol, position in self.positions.items():
            price = current_prices.get(symbol, position["avg_price"])
            total += position["quantity"] * price

        return total

    def calculate_position_pnl(self, symbol: str, current_price: float) -> Dict:
        """
        Calculate P&L for a position.

        Args:
            symbol: Stock symbol
            current_price: Current stock price

        Returns:
            Dictionary with P&L details
        """
        position = self.positions.get(symbol)
        if not position:
            return {"error": "No position"}

        current_value = position["quantity"] * current_price
        cost_basis = position["total_cost"]
        pnl = current_value - cost_basis
        pnl_percent = (pnl / cost_basis) * 100 if cost_basis > 0 else 0

        return {
            "symbol": symbol,
            "quantity": position["quantity"],
            "avg_price": position["avg_price"],
            "current_price": current_price,
            "cost_basis": cost_basis,
            "current_value": current_value,
            "pnl": pnl,
            "pnl_percent": pnl_percent
        }

    def can_buy(self, symbol: str, quantity: int, price: float) -> Tuple[bool, str]:
        """
        Check if a buy order can be executed.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Price per share

        Returns:
            Tuple of (can_execute, reason)
        """
        total_cost = quantity * price

        if total_cost > self.cash:
            return False, f"Insufficient cash: need ${total_cost:.2f}, have ${self.cash:.2f}"

        # Check position size limit
        if symbol in self.positions:
            current_value = self.positions[symbol]["quantity"] * price
            new_total = current_value + total_cost
        else:
            new_total = total_cost

        portfolio_value = self.cash + sum(
            p["quantity"] * price for p in self.positions.values()
        )

        if new_total / portfolio_value > config.MAX_POSITION_SIZE:
            return False, f"Would exceed max position size of {config.MAX_POSITION_SIZE * 100}%"

        return True, "OK"

    def can_sell(self, symbol: str, quantity: int) -> Tuple[bool, str]:
        """
        Check if a sell order can be executed.

        Args:
            symbol: Stock symbol
            quantity: Number of shares to sell

        Returns:
            Tuple of (can_execute, reason)
        """
        position = self.positions.get(symbol)

        if not position:
            return False, f"No position in {symbol}"

        if quantity > position["quantity"]:
            return False, f"Insufficient shares: have {position['quantity']}, want to sell {quantity}"

        return True, "OK"

    def buy(
        self,
        symbol: str,
        quantity: int,
        price: float,
        reasoning: str = None,
        confidence: float = None
    ) -> Dict:
        """
        Execute a buy order.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Price per share
            reasoning: Why this trade was made
            confidence: Model confidence

        Returns:
            Trade execution result
        """
        can_execute, reason = self.can_buy(symbol, quantity, price)

        if not can_execute:
            logger.warning(f"Cannot buy {symbol}: {reason}")
            return {"success": False, "reason": reason}

        total_cost = quantity * price

        # Update position
        if symbol in self.positions:
            existing = self.positions[symbol]
            new_quantity = existing["quantity"] + quantity
            new_total_cost = existing["total_cost"] + total_cost
            new_avg_price = new_total_cost / new_quantity

            self.positions[symbol] = {
                "quantity": new_quantity,
                "avg_price": new_avg_price,
                "total_cost": new_total_cost
            }
        else:
            self.positions[symbol] = {
                "quantity": quantity,
                "avg_price": price,
                "total_cost": total_cost
            }

        # Deduct cash
        self.cash -= total_cost

        # Record trade
        trade = {
            "symbol": symbol,
            "action": "buy",
            "quantity": quantity,
            "price": price,
            "total_value": total_cost,
            "timestamp": datetime.now(),
            "reasoning": reasoning,
            "confidence": confidence,
            "cash_after": self.cash
        }
        self.trade_history.append(trade)

        # Persist to database
        self.storage.record_trade(
            symbol=symbol,
            action="buy",
            quantity=quantity,
            price=price,
            reasoning=reasoning,
            confidence=confidence
        )

        logger.info(f"BUY {quantity} {symbol} @ ${price:.2f} = ${total_cost:.2f}")

        return {
            "success": True,
            "trade": trade,
            "position": self.positions[symbol]
        }

    def sell(
        self,
        symbol: str,
        quantity: int,
        price: float,
        reasoning: str = None,
        confidence: float = None
    ) -> Dict:
        """
        Execute a sell order.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Price per share
            reasoning: Why this trade was made
            confidence: Model confidence

        Returns:
            Trade execution result
        """
        can_execute, reason = self.can_sell(symbol, quantity)

        if not can_execute:
            logger.warning(f"Cannot sell {symbol}: {reason}")
            return {"success": False, "reason": reason}

        total_value = quantity * price

        # Calculate P&L for this trade
        position = self.positions[symbol]
        cost_basis = position["avg_price"] * quantity
        trade_pnl = total_value - cost_basis

        # Update position
        new_quantity = position["quantity"] - quantity

        if new_quantity == 0:
            del self.positions[symbol]
        else:
            new_total_cost = position["total_cost"] - cost_basis
            self.positions[symbol] = {
                "quantity": new_quantity,
                "avg_price": position["avg_price"],  # Avg price stays same
                "total_cost": new_total_cost
            }

        # Add cash
        self.cash += total_value

        # Record trade
        trade = {
            "symbol": symbol,
            "action": "sell",
            "quantity": quantity,
            "price": price,
            "total_value": total_value,
            "pnl": trade_pnl,
            "timestamp": datetime.now(),
            "reasoning": reasoning,
            "confidence": confidence,
            "cash_after": self.cash
        }
        self.trade_history.append(trade)

        # Persist to database
        self.storage.record_trade(
            symbol=symbol,
            action="sell",
            quantity=quantity,
            price=price,
            reasoning=reasoning,
            confidence=confidence
        )

        logger.info(f"SELL {quantity} {symbol} @ ${price:.2f} = ${total_value:.2f} (P&L: ${trade_pnl:.2f})")

        return {
            "success": True,
            "trade": trade,
            "pnl": trade_pnl,
            "remaining_position": self.positions.get(symbol)
        }

    def calculate_shares_to_buy(
        self,
        symbol: str,
        price: float,
        risk_percent: float = config.RISK_PER_TRADE
    ) -> int:
        """
        Calculate number of shares to buy based on risk management.

        Args:
            symbol: Stock symbol
            price: Current price
            risk_percent: Percentage of portfolio to risk

        Returns:
            Number of shares to buy
        """
        # Calculate max amount to invest
        portfolio_value = self.cash + sum(
            p["total_cost"] for p in self.positions.values()
        )
        max_investment = portfolio_value * risk_percent

        # Don't exceed available cash
        max_investment = min(max_investment, self.cash * 0.95)  # Keep 5% cash buffer

        # Calculate shares
        shares = int(max_investment / price)

        return max(shares, 0)

    def get_trade_history(self, symbol: str = None, limit: int = 50) -> List[Dict]:
        """Get trade history."""
        if symbol:
            trades = [t for t in self.trade_history if t["symbol"] == symbol]
        else:
            trades = self.trade_history

        return trades[-limit:]

    def get_summary(self, current_prices: Dict[str, float]) -> Dict:
        """
        Get portfolio summary.

        Args:
            current_prices: Current prices for all held symbols

        Returns:
            Portfolio summary
        """
        total_value = self.calculate_portfolio_value(current_prices)
        total_pnl = total_value - self.initial_capital
        pnl_percent = (total_pnl / self.initial_capital) * 100

        positions_summary = []
        for symbol, position in self.positions.items():
            price = current_prices.get(symbol, position["avg_price"])
            pnl_info = self.calculate_position_pnl(symbol, price)
            positions_summary.append(pnl_info)

        return {
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "positions_value": total_value - self.cash,
            "total_value": total_value,
            "total_pnl": total_pnl,
            "total_pnl_percent": pnl_percent,
            "num_positions": len(self.positions),
            "positions": positions_summary,
            "num_trades": len(self.trade_history)
        }

    def record_snapshot(self, current_prices: Dict[str, float]):
        """Record portfolio snapshot to database."""
        total_value = self.calculate_portfolio_value(current_prices)

        positions_data = {}
        for symbol, position in self.positions.items():
            price = current_prices.get(symbol, position["avg_price"])
            positions_data[symbol] = {
                **position,
                "current_price": price,
                "current_value": position["quantity"] * price
            }

        self.storage.record_portfolio_snapshot(
            total_value=total_value,
            cash=self.cash,
            positions=positions_data
        )

    def reset(self):
        """Reset to initial state."""
        self.cash = self.initial_capital
        self.positions = {}
        self.trade_history = []
        logger.info(f"Portfolio reset to ${self.initial_capital:.2f}")

"""
Performance Analyzer - Track and analyze trading performance
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from data.storage import DataStorage

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """Analyzes trading and prediction performance."""

    def __init__(self):
        self.storage = DataStorage()

    def analyze_predictions(self, days: int = config.PERFORMANCE_WINDOW) -> Dict:
        """
        Analyze prediction accuracy over a time period.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with prediction analysis
        """
        predictions_df = self.storage.get_predictions_for_analysis(days=days)

        if predictions_df.empty:
            return {
                "total_predictions": 0,
                "accuracy": 0,
                "error": "No predictions with actual results"
            }

        # Overall accuracy
        correct = (predictions_df["predicted_direction"] == predictions_df["actual_direction"]).sum()
        total = len(predictions_df)
        accuracy = correct / total if total > 0 else 0

        # Accuracy by direction
        up_mask = predictions_df["predicted_direction"] == "up"
        down_mask = predictions_df["predicted_direction"] == "down"

        up_correct = (predictions_df[up_mask]["actual_direction"] == "up").sum()
        up_total = up_mask.sum()

        down_correct = (predictions_df[down_mask]["actual_direction"] == "down").sum()
        down_total = down_mask.sum()

        # Accuracy by model
        model_accuracy = {}
        if "model_name" in predictions_df.columns:
            for model in predictions_df["model_name"].unique():
                if pd.isna(model):
                    continue
                model_df = predictions_df[predictions_df["model_name"] == model]
                model_correct = (model_df["predicted_direction"] == model_df["actual_direction"]).sum()
                model_accuracy[model] = model_correct / len(model_df) if len(model_df) > 0 else 0

        # Confidence analysis
        avg_confidence = predictions_df["confidence"].mean() if "confidence" in predictions_df.columns else None

        # High confidence accuracy
        if "confidence" in predictions_df.columns:
            high_conf = predictions_df[predictions_df["confidence"] > 0.7]
            high_conf_accuracy = (
                (high_conf["predicted_direction"] == high_conf["actual_direction"]).sum() / len(high_conf)
                if len(high_conf) > 0 else None
            )
        else:
            high_conf_accuracy = None

        return {
            "total_predictions": total,
            "correct_predictions": correct,
            "accuracy": accuracy,
            "up_predictions": up_total,
            "up_accuracy": up_correct / up_total if up_total > 0 else 0,
            "down_predictions": down_total,
            "down_accuracy": down_correct / down_total if down_total > 0 else 0,
            "model_accuracy": model_accuracy,
            "average_confidence": avg_confidence,
            "high_confidence_accuracy": high_conf_accuracy
        }

    def analyze_trades(self, days: int = config.PERFORMANCE_WINDOW) -> Dict:
        """
        Analyze trading performance.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with trading analysis
        """
        trades = self.storage.get_trades(days=days, limit=1000)

        if not trades:
            return {
                "total_trades": 0,
                "error": "No trades found"
            }

        df = pd.DataFrame(trades)

        # Basic stats
        total_trades = len(df)
        buys = len(df[df["action"] == "buy"])
        sells = len(df[df["action"] == "sell"])

        # P&L analysis (for sells that have P&L)
        sell_df = df[df["action"] == "sell"].copy()

        # Calculate win rate from trade history
        # This is simplified - in real implementation we'd track paired trades
        total_value_bought = df[df["action"] == "buy"]["total_value"].sum()
        total_value_sold = df[df["action"] == "sell"]["total_value"].sum()

        # Group by symbol
        symbol_stats = {}
        for symbol in df["symbol"].unique():
            symbol_df = df[df["symbol"] == symbol]
            symbol_stats[symbol] = {
                "total_trades": len(symbol_df),
                "buys": len(symbol_df[symbol_df["action"] == "buy"]),
                "sells": len(symbol_df[symbol_df["action"] == "sell"]),
                "total_bought": symbol_df[symbol_df["action"] == "buy"]["total_value"].sum(),
                "total_sold": symbol_df[symbol_df["action"] == "sell"]["total_value"].sum()
            }

        # Trading frequency
        if len(df) > 1:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            trading_days = (df["timestamp"].max() - df["timestamp"].min()).days + 1
            trades_per_day = total_trades / trading_days if trading_days > 0 else 0
        else:
            trades_per_day = 0

        return {
            "total_trades": total_trades,
            "buy_trades": buys,
            "sell_trades": sells,
            "total_bought": total_value_bought,
            "total_sold": total_value_sold,
            "trades_per_day": trades_per_day,
            "symbol_breakdown": symbol_stats
        }

    def calculate_returns(self, days: int = config.PERFORMANCE_WINDOW) -> Dict:
        """
        Calculate portfolio returns.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with return metrics
        """
        history = self.storage.get_portfolio_history(days=days)

        if history.empty or len(history) < 2:
            return {
                "error": "Insufficient portfolio history"
            }

        values = history["total_value"]
        returns = values.pct_change().dropna()

        if len(returns) == 0:
            return {"error": "Cannot calculate returns"}

        # Basic return metrics
        total_return = (values.iloc[-1] / values.iloc[0]) - 1
        avg_daily_return = returns.mean()
        std_daily_return = returns.std()

        # Annualized metrics
        trading_days = len(returns)
        annualized_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0
        annualized_volatility = std_daily_return * np.sqrt(252)

        # Risk-adjusted returns
        risk_free_rate = 0.05  # 5% annual
        daily_rf = risk_free_rate / 252
        excess_returns = returns - daily_rf

        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = (excess_returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0

        # Drawdown analysis
        rolling_max = values.expanding().max()
        drawdown = (values - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        current_drawdown = drawdown.iloc[-1]

        # Win/Loss days
        winning_days = (returns > 0).sum()
        losing_days = (returns < 0).sum()
        win_rate = winning_days / len(returns) if len(returns) > 0 else 0

        return {
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "annualized_return": annualized_return,
            "annualized_return_pct": annualized_return * 100,
            "avg_daily_return": avg_daily_return,
            "daily_volatility": std_daily_return,
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown * 100,
            "current_drawdown": current_drawdown,
            "winning_days": winning_days,
            "losing_days": losing_days,
            "win_rate": win_rate,
            "trading_days": trading_days,
            "start_value": values.iloc[0],
            "end_value": values.iloc[-1]
        }

    def get_model_rankings(self) -> List[Tuple[str, float]]:
        """
        Rank models by recent performance.

        Returns:
            List of (model_name, score) tuples, sorted by score
        """
        performance_df = self.storage.get_model_performance(days=config.PERFORMANCE_WINDOW)

        if performance_df.empty:
            return []

        # Get latest performance per model
        latest = performance_df.sort_values("evaluation_date").groupby("model_name").last()

        rankings = []
        for model_name, row in latest.iterrows():
            # Combined score: weighted average of accuracy and F1
            accuracy = row.get("accuracy", 0) or 0
            f1 = row.get("f1_score", 0) or 0
            score = accuracy * 0.4 + f1 * 0.6
            rankings.append((model_name, score))

        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def generate_report(self, current_prices: Dict[str, float] = None) -> str:
        """
        Generate a comprehensive performance report.

        Args:
            current_prices: Current stock prices (for portfolio value)

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("TRADING PERFORMANCE REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 60)

        # Prediction analysis
        pred_analysis = self.analyze_predictions()
        lines.append("\n--- PREDICTION PERFORMANCE ---")
        lines.append(f"Total Predictions: {pred_analysis.get('total_predictions', 0)}")
        lines.append(f"Overall Accuracy: {pred_analysis.get('accuracy', 0):.1%}")

        if pred_analysis.get("model_accuracy"):
            lines.append("\nBy Model:")
            for model, acc in pred_analysis["model_accuracy"].items():
                lines.append(f"  {model}: {acc:.1%}")

        # Returns analysis
        returns = self.calculate_returns()
        if "error" not in returns:
            lines.append("\n--- PORTFOLIO RETURNS ---")
            lines.append(f"Total Return: {returns.get('total_return_pct', 0):.2f}%")
            lines.append(f"Annualized Return: {returns.get('annualized_return_pct', 0):.2f}%")
            lines.append(f"Sharpe Ratio: {returns.get('sharpe_ratio', 0):.2f}")
            lines.append(f"Max Drawdown: {returns.get('max_drawdown_pct', 0):.2f}%")
            lines.append(f"Win Rate: {returns.get('win_rate', 0):.1%}")

        # Trade analysis
        trades = self.analyze_trades()
        if trades.get("total_trades", 0) > 0:
            lines.append("\n--- TRADING ACTIVITY ---")
            lines.append(f"Total Trades: {trades['total_trades']}")
            lines.append(f"Buys: {trades['buy_trades']}, Sells: {trades['sell_trades']}")
            lines.append(f"Trades per Day: {trades.get('trades_per_day', 0):.1f}")

        # Model rankings
        rankings = self.get_model_rankings()
        if rankings:
            lines.append("\n--- MODEL RANKINGS ---")
            for i, (model, score) in enumerate(rankings, 1):
                lines.append(f"  {i}. {model}: {score:.3f}")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)

    def should_retrain_models(self) -> Tuple[bool, str]:
        """
        Determine if models should be retrained.

        Returns:
            Tuple of (should_retrain, reason)
        """
        pred_analysis = self.analyze_predictions()

        if pred_analysis.get("total_predictions", 0) < config.MIN_TRADES_FOR_EVALUATION:
            return False, "Insufficient predictions for evaluation"

        accuracy = pred_analysis.get("accuracy", 0)

        # Retrain if accuracy drops below 50% (worse than random)
        if accuracy < 0.50:
            return True, f"Low accuracy: {accuracy:.1%}"

        # Check if high-confidence predictions are underperforming
        high_conf_acc = pred_analysis.get("high_confidence_accuracy")
        if high_conf_acc and high_conf_acc < 0.55:
            return True, f"High confidence predictions underperforming: {high_conf_acc:.1%}"

        return False, "Performance acceptable"

    def get_improvement_suggestions(self) -> List[str]:
        """
        Generate suggestions for improving trading performance.

        Returns:
            List of suggestion strings
        """
        suggestions = []

        pred_analysis = self.analyze_predictions()
        returns = self.calculate_returns()

        # Check overall accuracy
        accuracy = pred_analysis.get("accuracy", 0)
        if accuracy < 0.55:
            suggestions.append("Overall prediction accuracy is low. Consider retraining models with more recent data.")

        # Check directional bias
        up_acc = pred_analysis.get("up_accuracy", 0)
        down_acc = pred_analysis.get("down_accuracy", 0)

        if abs(up_acc - down_acc) > 0.2:
            if up_acc > down_acc:
                suggestions.append("Models are better at predicting upward moves. Consider being more selective with sell signals.")
            else:
                suggestions.append("Models are better at predicting downward moves. Consider being more selective with buy signals.")

        # Check Sharpe ratio
        if "sharpe_ratio" in returns:
            sharpe = returns["sharpe_ratio"]
            if sharpe < 0:
                suggestions.append("Negative Sharpe ratio indicates poor risk-adjusted returns. Consider reducing position sizes.")
            elif sharpe < 1:
                suggestions.append("Sharpe ratio below 1. Consider being more selective with trade entries.")

        # Check drawdown
        if "max_drawdown_pct" in returns:
            dd = abs(returns["max_drawdown_pct"])
            if dd > 20:
                suggestions.append(f"High max drawdown ({dd:.1f}%). Consider implementing stop-losses.")

        # Check win rate
        if "win_rate" in returns:
            win_rate = returns["win_rate"]
            if win_rate < 0.45:
                suggestions.append("Low win rate. Consider increasing confidence thresholds for trade execution.")

        if not suggestions:
            suggestions.append("Performance metrics look reasonable. Continue monitoring.")

        return suggestions

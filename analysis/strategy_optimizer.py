"""
Strategy Optimizer - Self-improving strategy adjustment
"""
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from analysis.performance import PerformanceAnalyzer
from data.storage import DataStorage

logger = logging.getLogger(__name__)


class StrategyOptimizer:
    """Self-improving strategy optimization based on performance analysis."""

    def __init__(self):
        self.storage = DataStorage()
        self.analyzer = PerformanceAnalyzer()
        self.current_weights = self._load_weights()
        self.optimization_history: List[Dict] = []

    def _load_weights(self) -> Dict[str, float]:
        """Load latest weights from storage or use defaults."""
        saved_weights = self.storage.get_latest_weights()
        if saved_weights:
            logger.info(f"Loaded saved weights: {saved_weights}")
            return saved_weights
        return config.DEFAULT_MODEL_WEIGHTS.copy()

    def analyze_model_performance(self) -> Dict[str, Dict]:
        """
        Analyze each model's recent performance.

        Returns:
            Dictionary of model performance metrics
        """
        predictions_df = self.storage.get_predictions_for_analysis(days=config.PERFORMANCE_WINDOW)

        if predictions_df.empty:
            return {}

        model_stats = {}

        for model in predictions_df["model_name"].dropna().unique():
            model_df = predictions_df[predictions_df["model_name"] == model]

            if len(model_df) < 5:  # Need minimum predictions
                continue

            correct = (model_df["predicted_direction"] == model_df["actual_direction"]).sum()
            total = len(model_df)

            # Calculate metrics
            accuracy = correct / total

            # Profit factor approximation
            # (sum of correct prediction magnitudes / sum of incorrect)
            if "predicted_change" in model_df.columns and "actual_change" in model_df.columns:
                correct_mask = model_df["predicted_direction"] == model_df["actual_direction"]
                correct_gains = model_df[correct_mask]["actual_change"].abs().sum()
                incorrect_losses = model_df[~correct_mask]["actual_change"].abs().sum()
                profit_factor = correct_gains / incorrect_losses if incorrect_losses > 0 else float('inf')
            else:
                profit_factor = None

            # Average confidence when correct vs incorrect
            if "confidence" in model_df.columns:
                correct_conf = model_df[model_df["predicted_direction"] == model_df["actual_direction"]]["confidence"].mean()
                incorrect_conf = model_df[model_df["predicted_direction"] != model_df["actual_direction"]]["confidence"].mean()
            else:
                correct_conf = incorrect_conf = None

            model_stats[model] = {
                "total_predictions": total,
                "accuracy": accuracy,
                "correct": correct,
                "profit_factor": profit_factor,
                "avg_confidence_when_correct": correct_conf,
                "avg_confidence_when_incorrect": incorrect_conf
            }

        return model_stats

    def calculate_optimal_weights(self, model_stats: Dict[str, Dict]) -> Dict[str, float]:
        """
        Calculate optimal model weights based on performance.

        Args:
            model_stats: Performance statistics for each model

        Returns:
            Dictionary of optimized weights
        """
        if not model_stats:
            return self.current_weights

        scores = {}

        for model, stats in model_stats.items():
            if model not in self.current_weights:
                continue

            # Base score from accuracy
            accuracy = stats.get("accuracy", 0.5)

            # Bonus for high accuracy
            if accuracy > 0.6:
                accuracy_score = accuracy * 1.2
            elif accuracy < 0.45:
                accuracy_score = accuracy * 0.8
            else:
                accuracy_score = accuracy

            # Bonus for good profit factor
            pf = stats.get("profit_factor")
            if pf and pf > 1.5:
                pf_bonus = 0.1
            elif pf and pf < 0.8:
                pf_bonus = -0.1
            else:
                pf_bonus = 0

            # Bonus for confidence calibration
            # (higher confidence when correct is good)
            conf_correct = stats.get("avg_confidence_when_correct")
            conf_incorrect = stats.get("avg_confidence_when_incorrect")
            if conf_correct and conf_incorrect:
                if conf_correct > conf_incorrect:
                    conf_bonus = 0.05
                else:
                    conf_bonus = -0.05
            else:
                conf_bonus = 0

            scores[model] = max(accuracy_score + pf_bonus + conf_bonus, 0.1)

        # Normalize to get weights
        total_score = sum(scores.values())
        if total_score > 0:
            return {model: score / total_score for model, score in scores.items()}

        return self.current_weights

    def optimize_weights(self, force: bool = False) -> Dict:
        """
        Optimize model weights based on recent performance.

        Args:
            force: Force optimization even if not enough data

        Returns:
            Optimization result
        """
        # Check if we have enough data
        pred_analysis = self.analyzer.analyze_predictions()

        if not force and pred_analysis.get("total_predictions", 0) < config.MIN_TRADES_FOR_EVALUATION:
            return {
                "optimized": False,
                "reason": "Insufficient predictions for optimization",
                "current_weights": self.current_weights
            }

        # Analyze model performance
        model_stats = self.analyze_model_performance()

        if not model_stats:
            return {
                "optimized": False,
                "reason": "No model statistics available",
                "current_weights": self.current_weights
            }

        # Calculate optimal weights
        old_weights = self.current_weights.copy()
        new_weights = self.calculate_optimal_weights(model_stats)

        # Smooth transition (don't change too drastically)
        smoothed_weights = {}
        for model in self.current_weights:
            old = old_weights.get(model, 0.33)
            new = new_weights.get(model, 0.33)
            # Move 40% toward new weight
            smoothed_weights[model] = old * 0.6 + new * 0.4

        # Normalize
        total = sum(smoothed_weights.values())
        smoothed_weights = {k: v / total for k, v in smoothed_weights.items()}

        # Check if weights changed significantly
        weight_change = sum(abs(smoothed_weights[m] - old_weights[m]) for m in smoothed_weights)

        if weight_change < 0.05:
            return {
                "optimized": False,
                "reason": "Weights already optimal",
                "current_weights": self.current_weights,
                "model_stats": model_stats
            }

        # Update weights
        self.current_weights = smoothed_weights

        # Save to storage
        self.storage.save_strategy_weights(
            weights=smoothed_weights,
            reason="Automated optimization based on performance",
            performance_metrics=model_stats
        )

        # Record optimization
        optimization_record = {
            "timestamp": datetime.now().isoformat(),
            "old_weights": old_weights,
            "new_weights": smoothed_weights,
            "model_stats": model_stats,
            "weight_change": weight_change
        }
        self.optimization_history.append(optimization_record)

        logger.info(f"Optimized weights: {old_weights} -> {smoothed_weights}")

        return {
            "optimized": True,
            "old_weights": old_weights,
            "new_weights": smoothed_weights,
            "model_stats": model_stats,
            "weight_change": weight_change
        }

    def suggest_parameter_adjustments(self) -> List[Dict]:
        """
        Suggest parameter adjustments based on performance analysis.

        Returns:
            List of suggested adjustments
        """
        suggestions = []
        pred_analysis = self.analyzer.analyze_predictions()
        returns = self.analyzer.calculate_returns()

        # Threshold adjustments
        accuracy = pred_analysis.get("accuracy", 0.5)

        if accuracy < 0.50:
            suggestions.append({
                "parameter": "BUY_THRESHOLD",
                "current": config.BUY_THRESHOLD,
                "suggested": min(config.BUY_THRESHOLD + 0.05, 0.75),
                "reason": "Increase buy threshold due to low accuracy"
            })
            suggestions.append({
                "parameter": "SELL_THRESHOLD",
                "current": config.SELL_THRESHOLD,
                "suggested": max(config.SELL_THRESHOLD - 0.05, 0.25),
                "reason": "Decrease sell threshold due to low accuracy"
            })

        # Risk adjustments based on drawdown
        if "max_drawdown_pct" in returns:
            dd = abs(returns["max_drawdown_pct"])
            if dd > 15:
                new_risk = max(config.RISK_PER_TRADE * 0.8, 0.01)
                suggestions.append({
                    "parameter": "RISK_PER_TRADE",
                    "current": config.RISK_PER_TRADE,
                    "suggested": new_risk,
                    "reason": f"Reduce position size due to high drawdown ({dd:.1f}%)"
                })

        # High confidence accuracy check
        high_conf_acc = pred_analysis.get("high_confidence_accuracy")
        if high_conf_acc and high_conf_acc > 0.65:
            suggestions.append({
                "parameter": "CONFIDENCE_MULTIPLIER",
                "current": 1.0,
                "suggested": 1.2,
                "reason": "Increase position size on high-confidence signals"
            })

        return suggestions

    def adapt_to_market_conditions(self, market_data: Dict) -> Dict:
        """
        Adapt strategy based on current market conditions.

        Args:
            market_data: Dictionary with market indicators

        Returns:
            Adaptation recommendations
        """
        adaptations = {}

        # Volatility adaptation
        volatility = market_data.get("volatility")
        if volatility:
            if volatility > 0.03:  # High volatility (>3% daily)
                adaptations["position_size_multiplier"] = 0.7
                adaptations["reason_volatility"] = "Reduce position size in high volatility"
            elif volatility < 0.01:  # Low volatility
                adaptations["position_size_multiplier"] = 1.2
                adaptations["reason_volatility"] = "Can increase position size in low volatility"

        # Trend adaptation
        trend = market_data.get("trend")
        if trend:
            if trend == "strong_up":
                adaptations["buy_bias"] = 0.1
                adaptations["reason_trend"] = "Add bullish bias in strong uptrend"
            elif trend == "strong_down":
                adaptations["sell_bias"] = 0.1
                adaptations["reason_trend"] = "Add bearish bias in strong downtrend"

        # Sentiment adaptation
        sentiment = market_data.get("sentiment_score")
        if sentiment:
            if sentiment > 0.3:
                adaptations["sentiment_multiplier"] = 1.1
            elif sentiment < -0.3:
                adaptations["sentiment_multiplier"] = 1.1  # Also useful for shorting

        return adaptations

    def get_optimization_status(self) -> Dict:
        """Get current optimization status and history."""
        return {
            "current_weights": self.current_weights,
            "optimization_count": len(self.optimization_history),
            "last_optimization": self.optimization_history[-1] if self.optimization_history else None,
            "improvement_suggestions": self.analyzer.get_improvement_suggestions()
        }

    def reset_to_defaults(self):
        """Reset weights to default values."""
        self.current_weights = config.DEFAULT_MODEL_WEIGHTS.copy()
        self.storage.save_strategy_weights(
            weights=self.current_weights,
            reason="Manual reset to defaults"
        )
        logger.info("Reset weights to defaults")

    def get_weights(self) -> Dict[str, float]:
        """Get current model weights."""
        return self.current_weights.copy()

    def set_weights(self, weights: Dict[str, float], reason: str = "Manual adjustment"):
        """
        Manually set model weights.

        Args:
            weights: New weights dictionary
            reason: Reason for the change
        """
        # Normalize weights
        total = sum(weights.values())
        self.current_weights = {k: v / total for k, v in weights.items()}

        self.storage.save_strategy_weights(
            weights=self.current_weights,
            reason=reason
        )

        logger.info(f"Manually set weights: {self.current_weights}")

    def run_optimization_cycle(self) -> Dict:
        """
        Run a complete optimization cycle.

        Returns:
            Complete optimization results
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "steps": []
        }

        # Step 1: Analyze predictions
        pred_analysis = self.analyzer.analyze_predictions()
        results["prediction_analysis"] = pred_analysis
        results["steps"].append("Analyzed prediction performance")

        # Step 2: Analyze model performance
        model_stats = self.analyze_model_performance()
        results["model_stats"] = model_stats
        results["steps"].append("Analyzed individual model performance")

        # Step 3: Optimize weights
        optimization = self.optimize_weights()
        results["weight_optimization"] = optimization
        results["steps"].append("Optimized model weights")

        # Step 4: Generate suggestions
        suggestions = self.suggest_parameter_adjustments()
        results["parameter_suggestions"] = suggestions
        results["steps"].append("Generated parameter adjustment suggestions")

        # Step 5: Should we retrain?
        should_retrain, retrain_reason = self.analyzer.should_retrain_models()
        results["should_retrain"] = should_retrain
        results["retrain_reason"] = retrain_reason
        results["steps"].append("Evaluated need for model retraining")

        return results

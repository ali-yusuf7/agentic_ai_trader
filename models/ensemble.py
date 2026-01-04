"""
Ensemble Model - Combines predictions from multiple models
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from models.predictor import StockPredictor

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """Combines predictions from multiple ML models with adaptive weighting."""

    def __init__(self):
        self.predictor = StockPredictor()
        self.model_weights = config.DEFAULT_MODEL_WEIGHTS.copy()
        self.model_performance_history: Dict[str, List[float]] = {
            name: [] for name in self.model_weights
        }
        self.prediction_history: List[Dict] = []

    def train_all_models(self, df: pd.DataFrame) -> Dict:
        """
        Train all models in the ensemble.

        Args:
            df: DataFrame with features and target

        Returns:
            Training metrics for all models
        """
        # Create target variable
        df = self.predictor.create_target(df)

        # Train all models
        metrics = self.predictor.train(df)

        return metrics

    def predict(self, df: pd.DataFrame) -> Dict:
        """
        Get ensemble prediction combining all models.

        Args:
            df: DataFrame with features

        Returns:
            Dictionary with ensemble prediction and individual model predictions
        """
        individual_predictions = {}
        weighted_score = 0
        total_weight = 0

        for model_name, weight in self.model_weights.items():
            prediction = self.predictor.predict_latest(df, model_name)

            if "error" not in prediction:
                individual_predictions[model_name] = prediction

                # Weighted voting
                confidence = prediction.get("confidence", 0.5)
                pred_value = prediction["prediction_value"]

                # Score: +1 for up prediction, -1 for down prediction, scaled by confidence
                score = (pred_value * 2 - 1) * confidence
                weighted_score += score * weight
                total_weight += weight

        if total_weight == 0:
            return {"error": "No models could make predictions"}

        # Normalize weighted score to [-1, 1]
        ensemble_score = weighted_score / total_weight

        # Convert to prediction and confidence
        if ensemble_score > 0:
            prediction = "up"
            confidence = abs(ensemble_score)
        else:
            prediction = "down"
            confidence = abs(ensemble_score)

        # Determine trading signal based on thresholds
        if ensemble_score > config.BUY_THRESHOLD - 0.5:
            signal = "buy"
        elif ensemble_score < config.SELL_THRESHOLD - 0.5:
            signal = "sell"
        else:
            signal = "hold"

        result = {
            "ensemble_prediction": prediction,
            "ensemble_score": ensemble_score,
            "confidence": confidence,
            "signal": signal,
            "individual_predictions": individual_predictions,
            "model_weights": self.model_weights.copy()
        }

        # Record prediction for later analysis
        self.prediction_history.append({
            "timestamp": pd.Timestamp.now(),
            "result": result
        })

        return result

    def update_weights(self, performance_metrics: Dict[str, Dict]):
        """
        Update model weights based on recent performance.

        Args:
            performance_metrics: Dictionary of model performance metrics
        """
        if not performance_metrics:
            return

        # Calculate new weights based on accuracy and F1 score
        scores = {}
        for model_name, metrics in performance_metrics.items():
            if model_name in self.model_weights:
                accuracy = metrics.get("accuracy", 0.5)
                f1 = metrics.get("f1", 0.5)

                # Combined score: weighted average of accuracy and F1
                combined_score = accuracy * 0.4 + f1 * 0.6
                scores[model_name] = max(combined_score, 0.1)  # Minimum score to keep models active

                # Track performance history
                self.model_performance_history[model_name].append(combined_score)

        # Normalize to get new weights
        total_score = sum(scores.values())
        if total_score > 0:
            new_weights = {name: score / total_score for name, score in scores.items()}

            # Smooth weight updates (don't change too drastically)
            for name in self.model_weights:
                if name in new_weights:
                    old_weight = self.model_weights[name]
                    target_weight = new_weights[name]
                    # Move 30% toward new weight
                    self.model_weights[name] = old_weight * 0.7 + target_weight * 0.3

            logger.info(f"Updated model weights: {self.model_weights}")

    def get_model_agreement(self, predictions: Dict) -> Dict:
        """
        Analyze agreement between models.

        Args:
            predictions: Dictionary of individual model predictions

        Returns:
            Agreement analysis
        """
        individual = predictions.get("individual_predictions", {})

        if len(individual) < 2:
            return {"agreement": 1.0, "status": "insufficient_models"}

        pred_values = [p["prediction_value"] for p in individual.values()]

        # Calculate agreement ratio
        up_count = sum(pred_values)
        down_count = len(pred_values) - up_count

        agreement = max(up_count, down_count) / len(pred_values)

        if agreement == 1.0:
            status = "unanimous"
        elif agreement >= 0.67:
            status = "majority"
        else:
            status = "split"

        return {
            "agreement": agreement,
            "status": status,
            "up_votes": up_count,
            "down_votes": down_count,
            "total_models": len(pred_values)
        }

    def get_confidence_analysis(self, predictions: Dict) -> Dict:
        """
        Analyze confidence levels across models.

        Args:
            predictions: Dictionary of predictions

        Returns:
            Confidence analysis
        """
        individual = predictions.get("individual_predictions", {})

        if not individual:
            return {}

        confidences = [p.get("confidence", 0.5) for p in individual.values()]

        return {
            "mean_confidence": np.mean(confidences),
            "min_confidence": np.min(confidences),
            "max_confidence": np.max(confidences),
            "confidence_std": np.std(confidences),
            "high_confidence_count": sum(1 for c in confidences if c > 0.7),
            "low_confidence_count": sum(1 for c in confidences if c < 0.4)
        }

    def should_trade(self, predictions: Dict, min_confidence: float = 0.55) -> Tuple[bool, str]:
        """
        Determine if conditions are favorable for trading.

        Args:
            predictions: Ensemble prediction result
            min_confidence: Minimum confidence threshold

        Returns:
            Tuple of (should_trade, reason)
        """
        if "error" in predictions:
            return False, "Prediction error"

        confidence = predictions.get("confidence", 0)
        signal = predictions.get("signal", "hold")
        agreement = self.get_model_agreement(predictions)

        # Don't trade on hold signal
        if signal == "hold":
            return False, "Signal is hold"

        # Check confidence threshold
        if confidence < min_confidence:
            return False, f"Low confidence: {confidence:.2f} < {min_confidence}"

        # Check model agreement
        if agreement["status"] == "split":
            return False, "Models are split on prediction"

        return True, f"Good conditions: {signal} with {confidence:.2f} confidence"

    def get_weights(self) -> Dict[str, float]:
        """Get current model weights."""
        return self.model_weights.copy()

    def set_weights(self, weights: Dict[str, float]):
        """Set model weights manually."""
        for name, weight in weights.items():
            if name in self.model_weights:
                self.model_weights[name] = weight

        # Normalize weights
        total = sum(self.model_weights.values())
        if total > 0:
            self.model_weights = {k: v / total for k, v in self.model_weights.items()}

    def save_models(self, symbol: str):
        """Save all models."""
        self.predictor.save_models(symbol)

    def load_models(self, symbol: str) -> bool:
        """Load all models."""
        return self.predictor.load_models(symbol)

    def get_feature_importance_summary(self, top_n: int = 5) -> Dict[str, List]:
        """
        Get top features across all models.

        Args:
            top_n: Number of top features per model

        Returns:
            Dictionary of model feature importances
        """
        summary = {}
        for model_name in self.model_weights:
            importance = self.predictor.get_feature_importance(model_name, top_n)
            summary[model_name] = importance
        return summary

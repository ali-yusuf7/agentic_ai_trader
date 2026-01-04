"""
Predictor - ML models for stock price prediction
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import config

logger = logging.getLogger(__name__)


class StockPredictor:
    """ML models for predicting stock price movements."""

    def __init__(self):
        self.models: Dict[str, object] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_importance: Dict[str, Dict[str, float]] = {}
        self.model_metrics: Dict[str, Dict] = {}
        self.models_dir = config.MODELS_DIR

        self._init_models()

    def _init_models(self):
        """Initialize ML models."""
        self.models = {
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            "xgboost": XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss"
            ),
            "linear_regression": LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        }

        # Initialize scalers for each model
        for name in self.models:
            self.scalers[name] = StandardScaler()

    def prepare_features(
        self,
        df: pd.DataFrame,
        target_col: str = "target",
        feature_cols: List[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features and target for training.

        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            feature_cols: List of feature columns (if None, auto-detect)

        Returns:
            Tuple of (X, y, feature_names)
        """
        # Default feature columns (exclude non-feature columns)
        exclude_cols = ["date", "open", "high", "low", "close", "volume",
                       "dividends", "stock_splits", target_col]

        if feature_cols is None:
            feature_cols = [c for c in df.columns
                          if c not in exclude_cols
                          and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]

        # Remove rows with NaN
        df_clean = df[feature_cols + [target_col]].dropna()

        if len(df_clean) < config.MIN_DATA_POINTS:
            logger.warning(f"Not enough data points: {len(df_clean)} < {config.MIN_DATA_POINTS}")
            return None, None, feature_cols

        X = df_clean[feature_cols].values
        y = df_clean[target_col].values

        return X, y, feature_cols

    def create_target(
        self,
        df: pd.DataFrame,
        horizon: int = config.PREDICTION_HORIZON,
        threshold: float = 0.0
    ) -> pd.DataFrame:
        """
        Create target variable for prediction.

        Target: 1 if price goes up by more than threshold, 0 otherwise

        Args:
            df: DataFrame with close prices
            horizon: Number of days ahead to predict
            threshold: Minimum return to classify as 'up'

        Returns:
            DataFrame with target column added
        """
        df = df.copy()

        # Future return
        df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1

        # Binary target: 1 = price goes up, 0 = price goes down
        df["target"] = (df["future_return"] > threshold).astype(int)

        # For regression: actual future return
        df["target_return"] = df["future_return"]

        return df

    def train(
        self,
        df: pd.DataFrame,
        model_name: str = None,
        feature_cols: List[str] = None
    ) -> Dict[str, Dict]:
        """
        Train prediction models.

        Args:
            df: DataFrame with features and target
            model_name: Specific model to train (if None, train all)
            feature_cols: Feature columns to use

        Returns:
            Dictionary of training metrics for each model
        """
        X, y, features = self.prepare_features(df, feature_cols=feature_cols)

        if X is None:
            logger.error("Could not prepare features for training")
            return {}

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1 - config.TRAIN_TEST_SPLIT,
            shuffle=False  # Time series: don't shuffle
        )

        models_to_train = [model_name] if model_name else list(self.models.keys())
        results = {}

        for name in models_to_train:
            if name not in self.models:
                logger.warning(f"Unknown model: {name}")
                continue

            logger.info(f"Training {name}...")

            try:
                model = self.models[name]
                scaler = self.scalers[name]

                # Scale features
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Train
                model.fit(X_train_scaled, y_train)

                # Evaluate
                y_pred = model.predict(X_test_scaled)

                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, zero_division=0),
                    "recall": recall_score(y_test, y_pred, zero_division=0),
                    "f1": f1_score(y_test, y_pred, zero_division=0),
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "trained_at": datetime.now().isoformat()
                }

                self.model_metrics[name] = metrics
                results[name] = metrics

                # Store feature importance
                if hasattr(model, "feature_importances_"):
                    importance = dict(zip(features, model.feature_importances_))
                    self.feature_importance[name] = importance
                elif hasattr(model, "coef_"):
                    importance = dict(zip(features, np.abs(model.coef_[0])))
                    self.feature_importance[name] = importance

                logger.info(f"{name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                results[name] = {"error": str(e)}

        return results

    def predict(
        self,
        df: pd.DataFrame,
        model_name: str = "random_forest",
        feature_cols: List[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using a trained model.

        Args:
            df: DataFrame with features
            model_name: Model to use for prediction
            feature_cols: Feature columns to use

        Returns:
            Tuple of (predictions, probabilities)
        """
        if model_name not in self.models:
            logger.error(f"Unknown model: {model_name}")
            return None, None

        # Prepare features
        exclude_cols = ["date", "open", "high", "low", "close", "volume",
                       "dividends", "stock_splits", "target", "target_return", "future_return"]

        if feature_cols is None:
            feature_cols = [c for c in df.columns
                          if c not in exclude_cols
                          and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]

        # Get last row with complete data
        df_clean = df[feature_cols].dropna()

        if df_clean.empty:
            logger.warning("No valid data for prediction")
            return None, None

        X = df_clean.values
        model = self.models[model_name]
        scaler = self.scalers[model_name]

        try:
            X_scaled = scaler.transform(X)
            predictions = model.predict(X_scaled)

            # Get probabilities if available
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(X_scaled)
            else:
                probabilities = None

            return predictions, probabilities

        except Exception as e:
            logger.error(f"Error making prediction with {model_name}: {e}")
            return None, None

    def predict_latest(
        self,
        df: pd.DataFrame,
        model_name: str = "random_forest"
    ) -> Dict:
        """
        Get prediction for the latest data point.

        Args:
            df: DataFrame with features
            model_name: Model to use

        Returns:
            Dictionary with prediction details
        """
        predictions, probabilities = self.predict(df, model_name)

        if predictions is None:
            return {"error": "Could not make prediction"}

        # Get latest prediction
        latest_pred = int(predictions[-1])
        latest_prob = probabilities[-1] if probabilities is not None else None

        result = {
            "prediction": "up" if latest_pred == 1 else "down",
            "prediction_value": latest_pred,
            "model": model_name
        }

        if latest_prob is not None:
            result["confidence"] = float(max(latest_prob))
            result["probability_up"] = float(latest_prob[1]) if len(latest_prob) > 1 else None
            result["probability_down"] = float(latest_prob[0]) if len(latest_prob) > 1 else None

        return result

    def get_feature_importance(self, model_name: str = "random_forest", top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top feature importances for a model.

        Args:
            model_name: Model name
            top_n: Number of top features to return

        Returns:
            List of (feature_name, importance) tuples
        """
        if model_name not in self.feature_importance:
            return []

        importance = self.feature_importance[model_name]
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        return sorted_importance[:top_n]

    def save_models(self, symbol: str = "default"):
        """Save trained models to disk."""
        save_dir = self.models_dir / symbol
        save_dir.mkdir(parents=True, exist_ok=True)

        for name, model in self.models.items():
            model_path = save_dir / f"{name}_model.joblib"
            scaler_path = save_dir / f"{name}_scaler.joblib"

            joblib.dump(model, model_path)
            joblib.dump(self.scalers[name], scaler_path)

        logger.info(f"Models saved to {save_dir}")

    def load_models(self, symbol: str = "default") -> bool:
        """Load trained models from disk."""
        load_dir = self.models_dir / symbol

        if not load_dir.exists():
            logger.warning(f"No saved models found at {load_dir}")
            return False

        for name in self.models:
            model_path = load_dir / f"{name}_model.joblib"
            scaler_path = load_dir / f"{name}_scaler.joblib"

            if model_path.exists() and scaler_path.exists():
                self.models[name] = joblib.load(model_path)
                self.scalers[name] = joblib.load(scaler_path)
                logger.info(f"Loaded {name} model")
            else:
                logger.warning(f"Could not find saved {name} model")

        return True

    def get_model_metrics(self, model_name: str = None) -> Dict:
        """Get metrics for trained models."""
        if model_name:
            return self.model_metrics.get(model_name, {})
        return self.model_metrics

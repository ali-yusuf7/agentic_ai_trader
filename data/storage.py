"""
Data Storage - SQLite persistence layer for trades, predictions, and performance data
"""
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

import config

logger = logging.getLogger(__name__)


class DataStorage:
    """SQLite-based storage for trading data and model performance."""

    def __init__(self, db_path: Path = config.DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_database(self):
        """Initialize database tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    total_value REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    reasoning TEXT,
                    confidence REAL
                )
            """)

            # Predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    prediction_date DATETIME NOT NULL,
                    target_date DATETIME NOT NULL,
                    predicted_direction TEXT NOT NULL,
                    predicted_change REAL,
                    confidence REAL,
                    actual_direction TEXT,
                    actual_change REAL,
                    model_name TEXT,
                    features_used TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Portfolio history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    total_value REAL NOT NULL,
                    cash REAL NOT NULL,
                    positions TEXT NOT NULL
                )
            """)

            # Model performance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    symbol TEXT,
                    accuracy REAL,
                    precision_score REAL,
                    recall_score REAL,
                    f1_score REAL,
                    sharpe_ratio REAL,
                    total_predictions INTEGER,
                    correct_predictions INTEGER,
                    evaluation_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                    parameters TEXT
                )
            """)

            # Strategy weights table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_weights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    weights TEXT NOT NULL,
                    reason TEXT,
                    performance_metrics TEXT,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Agent decisions log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    decision_type TEXT NOT NULL,
                    symbol TEXT,
                    decision TEXT NOT NULL,
                    reasoning TEXT,
                    market_conditions TEXT,
                    model_outputs TEXT
                )
            """)

            conn.commit()
            logger.info("Database initialized successfully")

    # Trade operations
    def record_trade(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        reasoning: str = None,
        confidence: float = None
    ) -> int:
        """Record a trade."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trades (symbol, action, quantity, price, total_value, reasoning, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (symbol, action, quantity, price, quantity * price, reasoning, confidence))
            conn.commit()
            return cursor.lastrowid

    def get_trades(
        self,
        symbol: str = None,
        days: int = None,
        limit: int = 100
    ) -> List[dict]:
        """Get trade history."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM trades WHERE 1=1"
            params = []

            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            if days:
                query += " AND timestamp >= datetime('now', ?)"
                params.append(f"-{days} days")

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    # Prediction operations
    def record_prediction(
        self,
        symbol: str,
        prediction_date: datetime,
        target_date: datetime,
        predicted_direction: str,
        predicted_change: float = None,
        confidence: float = None,
        model_name: str = None,
        features_used: List[str] = None
    ) -> int:
        """Record a prediction."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions
                (symbol, prediction_date, target_date, predicted_direction,
                 predicted_change, confidence, model_name, features_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, prediction_date, target_date, predicted_direction,
                predicted_change, confidence, model_name,
                json.dumps(features_used) if features_used else None
            ))
            conn.commit()
            return cursor.lastrowid

    def update_prediction_actual(
        self,
        prediction_id: int,
        actual_direction: str,
        actual_change: float
    ):
        """Update a prediction with actual results."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE predictions
                SET actual_direction = ?, actual_change = ?
                WHERE id = ?
            """, (actual_direction, actual_change, prediction_id))
            conn.commit()

    def get_pending_predictions(self) -> List[dict]:
        """Get predictions that need actual results updated."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM predictions
                WHERE actual_direction IS NULL
                AND target_date <= datetime('now')
                ORDER BY target_date
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_predictions_for_analysis(
        self,
        model_name: str = None,
        days: int = 30
    ) -> pd.DataFrame:
        """Get predictions with actuals for analysis."""
        with self._get_connection() as conn:
            query = """
                SELECT * FROM predictions
                WHERE actual_direction IS NOT NULL
                AND created_at >= datetime('now', ?)
            """
            params = [f"-{days} days"]

            if model_name:
                query += " AND model_name = ?"
                params.append(model_name)

            return pd.read_sql_query(query, conn, params=params)

    # Portfolio operations
    def record_portfolio_snapshot(
        self,
        total_value: float,
        cash: float,
        positions: Dict[str, dict]
    ):
        """Record portfolio state."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO portfolio_history (total_value, cash, positions)
                VALUES (?, ?, ?)
            """, (total_value, cash, json.dumps(positions)))
            conn.commit()

    def get_portfolio_history(self, days: int = 30) -> pd.DataFrame:
        """Get portfolio history."""
        with self._get_connection() as conn:
            query = """
                SELECT * FROM portfolio_history
                WHERE timestamp >= datetime('now', ?)
                ORDER BY timestamp
            """
            return pd.read_sql_query(query, conn, params=[f"-{days} days"])

    # Model performance operations
    def record_model_performance(
        self,
        model_name: str,
        accuracy: float,
        precision_score: float = None,
        recall_score: float = None,
        f1_score: float = None,
        sharpe_ratio: float = None,
        total_predictions: int = None,
        correct_predictions: int = None,
        symbol: str = None,
        parameters: dict = None
    ):
        """Record model performance metrics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO model_performance
                (model_name, symbol, accuracy, precision_score, recall_score,
                 f1_score, sharpe_ratio, total_predictions, correct_predictions, parameters)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_name, symbol, accuracy, precision_score, recall_score,
                f1_score, sharpe_ratio, total_predictions, correct_predictions,
                json.dumps(parameters) if parameters else None
            ))
            conn.commit()

    def get_model_performance(self, model_name: str = None, days: int = 30) -> pd.DataFrame:
        """Get model performance history."""
        with self._get_connection() as conn:
            query = """
                SELECT * FROM model_performance
                WHERE evaluation_date >= datetime('now', ?)
            """
            params = [f"-{days} days"]

            if model_name:
                query += " AND model_name = ?"
                params.append(model_name)

            query += " ORDER BY evaluation_date DESC"
            return pd.read_sql_query(query, conn, params=params)

    # Strategy weights operations
    def save_strategy_weights(
        self,
        weights: Dict[str, float],
        reason: str = None,
        performance_metrics: dict = None
    ):
        """Save updated strategy weights."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO strategy_weights (weights, reason, performance_metrics)
                VALUES (?, ?, ?)
            """, (
                json.dumps(weights),
                reason,
                json.dumps(performance_metrics) if performance_metrics else None
            ))
            conn.commit()

    def get_latest_weights(self) -> Optional[Dict[str, float]]:
        """Get the most recent strategy weights."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT weights FROM strategy_weights
                ORDER BY updated_at DESC LIMIT 1
            """)
            row = cursor.fetchone()
            if row:
                return json.loads(row["weights"])
            return None

    # Agent decision logging
    def log_decision(
        self,
        decision_type: str,
        decision: str,
        symbol: str = None,
        reasoning: str = None,
        market_conditions: dict = None,
        model_outputs: dict = None
    ):
        """Log an agent decision."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO agent_decisions
                (decision_type, symbol, decision, reasoning, market_conditions, model_outputs)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                decision_type, symbol, decision, reasoning,
                json.dumps(market_conditions) if market_conditions else None,
                json.dumps(model_outputs) if model_outputs else None
            ))
            conn.commit()

    def get_decision_log(self, days: int = 7, limit: int = 100) -> List[dict]:
        """Get recent agent decisions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM agent_decisions
                WHERE timestamp >= datetime('now', ?)
                ORDER BY timestamp DESC LIMIT ?
            """, (f"-{days} days", limit))
            return [dict(row) for row in cursor.fetchall()]

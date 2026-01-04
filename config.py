"""
Configuration settings for the Agentic AI Trader
"""
import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "db" / "trades.db"
MODELS_DIR = BASE_DIR / "saved_models"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Stock symbols to track
STOCKS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

# Trading parameters
INITIAL_CAPITAL = 100000.0  # Starting paper trading capital
RISK_PER_TRADE = 0.02  # Maximum 2% of portfolio per trade
MAX_POSITION_SIZE = 0.20  # Maximum 20% of portfolio in single stock
TRADING_INTERVAL_MINUTES = 60  # Check for opportunities every hour

# Data parameters
HISTORICAL_DAYS = 365  # Days of historical data to fetch
MIN_DATA_POINTS = 100  # Minimum data points needed for training

# Technical indicator parameters
SMA_SHORT = 20
SMA_LONG = 50
EMA_SHORT = 12
EMA_LONG = 26
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

# Model parameters
TRAIN_TEST_SPLIT = 0.8
PREDICTION_HORIZON = 5  # Predict price movement N days ahead
MODEL_RETRAIN_DAYS = 7  # Retrain models every N days

# Ensemble weights (will be adjusted by strategy optimizer)
DEFAULT_MODEL_WEIGHTS = {
    "random_forest": 0.4,
    "xgboost": 0.4,
    "linear_regression": 0.2
}

# Trading signals thresholds
BUY_THRESHOLD = 0.6  # Confidence threshold for buy signal
SELL_THRESHOLD = 0.4  # Confidence threshold for sell signal

# News RSS feeds for sentiment analysis
NEWS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US",
    "https://www.investing.com/rss/news.rss",
]

# Performance tracking
PERFORMANCE_WINDOW = 30  # Days to evaluate recent performance
MIN_TRADES_FOR_EVALUATION = 10  # Minimum trades before strategy adjustment

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

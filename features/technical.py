"""
Technical Indicators - Calculate standard quant indicators for trading signals
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calculate technical indicators for stock data."""

    def __init__(self):
        self.sma_short = config.SMA_SHORT
        self.sma_long = config.SMA_LONG
        self.ema_short = config.EMA_SHORT
        self.ema_long = config.EMA_LONG
        self.rsi_period = config.RSI_PERIOD
        self.macd_fast = config.MACD_FAST
        self.macd_slow = config.MACD_SLOW
        self.macd_signal = config.MACD_SIGNAL
        self.bb_period = config.BOLLINGER_PERIOD
        self.bb_std = config.BOLLINGER_STD

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators and add them to the dataframe.

        Args:
            df: DataFrame with OHLCV columns (open, high, low, close, volume)

        Returns:
            DataFrame with all technical indicators added
        """
        df = df.copy()

        # Ensure we have the required columns
        required = ["close", "high", "low", "volume"]
        for col in required:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                return df

        # Moving Averages
        df = self.add_sma(df)
        df = self.add_ema(df)

        # Momentum Indicators
        df = self.add_rsi(df)
        df = self.add_macd(df)

        # Volatility Indicators
        df = self.add_bollinger_bands(df)
        df = self.add_atr(df)

        # Volume Indicators
        df = self.add_volume_indicators(df)

        # Price-based features
        df = self.add_price_features(df)

        # Trend features
        df = self.add_trend_features(df)

        return df

    def add_sma(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Simple Moving Averages."""
        df["sma_short"] = df["close"].rolling(window=self.sma_short).mean()
        df["sma_long"] = df["close"].rolling(window=self.sma_long).mean()
        df["sma_crossover"] = (df["sma_short"] > df["sma_long"]).astype(int)
        df["price_above_sma_short"] = (df["close"] > df["sma_short"]).astype(int)
        df["price_above_sma_long"] = (df["close"] > df["sma_long"]).astype(int)
        return df

    def add_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Exponential Moving Averages."""
        df["ema_short"] = df["close"].ewm(span=self.ema_short, adjust=False).mean()
        df["ema_long"] = df["close"].ewm(span=self.ema_long, adjust=False).mean()
        df["ema_crossover"] = (df["ema_short"] > df["ema_long"]).astype(int)
        return df

    def add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Relative Strength Index."""
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()

        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # RSI zones
        df["rsi_oversold"] = (df["rsi"] < 30).astype(int)
        df["rsi_overbought"] = (df["rsi"] > 70).astype(int)

        return df

    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD (Moving Average Convergence Divergence)."""
        ema_fast = df["close"].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=self.macd_slow, adjust=False).mean()

        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=self.macd_signal, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]
        df["macd_crossover"] = (df["macd"] > df["macd_signal"]).astype(int)

        return df

    def add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Bands."""
        sma = df["close"].rolling(window=self.bb_period).mean()
        std = df["close"].rolling(window=self.bb_period).std()

        df["bb_middle"] = sma
        df["bb_upper"] = sma + (std * self.bb_std)
        df["bb_lower"] = sma - (std * self.bb_std)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

        # Position within bands (0 to 1)
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        return df

    def add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average True Range."""
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=period).mean()
        df["atr_percent"] = df["atr"] / df["close"] * 100

        return df

    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        # Volume moving average
        df["volume_sma"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"]

        # On-Balance Volume (OBV)
        df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).cumsum()

        # Volume Price Trend
        df["vpt"] = (df["close"].pct_change() * df["volume"]).cumsum()

        return df

    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        # Returns
        df["returns_1d"] = df["close"].pct_change()
        df["returns_5d"] = df["close"].pct_change(5)
        df["returns_10d"] = df["close"].pct_change(10)
        df["returns_20d"] = df["close"].pct_change(20)

        # Volatility
        df["volatility_10d"] = df["returns_1d"].rolling(window=10).std()
        df["volatility_20d"] = df["returns_1d"].rolling(window=20).std()

        # High/Low range
        df["daily_range"] = (df["high"] - df["low"]) / df["close"]

        # Distance from 52-week high/low
        df["high_52w"] = df["high"].rolling(window=252).max()
        df["low_52w"] = df["low"].rolling(window=252).min()
        df["dist_from_high"] = (df["close"] - df["high_52w"]) / df["high_52w"]
        df["dist_from_low"] = (df["close"] - df["low_52w"]) / df["low_52w"]

        return df

    def add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-related features."""
        # Price momentum
        df["momentum_10"] = df["close"] - df["close"].shift(10)
        df["momentum_20"] = df["close"] - df["close"].shift(20)

        # Rate of change
        df["roc_10"] = ((df["close"] - df["close"].shift(10)) / df["close"].shift(10)) * 100
        df["roc_20"] = ((df["close"] - df["close"].shift(20)) / df["close"].shift(20)) * 100

        # Trend strength (ADX approximation)
        df["trend_strength"] = abs(df["sma_short"] - df["sma_long"]) / df["close"] * 100

        return df

    def get_feature_names(self) -> list:
        """Get list of all feature names generated by this class."""
        return [
            # Moving averages
            "sma_short", "sma_long", "sma_crossover",
            "price_above_sma_short", "price_above_sma_long",
            "ema_short", "ema_long", "ema_crossover",
            # RSI
            "rsi", "rsi_oversold", "rsi_overbought",
            # MACD
            "macd", "macd_signal", "macd_histogram", "macd_crossover",
            # Bollinger Bands
            "bb_middle", "bb_upper", "bb_lower", "bb_width", "bb_position",
            # ATR
            "atr", "atr_percent",
            # Volume
            "volume_sma", "volume_ratio", "obv", "vpt",
            # Price features
            "returns_1d", "returns_5d", "returns_10d", "returns_20d",
            "volatility_10d", "volatility_20d", "daily_range",
            "high_52w", "low_52w", "dist_from_high", "dist_from_low",
            # Trend
            "momentum_10", "momentum_20", "roc_10", "roc_20", "trend_strength"
        ]

    def get_signal_summary(self, df: pd.DataFrame) -> dict:
        """
        Get a summary of technical signals from the latest data point.

        Args:
            df: DataFrame with technical indicators

        Returns:
            Dictionary with signal summary
        """
        if df.empty:
            return {}

        latest = df.iloc[-1]

        signals = {
            "trend": {
                "sma_crossover": "bullish" if latest.get("sma_crossover", 0) == 1 else "bearish",
                "ema_crossover": "bullish" if latest.get("ema_crossover", 0) == 1 else "bearish",
                "price_vs_sma": "above" if latest.get("price_above_sma_long", 0) == 1 else "below"
            },
            "momentum": {
                "rsi": latest.get("rsi", 50),
                "rsi_signal": "oversold" if latest.get("rsi_oversold", 0) == 1 else (
                    "overbought" if latest.get("rsi_overbought", 0) == 1 else "neutral"
                ),
                "macd_crossover": "bullish" if latest.get("macd_crossover", 0) == 1 else "bearish"
            },
            "volatility": {
                "bb_position": latest.get("bb_position", 0.5),
                "atr_percent": latest.get("atr_percent", 0)
            },
            "volume": {
                "volume_ratio": latest.get("volume_ratio", 1),
                "volume_signal": "high" if latest.get("volume_ratio", 1) > 1.5 else (
                    "low" if latest.get("volume_ratio", 1) < 0.5 else "normal"
                )
            }
        }

        # Calculate overall signal score (-1 to 1)
        bullish_signals = 0
        total_signals = 0

        if latest.get("sma_crossover") == 1:
            bullish_signals += 1
        total_signals += 1

        if latest.get("ema_crossover") == 1:
            bullish_signals += 1
        total_signals += 1

        if latest.get("macd_crossover") == 1:
            bullish_signals += 1
        total_signals += 1

        rsi = latest.get("rsi", 50)
        if rsi < 30:
            bullish_signals += 1  # Oversold = potential buy
        elif rsi > 70:
            bullish_signals -= 0.5  # Overbought = potential sell
        total_signals += 1

        signals["overall_score"] = (bullish_signals / total_signals) * 2 - 1

        return signals

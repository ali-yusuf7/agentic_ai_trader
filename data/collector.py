"""
Data Collector - Fetches stock prices and news data
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import feedparser
import pandas as pd
import yfinance as yf

import config

logger = logging.getLogger(__name__)


class DataCollector:
    """Collects stock price data and news from various sources."""

    def __init__(self):
        self.price_cache: Dict[str, pd.DataFrame] = {}
        self.news_cache: Dict[str, List[dict]] = {}
        self.cache_expiry: Dict[str, datetime] = {}
        self.cache_duration = timedelta(minutes=15)

    def get_stock_data(
        self,
        symbol: str,
        days: int = config.HISTORICAL_DAYS,
        force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical stock data from Yahoo Finance.

        Args:
            symbol: Stock ticker symbol
            days: Number of days of historical data
            force_refresh: Force refresh even if cached

        Returns:
            DataFrame with OHLCV data or None if fetch fails
        """
        cache_key = f"{symbol}_{days}"

        # Check cache
        if not force_refresh and cache_key in self.price_cache:
            if datetime.now() < self.cache_expiry.get(cache_key, datetime.min):
                logger.debug(f"Using cached data for {symbol}")
                return self.price_cache[cache_key]

        try:
            logger.info(f"Fetching {days} days of data for {symbol}")
            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            df = ticker.history(start=start_date, end=end_date)

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return None

            # Clean column names
            df.columns = [col.lower().replace(" ", "_") for col in df.columns]
            df.index.name = "date"
            df = df.reset_index()

            # Cache the result
            self.price_cache[cache_key] = df
            self.cache_expiry[cache_key] = datetime.now() + self.cache_duration

            logger.info(f"Fetched {len(df)} data points for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def get_multiple_stocks(
        self,
        symbols: List[str],
        days: int = config.HISTORICAL_DAYS
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks.

        Args:
            symbols: List of stock ticker symbols
            days: Number of days of historical data

        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        result = {}
        for symbol in symbols:
            data = self.get_stock_data(symbol, days)
            if data is not None:
                result[symbol] = data
        return result

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get the current/latest price for a stock.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Current price or None if unavailable
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            if not data.empty:
                return float(data["Close"].iloc[-1])
            return None
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None

    def get_stock_info(self, symbol: str) -> Optional[dict]:
        """
        Get company information for a stock.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with company info or None
        """
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info
        except Exception as e:
            logger.error(f"Error getting info for {symbol}: {e}")
            return None

    def get_news(
        self,
        symbol: str,
        max_articles: int = 10,
        force_refresh: bool = False
    ) -> List[dict]:
        """
        Fetch news articles for a stock from RSS feeds.

        Args:
            symbol: Stock ticker symbol
            max_articles: Maximum number of articles to return
            force_refresh: Force refresh even if cached

        Returns:
            List of news article dictionaries
        """
        cache_key = f"news_{symbol}"

        # Check cache
        if not force_refresh and cache_key in self.news_cache:
            if datetime.now() < self.cache_expiry.get(cache_key, datetime.min):
                logger.debug(f"Using cached news for {symbol}")
                return self.news_cache[cache_key][:max_articles]

        articles = []

        for feed_url in config.NEWS_FEEDS:
            try:
                url = feed_url.format(symbol=symbol)
                feed = feedparser.parse(url)

                for entry in feed.entries[:max_articles]:
                    article = {
                        "title": entry.get("title", ""),
                        "summary": entry.get("summary", ""),
                        "link": entry.get("link", ""),
                        "published": entry.get("published", ""),
                        "source": feed.feed.get("title", "Unknown"),
                        "symbol": symbol
                    }
                    articles.append(article)

            except Exception as e:
                logger.warning(f"Error fetching news from {feed_url}: {e}")
                continue

        # Cache the results
        self.news_cache[cache_key] = articles
        self.cache_expiry[cache_key] = datetime.now() + self.cache_duration

        logger.info(f"Fetched {len(articles)} news articles for {symbol}")
        return articles[:max_articles]

    def get_news_for_multiple_stocks(
        self,
        symbols: List[str],
        max_articles_per_stock: int = 5
    ) -> Dict[str, List[dict]]:
        """
        Fetch news for multiple stocks.

        Args:
            symbols: List of stock ticker symbols
            max_articles_per_stock: Max articles per stock

        Returns:
            Dictionary mapping symbols to their news articles
        """
        result = {}
        for symbol in symbols:
            result[symbol] = self.get_news(symbol, max_articles_per_stock)
        return result

    def clear_cache(self):
        """Clear all cached data."""
        self.price_cache.clear()
        self.news_cache.clear()
        self.cache_expiry.clear()
        logger.info("Cache cleared")

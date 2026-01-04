"""
Trading Agent - The main agentic AI that orchestrates all trading operations
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

import config
from analysis.performance import PerformanceAnalyzer
from analysis.strategy_optimizer import StrategyOptimizer
from data.collector import DataCollector
from data.storage import DataStorage
from features.sentiment import SentimentAnalyzer
from features.technical import TechnicalIndicators
from models.ensemble import EnsemblePredictor
from trading.paper_trader import PaperTrader
from trading.portfolio import PortfolioManager

logger = logging.getLogger(__name__)


class TradingAgent:
    """
    Agentic AI Trading System

    This agent autonomously:
    1. Collects market data and news
    2. Generates features (technical indicators + sentiment)
    3. Makes predictions using ML ensemble
    4. Executes paper trades based on signals
    5. Analyzes performance and improves strategies
    """

    def __init__(self, stocks: List[str] = None):
        self.stocks = stocks or config.STOCKS

        # Initialize components
        self.collector = DataCollector()
        self.storage = DataStorage()
        self.technical = TechnicalIndicators()
        self.sentiment = SentimentAnalyzer()
        self.ensemble = EnsemblePredictor()
        self.trader = PaperTrader()
        self.portfolio = PortfolioManager(self.trader)
        self.performance = PerformanceAnalyzer()
        self.optimizer = StrategyOptimizer()

        # Agent state
        self.is_initialized = False
        self.last_trade_time: Dict[str, datetime] = {}
        self.models_trained = False

        logger.info(f"Trading Agent initialized for stocks: {self.stocks}")

    def initialize(self) -> Dict:
        """
        Initialize the agent: fetch data, train models, etc.

        Returns:
            Initialization status
        """
        logger.info("Initializing Trading Agent...")
        results = {"steps": [], "errors": []}

        # Step 1: Fetch historical data for all stocks
        logger.info("Fetching historical data...")
        stock_data = {}
        for symbol in self.stocks:
            data = self.collector.get_stock_data(symbol, days=config.HISTORICAL_DAYS)
            if data is not None and len(data) >= config.MIN_DATA_POINTS:
                stock_data[symbol] = data
                results["steps"].append(f"Fetched {len(data)} days of data for {symbol}")
            else:
                results["errors"].append(f"Insufficient data for {symbol}")

        if not stock_data:
            results["success"] = False
            results["error"] = "Could not fetch data for any stocks"
            return results

        # Step 2: Generate features and train models for each stock
        logger.info("Training models...")
        for symbol, data in stock_data.items():
            try:
                # Add technical indicators
                data_with_features = self.technical.calculate_all(data)

                # Create target variable
                data_with_target = self.ensemble.predictor.create_target(data_with_features)

                # Train models
                metrics = self.ensemble.train_all_models(data_with_target)

                if metrics:
                    # Save models
                    self.ensemble.save_models(symbol)
                    results["steps"].append(f"Trained models for {symbol}: {list(metrics.keys())}")

                    # Record model performance
                    for model_name, model_metrics in metrics.items():
                        if "error" not in model_metrics:
                            self.storage.record_model_performance(
                                model_name=model_name,
                                symbol=symbol,
                                accuracy=model_metrics.get("accuracy"),
                                f1_score=model_metrics.get("f1")
                            )
            except Exception as e:
                results["errors"].append(f"Error training models for {symbol}: {e}")
                logger.error(f"Error training models for {symbol}: {e}")

        self.models_trained = True
        self.is_initialized = True
        results["success"] = True
        results["stocks_ready"] = list(stock_data.keys())

        logger.info("Trading Agent initialization complete")
        return results

    def analyze_stock(self, symbol: str) -> Dict:
        """
        Perform complete analysis on a single stock.

        Args:
            symbol: Stock symbol

        Returns:
            Complete analysis including prediction and sentiment
        """
        analysis = {"symbol": symbol, "timestamp": datetime.now().isoformat()}

        # Get latest data
        data = self.collector.get_stock_data(symbol, days=100)
        if data is None or data.empty:
            analysis["error"] = "Could not fetch data"
            return analysis

        # Current price
        current_price = float(data["close"].iloc[-1])
        analysis["current_price"] = current_price

        # Technical indicators
        data_with_features = self.technical.calculate_all(data)
        technical_signals = self.technical.get_signal_summary(data_with_features)
        analysis["technical_signals"] = technical_signals

        # News sentiment
        news = self.collector.get_news(symbol, max_articles=10)
        sentiment_features = self.sentiment.get_sentiment_features(news)
        analysis["sentiment"] = sentiment_features

        # Load trained models and make prediction
        self.ensemble.load_models(symbol)
        prediction = self.ensemble.predict(data_with_features)

        if "error" not in prediction:
            analysis["prediction"] = {
                "direction": prediction["ensemble_prediction"],
                "confidence": prediction["confidence"],
                "signal": prediction["signal"],
                "individual_models": prediction["individual_predictions"]
            }

            # Model agreement analysis
            agreement = self.ensemble.get_model_agreement(prediction)
            analysis["model_agreement"] = agreement

            # Should we trade?
            should_trade, trade_reason = self.ensemble.should_trade(prediction)
            analysis["should_trade"] = should_trade
            analysis["trade_reason"] = trade_reason
        else:
            analysis["prediction"] = {"error": prediction["error"]}

        # Current position
        position = self.trader.get_position(symbol)
        if position:
            pnl = self.trader.calculate_position_pnl(symbol, current_price)
            analysis["current_position"] = pnl
        else:
            analysis["current_position"] = None

        return analysis

    def make_trading_decision(self, analysis: Dict) -> Dict:
        """
        Make a trading decision based on analysis.

        Args:
            analysis: Stock analysis from analyze_stock()

        Returns:
            Trading decision with reasoning
        """
        symbol = analysis.get("symbol")
        decision = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "action": "hold",
            "reasoning": []
        }

        # Check if we should trade at all
        if not analysis.get("should_trade", False):
            decision["action"] = "hold"
            decision["reasoning"].append(analysis.get("trade_reason", "No trade signal"))
            return decision

        prediction = analysis.get("prediction", {})
        signal = prediction.get("signal", "hold")
        confidence = prediction.get("confidence", 0)
        current_price = analysis.get("current_price", 0)

        # Get technical and sentiment signals
        tech_signals = analysis.get("technical_signals", {})
        sentiment = analysis.get("sentiment", {})

        # Build reasoning
        reasoning = []

        # Technical analysis
        overall_tech_score = tech_signals.get("overall_score", 0)
        if overall_tech_score > 0.3:
            reasoning.append(f"Technical indicators bullish (score: {overall_tech_score:.2f})")
        elif overall_tech_score < -0.3:
            reasoning.append(f"Technical indicators bearish (score: {overall_tech_score:.2f})")
        else:
            reasoning.append(f"Technical indicators neutral (score: {overall_tech_score:.2f})")

        # Sentiment analysis
        sentiment_score = sentiment.get("sentiment_score", 0)
        sentiment_label = sentiment.get("sentiment_label", "neutral")
        reasoning.append(f"News sentiment: {sentiment_label} ({sentiment_score:.2f})")

        # ML prediction
        reasoning.append(f"ML ensemble predicts: {prediction.get('direction', 'unknown')} "
                        f"with {confidence:.1%} confidence")

        # Model agreement
        agreement = analysis.get("model_agreement", {})
        reasoning.append(f"Model agreement: {agreement.get('status', 'unknown')} "
                        f"({agreement.get('up_votes', 0)} up, {agreement.get('down_votes', 0)} down)")

        decision["reasoning"] = reasoning

        # Make final decision
        if signal == "buy":
            # Additional checks for buy
            if sentiment_score < -0.2:
                decision["action"] = "hold"
                decision["reasoning"].append("Holding due to negative news sentiment")
            elif overall_tech_score < -0.3:
                decision["action"] = "hold"
                decision["reasoning"].append("Holding due to bearish technicals")
            else:
                decision["action"] = "buy"
                # Calculate position size
                rec = self.portfolio.get_position_recommendation(
                    symbol, "buy", confidence, current_price
                )
                decision["quantity"] = rec.get("quantity", 0)
                decision["price"] = current_price

        elif signal == "sell":
            # Check if we have a position
            if analysis.get("current_position") is None:
                decision["action"] = "hold"
                decision["reasoning"].append("No position to sell")
            else:
                decision["action"] = "sell"
                rec = self.portfolio.get_position_recommendation(
                    symbol, "sell", confidence, current_price
                )
                decision["quantity"] = rec.get("quantity", 0)
                decision["price"] = current_price

        else:
            decision["action"] = "hold"

        decision["confidence"] = confidence

        # Log decision
        self.storage.log_decision(
            decision_type="trade_decision",
            symbol=symbol,
            decision=decision["action"],
            reasoning="\n".join(reasoning),
            market_conditions={
                "price": current_price,
                "technical_score": overall_tech_score,
                "sentiment_score": sentiment_score
            },
            model_outputs=prediction
        )

        return decision

    def execute_decision(self, decision: Dict) -> Dict:
        """
        Execute a trading decision.

        Args:
            decision: Trading decision from make_trading_decision()

        Returns:
            Execution result
        """
        action = decision.get("action", "hold")

        if action == "hold":
            return {"success": True, "action": "hold", "reason": "No trade executed"}

        symbol = decision["symbol"]
        quantity = decision.get("quantity", 0)
        price = decision.get("price", 0)
        reasoning = "\n".join(decision.get("reasoning", []))
        confidence = decision.get("confidence")

        if quantity <= 0:
            return {"success": False, "reason": "Invalid quantity"}

        if action == "buy":
            result = self.trader.buy(
                symbol=symbol,
                quantity=quantity,
                price=price,
                reasoning=reasoning,
                confidence=confidence
            )
        elif action == "sell":
            result = self.trader.sell(
                symbol=symbol,
                quantity=quantity,
                price=price,
                reasoning=reasoning,
                confidence=confidence
            )
        else:
            result = {"success": False, "reason": f"Unknown action: {action}"}

        return result

    def run_trading_cycle(self) -> Dict:
        """
        Run a complete trading cycle for all stocks.

        Returns:
            Cycle results including all decisions and executions
        """
        if not self.is_initialized:
            init_result = self.initialize()
            if not init_result.get("success"):
                return {"error": "Initialization failed", "details": init_result}

        logger.info("Starting trading cycle...")
        cycle_results = {
            "timestamp": datetime.now().isoformat(),
            "stocks_analyzed": [],
            "decisions": [],
            "executions": [],
            "errors": []
        }

        # Get current prices for portfolio valuation
        current_prices = {}
        for symbol in self.stocks:
            price = self.collector.get_current_price(symbol)
            if price:
                current_prices[symbol] = price

        # Analyze each stock and make decisions
        for symbol in self.stocks:
            try:
                # Analyze
                analysis = self.analyze_stock(symbol)
                cycle_results["stocks_analyzed"].append(analysis)

                # Make decision
                decision = self.make_trading_decision(analysis)
                cycle_results["decisions"].append(decision)

                # Execute if actionable
                if decision["action"] != "hold":
                    execution = self.execute_decision(decision)
                    cycle_results["executions"].append(execution)

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                cycle_results["errors"].append(f"{symbol}: {str(e)}")

        # Record portfolio snapshot
        self.trader.record_snapshot(current_prices)

        # Get portfolio summary
        cycle_results["portfolio_summary"] = self.trader.get_summary(current_prices)

        logger.info(f"Trading cycle complete. Analyzed {len(cycle_results['stocks_analyzed'])} stocks, "
                   f"executed {len(cycle_results['executions'])} trades")

        return cycle_results

    def run_optimization_cycle(self) -> Dict:
        """
        Run strategy optimization cycle.

        Returns:
            Optimization results
        """
        logger.info("Starting optimization cycle...")

        # Run optimization
        opt_results = self.optimizer.run_optimization_cycle()

        # Update ensemble weights
        new_weights = self.optimizer.get_weights()
        self.ensemble.set_weights(new_weights)

        # Check if models need retraining
        if opt_results.get("should_retrain", False):
            logger.info(f"Retraining models: {opt_results.get('retrain_reason')}")
            self.models_trained = False
            self.initialize()  # Re-initialize will retrain

        return opt_results

    def get_status(self) -> Dict:
        """Get current agent status."""
        current_prices = {}
        for symbol in self.stocks:
            price = self.collector.get_current_price(symbol)
            if price:
                current_prices[symbol] = price

        return {
            "initialized": self.is_initialized,
            "models_trained": self.models_trained,
            "stocks": self.stocks,
            "portfolio": self.trader.get_summary(current_prices),
            "model_weights": self.ensemble.get_weights(),
            "optimization_status": self.optimizer.get_optimization_status()
        }

    def get_performance_report(self) -> str:
        """Get formatted performance report."""
        return self.performance.generate_report()

    def run_backtest(self, symbol: str, days: int = 60) -> Dict:
        """
        Run a simple backtest on historical data.

        Args:
            symbol: Stock symbol
            days: Number of days to backtest

        Returns:
            Backtest results
        """
        logger.info(f"Running backtest for {symbol} over {days} days...")

        # Get historical data
        data = self.collector.get_stock_data(symbol, days=days + config.HISTORICAL_DAYS)
        if data is None:
            return {"error": "Could not fetch data"}

        # Add features
        data_with_features = self.technical.calculate_all(data)
        data_with_target = self.ensemble.predictor.create_target(data_with_features)

        # Use first portion for training
        train_size = len(data_with_target) - days
        train_data = data_with_target.iloc[:train_size]
        test_data = data_with_target.iloc[train_size:]

        # Train on historical data
        self.ensemble.train_all_models(train_data)

        # Simulate trading on test data
        backtest_trader = PaperTrader()
        results = {
            "symbol": symbol,
            "test_days": len(test_data),
            "trades": [],
            "predictions": []
        }

        for i in range(len(test_data) - config.PREDICTION_HORIZON):
            current_data = test_data.iloc[:i + 1]
            current_price = float(current_data["close"].iloc[-1])
            actual_future = float(test_data["close"].iloc[i + config.PREDICTION_HORIZON])

            # Make prediction
            prediction = self.ensemble.predict(current_data)

            if "error" not in prediction:
                predicted_direction = prediction["ensemble_prediction"]
                actual_direction = "up" if actual_future > current_price else "down"

                results["predictions"].append({
                    "predicted": predicted_direction,
                    "actual": actual_direction,
                    "confidence": prediction["confidence"],
                    "correct": predicted_direction == actual_direction
                })

        # Calculate backtest metrics
        if results["predictions"]:
            correct = sum(1 for p in results["predictions"] if p["correct"])
            total = len(results["predictions"])
            results["accuracy"] = correct / total
            results["total_predictions"] = total
            results["correct_predictions"] = correct

        return results

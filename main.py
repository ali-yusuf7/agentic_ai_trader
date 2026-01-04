#!/usr/bin/env python3
"""
Agentic AI Trader - Main Entry Point

An autonomous AI trading system that:
- Predicts stock prices using ML ensemble
- Analyzes news sentiment
- Executes paper trades
- Learns and improves over time

Usage:
    python main.py                  # Run trading cycle
    python main.py --init           # Initialize and train models
    python main.py --analyze AAPL   # Analyze single stock
    python main.py --status         # Show agent status
    python main.py --report         # Show performance report
    python main.py --optimize       # Run optimization cycle
    python main.py --backtest AAPL  # Run backtest on a stock
    python main.py --daemon         # Run continuously
"""
import argparse
import logging
import sys
import time
from datetime import datetime

import schedule

import config
from agents.trading_agent import TradingAgent

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOGS_DIR / f"trading_{datetime.now().strftime('%Y%m%d')}.log")
    ]
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print application banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                   AGENTIC AI TRADER                           ║
    ║         Autonomous Stock Prediction & Trading System          ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_analysis(analysis: dict):
    """Pretty print stock analysis."""
    print(f"\n{'='*60}")
    print(f"ANALYSIS: {analysis.get('symbol', 'Unknown')}")
    print(f"Time: {analysis.get('timestamp', 'N/A')}")
    print(f"{'='*60}")

    print(f"\nCurrent Price: ${analysis.get('current_price', 0):.2f}")

    # Technical signals
    tech = analysis.get('technical_signals', {})
    if tech:
        print(f"\n--- Technical Indicators ---")
        trend = tech.get('trend', {})
        print(f"  SMA Crossover: {trend.get('sma_crossover', 'N/A')}")
        print(f"  EMA Crossover: {trend.get('ema_crossover', 'N/A')}")

        momentum = tech.get('momentum', {})
        print(f"  RSI: {momentum.get('rsi', 0):.1f} ({momentum.get('rsi_signal', 'N/A')})")
        print(f"  MACD Crossover: {momentum.get('macd_crossover', 'N/A')}")

        print(f"  Overall Score: {tech.get('overall_score', 0):.2f}")

    # Sentiment
    sentiment = analysis.get('sentiment', {})
    if sentiment:
        print(f"\n--- News Sentiment ---")
        print(f"  Score: {sentiment.get('sentiment_score', 0):.2f}")
        print(f"  Label: {sentiment.get('sentiment_label', 'N/A')}")
        print(f"  Articles: {sentiment.get('news_volume', 0)}")

    # Prediction
    pred = analysis.get('prediction', {})
    if pred and 'error' not in pred:
        print(f"\n--- ML Prediction ---")
        print(f"  Direction: {pred.get('direction', 'N/A').upper()}")
        print(f"  Confidence: {pred.get('confidence', 0):.1%}")
        print(f"  Signal: {pred.get('signal', 'N/A').upper()}")

        print(f"\n  Individual Models:")
        for model, model_pred in pred.get('individual_models', {}).items():
            print(f"    {model}: {model_pred.get('prediction', 'N/A')} "
                  f"({model_pred.get('confidence', 0):.1%})")

    # Model agreement
    agreement = analysis.get('model_agreement', {})
    if agreement:
        print(f"\n--- Model Agreement ---")
        print(f"  Status: {agreement.get('status', 'N/A')}")
        print(f"  Votes: {agreement.get('up_votes', 0)} up / {agreement.get('down_votes', 0)} down")

    # Trading recommendation
    print(f"\n--- Trading Recommendation ---")
    print(f"  Should Trade: {'Yes' if analysis.get('should_trade') else 'No'}")
    print(f"  Reason: {analysis.get('trade_reason', 'N/A')}")

    # Current position
    position = analysis.get('current_position')
    if position:
        print(f"\n--- Current Position ---")
        print(f"  Shares: {position.get('quantity', 0)}")
        print(f"  Avg Price: ${position.get('avg_price', 0):.2f}")
        print(f"  P&L: ${position.get('pnl', 0):.2f} ({position.get('pnl_percent', 0):.1f}%)")

    print(f"\n{'='*60}\n")


def print_decision(decision: dict):
    """Pretty print trading decision."""
    print(f"\n{'='*60}")
    print(f"DECISION: {decision.get('symbol', 'Unknown')}")
    print(f"{'='*60}")
    print(f"Action: {decision.get('action', 'N/A').upper()}")

    if decision.get('quantity'):
        print(f"Quantity: {decision['quantity']} shares")
        print(f"Price: ${decision.get('price', 0):.2f}")

    print(f"Confidence: {decision.get('confidence', 0):.1%}")

    print(f"\nReasoning:")
    for reason in decision.get('reasoning', []):
        print(f"  - {reason}")

    print(f"{'='*60}\n")


def print_portfolio(summary: dict):
    """Pretty print portfolio summary."""
    print(f"\n{'='*60}")
    print("PORTFOLIO SUMMARY")
    print(f"{'='*60}")
    print(f"Initial Capital: ${summary.get('initial_capital', 0):,.2f}")
    print(f"Current Cash: ${summary.get('cash', 0):,.2f}")
    print(f"Positions Value: ${summary.get('positions_value', 0):,.2f}")
    print(f"Total Value: ${summary.get('total_value', 0):,.2f}")
    print(f"\nTotal P&L: ${summary.get('total_pnl', 0):,.2f} "
          f"({summary.get('total_pnl_percent', 0):.2f}%)")
    print(f"Number of Positions: {summary.get('num_positions', 0)}")
    print(f"Total Trades: {summary.get('num_trades', 0)}")

    positions = summary.get('positions', [])
    if positions:
        print(f"\n--- Open Positions ---")
        for pos in positions:
            print(f"  {pos.get('symbol', 'N/A')}: {pos.get('quantity', 0)} shares @ "
                  f"${pos.get('avg_price', 0):.2f} | P&L: ${pos.get('pnl', 0):.2f} "
                  f"({pos.get('pnl_percent', 0):.1f}%)")

    print(f"{'='*60}\n")


def run_init(agent: TradingAgent):
    """Initialize the trading agent."""
    print("\nInitializing Trading Agent...")
    print("This will fetch historical data and train ML models.\n")

    result = agent.initialize()

    if result.get('success'):
        print("\n✓ Initialization successful!")
        print(f"\nSteps completed:")
        for step in result.get('steps', []):
            print(f"  - {step}")

        if result.get('errors'):
            print(f"\nWarnings:")
            for error in result['errors']:
                print(f"  - {error}")

        print(f"\nStocks ready: {', '.join(result.get('stocks_ready', []))}")
    else:
        print(f"\n✗ Initialization failed: {result.get('error', 'Unknown error')}")


def run_analyze(agent: TradingAgent, symbol: str):
    """Analyze a single stock."""
    print(f"\nAnalyzing {symbol}...")

    if not agent.is_initialized:
        print("Agent not initialized. Initializing first...")
        agent.initialize()

    analysis = agent.analyze_stock(symbol)
    print_analysis(analysis)

    decision = agent.make_trading_decision(analysis)
    print_decision(decision)


def run_trading_cycle(agent: TradingAgent):
    """Run a complete trading cycle."""
    print("\nRunning trading cycle...")

    results = agent.run_trading_cycle()

    if 'error' in results:
        print(f"\n✗ Error: {results['error']}")
        return

    print(f"\nAnalyzed {len(results.get('stocks_analyzed', []))} stocks")

    # Show decisions
    for decision in results.get('decisions', []):
        if decision.get('action') != 'hold':
            print_decision(decision)

    # Show executions
    executions = results.get('executions', [])
    if executions:
        print(f"\n--- Trade Executions ---")
        for exec_result in executions:
            if exec_result.get('success'):
                trade = exec_result.get('trade', {})
                print(f"  ✓ {trade.get('action', '').upper()} {trade.get('quantity', 0)} "
                      f"{trade.get('symbol', '')} @ ${trade.get('price', 0):.2f}")
            else:
                print(f"  ✗ Failed: {exec_result.get('reason', 'Unknown')}")

    # Show portfolio summary
    print_portfolio(results.get('portfolio_summary', {}))


def run_status(agent: TradingAgent):
    """Show agent status."""
    status = agent.get_status()

    print(f"\n{'='*60}")
    print("AGENT STATUS")
    print(f"{'='*60}")
    print(f"Initialized: {'Yes' if status.get('initialized') else 'No'}")
    print(f"Models Trained: {'Yes' if status.get('models_trained') else 'No'}")
    print(f"Tracked Stocks: {', '.join(status.get('stocks', []))}")

    print(f"\n--- Model Weights ---")
    for model, weight in status.get('model_weights', {}).items():
        print(f"  {model}: {weight:.1%}")

    print_portfolio(status.get('portfolio', {}))

    opt_status = status.get('optimization_status', {})
    if opt_status.get('improvement_suggestions'):
        print(f"\n--- Improvement Suggestions ---")
        for suggestion in opt_status['improvement_suggestions']:
            print(f"  - {suggestion}")


def run_optimize(agent: TradingAgent):
    """Run optimization cycle."""
    print("\nRunning optimization cycle...")

    results = agent.run_optimization_cycle()

    print(f"\n{'='*60}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*60}")

    for step in results.get('steps', []):
        print(f"  ✓ {step}")

    # Prediction analysis
    pred = results.get('prediction_analysis', {})
    print(f"\n--- Prediction Performance ---")
    print(f"  Total Predictions: {pred.get('total_predictions', 0)}")
    print(f"  Accuracy: {pred.get('accuracy', 0):.1%}")

    # Weight optimization
    weight_opt = results.get('weight_optimization', {})
    if weight_opt.get('optimized'):
        print(f"\n--- Weight Changes ---")
        for model in weight_opt.get('old_weights', {}):
            old = weight_opt['old_weights'].get(model, 0)
            new = weight_opt['new_weights'].get(model, 0)
            change = new - old
            print(f"  {model}: {old:.1%} -> {new:.1%} ({'+' if change >= 0 else ''}{change:.1%})")
    else:
        print(f"\n  Weights not changed: {weight_opt.get('reason', 'N/A')}")

    # Parameter suggestions
    suggestions = results.get('parameter_suggestions', [])
    if suggestions:
        print(f"\n--- Suggested Parameter Changes ---")
        for s in suggestions:
            print(f"  {s['parameter']}: {s['current']} -> {s['suggested']}")
            print(f"    Reason: {s['reason']}")

    # Retrain recommendation
    if results.get('should_retrain'):
        print(f"\n⚠ Models should be retrained: {results.get('retrain_reason')}")


def run_backtest(agent: TradingAgent, symbol: str, days: int = 60):
    """Run backtest."""
    print(f"\nRunning backtest for {symbol} over {days} days...")

    if not agent.is_initialized:
        agent.initialize()

    results = agent.run_backtest(symbol, days)

    if 'error' in results:
        print(f"\n✗ Error: {results['error']}")
        return

    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS: {symbol}")
    print(f"{'='*60}")
    print(f"Test Period: {results.get('test_days', 0)} days")
    print(f"Total Predictions: {results.get('total_predictions', 0)}")
    print(f"Correct Predictions: {results.get('correct_predictions', 0)}")
    print(f"Accuracy: {results.get('accuracy', 0):.1%}")


def run_daemon(agent: TradingAgent):
    """Run agent continuously."""
    print("\nStarting daemon mode...")
    print(f"Will run trading cycle every {config.TRADING_INTERVAL_MINUTES} minutes")
    print("Press Ctrl+C to stop\n")

    # Initialize first
    if not agent.is_initialized:
        agent.initialize()

    def trading_job():
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running trading cycle...")
        try:
            run_trading_cycle(agent)
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")

    def optimization_job():
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running optimization cycle...")
        try:
            run_optimize(agent)
        except Exception as e:
            logger.error(f"Error in optimization cycle: {e}")

    # Schedule jobs
    schedule.every(config.TRADING_INTERVAL_MINUTES).minutes.do(trading_job)
    schedule.every().day.at("18:00").do(optimization_job)  # Run optimization daily

    # Run immediately
    trading_job()

    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Agentic AI Trader")
    parser.add_argument("--init", action="store_true", help="Initialize and train models")
    parser.add_argument("--analyze", metavar="SYMBOL", help="Analyze a single stock")
    parser.add_argument("--status", action="store_true", help="Show agent status")
    parser.add_argument("--report", action="store_true", help="Show performance report")
    parser.add_argument("--optimize", action="store_true", help="Run optimization cycle")
    parser.add_argument("--backtest", metavar="SYMBOL", help="Run backtest on a stock")
    parser.add_argument("--backtest-days", type=int, default=60, help="Days for backtest")
    parser.add_argument("--daemon", action="store_true", help="Run continuously")
    parser.add_argument("--stocks", nargs="+", help="Stock symbols to track")

    args = parser.parse_args()

    print_banner()

    # Create agent
    stocks = args.stocks if args.stocks else config.STOCKS
    agent = TradingAgent(stocks=stocks)

    try:
        if args.init:
            run_init(agent)
        elif args.analyze:
            run_analyze(agent, args.analyze.upper())
        elif args.status:
            run_status(agent)
        elif args.report:
            print(agent.get_performance_report())
        elif args.optimize:
            run_optimize(agent)
        elif args.backtest:
            run_backtest(agent, args.backtest.upper(), args.backtest_days)
        elif args.daemon:
            run_daemon(agent)
        else:
            # Default: run single trading cycle
            run_trading_cycle(agent)

    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()

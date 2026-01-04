# Agentic AI Trader

An autonomous AI-powered stock trading system that predicts market prices, executes paper trades, analyzes performance, and continuously improves its strategies over time.

## Features

- **ML Ensemble Predictions** - Combines Random Forest, XGBoost, and Logistic Regression with adaptive weighting
- **Technical Analysis** - 30+ indicators including RSI, MACD, Bollinger Bands, moving averages, volume analysis
- **News Sentiment Analysis** - Fetches financial news and analyzes sentiment using NLTK VADER
- **Paper Trading** - Simulates trades with realistic portfolio management and risk controls
- **Self-Improvement** - Automatically adjusts model weights and strategies based on performance
- **Performance Tracking** - Calculates Sharpe ratio, max drawdown, win rate, and other metrics

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Main Trading Agent                         │
│         (Orchestrates all components autonomously)              │
└─────────────────────────────────────────────────────────────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   Data      │ │  Feature    │ │ Prediction  │ │  Strategy   │
│  Collector  │ │  Engineer   │ │   Engine    │ │  Optimizer  │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/ali-yusuf7/agentic_ai_trader.git
cd agentic_ai_trader

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (one-time setup)
python -c "import nltk; nltk.download('vader_lexicon')"
```

## Usage

### Initialize & Train Models
```bash
python main.py --init
```
Fetches historical data and trains ML models for all tracked stocks.

### Run Trading Cycle
```bash
python main.py
```
Analyzes all stocks, makes predictions, and executes paper trades.

### Analyze Single Stock
```bash
python main.py --analyze AAPL
```
Deep analysis of a specific stock with technical indicators, sentiment, and ML prediction.

### View Status
```bash
python main.py --status
```
Shows current portfolio, positions, and model weights.

### Performance Report
```bash
python main.py --report
```
Generates comprehensive performance report with metrics.

### Run Optimization
```bash
python main.py --optimize
```
Analyzes recent performance and adjusts model weights.

### Backtest
```bash
python main.py --backtest AAPL --backtest-days 60
```
Tests prediction accuracy on historical data.

### Daemon Mode (Continuous)
```bash
python main.py --daemon
```
Runs continuously, executing trading cycles at configured intervals.

### Custom Stocks
```bash
python main.py --stocks AAPL GOOGL NVDA
```

## Project Structure

```
agentic_ai_trader/
├── main.py                    # CLI entry point
├── config.py                  # Configuration settings
├── requirements.txt           # Dependencies
├── agents/
│   └── trading_agent.py       # Main AI orchestrator
├── data/
│   ├── collector.py           # Yahoo Finance & news fetching
│   └── storage.py             # SQLite persistence
├── features/
│   ├── technical.py           # Technical indicators
│   └── sentiment.py           # News sentiment analysis
├── models/
│   ├── predictor.py           # ML models
│   └── ensemble.py            # Weighted ensemble
├── trading/
│   ├── paper_trader.py        # Paper trading simulation
│   └── portfolio.py           # Portfolio management
└── analysis/
    ├── performance.py         # Performance metrics
    └── strategy_optimizer.py  # Self-improvement logic
```

## Configuration

Edit `config.py` to customize:

| Setting | Default | Description |
|---------|---------|-------------|
| `STOCKS` | AAPL, GOOGL, MSFT, AMZN, TSLA | Stocks to track |
| `INITIAL_CAPITAL` | $100,000 | Starting paper trading capital |
| `RISK_PER_TRADE` | 2% | Max portfolio % per trade |
| `PREDICTION_HORIZON` | 5 days | How far ahead to predict |
| `TRADING_INTERVAL_MINUTES` | 60 | Cycle frequency in daemon mode |

## How It Works

1. **Data Collection** - Fetches OHLCV data from Yahoo Finance and news from RSS feeds
2. **Feature Engineering** - Calculates technical indicators and sentiment scores
3. **Prediction** - ML ensemble votes on price direction with confidence scores
4. **Decision Making** - Agent combines signals, checks risk limits, decides action
5. **Execution** - Paper trades executed with full logging
6. **Analysis** - Tracks prediction accuracy and trading performance
7. **Optimization** - Adjusts model weights based on which models perform best

## Technical Indicators

- Moving Averages (SMA, EMA)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- ATR (Average True Range)
- Volume indicators (OBV, VPT)
- Price momentum and rate of change
- 52-week high/low distance

## Disclaimer

This is an educational project for paper trading only. It does not execute real trades or provide financial advice. Past performance does not guarantee future results. Use at your own risk.

## License

MIT License - see [LICENSE](LICENSE) for details.

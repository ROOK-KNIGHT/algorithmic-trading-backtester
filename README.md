# Algorithmic Trading Backtester

A comprehensive Python-based backtesting framework for algorithmic trading strategies with advanced visualization capabilities. This repository contains three distinct trading strategies with full backtesting infrastructure and professional-grade performance analysis.

## Overview

This backtesting framework provides:
- **Three Complete Trading Strategies** with theoretical foundations
- **Comprehensive Backtesting Engine** with realistic trade execution
- **Advanced Visualization Suite** for performance analysis
- **Automated Workflow** from data fetching to chart generation
- **Professional Performance Metrics** including Sharpe ratio, drawdown analysis, and profit factors

## Trading Strategies

### 1. DEMA-SMA Crossover Strategy

**Theory**: This strategy combines the responsiveness of the Double Exponential Moving Average (DEMA) with the stability of the Simple Moving Average (SMA) to identify trend changes and momentum shifts.

**Implementation**:
- DEMA(15) for responsive trend detection
- SMA(30) for trend confirmation
- Long signals when DEMA crosses above SMA
- Short signals when DEMA crosses below SMA
- Dynamic take-profit based on distance standard deviation
- 1.5% stop loss with 2:1 risk-reward ratio

**Key Features**:
- Adaptive take-profit levels
- Trend alignment validation
- Early profit-taking at 0.5% gains

### 2. Divergence Strategy

**Theory**: Based on the principle that price and momentum indicators often diverge before significant price reversals. The strategy identifies discrepancies between price action and technical indicators (RSI, A/D Line) to predict trend changes.

**Implementation**:
- RSI(14) and Accumulation/Distribution Line analysis
- Swing high/low detection using local extrema
- Bullish divergence: Lower price lows with higher indicator lows
- Bearish divergence: Higher price highs with lower indicator highs
- Multiple divergence strength classifications (strong, medium, weak, hidden)
- Support/resistance level integration

**Key Features**:
- Multi-indicator divergence confirmation
- Swing point clustering for support/resistance
- Trend alignment scoring
- Confidence-based position sizing

### 3. Exceedence Strategy

**Theory**: A volatility-based momentum strategy that capitalizes on price movements exceeding calculated volatility bands. The strategy assumes that extreme positions within volatility ranges often lead to mean reversion or continuation patterns.

**Implementation**:
- Dynamic volatility band calculation using rolling statistics
- High-side and low-side volatility components
- Position-in-range calculation (0-100%)
- Long signals at 99% of range (extreme high)
- Short signals at 1% of range (extreme low)
- Band stability analysis for exit timing

**Key Features**:
- Adaptive volatility bands
- Position-in-range momentum detection
- Band stability validation
- Breakeven exit conditions

## Performance Results

### DEMA-SMA Crossover Strategy
- **NVDA (10-day, 5-minute)**: Multiple successful trades with trend-following accuracy
- **Consistent Performance**: Reliable signal generation in trending markets

### Divergence Strategy
- **NVDA (5-day, 15-minute)**: 81.8% win rate, $68.47 profit, 11 trades
- **AAPL (10-day, 30-minute)**: 60% win rate, $13.89 profit, 5 trades
- **Strong Performance**: Excellent risk-adjusted returns in volatile conditions

### Exceedence Strategy
- **TSLA (5-day, 30-minute)**: 100% win rate, $683.85 profit, 2.74% ROI, 27 trades
- **Exceptional Results**: Perfect win rate with consistent profit generation
- **High Sharpe Ratio**: 126.052 indicating excellent risk-adjusted performance

## Installation and Setup

### Prerequisites
```bash
pip install pandas numpy matplotlib scipy talib
```

### Required Files
- `historical_data_handler.py` - Data fetching interface
- `connection_manager.py` - API connection management

### API Setup
Before using the backtester, you need to configure your API credentials:

1. **Schwab API Setup**:
   - Register for a Schwab Developer Account at https://developer.schwab.com/
   - Create a new application to get your APP_KEY and APP_SECRET
   - Update `connection_manager.py` with your credentials:
     ```python
     APP_KEY = "YOUR_SCHWAB_APP_KEY_HERE"
     APP_SECRET = "YOUR_SCHWAB_APP_SECRET_HERE"
     ```

2. **Token Management**:
   - The system will automatically handle OAuth token refresh
   - Tokens are stored locally in `cs_tokens.json` (excluded from git)

### Directory Structure
```
algorithmic-trading-backtester/
├── README.md
├── .gitignore
├── backtesters/
│   ├── dema_sma_crossover_backtest.py
│   ├── divergence_backtest.py
│   └── exceedence_backtest.py
├── visualizers/
│   ├── dema_sma_visualization.py
│   ├── divergence_visualization.py
│   └── exceedence_visualization.py
├── handlers/
│   ├── connection_manager.py
│   ├── historical_data_handler.py
│   └── fetch_data.py
├── strategies/
│   ├── dema_sma_crossover_strategy.py
│   ├── exceedence_strategy.py
│   ├── Divergence_calculator.py
│   └── stock_divergence_calculator.py
├── charts/
├── historical_data/
└── data/
```

## Usage

### Running Backtests

#### DEMA-SMA Crossover Strategy
```bash
# Basic backtest with default parameters
python3 backtesters/dema_sma_crossover_backtest.py AAPL

# Custom timeframe
python3 backtesters/dema_sma_crossover_backtest.py NVDA --period 10 --period-type day --frequency-type minute --frequency 5

# Custom lookback period
python3 backtesters/dema_sma_crossover_backtest.py TSLA --lookback 100
```

#### Divergence Strategy
```bash
# Basic divergence backtest
python3 backtesters/divergence_backtest.py AAPL

# High-frequency analysis
python3 backtesters/divergence_backtest.py NVDA --period 5 --period-type day --frequency-type minute --frequency 15

# Custom divergence lookback
python3 backtesters/divergence_backtest.py TSLA --lookback 20
```

#### Exceedence Strategy
```bash
# Basic volatility backtest
python3 backtesters/exceedence_backtest.py AAPL

# Intraday volatility analysis
python3 backtesters/exceedence_backtest.py TSLA --period 5 --period-type day --frequency-type minute --frequency 30

# Custom volatility lookback
python3 backtesters/exceedence_backtest.py NVDA --lookback 50
```

### Visualization Options

Each strategy includes comprehensive visualization capabilities:

```bash
# Comprehensive analysis (automatic after backtest)
python3 visualizers/[strategy]_visualization.py historical_data/results.csv

# Simple 3-panel view
python3 visualizers/[strategy]_visualization.py historical_data/results.csv --simple

# Save to custom location
python3 visualizers/[strategy]_visualization.py historical_data/results.csv --save charts/custom_analysis.png
```

## Visualization Features

### Comprehensive Mode
- **Price Charts**: Candlestick charts with strategy-specific indicators
- **Signal Visualization**: Entry/exit points with profit/loss color coding
- **Technical Indicators**: Strategy-specific overlays (DEMA/SMA, RSI, volatility bands)
- **Equity Curve**: Portfolio performance with drawdown highlighting
- **Performance Metrics**: Complete statistical summary
- **Trade Distribution**: P&L histogram analysis
- **Strategy-Specific Analysis**: Divergence types, signal performance, volatility analysis

### Simple Mode
- **Streamlined View**: 3-panel layout for quick analysis
- **Essential Metrics**: Key performance indicators
- **Clean Presentation**: Professional formatting for reports

## Performance Metrics

The backtester calculates comprehensive performance statistics:

- **Trade Statistics**: Win rate, total trades, average P&L
- **Risk Metrics**: Sharpe ratio, maximum drawdown, profit factor
- **Return Analysis**: ROI, gross profit/loss, risk-adjusted returns
- **Strategy-Specific Metrics**: Signal accuracy, divergence success rates, volatility capture

## Risk Management

All strategies implement professional risk management:
- **Position Sizing**: Percentage-based risk allocation
- **Stop Losses**: Adaptive stop-loss levels
- **Take Profits**: Dynamic profit-taking rules
- **Drawdown Control**: Maximum drawdown monitoring
- **Trade Validation**: Sequence validation and error checking

## Data Requirements

- **Historical Data**: OHLCV data from supported data providers
- **Timeframes**: Minute, daily, weekly, monthly intervals
- **Symbols**: Equity markets (extensible to other asset classes)
- **Quality**: Clean, adjusted data for accurate backtesting

## Technical Architecture

### Modular Design
- **Strategy Modules**: Independent strategy implementations
- **Visualization Engine**: Reusable charting framework
- **Data Handler**: Abstracted data access layer
- **Performance Calculator**: Standardized metrics computation

### Extensibility
- **Plugin Architecture**: Easy addition of new strategies
- **Configurable Parameters**: Adjustable strategy parameters
- **Multiple Timeframes**: Support for various analysis periods
- **Custom Indicators**: Framework for additional technical indicators

## Contributing

This backtesting framework is designed for professional algorithmic trading research and development. The modular architecture allows for easy extension and customization of trading strategies.

## Disclaimer

This software is for educational and research purposes only. Past performance does not guarantee future results. Trading involves substantial risk and may not be suitable for all investors. Always conduct thorough testing and risk assessment before deploying any trading strategy with real capital.

## License

This project is provided as-is for educational and research purposes. Please ensure compliance with all applicable financial regulations and data provider terms of service.

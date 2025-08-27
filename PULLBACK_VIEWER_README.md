# Interactive Pullback Viewer

Based on TradingView's Pullback Viewer by emka, this interactive chart identifies valid pullback points in trending markets using live data from your Schwab API handlers.

## Features

- **Real-time data fetching** using existing Schwab API handlers
- **Interactive plotly charts** with zoom, pan, and hover functionality
- **Pullback identification** for both bullish and bearish trends
- **Configurable parameters** for different timeframes and analysis types
- **Volume analysis** with optional volume subplot
- **Structure-only mode** to show only the most recent pullbacks of each type

## What is a Valid Pullback?

According to the TradingView indicator:

- **Bearish Pullback**: Current candle body closes above the previous candle's high (bullish candle breaking above resistance)
- **Bullish Pullback**: Current candle body closes below the previous candle's low (bearish candle breaking below support)
- Must be a **clean body close**, not just a wick touching the level

## Usage

### Basic Usage
```bash
python3 pullback_viewer_interactive.py AAPL
```

### Advanced Options
```bash
# 7 days of 1-minute data
python3 pullback_viewer_interactive.py NVDA --days 7 --frequency 1

# Structure-only mode (shows only most recent pullbacks)
python3 pullback_viewer_interactive.py TSLA --structure-only

# Hide volume chart
python3 pullback_viewer_interactive.py MSFT --no-volume

# Combine options
python3 pullback_viewer_interactive.py GOOGL --days 5 --frequency 15 --structure-only --no-volume
```

### Command Line Options

- `symbol`: Stock symbol (required)
- `--days`: Number of days of data (default: 7)
- `--frequency`: Frequency in minutes - choices: 1, 5, 15, 30, 60 (default: 5)
- `--structure-only`: Show only structural pullbacks (last of each type)
- `--no-volume`: Hide volume chart

## Output

The script provides:

1. **Console Analysis**: 
   - Total pullback count
   - Bullish vs bearish ratio
   - Recent pullback details

2. **Interactive Chart**: 
   - Candlestick price chart
   - Red dots (●) for bearish pullbacks
   - Green dots (●) for bullish pullbacks
   - Volume bars (optional)
   - Hover tooltips with details

3. **HTML File**: 
   - Saved automatically with timestamp
   - Can be shared or viewed later

## Examples

### Regular Mode
Shows all pullback points identified in the data:
```bash
python3 pullback_viewer_interactive.py AAPL --days 3 --frequency 5
```
Output: `Identified 76 pullback points`

### Structure-Only Mode
Shows only the most recent pullback of each type:
```bash
python3 pullback_viewer_interactive.py NVDA --days 2 --frequency 15 --structure-only
```
Output: `Identified 2 pullback points`

## Use Cases

As mentioned in the original TradingView indicator, pullbacks can be used to:

- **Identify supply and demand zones**
- **Spot key levels for support and resistance**
- **Use as anchor points for trendlines**
- **Find potential reaction points in trending markets**

## Requirements

- Valid Schwab API tokens (automatically refreshed)
- Python packages: plotly, pandas, numpy, requests
- API keys configured in `/Users/isaac/Desktop/Projects/CS_KEYS/KEYS.json`

## Authentication

The script automatically handles token refresh. If you get authentication errors, ensure your Schwab API credentials are properly configured and you have valid refresh tokens.

## Chart Features

The interactive chart includes:
- **Zoom and Pan**: Mouse wheel and drag
- **Hover Details**: Price and pullback information
- **Legend Toggle**: Click to show/hide data series
- **Dark Theme**: Professional trading interface
- **Time Navigation**: Click and drag on time axis
- **Volume Correlation**: Optional volume bars with price-matched colors

## Files Generated

Each run creates an HTML file with format:
`{SYMBOL}_pullback_viewer_{YYYYMMDD_HHMMSS}.html`

Example: `AAPL_pullback_viewer_20250826_210331.html`

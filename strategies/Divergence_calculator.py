import base64
import requests
import json
import urllib.parse
import os
import time
import pandas as pd
import numpy as np
# Try different import approaches for TA-Lib
try:
    import talib
except ImportError:
    try:
        import TA_Lib as talib  # Using underscore instead of hyphen
    except ImportError:
        try:
            import talib.abstract as talib
        except ImportError:
            print("TA-Lib not found. Please ensure it's installed properly.")
            sys.exit(1)
from datetime import datetime, timedelta
import logging
import threading
import sys
import concurrent.futures
from discord_webhook import DiscordWebhook, DiscordEmbed
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from historical_data_handler import HistoricalDataHandler
from divergence_notifier import DivergenceNotifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("divergence_scanner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create directory for chart images
CHARTS_DIR = "/Users/isaac/Desktop/Over_night/data/divergence"
os.makedirs(CHARTS_DIR, exist_ok=True)
print(f"Charts will be saved to: {CHARTS_DIR}")


# Scanning parameters
BATCH_SIZE = 200  # Process symbols in batches to avoid rate limiting
MAX_WORKERS = 100  # Maximum number of parallel workers for scanning

def get_next_interval(frequency=1):
    """
    Calculate seconds until next candle interval based on frequency
    Args:
        frequency: Number of minutes per candle (default: 1)
    """
    now = datetime.now()
    current_minute = now.minute
    # Calculate next interval based on frequency
    next_interval = ((current_minute // frequency) + 1) * frequency
    # If we're already past the last interval in this hour
    if next_interval >= 60:
        next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        return (next_hour - now).total_seconds()
    # Get to the next interval
    next_time = now.replace(minute=next_interval, second=0, microsecond=0)
    if next_time <= now:  # If we've passed the time already
        next_time += timedelta(minutes=frequency)
    return (next_time - now).total_seconds()

# S&P 500 symbols - static list, can be replaced with dynamic fetching
SP500_SYMBOLS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "GOOG", "TSLA", "BRK.B", "UNH", 
    "LLY", "JPM", "XOM", "V", "AVGO", "PG", "MA", "HD", "COST", "MRK", 
    "CVX", "ABBV", "PEP", "KO", "ADBE", "WMT", "BAC", "CRM", "MCD", "ABT", 
    "ACN", "LIN", "CSCO", "AMD", "TMO", "CMCSA", "ORCL", "NKE", "DHR", "PFE", 
    "INTC", "PM", "NFLX", "WFC", "TXN", "VZ", "COP", "IBM", "QCOM", "UPS"
]  # Top 50 companies in S&P 500 by market cap

# Include futures symbols for comprehensive market analysis
FUTURES_SYMBOLS = ['/ES', '/NQ', '/RTY', '/YM', '/GC', '/CL']

# Combine both lists
SYMBOLS = FUTURES_SYMBOLS + SP500_SYMBOLS

# Add function to dynamically fetch S&P 500 symbols
def get_sp500_symbols():
    """
    Dynamically fetch S&P 500 symbols from Wikipedia.
    Returns a list of symbols.
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        sp500_table = tables[0]
        symbols = sp500_table['Symbol'].tolist()
        # Clean up symbols (remove dots, etc.)
        symbols = [symbol.replace('.', '-') for symbol in symbols]
        logger.info(f"Successfully fetched {len(symbols)} S&P 500 symbols")
        return symbols
    except Exception as e:
        logger.error(f"Error fetching S&P 500 symbols: {str(e)}")
        # Fallback to static list if unable to fetch
        logger.info(f"Using static list of {len(SP500_SYMBOLS)} top S&P 500 symbols")
        return SP500_SYMBOLS

# Market Hours (times in Pacific Time)
MARKET_HOURS = {
    "FUTURES_OPEN": "15:00",  # 3:00 PM PT Sunday
    "FUTURES_CLOSE": "13:00"  # 1:00 PM PT Friday
}

# Market data timeframes
TIMEFRAMES = ['1min', '5min', '15min', '1hour', '4hour']

# Indicator parameters
RSI_PERIOD = 14
ADV_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
EMA_SHORT = 9
EMA_MEDIUM = 21
EMA_LONG = 50
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
ATR_PERIOD = 14

# Divergence parameters
SWING_LOOKBACK = 10
DIVERGENCE_THRESHOLD = 0.05
MIN_SWING_PERCENT = 0.2

# Signal cooldown to prevent excessive alerts
COOLDOWN_PERIOD = 0  # 4 hours
last_signal_time = {}

# Define market sessions (times in UTC)
MARKET_SESSIONS = {
    "Asian": {"start": "21:00", "end": "03:00"},
    "European": {"start": "07:00", "end": "16:00"},
    "US": {"start": "13:30", "end": "20:00"}
}

# Trading strategy parameters
RISK_REWARD_RATIO = 2.5  # Minimum reward:risk ratio
MAX_RISK_PERCENT = 1.0   # Maximum risk percentage per trade
STOP_LOSS_ATR_MULT = 1.5 # Stop loss multiplier based on ATR

class DivergenceScanner:
    def __init__(self):
        # Initialize without tokens for now
        # self.tokens = self.ensure_valid_tokens()  # This method doesn't exist
        self.historical_data = {}
        self.results = {}
        self.stop_event = threading.Event()
        self.trade_signals = []  # Track active trade signals
        self.logger = logging.getLogger(__name__)
        self.last_data_frequency = 1  # Track the last used data frequency
        
        # Initialize last signal time for each symbol with expanded divergence types
        for symbol in SYMBOLS:
            last_signal_time[symbol] = {}
            # RSI divergences
            for direction in ["bullish", "bearish"]:
                for strength in ["strong", "medium", "weak", "hidden"]:
                    last_signal_time[symbol][f"{direction}_rsi_{strength}"] = 0
            
            # AD Volume divergences
            for direction in ["bullish", "bearish"]:
                for strength in ["strong", "medium", "weak", "hidden"]:
                    last_signal_time[symbol][f"{direction}_adv_{strength}"] = 0
        
        logger.info("Enhanced Divergence Scanner initialized for symbols: %s", ", ".join(SYMBOLS))
        self.historical_data_handler = HistoricalDataHandler()

    def get_historical_data(self, symbol, period_type="day", period=10, frequency_type="minute", frequency=5):
        """Get historical data for a symbol with specified parameters"""
        try:
            logger.info(f"Loading data for {symbol} with {frequency_type}:{frequency} timeframe...")
            # Calculate current time in epoch milliseconds for endDate
            current_epoch_ms = int(pd.Timestamp.now().timestamp() * 1000)
            
            # Store the frequency for interval timing
            self.last_data_frequency = frequency
            
            # Fetch data with current time as endDate to ensure latest data
            data = self.historical_data_handler.fetch_historical_data(
                symbol=symbol,
                periodType=period_type,
                period=period,
                frequencyType=frequency_type,
                freq=frequency,
                endDate=current_epoch_ms,
                needExtendedHoursData=True
            )
            
            if not data or 'candles' not in data:
                raise ValueError(f"Failed to fetch data for {symbol}")
            
            df = pd.DataFrame(data['candles'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            return df
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {str(e)}")
            raise


    def calculate_indicators(self, df):
        """
        Calculate technical indicators for divergence detection and additional analysis.
        """
        if df is None or len(df) < 50:
            logger.warning("Not enough data to calculate indicators")
            return None
        
        df_indicators = df.copy()
        
        # Convert data to proper type for TA-Lib (float64/double)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df_indicators[col] = df_indicators[col].astype(float)
            
        # Calculate RSI
        df_indicators['rsi'] = talib.RSI(df_indicators['close'].values, timeperiod=RSI_PERIOD)
        
        # Calculate AD (Accumulation/Distribution) Line
        df_indicators['ad'] = talib.AD(
            df_indicators['high'].values,
            df_indicators['low'].values,
            df_indicators['close'].values,
            df_indicators['volume'].values
        )
        
        # Calculate OBV (On-Balance Volume)
        df_indicators['obv'] = talib.OBV(df_indicators['close'].values, df_indicators['volume'].values)
        
        # Calculate MACD (for additional context)
        macd, macd_signal, macd_hist = talib.MACD(
            df_indicators['close'].values,
            fastperiod=MACD_FAST,
            slowperiod=MACD_SLOW,
            signalperiod=MACD_SIGNAL
        )
        df_indicators['macd'] = macd
        df_indicators['macd_signal'] = macd_signal
        df_indicators['macd_hist'] = macd_hist
        
        # Calculate EMAs for trend context
        df_indicators['ema_short'] = talib.EMA(df_indicators['close'].values, timeperiod=EMA_SHORT)
        df_indicators['ema_medium'] = talib.EMA(df_indicators['close'].values, timeperiod=EMA_MEDIUM)
        df_indicators['ema_long'] = talib.EMA(df_indicators['close'].values, timeperiod=EMA_LONG)
        
        # Calculate Bollinger Bands
        df_indicators['bb_upper'], df_indicators['bb_middle'], df_indicators['bb_lower'] = talib.BBANDS(
            df_indicators['close'].values, 
            timeperiod=BOLLINGER_PERIOD,
            nbdevup=BOLLINGER_STD,
            nbdevdn=BOLLINGER_STD
        )
        
        # Calculate ATR for volatility and stop loss/take profit
        df_indicators['atr'] = talib.ATR(
            df_indicators['high'].values,
            df_indicators['low'].values,
            df_indicators['close'].values,
            timeperiod=ATR_PERIOD
        )
        
        # Add trend strength indicator (ADX)
        df_indicators['adx'] = talib.ADX(
            df_indicators['high'].values,
            df_indicators['low'].values,
            df_indicators['close'].values,
            timeperiod=14
        )
        
        # Calculate momentum indicators
        df_indicators['mom'] = talib.MOM(df_indicators['close'].values, timeperiod=10)
        
        # Identify market session
        df_indicators['market_session'] = self.identify_market_session(df_indicators.index)
        
        # Add trend direction
        df_indicators['trend'] = np.where(
            df_indicators['ema_short'] > df_indicators['ema_long'], 
            'bullish', 
            np.where(
                df_indicators['ema_short'] < df_indicators['ema_long'], 
                'bearish', 
                'neutral'
            )
        )
        
        # Add volatility band
        df_indicators['volatility_ratio'] = (df_indicators['bb_upper'] - df_indicators['bb_lower']) / df_indicators['bb_middle']
        
        return df_indicators

    def identify_market_session(self, datetime_index):
        """
        Identify which market session each timestamp belongs to.
        """
        sessions = []
        
        for dt in datetime_index:
            time_str = dt.strftime('%H:%M')
            if self.is_in_session(time_str, MARKET_SESSIONS["Asian"]):
                sessions.append("Asian")
            elif self.is_in_session(time_str, MARKET_SESSIONS["European"]):
                sessions.append("European")
            elif self.is_in_session(time_str, MARKET_SESSIONS["US"]):
                sessions.append("US")
            else:
                sessions.append("Off-Hours")
                
        return sessions
    
    def is_in_session(self, time_str, session):
        """
        Check if a time is within a market session.
        """
        # Handle sessions that span across midnight
        if session["start"] > session["end"]:
            return time_str >= session["start"] or time_str <= session["end"]
        else:
            return session["start"] <= time_str <= session["end"]

    def detect_price_swings(self, df, min_points=5):
        """
        Detect significant price swing highs and lows.
        Returns DataFrame with swing points identified.
        """
        try:
            df_swings = df.copy()
            
            # Ensure close is float type
            df_swings['close'] = df_swings['close'].astype(float)
            
            # Find local maxima and minima
            high_indices = argrelextrema(df_swings['close'].values, np.greater, order=min_points)[0]
            low_indices = argrelextrema(df_swings['close'].values, np.less, order=min_points)[0]
            
            # Initialize swing columns with NaN
            df_swings['swing_high'] = np.nan
            df_swings['swing_low'] = np.nan
            
            # Assign swing values
            if len(high_indices) > 0:
                df_swings.loc[df_swings.index[high_indices], 'swing_high'] = df_swings.iloc[high_indices]['close'].values
            
            if len(low_indices) > 0:
                df_swings.loc[df_swings.index[low_indices], 'swing_low'] = df_swings.iloc[low_indices]['close'].values
            
            # Calculate the percentage change for swings
            avg_price = df_swings['close'].mean()
            min_swing_value = avg_price * (MIN_SWING_PERCENT / 100)
            
            # Filter swings that are significant enough
            for i in range(len(df_swings)):
                if pd.notna(df_swings['swing_high'].iloc[i]):
                    # Check if this high is significant compared to nearby lows
                    nearby_lows = df_swings['swing_low'].iloc[max(0, i-SWING_LOOKBACK):min(len(df_swings), i+SWING_LOOKBACK)]
                    nearby_lows_clean = nearby_lows.dropna()
                    if nearby_lows_clean.empty or (df_swings['swing_high'].iloc[i] - nearby_lows_clean.min()) < min_swing_value:
                        df_swings.loc[df_swings.index[i], 'swing_high'] = np.nan
                
                if pd.notna(df_swings['swing_low'].iloc[i]):
                    # Check if this low is significant compared to nearby highs
                    nearby_highs = df_swings['swing_high'].iloc[max(0, i-SWING_LOOKBACK):min(len(df_swings), i+SWING_LOOKBACK)]
                    nearby_highs_clean = nearby_highs.dropna()
                    if nearby_highs_clean.empty or (nearby_highs_clean.max() - df_swings['swing_low'].iloc[i]) < min_swing_value:
                        df_swings.loc[df_swings.index[i], 'swing_low'] = np.nan
            
            # Identify potential support/resistance levels
            df_swings['support'] = np.nan
            df_swings['resistance'] = np.nan
            
            # Extract multiple swing lows/highs for support/resistance detection
            swing_lows = df_swings.loc[~df_swings['swing_low'].isna()]['swing_low'].tolist()
            swing_highs = df_swings.loc[~df_swings['swing_high'].isna()]['swing_high'].tolist()
            
            # Find clusters of swing levels (simple implementation)
            if swing_lows:
                # Check each price level to see if it's a potential support 
                # by counting nearby swing lows
                last_price = df_swings['close'].iloc[-1]
                for i, level in enumerate(swing_lows):
                    # Only consider levels below current price for support
                    if level < last_price:
                        cluster_count = sum(1 for lvl in swing_lows if abs(lvl - level)/level < 0.005)
                        if cluster_count >= 2:
                            # Mark this as a support level on the most recent occurrence
                            matching_idx = df_swings[df_swings['swing_low'] == level].index[-1]
                            df_swings.loc[matching_idx, 'support'] = level
            
            if swing_highs:
                # Check each price level to see if it's a potential resistance
                # by counting nearby swing highs
                last_price = df_swings['close'].iloc[-1]
                for i, level in enumerate(swing_highs):
                    # Only consider levels above current price for resistance
                    if level > last_price:
                        cluster_count = sum(1 for lvl in swing_highs if abs(lvl - level)/level < 0.005)
                        if cluster_count >= 2:
                            # Mark this as a resistance level on the most recent occurrence
                            matching_idx = df_swings[df_swings['swing_high'] == level].index[-1]
                            df_swings.loc[matching_idx, 'resistance'] = level
                        
            return df_swings
            
        except Exception as e:
            logger.error(f"Error in detect_price_swings: {str(e)}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            # Return original dataframe with empty swing columns if there's an error
            df['swing_high'] = np.nan
            df['swing_low'] = np.nan
            return df

    def detect_divergences(self, df_with_swings, symbol):
        """
        Detect price/RSI and price/AD divergences with classification by strength.
        Based on the RSI Divergence Cheat Sheet structure:
        - Strong: Clear divergence with lower/higher swings
        - Medium: Divergence with equal swings on one side
        - Weak: Divergence with weaker swings
        - Hidden: Countertrend divergence
        Returns a dict with detected divergences.
        """
        try:
            df = df_with_swings.copy()
            
            if df is None or 'swing_high' not in df.columns or 'swing_low' not in df.columns:
                logger.warning(f"Cannot detect divergences for {symbol} - missing swing data")
                return None
                
            # Initialize time variables for signal age checks
            current_time = df.index[-1]  # Get the latest timestamp from the data
            max_age = pd.Timedelta(minutes=5)  # Maximum age for valid signals
            
            # Initialize results dictionary
            divergences = {
                "rsi_divergences": {
                    "bullish": {
                        "strong": False,
                        "medium": False,
                        "weak": False,
                        "hidden": False
                    },
                    "bearish": {
                        "strong": False,
                        "medium": False,
                        "weak": False,
                        "hidden": False
                    }
                },
                "adv_divergences": {
                    "bullish": {
                        "strong": False,
                        "medium": False,
                        "weak": False,
                        "hidden": False
                    },
                    "bearish": {
                        "strong": False,
                        "medium": False,
                        "weak": False,
                        "hidden": False
                    }
                },
                "details": {},
                "trade_signals": {}
            }
            
            # Get the swings
            swing_highs = df['swing_high'].dropna()
            swing_lows = df['swing_low'].dropna()
            
            # Check if we have enough swing points to analyze
            if len(swing_highs) < 2 and len(swing_lows) < 2:
                logger.info(f"Not enough swing points for {symbol} to detect divergences (highs: {len(swing_highs)}, lows: {len(swing_lows)})")
                return divergences
            
            # =============== RSI DIVERGENCE DETECTION ===============
            # Check RSI divergences for swing lows (bullish)
            if len(swing_lows) >= 2:
                lows_indices = swing_lows.index.tolist()
                lows_indices.sort()
                
                if len(lows_indices) >= 2:
                    # Get the last two low points
                    idx1, idx2 = lows_indices[-2], lows_indices[-1]
                    
                    # Price and RSI values for comparison
                    price1, price2 = float(df.loc[idx1, 'close']), float(df.loc[idx2, 'close'])
                    rsi1, rsi2 = float(df.loc[idx1, 'rsi']), float(df.loc[idx2, 'rsi'])
                    
                    # Calculate percentage changes to determine the type
                    price_change_pct = ((price2 - price1) / price1) * 100
                    rsi_change_pct = ((rsi2 - rsi1) / rsi1) * 100 if rsi1 != 0 else 0
                    
                    # Tolerance for "equal" determination (1% change)
                    equal_tolerance = 1.0
                    
                    # Bullish Regular Divergences
                    if price2 < price1:  # Lower low in price
                        if rsi2 > rsi1:  # Higher low in RSI
                            # Strong Bullish: Lower price low, higher RSI low
                            divergences["rsi_divergences"]["bullish"]["strong"] = True
                            divergences["details"]["bullish_rsi_strong"] = {
                                "type": "Strong Bullish RSI Divergence",
                                "first_swing": {"date": idx1, "price": price1, "rsi": rsi1},
                                "second_swing": {"date": idx2, "price": price2, "rsi": rsi2}
                            }
                            
                            # Add trade signal with entry/exit levels
                            atr_value = df.loc[idx2, 'atr'] if 'atr' in df.columns else (price1 - price2) * 0.1
                            entry_price = price2  # Entry at the second swing low
                            stop_loss = price2 - (atr_value * STOP_LOSS_ATR_MULT)
                            risk_amount = entry_price - stop_loss
                            take_profit = entry_price + (risk_amount * RISK_REWARD_RATIO)
                            
                            # Get market context
                            trend = df.loc[idx2, 'trend'] if 'trend' in df.columns else "unknown"
                            market_session = df.loc[idx2, 'market_session'] if 'market_session' in df.columns else "unknown"
                            
                            # Find potential resistance level
                            resistance_level = None
                            if 'resistance' in df.columns:
                                resistance_levels = df['resistance'].dropna()
                                if not resistance_levels.empty:
                                    potential_levels = resistance_levels[resistance_levels > entry_price]
                                    if not potential_levels.empty:
                                        resistance_level = potential_levels.min()  # Closest resistance above entry
                            
                            # Adjust take profit to nearest resistance if available
                            if resistance_level is not None and resistance_level < take_profit:
                                take_profit = resistance_level
                                
                            # Calculate reward:risk ratio
                            reward_risk_ratio = (take_profit - entry_price) / risk_amount if risk_amount > 0 else 0
                            print(f"Reward/Risk Ratio: {reward_risk_ratio}")
                            current_time = df.index[-1]  # Get the latest timestamp
                            print(f"Current time: {current_time}")
                            swing_time = idx2  # Second swing point timestamp
                            print(f"Swing time: {swing_time}")
                            max_age = pd.Timedelta(minutes=5)  # Adjust this timeframe as needed
                            print(f"Max age: {max_age}")

                            # Only add trade signal if reward:risk ratio meets our criteria
                            if reward_risk_ratio >= RISK_REWARD_RATIO and (current_time - swing_time) <= max_age:
                                divergences["trade_signals"]["bullish_rsi_strong"] = {
                                    "signal_type": "BUY",
                                    "entry_price": entry_price,
                                    "stop_loss": stop_loss,
                                    "take_profit": take_profit,
                                    "risk_amount": risk_amount,
                                    "reward_amount": take_profit - entry_price,
                                    "reward_risk_ratio": reward_risk_ratio,
                                    "trend_alignment": trend == "bullish",
                                    "market_session": market_session,
                                    "signal_time": idx2,
                                    "confidence": "high"
                                }
                            
                            logger.info(f"Strong Bullish RSI divergence detected for {symbol}")
                        elif abs(rsi_change_pct) <= equal_tolerance:  # Equal RSI low (within tolerance)
                            # Medium Bullish: Lower price low, equal RSI low
                            divergences["rsi_divergences"]["bullish"]["medium"] = True
                            divergences["details"]["bullish_rsi_medium"] = {
                                "type": "Medium Bullish RSI Divergence",
                                "first_swing": {"date": idx1, "price": price1, "rsi": rsi1},
                                "second_swing": {"date": idx2, "price": price2, "rsi": rsi2}
                            }
                            logger.info(f"Medium Bullish RSI divergence detected for {symbol}")
                        elif rsi2 < rsi1 and abs(rsi_change_pct) < abs(price_change_pct):  # Lower RSI low but not as much as price
                            # Weak Bullish: Lower price low, lower RSI low but less percentage decline
                            divergences["rsi_divergences"]["bullish"]["weak"] = True
                            divergences["details"]["bullish_rsi_weak"] = {
                                "type": "Weak Bullish RSI Divergence",
                                "first_swing": {"date": idx1, "price": price1, "rsi": rsi1},
                                "second_swing": {"date": idx2, "price": price2, "rsi": rsi2}
                            }
                            logger.info(f"Weak Bullish RSI divergence detected for {symbol}")
                    elif price2 > price1:  # Higher low in price (hidden divergence)
                        if rsi2 < rsi1:  # Lower low in RSI
                            # Hidden Bullish: Higher price low, lower RSI low
                            divergences["rsi_divergences"]["bullish"]["hidden"] = True
                            divergences["details"]["bullish_rsi_hidden"] = {
                                "type": "Hidden Bullish RSI Divergence",
                                "first_swing": {"date": idx1, "price": price1, "rsi": rsi1},
                                "second_swing": {"date": idx2, "price": price2, "rsi": rsi2}
                            }
                            logger.info(f"Hidden Bullish RSI divergence detected for {symbol}")
            
            # Check RSI divergences for swing highs (bearish)
            if len(swing_highs) >= 2:
                highs_indices = swing_highs.index.tolist()
                highs_indices.sort()
                
                if len(highs_indices) >= 2:
                    # Get the last two high points
                    idx1, idx2 = highs_indices[-2], highs_indices[-1]
                    
                    # Price and RSI values for comparison
                    price1, price2 = float(df.loc[idx1, 'close']), float(df.loc[idx2, 'close'])
                    rsi1, rsi2 = float(df.loc[idx1, 'rsi']), float(df.loc[idx2, 'rsi'])
                    
                    # Calculate percentage changes to determine the type
                    price_change_pct = ((price2 - price1) / price1) * 100
                    rsi_change_pct = ((rsi2 - rsi1) / rsi1) * 100 if rsi1 != 0 else 0
                    
                    # Tolerance for "equal" determination (1% change)
                    equal_tolerance = 1.0
                    
                    # Bearish Regular Divergences
                    if price2 > price1:  # Higher high in price
                        if rsi2 < rsi1:  # Lower high in RSI
                            # Strong Bearish: Higher price high, lower RSI high
                            divergences["rsi_divergences"]["bearish"]["strong"] = True
                            divergences["details"]["bearish_rsi_strong"] = {
                                "type": "Strong Bearish RSI Divergence",
                                "first_swing": {"date": idx1, "price": price1, "rsi": rsi1},
                                "second_swing": {"date": idx2, "price": price2, "rsi": rsi2}
                            }
                            
                            # Add trade signal with entry/exit levels
                            atr_value = df.loc[idx2, 'atr'] if 'atr' in df.columns else (price2 - price1) * 0.1
                            entry_price = price2  # Entry at the second swing high
                            stop_loss = price2 + (atr_value * STOP_LOSS_ATR_MULT)
                            risk_amount = stop_loss - entry_price
                            take_profit = entry_price - (risk_amount * RISK_REWARD_RATIO)
                            
                            # Get market context
                            trend = df.loc[idx2, 'trend'] if 'trend' in df.columns else "unknown"
                            market_session = df.loc[idx2, 'market_session'] if 'market_session' in df.columns else "unknown"
                            
                            # Get current time and swing time for signal age check
                            current_time = pd.Timestamp.now()
                            swing_time = idx2
                            max_age = pd.Timedelta(minutes=5)  # Maximum age for valid signals
                            
                            # Find potential support level
                            support_level = None
                            if 'support' in df.columns:
                                support_levels = df['support'].dropna()
                                if not support_levels.empty:
                                    potential_levels = support_levels[support_levels < entry_price]
                                    if not potential_levels.empty:
                                        support_level = potential_levels.max()  # Closest support below entry
                            
                            # Adjust take profit to nearest support if available
                            if support_level is not None and support_level > take_profit:
                                take_profit = support_level
                                
                            # Calculate reward:risk ratio
                            reward_risk_ratio = (entry_price - take_profit) / risk_amount if risk_amount > 0 else 0
                            
                            # Only add trade signal if reward:risk ratio meets our criteria
                            if reward_risk_ratio >= RISK_REWARD_RATIO and (current_time - swing_time) <= max_age:
                                divergences["trade_signals"]["bearish_rsi_strong"] = {
                                    "signal_type": "SELL",
                                    "entry_price": entry_price,
                                    "stop_loss": stop_loss,
                                    "take_profit": take_profit,
                                    "risk_amount": risk_amount,
                                    "reward_amount": entry_price - take_profit,
                                    "reward_risk_ratio": reward_risk_ratio,
                                    "trend_alignment": trend == "bearish",
                                    "market_session": market_session,
                                    "signal_time": idx2,
                                    "confidence": "high"
                                }
                            
                            logger.info(f"Strong Bearish RSI divergence detected for {symbol}")
                        elif abs(rsi_change_pct) <= equal_tolerance:  # Equal RSI high (within tolerance)
                            # Medium Bearish: Higher price high, equal RSI high
                            divergences["rsi_divergences"]["bearish"]["medium"] = True
                            divergences["details"]["bearish_rsi_medium"] = {
                                "type": "Medium Bearish RSI Divergence",
                                "first_swing": {"date": idx1, "price": price1, "rsi": rsi1},
                                "second_swing": {"date": idx2, "price": price2, "rsi": rsi2}
                            }
                            logger.info(f"Medium Bearish RSI divergence detected for {symbol}")
                        elif rsi2 > rsi1 and abs(rsi_change_pct) < abs(price_change_pct):  # Higher RSI high but not as much as price
                            # Weak Bearish: Higher price high, higher RSI high but less percentage increase
                            divergences["rsi_divergences"]["bearish"]["weak"] = True
                            divergences["details"]["bearish_rsi_weak"] = {
                                "type": "Weak Bearish RSI Divergence",
                                "first_swing": {"date": idx1, "price": price1, "rsi": rsi1},
                                "second_swing": {"date": idx2, "price": price2, "rsi": rsi2}
                            }
                            logger.info(f"Weak Bearish RSI divergence detected for {symbol}")
                    elif price2 < price1:  # Lower high in price (hidden divergence)
                        if rsi2 > rsi1:  # Higher high in RSI
                            # Hidden Bearish: Lower price high, higher RSI high
                            divergences["rsi_divergences"]["bearish"]["hidden"] = True
                            divergences["details"]["bearish_rsi_hidden"] = {
                                "type": "Hidden Bearish RSI Divergence",
                                "first_swing": {"date": idx1, "price": price1, "rsi": rsi1},
                                "second_swing": {"date": idx2, "price": price2, "rsi": rsi2}
                            }
                            logger.info(f"Hidden Bearish RSI divergence detected for {symbol}")
            
            # =============== AD VOLUME DIVERGENCE DETECTION ===============
            # Check for AD divergences only if AD is present
            if 'ad' in df.columns:
                # Check AD divergences for swing lows (bullish)
                if len(swing_lows) >= 2:
                    lows_indices = swing_lows.index.tolist()
                    lows_indices.sort()
                    
                    if len(lows_indices) >= 2:
                        # Get the last two low points
                        idx1, idx2 = lows_indices[-2], lows_indices[-1]
                        
                        # Price and AD values for comparison
                        price1, price2 = float(df.loc[idx1, 'close']), float(df.loc[idx2, 'close'])
                        ad1, ad2 = float(df.loc[idx1, 'ad']), float(df.loc[idx2, 'ad'])
                        
                        # Calculate percentage changes to determine the type
                        price_change_pct = ((price2 - price1) / price1) * 100
                        ad_change_pct = ((ad2 - ad1) / max(abs(ad1), 1)) * 100  # Using max to avoid division by zero
                        
                        # Tolerance for "equal" determination (1% change)
                        equal_tolerance = 1.0
                        
                        # Bullish Regular Divergences
                        if price2 < price1:  # Lower low in price
                            if ad2 > ad1:  # Higher low in AD
                                # Strong Bullish: Lower price low, higher AD low
                                divergences["adv_divergences"]["bullish"]["strong"] = True
                                divergences["details"]["bullish_adv_strong"] = {
                                    "type": "Strong Bullish AD Volume Divergence",
                                    "first_swing": {"date": idx1, "price": price1, "ad": ad1},
                                    "second_swing": {"date": idx2, "price": price2, "ad": ad2}
                                }
                                
                                # Only add trade signal if not already added by RSI divergence
                                if "bullish_rsi_strong" not in divergences["trade_signals"]:
                                    # Add trade signal with entry/exit levels
                                    atr_value = df.loc[idx2, 'atr'] if 'atr' in df.columns else (price1 - price2) * 0.1
                                    entry_price = price2  # Entry at the second swing low
                                    stop_loss = price2 - (atr_value * STOP_LOSS_ATR_MULT)
                                    risk_amount = entry_price - stop_loss
                                    take_profit = entry_price + (risk_amount * RISK_REWARD_RATIO)
                                    
                                    # Get market context
                                    trend = df.loc[idx2, 'trend'] if 'trend' in df.columns else "unknown"
                                    market_session = df.loc[idx2, 'market_session'] if 'market_session' in df.columns else "unknown"
                                    
                                    # Calculate reward:risk ratio
                                    reward_risk_ratio = (take_profit - entry_price) / risk_amount if risk_amount > 0 else 0
                                    
                                    # Only add trade signal if reward:risk ratio meets our criteria
                                    if reward_risk_ratio >= RISK_REWARD_RATIO and (current_time - swing_time) <= max_age:
                                        divergences["trade_signals"]["bullish_adv_strong"] = {
                                            "signal_type": "BUY",
                                            "entry_price": entry_price,
                                            "stop_loss": stop_loss,
                                            "take_profit": take_profit,
                                            "risk_amount": risk_amount,
                                            "reward_amount": take_profit - entry_price,
                                            "reward_risk_ratio": reward_risk_ratio,
                                            "trend_alignment": trend == "bullish",
                                            "market_session": market_session,
                                            "signal_time": idx2,
                                            "confidence": "medium"  # Volume divergence slightly less reliable than RSI
                                        }
                                
                                logger.info(f"Strong Bullish AD Volume divergence detected for {symbol}")
                            elif abs(ad_change_pct) <= equal_tolerance:  # Equal AD low (within tolerance)
                                # Medium Bullish: Lower price low, equal AD low
                                divergences["adv_divergences"]["bullish"]["medium"] = True
                                divergences["details"]["bullish_adv_medium"] = {
                                    "type": "Medium Bullish AD Volume Divergence",
                                    "first_swing": {"date": idx1, "price": price1, "ad": ad1},
                                    "second_swing": {"date": idx2, "price": price2, "ad": ad2}
                                }
                                logger.info(f"Medium Bullish AD Volume divergence detected for {symbol}")
                        elif price2 > price1:  # Higher low in price (hidden divergence)
                            if ad2 < ad1:  # Lower low in AD
                                # Hidden Bullish: Higher price low, lower AD low
                                divergences["adv_divergences"]["bullish"]["hidden"] = True
                                divergences["details"]["bullish_adv_hidden"] = {
                                    "type": "Hidden Bullish AD Volume Divergence",
                                    "first_swing": {"date": idx1, "price": price1, "ad": ad1},
                                    "second_swing": {"date": idx2, "price": price2, "ad": ad2}
                                }
                                logger.info(f"Hidden Bullish AD Volume divergence detected for {symbol}")
                
                # Check AD divergences for swing highs (bearish)
                if len(swing_highs) >= 2:
                    highs_indices = swing_highs.index.tolist()
                    highs_indices.sort()
                    
                    if len(highs_indices) >= 2:
                        # Get the last two high points
                        idx1, idx2 = highs_indices[-2], highs_indices[-1]
                        
                        # Price and AD values for comparison
                        price1, price2 = float(df.loc[idx1, 'close']), float(df.loc[idx2, 'close'])
                        ad1, ad2 = float(df.loc[idx1, 'ad']), float(df.loc[idx2, 'ad'])
                        
                        # Calculate percentage changes to determine the type
                        price_change_pct = ((price2 - price1) / price1) * 100
                        ad_change_pct = ((ad2 - ad1) / max(abs(ad1), 1)) * 100  # Using max to avoid division by zero
                        
                        # Tolerance for "equal" determination (1% change)
                        equal_tolerance = 1.0
                        
                        # Bearish Regular Divergences
                        if price2 > price1:  # Higher high in price
                            if ad2 < ad1:  # Lower high in AD
                                # Strong Bearish: Higher price high, lower AD high
                                divergences["adv_divergences"]["bearish"]["strong"] = True
                                divergences["details"]["bearish_adv_strong"] = {
                                    "type": "Strong Bearish AD Volume Divergence",
                                    "first_swing": {"date": idx1, "price": price1, "ad": ad1},
                                    "second_swing": {"date": idx2, "price": price2, "ad": ad2}
                                }
                                
                                # Only add trade signal if not already added by RSI divergence
                                if "bearish_rsi_strong" not in divergences["trade_signals"]:
                                    # Add trade signal with entry/exit levels
                                    atr_value = df.loc[idx2, 'atr'] if 'atr' in df.columns else (price2 - price1) * 0.1
                                    entry_price = price2  # Entry at the second swing high
                                    stop_loss = price2 + (atr_value * STOP_LOSS_ATR_MULT)
                                    risk_amount = stop_loss - entry_price
                                    take_profit = entry_price - (risk_amount * RISK_REWARD_RATIO)
                                    
                                    # Get market context
                                    trend = df.loc[idx2, 'trend'] if 'trend' in df.columns else "unknown"
                                    market_session = df.loc[idx2, 'market_session'] if 'market_session' in df.columns else "unknown"
                                    
                                    # Calculate reward:risk ratio
                                    reward_risk_ratio = (entry_price - take_profit) / risk_amount if risk_amount > 0 else 0
                                    
                                    # Only add trade signal if reward:risk ratio meets our criteria
                                    if reward_risk_ratio >= RISK_REWARD_RATIO and (current_time - swing_time) <= max_age:
                                        divergences["trade_signals"]["bearish_adv_strong"] = {
                                            "signal_type": "SELL",
                                            "entry_price": entry_price,
                                            "stop_loss": stop_loss,
                                            "take_profit": take_profit,
                                            "risk_amount": risk_amount,
                                            "reward_amount": entry_price - take_profit,
                                            "reward_risk_ratio": reward_risk_ratio,
                                            "trend_alignment": trend == "bearish",
                                            "market_session": market_session,
                                            "signal_time": idx2,
                                            "confidence": "medium"  # Volume divergence slightly less reliable than RSI
                                        }
                                
                                logger.info(f"Strong Bearish AD Volume divergence detected for {symbol}")
                            elif abs(ad_change_pct) <= equal_tolerance:  # Equal AD high (within tolerance)
                                # Medium Bearish: Higher price high, equal AD high
                                divergences["adv_divergences"]["bearish"]["medium"] = True
                                divergences["details"]["bearish_adv_medium"] = {
                                    "type": "Medium Bearish AD Volume Divergence",
                                    "first_swing": {"date": idx1, "price": price1, "ad": ad1},
                                    "second_swing": {"date": idx2, "price": price2, "ad": ad2}
                                }
                                logger.info(f"Medium Bearish AD Volume divergence detected for {symbol}")
                        elif price2 < price1:  # Lower high in price (hidden divergence)
                            if ad2 > ad1:  # Higher high in AD
                                # Hidden Bearish: Lower price high, higher AD high
                                divergences["adv_divergences"]["bearish"]["hidden"] = True
                                divergences["details"]["bearish_adv_hidden"] = {
                                    "type": "Hidden Bearish AD Volume Divergence",
                                    "first_swing": {"date": idx1, "price": price1, "ad": ad1},
                                    "second_swing": {"date": idx2, "price": price2, "ad": ad2}
                                }
                                logger.info(f"Hidden Bearish AD Volume divergence detected for {symbol}")
            
            return divergences
            
        except Exception as e:
            logger.error(f"Error in detect_divergences for {symbol}: {str(e)}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            # Return empty divergence dict with proper structure
            return {
                "rsi_divergences": {
                    "bullish": {"strong": False, "medium": False, "weak": False, "hidden": False},
                    "bearish": {"strong": False, "medium": False, "weak": False, "hidden": False}
                },
                "adv_divergences": {
                    "bullish": {"strong": False, "medium": False, "weak": False, "hidden": False},
                    "bearish": {"strong": False, "medium": False, "weak": False, "hidden": False}
                },
                "details": {},
                "trade_signals": {}
            }

    def scan_symbol(self, symbol):
        """
        Scan a single symbol for divergences with classification by strength.
        """
        current_time = time.time()

            
        try:
            # Fetch historical data
            logger.info(f"Fetching data for {symbol}...")
            df = self.get_historical_data(symbol, period_type="day", period=10, frequency_type="minute", frequency=5)
            if df is None or df.empty:
                logger.warning(f"No data available for {symbol}")
                return
                
            # Log data info for debugging
            logger.info(f"Retrieved data for {symbol}: {len(df)} rows, columns: {list(df.columns)}")
            logger.info(f"Data types: {df.dtypes}")
            
            # Calculate indicators
            logger.info(f"Calculating indicators for {symbol}...")
            df_with_swings = self.calculate_indicators(df)
            if df_with_swings is None:
                logger.warning(f"Failed to calculate indicators for {symbol}")
                return
            
            logger.info(f"Calculated indicators for {symbol}, columns: {list(df_with_swings.columns)}")
                
            # Detect swings
            logger.info(f"Detecting price swings for {symbol}...")
            df_with_swings = self.detect_price_swings(df_with_swings)
            
            # Log swing count for debugging
            swing_high_count = df_with_swings['swing_high'].dropna().count()
            swing_low_count = df_with_swings['swing_low'].dropna().count()
            logger.info(f"Detected {swing_high_count} swing highs and {swing_low_count} swing lows for {symbol}")
            
            # Detect divergences
            logger.info(f"Analyzing divergences for {symbol}...")
            divergence_results = self.detect_divergences(df_with_swings, symbol)
            if divergence_results is None:
                logger.warning(f"Failed to detect divergences for {symbol}")
                return
                
            # Check for new divergences (respecting cooldown)
            new_divergences_found = False
            
            # Check RSI divergences
            for direction in ["bullish", "bearish"]:
                for div_type in ["strong", "medium", "weak", "hidden"]:
                    div_key = f"{direction}_rsi_{div_type}"
                    
                    if divergence_results["rsi_divergences"][direction][div_type]:
                        # Create the key if it doesn't exist in last_signal_time
                        if div_key not in last_signal_time[symbol]:
                            last_signal_time[symbol][div_key] = 0
                            
                        if (current_time - last_signal_time[symbol][div_key]) > COOLDOWN_PERIOD:
                            new_divergences_found = True
                            last_signal_time[symbol][div_key] = current_time
                            logger.info(f"New {div_key} divergence detected on {symbol}")
            
            # Check AD volume divergences
            for direction in ["bullish", "bearish"]:
                for div_type in ["strong", "medium", "weak", "hidden"]:
                    div_key = f"{direction}_adv_{div_type}"
                    
                    if divergence_results["adv_divergences"][direction][div_type]:
                        # Create the key if it doesn't exist in last_signal_time
                        if div_key not in last_signal_time[symbol]:
                            last_signal_time[symbol][div_key] = 0
                            
                        if (current_time - last_signal_time[symbol][div_key]) > COOLDOWN_PERIOD:
                            new_divergences_found = True
                            last_signal_time[symbol][div_key] = current_time
                            logger.info(f"New {div_key} divergence detected on {symbol}")
            
            # Check for new trade signals and send notifications before storing them
            if "trade_signals" in divergence_results and divergence_results["trade_signals"]:
                has_new_signals = False
                new_signals = []
                
                for signal_type, signal_data in divergence_results["trade_signals"].items():
                    signal_exists = False
                    for existing_signal in self.trade_signals:
                        if (existing_signal["status"] == "active" and
                            existing_signal["symbol"] == symbol and
                            existing_signal["type"] == signal_type and
                            existing_signal["data"]["entry_price"] == signal_data["entry_price"] and
                            existing_signal["data"]["stop_loss"] == signal_data["stop_loss"] and
                            existing_signal["data"]["take_profit"] == signal_data["take_profit"] and
                            existing_signal["data"]["signal_time"] == signal_data["signal_time"]):
                            signal_exists = True
                            break
                    
                    if not signal_exists:
                        has_new_signals = True
                        new_signals.append({
                            "symbol": symbol,
                            "type": signal_type,
                            "data": signal_data,
                            "timestamp": current_time,
                            "status": "active"
                        })
                        logger.info(f"New trade signal detected: {symbol} {signal_type}")
                
                # Create notifier instance for sending signals
                if has_new_signals:
                    try:
                        notifier = DivergenceNotifier()
                        notifier.process_results({symbol: {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "data": divergence_results,
                            "last_price": df_with_swings['close'].iloc[-1],
                            "last_rsi": df_with_swings['rsi'].iloc[-1]
                        }}, {symbol: df_with_swings})
                        
                        self.trade_signals.extend(new_signals)
                        logger.info(f"Added {len(new_signals)} new trade signals to tracking and sent notifications")
                    except Exception as e:
                        logger.error(f"Error sending notifications for {symbol}: {str(e)}")
                        self.trade_signals.extend(new_signals)
                        logger.info(f"Added {len(new_signals)} new trade signals to tracking (notification failed)")
                else:
                    logger.info(f"No new trade signals for {symbol}")
            
            # Store results
            self.results[symbol] = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data": divergence_results,
                "last_price": df_with_swings['close'].iloc[-1],
                "last_rsi": df_with_swings['rsi'].iloc[-1]
            }
            
            # Add AD if available
            if 'ad' in df_with_swings.columns:
                self.results[symbol]["last_ad"] = df_with_swings['ad'].iloc[-1]
                
            logger.info(f"Completed scan for {symbol}")
            
        except Exception as e:
            logger.error(f"Error scanning {symbol}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    def print_summary(self):
        """
        Print a summary of current divergence findings with classification by strength.
        Optimized for handling large numbers of symbols by showing only those with divergences.
        """
        print("\n" + "="*80)
        print(f"DIVERGENCE SCANNER SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Count statistics
        total_symbols = len(SYMBOLS)
        scanned_symbols = len(self.results)
        symbols_with_divergences = 0
        bullish_signals = 0
        bearish_signals = 0
        futures_with_divergences = []
        stocks_with_divergences = []
        
        # First print divergence results for symbols with divergences only
        for symbol in SYMBOLS:
            if symbol not in self.results:
                continue
                
            result = self.results[symbol]
            
            # Check if this symbol has any divergences
            rsi_bull = result['data']["rsi_divergences"]["bullish"]
            rsi_bear = result['data']["rsi_divergences"]["bearish"]
            adv_bull = result['data']["adv_divergences"]["bullish"]
            adv_bear = result['data']["adv_divergences"]["bearish"]
            
            has_rsi_bull = any(rsi_bull.values())
            has_rsi_bear = any(rsi_bear.values())
            has_adv_bull = any(adv_bull.values())
            has_adv_bear = any(adv_bear.values())
            
            has_any_divergence = has_rsi_bull or has_rsi_bear or has_adv_bull or has_adv_bear
            
            # Count statistics
            if has_any_divergence:
                symbols_with_divergences += 1
                
                if symbol in FUTURES_SYMBOLS:
                    futures_with_divergences.append(symbol)
                else:
                    stocks_with_divergences.append(symbol)
                
                if has_rsi_bull or has_adv_bull:
                    bullish_signals += 1
                if has_rsi_bear or has_adv_bear:
                    bearish_signals += 1
            
            # Only print details for symbols with divergences
            if has_any_divergence:
                print(f"\nSymbol: {symbol} (Price: {result['last_price']:.2f})")
                
                # Print RSI divergences
                if has_rsi_bull:
                    print("  🟢 RSI Bullish Divergences:")
                    for div_type in ["strong", "medium", "weak", "hidden"]:
                        if rsi_bull[div_type]:
                            print(f"    - {div_type.capitalize()}")
                
                if has_rsi_bear:
                    print("  🔴 RSI Bearish Divergences:")
                    for div_type in ["strong", "medium", "weak", "hidden"]:
                        if rsi_bear[div_type]:
                            print(f"    - {div_type.capitalize()}")
                
                # Print AD Volume divergences
                if has_adv_bull:
                    print("  🟢 AD Volume Bullish Divergences:")
                    for div_type in ["strong", "medium", "weak", "hidden"]:
                        if adv_bull[div_type]:
                            print(f"    - {div_type.capitalize()}")
                
                if has_adv_bear:
                    print("  🔴 AD Volume Bearish Divergences:")
                    for div_type in ["strong", "medium", "weak", "hidden"]:
                        if adv_bear[div_type]:
                            print(f"    - {div_type.capitalize()}")
                
                # Print trade signals if any
                if "trade_signals" in result['data'] and result['data']["trade_signals"]:
                    print("  💰 Trade Signals:")
                    for signal_type, signal_data in result['data']["trade_signals"].items():
                        print(f"    - {signal_type}: {signal_data['signal_type']} at {signal_data['entry_price']:.2f} "
                              f"(Stop: {signal_data['stop_loss']:.2f}, Target: {signal_data['take_profit']:.2f}, "
                              f"R:R: {signal_data['reward_risk_ratio']:.2f})")
        
        # Print summary statistics
        print("\n" + "-"*80)
        print("SUMMARY STATISTICS")
        print("-"*80)
        print(f"Total Symbols: {total_symbols}")
        print(f"Symbols Scanned: {scanned_symbols}")
        print(f"Symbols with Divergences: {symbols_with_divergences} ({(symbols_with_divergences/scanned_symbols*100) if scanned_symbols > 0 else 0:.1f}%)")
        print(f"Bullish Signals: {bullish_signals}")
        print(f"Bearish Signals: {bearish_signals}")
        
        if futures_with_divergences:
            print(f"\nFutures with Divergences: {', '.join(futures_with_divergences)}")
            
        if stocks_with_divergences:
            print(f"\nTop Stocks with Divergences: {', '.join(stocks_with_divergences[:10])}" + 
                  (f" +{len(stocks_with_divergences) - 10} more" if len(stocks_with_divergences) > 10 else ""))
        
        # Print active trade signals
        if self.trade_signals:
            print("\n" + "-"*80)
            print("ACTIVE TRADE SIGNALS")
            print("-"*80)
            
            for signal in self.trade_signals:
                if signal["status"] == "active":
                    signal_data = signal["data"]
                    print(f"{signal['symbol']} - {signal_data['signal_type']} at {signal_data['entry_price']:.2f} "
                          f"(Stop: {signal_data['stop_loss']:.2f}, Target: {signal_data['take_profit']:.2f}, "
                          f"R:R: {signal_data['reward_risk_ratio']:.2f}, "
                          f"Signal time: {signal_data['signal_time'].strftime('%Y-%m-%d %H:%M')})")
                
        print("\n" + "="*80 + "\n")

    def is_market_open(self):
        """Check if the futures market is currently open"""
        current_time = datetime.now().strftime("%H:%M")
        current_day = datetime.now().strftime("%A")
        
        # Convert market hours to datetime objects for comparison
        open_time = datetime.strptime(MARKET_HOURS["FUTURES_OPEN"], "%H:%M").time()
        close_time = datetime.strptime(MARKET_HOURS["FUTURES_CLOSE"], "%H:%M").time()
        current = datetime.strptime(current_time, "%H:%M").time()
        
        # Market opens Sunday at open_time
        if current_day == "Sunday" and current >= open_time:
            return True
        # Market closes Friday at close_time
        elif current_day == "Friday" and current <= close_time:
            return True
        # Market is open Monday-Thursday
        elif current_day in ["Monday", "Tuesday", "Wednesday", "Thursday"]:
            return True
            
        return False

    def run_scanner(self):
        """
        Main scanner loop to continuously check all symbols using parallel processing.
        """
        logger.info("Starting parallel divergence scanner for S&P 500 stocks and futures...")
        logger.info(f"Using {MAX_WORKERS} parallel workers for scanning")
        
        # Try to dynamically get the S&P 500 symbols, otherwise use the predefined list
        try:
            sp500_symbols = get_sp500_symbols()
            if len(sp500_symbols) > 50:  # Sanity check that we got a good list
                # Update SYMBOLS list but keep the futures first
                global SYMBOLS
                SYMBOLS = FUTURES_SYMBOLS + sp500_symbols
                logger.info(f"Updated symbols list with {len(SYMBOLS)} symbols")
                
                # Initialize any new symbols in last_signal_time
                for symbol in SYMBOLS:
                    if symbol not in last_signal_time:
                        last_signal_time[symbol] = {}
                        # RSI divergences
                        for direction in ["bullish", "bearish"]:
                            for strength in ["strong", "medium", "weak", "hidden"]:
                                last_signal_time[symbol][f"{direction}_rsi_{strength}"] = 0
                        
                        # AD Volume divergences
                        for direction in ["bullish", "bearish"]:
                            for strength in ["strong", "medium", "weak", "hidden"]:
                                last_signal_time[symbol][f"{direction}_adv_{strength}"] = 0
        except Exception as e:
            logger.error(f"Error updating S&P 500 symbols: {e}")
            # Continue with the predefined list
        
        while not self.stop_event.is_set():
            try:
                if not self.is_market_open():
                    logger.info("Markets are closed. Waiting for market open...")
                    time.sleep(60)  # Check every minute
                    continue
                
                # Process symbols in batches to avoid rate limiting and memory issues
                symbol_batches = [SYMBOLS[i:i + BATCH_SIZE] for i in range(0, len(SYMBOLS), BATCH_SIZE)]
                
                # First prioritize futures and then stocks
                scan_start_time = time.time()
                total_symbols_processed = 0
                
                for batch_idx, batch in enumerate(symbol_batches):
                    if self.stop_event.is_set():
                        break
                        
                    logger.info(f"Processing batch {batch_idx+1}/{len(symbol_batches)} with {len(batch)} symbols in parallel")
                    
                    # Use ThreadPoolExecutor to process symbols in parallel
                    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                        # Submit all symbols in the batch to the thread pool
                        future_to_symbol = {executor.submit(self.scan_symbol, symbol): symbol for symbol in batch}
                        
                        # Process results as they complete
                        for future in concurrent.futures.as_completed(future_to_symbol):
                            symbol = future_to_symbol[future]
                            total_symbols_processed += 1
                            
                            try:
                                # The scan_symbol method doesn't return anything, it updates self.results directly
                                future.result()
                                logger.info(f"Completed parallel scanning for {symbol}")
                            except Exception as exc:
                                logger.error(f"Symbol {symbol} generated an exception: {exc}")
                    
                    # Small delay between batches to avoid overloading the API
                    if not self.stop_event.is_set() and batch_idx < len(symbol_batches) - 1:
                        logger.info(f"Completed batch {batch_idx+1}, pausing before next batch...")
                        time.sleep(.5)  # 2 second pause between batches
                
                scan_duration = time.time() - scan_start_time
                logger.info(f"Parallel scan complete. Processed {total_symbols_processed} symbols in {scan_duration:.2f} seconds")
                
                # Print summary of current results
                self.print_summary()
                
                # Calculate wait time until next interval based on stored frequency
                wait_seconds = get_next_interval(frequency=self.last_data_frequency)
                logger.info(f"Waiting {wait_seconds:.1f} seconds until next scan at {(datetime.now() + timedelta(seconds=wait_seconds)).strftime('%H:%M:%S')}")
                
                # Check for stop event during wait with smaller increments
                for _ in range(int(wait_seconds)):
                    if self.stop_event.is_set():
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in scanner main loop: {str(e)}")
                # Wait a bit before retrying
                time.sleep(60)
        
        logger.info("Scanner stopped.")



    def stop(self):
        """
        Signal the scanner to stop.
        """
        logger.info("Stopping scanner...")
        self.stop_event.set()


def main():
    scanner = DivergenceScanner()
    
    try:
        # Run the scanner in the main thread
        scanner.run_scanner()
    except KeyboardInterrupt:
        logger.info("Scanner interrupted by user")
    finally:
        scanner.stop()
        logger.info("Scanner shutdown complete")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Divergence Scanner for Futures and Stocks')
    parser.add_argument('--mode', choices=['futures', 'stocks', 'both'], default='both',
                      help='Scan mode: futures only, stocks only, or both (default: both)')
    
    args = parser.parse_args()
    
    # Set symbols based on mode
    if args.mode == 'futures':
        SYMBOLS = FUTURES_SYMBOLS
        print("Running Futures-only scanner...")
    elif args.mode == 'stocks':
        SYMBOLS = SP500_SYMBOLS
        print("Running Stocks-only scanner...")
    # else both - SYMBOLS already contains both
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")
    print(f"Scanning {len(SYMBOLS)} symbols")
    print(f"First 10 symbols: {SYMBOLS[:10]}")
    
    try:
        print("Initializing scanner...")
        scanner = DivergenceScanner()
        print("Scanner initialized successfully. Starting scan...")
        scanner.run_scanner()
    except KeyboardInterrupt:
        logger.info("Scanner interrupted by user")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
    finally:
        scanner.stop()
        logger.info("Scanner shutdown complete")

#!/usr/bin/env python3
"""
Stock Divergence Calculator
Detects price and indicator divergences in market data using modular connections
"""
import json
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
import logging
import threading
import pytz
from scipy.signal import argrelextrema
from typing import Dict, Any, List, Optional

# Import the modular API components
from modules import SchwabapiAuth, SchwabapiMarketData, SchwabapiAccount, SchwabapiTrading

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

# Scanning parameters
CHECK_INTERVAL = 5  # Check every 5 SECONDS
# Focus on S&P 500 stocks and major tech stocks
SYMBOLS = ['NVDA', 'AAPL', 'AMZN', 'GOOGL', 'MSFT', 'TSLA', 'META', 'NFLX', 'AMD', 'PLTR']  # Add more symbols as needed

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
SWING_LOOKBACK = 1  # Reduced to catch more recent swings
DIVERGENCE_THRESHOLD = 0.03  # Reduced threshold for more sensitive divergence detection
MIN_SWING_PERCENT = 0.1  # Reduced to catch smaller but significant swings

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
RISK_REWARD_RATIO = 2.0  # Minimum reward:risk ratio
MAX_RISK_PERCENT = 1.0   # Maximum risk percentage per trade
STOP_LOSS_ATR_MULT = 1.5 # Stop loss multiplier based on ATR

class DivergenceScanner:
    def __init__(self, auto_trade=False, risk_per_trade=0.5, shares=None):
        # Initialize authentication and API modules
        self.auth = SchwabapiAuth()
        self.market_data = SchwabapiMarketData(self.auth)
        self.account = SchwabapiAccount(self.auth)
        self.trading = SchwabapiTrading(self.auth, self.account)
        
        self.historical_data = {}
        self.results = {}
        self.stop_event = threading.Event()
        self.trade_signals = []  # Track active trade signals
        self.logger = logging.getLogger(__name__)
        
        # Trading configuration
        self.auto_trade = auto_trade
        self.risk_per_trade = risk_per_trade  # Max risk per trade as percentage of account
        self.shares = shares  # Fixed share size if specified
        
        # Position tracking
        self.positions = {}  # Track positions by symbol
        self.active_orders = {}  # Track active orders by symbol
        
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
            
            # Initialize position tracking for each symbol
            self.positions[symbol] = {
                "status": "FLAT",
                "size": 0,
                "entry_price": 0,
                "entry_time": None,
                "stop_loss": 0,
                "take_profit": 0,
                "profit_target_order_id": None
            }
        
        logger.info("Enhanced Divergence Scanner initialized for symbols: %s", ", ".join(SYMBOLS))
        
        # Sync positions at startup
        self.sync_all_positions()

    def sync_all_positions(self):
        """Sync positions for all tracked symbols."""
        try:
            for symbol in SYMBOLS:
                self.sync_position(symbol)
        except Exception as e:
            logger.error(f"Error syncing positions: {e}")
    
    def sync_position(self, symbol):
        """Get current position for a symbol and update internal tracking."""
        try:
            # Use the trading module to get position for the symbol
            position = self.trading.sync_positions(symbol)
            
            # Check if position exists and update our tracking variables
            position_size = abs(position.get('quantity', 0)) if position else 0
            
            if position_size <= 0:
                self.positions[symbol] = {
                    "status": "FLAT",
                    "size": 0,
                    "entry_price": 0,
                    "entry_time": None,
                    "stop_loss": 0,
                    "take_profit": 0,
                    "profit_target_order_id": None
                }
                return
            
            # Set position type based on quantity
            if position.get('quantity', 0) > 0:
                status = "LONG"
            else:
                status = "SHORT"
                
            # Update position tracking
            self.positions[symbol] = {
                "status": status,
                "size": position_size,
                "entry_price": position.get('average_price', 0),
                "entry_time": datetime.now(),
                "stop_loss": position.get('stop_loss', 0),
                "take_profit": position.get('take_profit', 0),
                "profit_target_order_id": position.get('profit_target_order_id')
            }
            
            logger.info(f"Synced position for {symbol}: {status} {position_size} shares at ${position.get('average_price', 0):.2f}")
            
        except Exception as e:
            logger.error(f"Error syncing position for {symbol}: {e}")
    
    def execute_trade_signal(self, symbol, signal):
        """Execute a trade based on the generated signal."""
        if not self.auto_trade:
            logger.info(f"Trade signal for {symbol}: {signal['signal_type']} at ${signal['entry_price']:.2f} (Auto-trading disabled)")
            return False
            
        try:
            # Check if we have an opposite position we need to close
            current_position = self.positions[symbol]["status"]
            
            if (signal['signal_type'] == "BUY" and current_position == "SHORT") or \
               (signal['signal_type'] == "SELL" and current_position == "LONG"):
                # Close opposite position first
                logger.info(f"Closing opposite {current_position} position before opening new {signal['signal_type']} position")
                success, message = self.trading.close_position(symbol)
                if not success:
                    logger.error(f"Failed to close opposite position: {message}")
                    return False
                # Reset position tracking
                self.positions[symbol]["status"] = "FLAT"
                self.positions[symbol]["size"] = 0
                
            # Calculate position size based on risk parameters if no fixed shares
            if self.shares:
                quantity = self.shares
            else:
                # Get account balance
                account_data = self.account.get_account_data()
                if not account_data:
                    logger.error("Failed to get account data for position sizing")
                    return False
                    
                # Use the first account's equity
                equity = account_data[0]['balances']['equity']
                risk_amount = equity * (self.risk_per_trade / 100)  # Convert percentage to decimal
                
                # Calculate shares based on risk amount and stop loss distance
                price_risk = abs(signal['entry_price'] - signal['stop_loss'])
                if price_risk <= 0:
                    logger.error(f"Invalid stop loss distance for {symbol}")
                    price_risk = signal['entry_price'] * 0.01  # Default to 1% risk
                    
                quantity = max(1, int(risk_amount / price_risk))
                
            # Execute the trade based on signal type
            if signal['signal_type'] == "BUY":
                # Place a buy market order
                if current_position == "LONG":
                    # Scale into existing position
                    success, message, result = self.trading.scale_into_position(symbol, "LONG", quantity)
                else:
                    # Open new position
                    result = self.trading.place_market_order(symbol, quantity, "BUY")
                    success = result.get("success", False)
                    message = f"Opened LONG position: {quantity} shares"
                    
            elif signal['signal_type'] == "SELL":
                # Place a sell short market order
                if current_position == "SHORT":
                    # Scale into existing position
                    success, message, result = self.trading.scale_into_position(symbol, "SHORT", quantity)
                else:
                    # Open new position
                    result = self.trading.place_market_order(symbol, quantity, "SELL_SHORT")
                    success = result.get("success", False)
                    message = f"Opened SHORT position: {quantity} shares"
                    
            if success:
                logger.info(f"Trade executed for {symbol}: {message}")
                
                # Set stop loss order
                self.set_stop_loss(symbol, signal['signal_type'], signal['entry_price'], signal['stop_loss'])
                
                # Set take profit order
                self.set_take_profit(symbol, signal['signal_type'], signal['entry_price'], signal['take_profit'])
                
                # Update position tracking and sync with actual position
                self.sync_position(symbol)
                return True
            else:
                error_msg = result.get("error", "Unknown error") if isinstance(result, dict) else "Unknown error"
                logger.error(f"Failed to execute trade for {symbol}: {error_msg}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def set_stop_loss(self, symbol, signal_type, entry_price, stop_price):
        """Set a stop loss order for the position."""
        try:
            # Validate the position exists
            if self.positions[symbol]["status"] == "FLAT" or self.positions[symbol]["size"] <= 0:
                logger.warning(f"No position found for {symbol} when setting stop loss")
                return False
                
            # Adjust stop price if needed
            if signal_type == "BUY" and stop_price >= entry_price:
                logger.warning(f"Invalid stop loss price for LONG position: ${stop_price:.2f} >= ${entry_price:.2f}")
                stop_price = entry_price * 0.99  # Default to 1% below entry
                
            if signal_type == "SELL" and stop_price <= entry_price:
                logger.warning(f"Invalid stop loss price for SHORT position: ${stop_price:.2f} <= ${entry_price:.2f}")
                stop_price = entry_price * 1.01  # Default to 1% above entry
                
            # Update position tracking
            self.positions[symbol]["stop_loss"] = stop_price
            
            logger.info(f"Set stop loss for {symbol} at ${stop_price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting stop loss for {symbol}: {e}")
            return False
    
    def set_take_profit(self, symbol, signal_type, entry_price, target_price):
        """Set a take profit order for the position."""
        try:
            # Validate the position exists
            if self.positions[symbol]["status"] == "FLAT" or self.positions[symbol]["size"] <= 0:
                logger.warning(f"No position found for {symbol} when setting take profit")
                return False
                
            # Get current position info
            position_size = self.positions[symbol]["size"]
            position_type = self.positions[symbol]["status"]
            
            # Cancel any existing profit target order
            profit_target_order_id = self.positions[symbol]["profit_target_order_id"]
            if profit_target_order_id:
                self.trading.cancel_order(profit_target_order_id)
                self.positions[symbol]["profit_target_order_id"] = None
                
            # Validate take profit price
            if position_type == "LONG" and target_price <= entry_price:
                logger.warning(f"Invalid take profit price for LONG position: ${target_price:.2f} <= ${entry_price:.2f}")
                target_price = entry_price * 1.02  # Default to 2% above entry
                
            if position_type == "SHORT" and target_price >= entry_price:
                logger.warning(f"Invalid take profit price for SHORT position: ${target_price:.2f} >= ${entry_price:.2f}")
                target_price = entry_price * 0.98  # Default to 2% below entry
                
            # Set appropriate order type
            order_type = "SELL" if position_type == "LONG" else "BUY_TO_COVER"
            
            # Place limit order for take profit
            result = self.trading.place_limit_order(symbol, position_size, target_price, order_type)
            
            if result.get("success", False):
                # Update position tracking
                self.positions[symbol]["take_profit"] = target_price
                self.positions[symbol]["profit_target_order_id"] = result.get("orderId")
                logger.info(f"Set take profit for {symbol} at ${target_price:.2f}")
                return True
            else:
                error_msg = result.get("error", "Unknown error")
                logger.error(f"Failed to set take profit order for {symbol}: {error_msg}")
                return False
                
        except Exception as e:
            logger.error(f"Error setting take profit for {symbol}: {e}")
            return False
            
    def get_historical_data(self, symbol, period_type="day", period=10, frequency_type="minute", frequency=15):
        """
        Fetch historical price data for a given symbol using the SchwabapiMarketData module.
        Returns DataFrame with OHLCV data.
        """
        try:
            logger.debug(f"Fetching historical data for {symbol} with frequency {frequency}")
            
            # Use the market_data module to fetch price history
            df = self.market_data.get_price_history(
                symbol=symbol,
                period_type=period_type,
                period=period,
                frequency_type=frequency_type,
                frequency=frequency,
                extended_hours=True
            )
            
            if df is not None and not df.empty:
                logger.info(f"Fetched {len(df)} candles for {symbol}")
                return df
            else:
                logger.warning(f"No historical data returned for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None

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
                            
                            # Check if signal is within last 5 bars and meets reward:risk criteria
                            current_idx = len(df) - 1
                            signal_idx = df.index.get_loc(idx2)
                            bars_ago = current_idx - signal_idx
                            
                            if bars_ago <= 5 and reward_risk_ratio >= RISK_REWARD_RATIO:
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
                            
                            # Check if signal is within last 5 bars and meets reward:risk criteria
                            current_idx = len(df) - 1
                            signal_idx = df.index.get_loc(idx2)
                            bars_ago = current_idx - signal_idx
                            
                            if bars_ago <= 5 and reward_risk_ratio >= RISK_REWARD_RATIO:
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
                                    
                                    # Check if signal is within last 5 bars and meets reward:risk criteria
                                    current_idx = len(df) - 1
                                    signal_idx = df.index.get_loc(idx2)
                                    bars_ago = current_idx - signal_idx
                                    
                                    if bars_ago <= 5 and reward_risk_ratio >= RISK_REWARD_RATIO:
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
                                    
                                    # Check if signal is within last 5 bars and meets reward:risk criteria
                                    current_idx = len(df) - 1
                                    signal_idx = df.index.get_loc(idx2)
                                    bars_ago = current_idx - signal_idx
                                    
                                    if bars_ago <= 5 and reward_risk_ratio >= RISK_REWARD_RATIO:
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
            df = self.get_historical_data(symbol, period_type="day", period=10, frequency_type="minute", frequency=15)
            if df is None or df.empty:
                logger.warning(f"No data available for {symbol}")
                return
                
            # Log data info for debugging
            logger.info(f"Retrieved data for {symbol}: {len(df)} rows, columns: {list(df.columns)}")
            
            # Calculate indicators
            logger.info(f"Calculating indicators for {symbol}...")
            df_with_indicators = self.calculate_indicators(df)
            if df_with_indicators is None:
                logger.warning(f"Failed to calculate indicators for {symbol}")
                return
            
            logger.info(f"Calculated indicators for {symbol}, columns: {list(df_with_indicators.columns)}")
                
            # Detect swings
            logger.info(f"Detecting price swings for {symbol}...")
            df_with_swings = self.detect_price_swings(df_with_indicators)
            
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
            
            # Process trade signals if any were detected
            if divergence_results and "trade_signals" in divergence_results and divergence_results["trade_signals"]:
                logger.info(f"Found {len(divergence_results['trade_signals'])} trade signals for {symbol}")
                
                # Execute each trade signal
                for signal_name, signal_data in divergence_results["trade_signals"].items():
                    # Log the signal 
                    logger.info(f"Signal: {signal_name} - {signal_data['signal_type']} at {signal_data['entry_price']:.2f}")
                    logger.info(f"  Stop: {signal_data['stop_loss']:.2f}, Target: {signal_data['take_profit']:.2f}, R:R: {signal_data['reward_risk_ratio']:.2f}")
                    
                    # Execute the trade
                    if self.execute_trade_signal(symbol, signal_data):
                        logger.info(f"Successfully executed {signal_data['signal_type']} trade for {symbol}")
                    else:
                        logger.info(f"Trade execution skipped or failed for {symbol}")
                
            logger.info(f"Completed scan for {symbol}")
            
        except Exception as e:
            logger.error(f"Error scanning {symbol}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    def print_summary(self):
        """
        Print a summary of current divergence findings with classification by strength.
        """
        print("\n" + "="*80)
        print(f"DIVERGENCE SCANNER SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # First print divergence results
        for symbol in SYMBOLS:
            if symbol in self.results:
                result = self.results[symbol]
                print(f"\nSymbol: {symbol} (Price: {result['last_price']:.2f})")
                
                # Print RSI divergences
                rsi_bull = result['data']["rsi_divergences"]["bullish"]
                rsi_bear = result['data']["rsi_divergences"]["bearish"]
                
                has_rsi_bull = any(rsi_bull.values())
                has_rsi_bear = any(rsi_bear.values())
                
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
                adv_bull = result['data']["adv_divergences"]["bullish"]
                adv_bear = result['data']["adv_divergences"]["bearish"]
                
                has_adv_bull = any(adv_bull.values())
                has_adv_bear = any(adv_bear.values())
                
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
                
                if not (has_rsi_bull or has_rsi_bear or has_adv_bull or has_adv_bear):
                    print("  No divergences detected")
            else:
                print(f"\nSymbol: {symbol} - No data available")
        
        print("\n" + "="*80 + "\n")

    def is_market_open(self):
        """Check if the market is currently open using the market_data module"""
        return self.market_data.is_market_open()

    def run_scanner(self):
        """
        Main scanner loop to continuously check all symbols.
        """
        logger.info("Starting divergence scanner...")
        
        while not self.stop_event.is_set():
            try:
                if not self.is_market_open():
                    logger.info("Markets are closed. Waiting for market open...")
                    time.sleep(60)  # Check every minute
                    continue
                    
                for symbol in SYMBOLS:
                    logger.info(f"Scanning {symbol} for divergences...")
                    self.scan_symbol(symbol)
                    
                    # Check if stop requested
                    if self.stop_event.is_set():
                        break
                    
                    # Small delay between symbols
                    time.sleep(1)
                
                # Print summary of current results
                self.print_summary()
                
                # Wait until next check interval
                logger.info(f"Scan complete. Waiting {CHECK_INTERVAL//60} minutes until next scan...")
                
                # Check for stop event during wait with smaller increments
                for _ in range(CHECK_INTERVAL):
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
    import argparse
    import sys
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Stock Divergence Scanner with Trade Execution')
    parser.add_argument('--auto', type=str, choices=['yes', 'no'], default='no',
                      help='Enable auto-trading mode (yes/no). Default: no')
    parser.add_argument('--risk', type=float, default=0.5,
                      help='Risk percentage per trade (default: 0.5%%)')
    parser.add_argument('--shares', type=int,
                      help='Fixed number of shares per trade (optional)')
    
    args = parser.parse_args()
    
    # Convert auto-trade argument to boolean
    auto_trade = args.auto.lower() == 'yes'
    
    # Validate parameters
    if auto_trade and args.shares is None and args.risk <= 0:
        print("Error: When auto-trading is enabled, either --shares or --risk (> 0) must be specified")
        sys.exit(1)
    
    # Print header with parameters
    print("\n" + "=" * 80)
    print("STOCK DIVERGENCE SCANNER WITH TRADE EXECUTION")
    print("=" * 80)
    print("PARAMETERS:")
    print(f"  Auto-Trading: {'ENABLED' if auto_trade else 'DISABLED'}")
    if args.shares:
        print(f"  Fixed Shares: {args.shares} shares per trade")
    else:
        print(f"  Risk Per Trade: {args.risk}% of account equity")
    print(f"  Risk/Reward Ratio: {RISK_REWARD_RATIO}")
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print("=" * 80 + "\n")
    
    # Create and run the scanner with the specified parameters
    scanner = DivergenceScanner(
        auto_trade=auto_trade,
        risk_per_trade=args.risk,
        shares=args.shares
    )
    
    try:
        # Run the scanner in the main thread
        scanner.run_scanner()
    except KeyboardInterrupt:
        logger.info("Scanner interrupted by user")
    finally:
        scanner.stop()
        logger.info("Scanner shutdown complete")


if __name__ == "__main__":
    import time
    main()

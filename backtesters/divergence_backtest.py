#!/usr/bin/env python3
"""
Backtest script for the Stock Divergence Calculator strategy
Uses historical data to test divergence-based trading signals
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
from historical_data_handler import HistoricalDataHandler
from scipy.signal import argrelextrema
import time
import subprocess

class DivergenceBacktester:
    def __init__(self):
        self.data_handler = HistoricalDataHandler()
        self.output_dir = 'historical_data'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Strategy parameters
        self.RSI_PERIOD = 14
        self.ADV_PERIOD = 14
        self.MACD_FAST = 12
        self.MACD_SLOW = 26
        self.MACD_SIGNAL = 9
        self.EMA_SHORT = 9
        self.EMA_MEDIUM = 21
        self.EMA_LONG = 50
        self.BOLLINGER_PERIOD = 20
        self.BOLLINGER_STD = 2
        self.ATR_PERIOD = 14
        
        # Divergence parameters 
        self.SWING_LOOKBACK = 10  # Lookback for nearby swings
        self.DIVERGENCE_THRESHOLD = 0.05
        self.MIN_SWING_PERCENT = .2  # Minimum swing size as a percentage


        
        # Trading strategy parameters
        self.RISK_REWARD_RATIO = 2.5
        self.MAX_RISK_PERCENT = 5
        self.STOP_LOSS_ATR_MULT = 1.5
    
    def calculate_performance_summary(self, csv_file_path):
        """Calculate and print performance summary from CSV data"""
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file_path)
            
            # Filter for completed trades (rows with PnL values)
            trades = df[df['pnl'].notna() & (df['pnl'] != 0)]
            
            if len(trades) == 0:
                print("\n📊 PERFORMANCE SUMMARY")
                print("=" * 50)
                print("No completed trades found in the data.")
                return
            
            # Basic trade statistics
            total_trades = len(trades)
            winning_trades = len(trades[trades['pnl'] > 0])
            losing_trades = len(trades[trades['pnl'] < 0])
            breakeven_trades = len(trades[trades['pnl'] == 0])
            
            # PnL calculations
            total_pnl = trades['pnl'].sum()
            avg_pnl_per_trade = trades['pnl'].mean()
            max_win = trades['pnl'].max()
            max_loss = trades['pnl'].min()
            
            # Win rate
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            # Account metrics (starting with $25,000, using 1% per trade = $250)
            initial_capital = 25000.0
            risk_per_trade = initial_capital * (self.MAX_RISK_PERCENT / 100)
            
            # Calculate ROI
            roi = (total_pnl / initial_capital) * 100
            
            # Calculate average returns for Sharpe ratio
            returns = trades['pnl'] / risk_per_trade  # Returns as percentage of risk capital
            avg_return = returns.mean()
            return_std = returns.std()
            
            # Sharpe ratio (assuming risk-free rate of 0 for simplicity)
            sharpe_ratio = avg_return / return_std if return_std != 0 else 0
            
            # Maximum drawdown calculation
            cumulative_pnl = trades['pnl'].cumsum()
            running_max = cumulative_pnl.expanding().max()
            drawdown = cumulative_pnl - running_max
            max_drawdown = drawdown.min()
            max_drawdown_pct = (max_drawdown / initial_capital) * 100
            
            # Profit factor
            gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
            gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
            
            # Get date range from the data
            if df['datetime'].dtype == 'object':
                df['datetime'] = pd.to_datetime(df['datetime'])
            
            date_range_start = df['datetime'].min().strftime('%Y-%m-%d %H:%M:%S')
            date_range_end = df['datetime'].max().strftime('%Y-%m-%d %H:%M:%S')
            
            # Print performance summary
            print("\n📊 DIVERGENCE STRATEGY PERFORMANCE SUMMARY")
            print("=" * 60)
            print(f"📅 DATE RANGE: {date_range_start} to {date_range_end}")
            print(f"📈 TRADE STATISTICS")
            print(f"   Total Trades: {total_trades}")
            print(f"   Winning Trades: {winning_trades}")
            print(f"   Losing Trades: {losing_trades}")
            print(f"   Breakeven Trades: {breakeven_trades}")
            print(f"   Win Rate: {win_rate:.1f}%")
            
            print(f"\n💰 PROFIT & LOSS")
            print(f"   Total PnL: ${total_pnl:.2f}")
            print(f"   Average PnL per Trade: ${avg_pnl_per_trade:.2f}")
            print(f"   Largest Win: ${max_win:.2f}")
            print(f"   Largest Loss: ${max_loss:.2f}")
            print(f"   Gross Profit: ${gross_profit:.2f}")
            print(f"   Gross Loss: ${gross_loss:.2f}")
            
            print(f"\n📊 PERFORMANCE METRICS")
            print(f"   Initial Capital: ${initial_capital:,.2f}")
            print(f"   Final Capital: ${initial_capital + total_pnl:,.2f}")
            print(f"   ROI: {roi:.2f}%")
            print(f"   Profit Factor: {profit_factor:.2f}")
            print(f"   Sharpe Ratio: {sharpe_ratio:.3f}")
            print(f"   Maximum Drawdown: ${max_drawdown:.2f} ({max_drawdown_pct:.2f}%)")
            
            # Divergence type analysis
            if 'divergence_type' in trades.columns:
                print(f"\n🔍 DIVERGENCE TYPE ANALYSIS")
                div_performance = trades.groupby('divergence_type').agg({
                    'pnl': ['count', 'sum', 'mean'],
                    'signal_type': 'first'
                }).round(2)
                
                for div_type in div_performance.index:
                    count = div_performance.loc[div_type, ('pnl', 'count')]
                    total_pnl = div_performance.loc[div_type, ('pnl', 'sum')]
                    avg_pnl = div_performance.loc[div_type, ('pnl', 'mean')]
                    signal_type = div_performance.loc[div_type, ('signal_type', 'first')]
                    print(f"   {div_type}: {count} trades, ${total_pnl:.2f} total, ${avg_pnl:.2f} avg ({signal_type})")
            
            # Additional insights
            if total_trades > 0:
                print(f"\n🔍 ADDITIONAL INSIGHTS")
                print(f"   Risk per Trade: ${risk_per_trade:.2f} ({self.MAX_RISK_PERCENT:.1f}% of capital)")
                print(f"   Risk/Reward Ratio Target: {self.RISK_REWARD_RATIO}:1")
                
                if win_rate > 50:
                    print(f"   ✅ Strategy shows positive win rate")
                else:
                    print(f"   ⚠️  Strategy has win rate below 50%")
                
                if roi > 0:
                    print(f"   ✅ Strategy is profitable overall")
                else:
                    print(f"   ❌ Strategy shows net loss")
                
                if sharpe_ratio > 1:
                    print(f"   ✅ Good risk-adjusted returns (Sharpe > 1)")
                elif sharpe_ratio > 0:
                    print(f"   ⚠️  Moderate risk-adjusted returns")
                else:
                    print(f"   ❌ Poor risk-adjusted returns")
            
            print("=" * 60)
            
        except Exception as e:
            print(f"Error calculating performance summary: {str(e)}")
    
    def calculate_indicators(self, df):
        """Calculate technical indicators for divergence detection"""
        if df is None or len(df) < 50:
            return None
        
        df_indicators = df.copy()
        
        # Convert data to proper type for TA-Lib
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df_indicators[col] = df_indicators[col].astype(float)
            
        # Calculate RSI
        df_indicators['rsi'] = talib.RSI(df_indicators['close'].values, timeperiod=self.RSI_PERIOD)
        
        # Calculate AD (Accumulation/Distribution) Line
        df_indicators['ad'] = talib.AD(
            df_indicators['high'].values,
            df_indicators['low'].values,
            df_indicators['close'].values,
            df_indicators['volume'].values
        )
        
        # Calculate OBV (On-Balance Volume)
        df_indicators['obv'] = talib.OBV(df_indicators['close'].values, df_indicators['volume'].values)
        
        # Calculate MACD
        macd, macd_signal, macd_hist = talib.MACD(
            df_indicators['close'].values,
            fastperiod=self.MACD_FAST,
            slowperiod=self.MACD_SLOW,
            signalperiod=self.MACD_SIGNAL
        )
        df_indicators['macd'] = macd
        df_indicators['macd_signal'] = macd_signal
        df_indicators['macd_hist'] = macd_hist
        
        # Calculate EMAs for trend context
        df_indicators['ema_short'] = talib.EMA(df_indicators['close'].values, timeperiod=self.EMA_SHORT)
        df_indicators['ema_medium'] = talib.EMA(df_indicators['close'].values, timeperiod=self.EMA_MEDIUM)
        df_indicators['ema_long'] = talib.EMA(df_indicators['close'].values, timeperiod=self.EMA_LONG)
        
        # Calculate Bollinger Bands
        df_indicators['bb_upper'], df_indicators['bb_middle'], df_indicators['bb_lower'] = talib.BBANDS(
            df_indicators['close'].values, 
            timeperiod=self.BOLLINGER_PERIOD,
            nbdevup=self.BOLLINGER_STD,
            nbdevdn=self.BOLLINGER_STD
        )
        
        # Calculate ATR for volatility and stop loss/take profit
        df_indicators['atr'] = talib.ATR(
            df_indicators['high'].values,
            df_indicators['low'].values,
            df_indicators['close'].values,
            timeperiod=self.ATR_PERIOD
        )
        
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
        
        return df_indicators

    def detect_price_swings(self, df, min_points=5):
        """Detect significant price swing highs and lows"""
        try:
            df_swings = df.copy()
            
            # Ensure close is float type
            df_swings['close'] = df_swings['close'].astype(float)
            
            # Find local maxima and minima using a smaller order for more sensitivity
            order = max(1, min_points // 2)  # Use smaller order for better detection
            high_indices = argrelextrema(df_swings['close'].values, np.greater, order=order)[0]
            low_indices = argrelextrema(df_swings['close'].values, np.less, order=order)[0]
            
            # Initialize swing columns with NaN
            df_swings['swing_high'] = np.nan
            df_swings['swing_low'] = np.nan
            
            # Assign swing values
            if len(high_indices) > 0:
                df_swings.loc[df_swings.index[high_indices], 'swing_high'] = df_swings.iloc[high_indices]['close'].values
            
            if len(low_indices) > 0:
                df_swings.loc[df_swings.index[low_indices], 'swing_low'] = df_swings.iloc[low_indices]['close'].values
            
            # Apply a more reasonable filtering approach
            # Only filter out swings that are too small relative to recent price action
            if len(df_swings) > 10:  # Only apply filtering if we have enough data
                # Calculate recent volatility for filtering
                recent_high = df_swings['close'].tail(20).max()
                recent_low = df_swings['close'].tail(20).min()
                recent_range = recent_high - recent_low
                min_swing_threshold = recent_range * (self.MIN_SWING_PERCENT / 100)
                
                # Filter swings that are too small
                swing_highs = df_swings['swing_high'].dropna()
                swing_lows = df_swings['swing_low'].dropna()
                
                for idx in swing_highs.index:
                    swing_high_price = df_swings.loc[idx, 'swing_high']
                    # Find nearby lows within lookback window
                    start_idx = max(0, idx - self.SWING_LOOKBACK)
                    end_idx = min(len(df_swings), idx + self.SWING_LOOKBACK + 1)
                    nearby_lows = df_swings.loc[start_idx:end_idx, 'swing_low'].dropna()
                    
                    if len(nearby_lows) > 0:
                        min_nearby_low = nearby_lows.min()
                        if (swing_high_price - min_nearby_low) < min_swing_threshold:
                            df_swings.loc[idx, 'swing_high'] = np.nan
                
                for idx in swing_lows.index:
                    swing_low_price = df_swings.loc[idx, 'swing_low']
                    # Find nearby highs within lookback window
                    start_idx = max(0, idx - self.SWING_LOOKBACK)
                    end_idx = min(len(df_swings), idx + self.SWING_LOOKBACK + 1)
                    nearby_highs = df_swings.loc[start_idx:end_idx, 'swing_high'].dropna()
                    
                    if len(nearby_highs) > 0:
                        max_nearby_high = nearby_highs.max()
                        if (max_nearby_high - swing_low_price) < min_swing_threshold:
                            df_swings.loc[idx, 'swing_low'] = np.nan
            
            # Identify potential support/resistance levels
            df_swings['support'] = np.nan
            df_swings['resistance'] = np.nan
            
            # Extract multiple swing lows/highs for support/resistance detection
            swing_lows_clean = df_swings.loc[~df_swings['swing_low'].isna()]['swing_low'].tolist()
            swing_highs_clean = df_swings.loc[~df_swings['swing_high'].isna()]['swing_high'].tolist()
            
            # Find clusters of swing levels (simple implementation)
            if swing_lows_clean:
                # Check each price level to see if it's a potential support 
                # by counting nearby swing lows
                last_price = df_swings['close'].iloc[-1]
                for i, level in enumerate(swing_lows_clean):
                    # Only consider levels below current price for support
                    if level < last_price:
                        cluster_count = sum(1 for lvl in swing_lows_clean if abs(lvl - level)/level < 0.005)
                        if cluster_count >= 2:
                            # Mark this as a support level on the most recent occurrence
                            matching_idx = df_swings[df_swings['swing_low'] == level].index[-1]
                            df_swings.loc[matching_idx, 'support'] = level
            
            if swing_highs_clean:
                # Check each price level to see if it's a potential resistance
                # by counting nearby swing highs
                last_price = df_swings['close'].iloc[-1]
                for i, level in enumerate(swing_highs_clean):
                    # Only consider levels above current price for resistance
                    if level > last_price:
                        cluster_count = sum(1 for lvl in swing_highs_clean if abs(lvl - level)/level < 0.005)
                        if cluster_count >= 2:
                            # Mark this as a resistance level on the most recent occurrence
                            matching_idx = df_swings[df_swings['swing_high'] == level].index[-1]
                            df_swings.loc[matching_idx, 'resistance'] = level
                        
            return df_swings
            
        except Exception as e:
            print(f"Error in detect_price_swings: {str(e)}")
            # Return original dataframe with empty swing columns if there's an error
            df['swing_high'] = np.nan
            df['swing_low'] = np.nan
            return df

    def detect_divergences(self, df_with_swings, symbol):
        """Detect price/RSI and price/AD divergences"""
        try:
            df = df_with_swings.copy()
            
            if df is None or 'swing_high' not in df.columns or 'swing_low' not in df.columns:
                return None
            
            # Initialize results dictionary
            divergences = {
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
            
            # Get the swings
            swing_highs = df['swing_high'].dropna()
            swing_lows = df['swing_low'].dropna()
            
            # Check if we have enough swing points to analyze
            if len(swing_highs) < 2 and len(swing_lows) < 2:
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
                    
                    # Calculate percentage changes
                    price_change_pct = ((price2 - price1) / price1) * 100
                    rsi_change_pct = ((rsi2 - rsi1) / rsi1) * 100 if rsi1 != 0 else 0
                    
                    # Tolerance for "equal" determination
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
                            entry_price = price2
                            stop_loss = price2 - (atr_value * self.STOP_LOSS_ATR_MULT)
                            risk_amount = entry_price - stop_loss
                            take_profit = entry_price + (risk_amount * self.RISK_REWARD_RATIO)
                            
                            # Find potential resistance level for take profit optimization
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
                            
                            # Get market context
                            trend = df.loc[idx2, 'trend'] if 'trend' in df.columns else "unknown"
                            
                            # Calculate reward:risk ratio
                            reward_risk_ratio = (take_profit - entry_price) / risk_amount if risk_amount > 0 else 0
                            
                            # Accept any positive reward:risk ratio for testing
                            if reward_risk_ratio > 0:
                                divergences["trade_signals"]["bullish_rsi_strong"] = {
                                    "signal_type": "BUY",
                                    "entry_price": entry_price,
                                    "stop_loss": stop_loss,
                                    "take_profit": take_profit,
                                    "risk_amount": risk_amount,
                                    "reward_amount": take_profit - entry_price,
                                    "reward_risk_ratio": reward_risk_ratio,
                                    "trend_alignment": trend == "bullish",
                                    "signal_time": idx2,
                                    "confidence": "high",
                                    "divergence_type": "bullish_rsi_strong"
                                }
                        elif abs(rsi_change_pct) <= equal_tolerance:
                            # Medium Bullish
                            divergences["rsi_divergences"]["bullish"]["medium"] = True
                        elif rsi2 < rsi1 and abs(rsi_change_pct) < abs(price_change_pct):
                            # Weak Bullish
                            divergences["rsi_divergences"]["bullish"]["weak"] = True
                    elif price2 > price1:  # Higher low in price (hidden divergence)
                        if rsi2 < rsi1:  # Lower low in RSI
                            # Hidden Bullish
                            divergences["rsi_divergences"]["bullish"]["hidden"] = True
            
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
                    
                    # Calculate percentage changes
                    price_change_pct = ((price2 - price1) / price1) * 100
                    rsi_change_pct = ((rsi2 - rsi1) / rsi1) * 100 if rsi1 != 0 else 0
                    
                    # Tolerance for "equal" determination
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
                            entry_price = price2
                            stop_loss = price2 + (atr_value * self.STOP_LOSS_ATR_MULT)
                            risk_amount = stop_loss - entry_price
                            take_profit = entry_price - (risk_amount * self.RISK_REWARD_RATIO)
                            
                            # Find potential support level for take profit optimization
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
                            
                            # Get market context
                            trend = df.loc[idx2, 'trend'] if 'trend' in df.columns else "unknown"
                            
                            # Calculate reward:risk ratio
                            reward_risk_ratio = (entry_price - take_profit) / risk_amount if risk_amount > 0 else 0
                            
                            # Accept any positive reward:risk ratio for testing
                            if reward_risk_ratio > 0:
                                divergences["trade_signals"]["bearish_rsi_strong"] = {
                                    "signal_type": "SELL",
                                    "entry_price": entry_price,
                                    "stop_loss": stop_loss,
                                    "take_profit": take_profit,
                                    "risk_amount": risk_amount,
                                    "reward_amount": entry_price - take_profit,
                                    "reward_risk_ratio": reward_risk_ratio,
                                    "trend_alignment": trend == "bearish",
                                    "signal_time": idx2,
                                    "confidence": "high",
                                    "divergence_type": "bearish_rsi_strong"
                                }
                        elif abs(rsi_change_pct) <= equal_tolerance:
                            # Medium Bearish
                            divergences["rsi_divergences"]["bearish"]["medium"] = True
                        elif rsi2 > rsi1 and abs(rsi_change_pct) < abs(price_change_pct):
                            # Weak Bearish
                            divergences["rsi_divergences"]["bearish"]["weak"] = True
                    elif price2 < price1:  # Lower high in price (hidden divergence)
                        if rsi2 > rsi1:  # Higher high in RSI
                            # Hidden Bearish
                            divergences["rsi_divergences"]["bearish"]["hidden"] = True
            
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
                                    stop_loss = price2 - (atr_value * self.STOP_LOSS_ATR_MULT)
                                    risk_amount = entry_price - stop_loss
                                    take_profit = entry_price + (risk_amount * self.RISK_REWARD_RATIO)
                                    
                                    # Find potential resistance level for take profit optimization
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
                                    
                                    # Get market context
                                    trend = df.loc[idx2, 'trend'] if 'trend' in df.columns else "unknown"
                                    
                                    # Calculate reward:risk ratio
                                    reward_risk_ratio = (take_profit - entry_price) / risk_amount if risk_amount > 0 else 0
                                    
                                    # Only add trade signal if reward:risk ratio meets our criteria
                                    if reward_risk_ratio > 0:
                                        divergences["trade_signals"]["bullish_adv_strong"] = {
                                            "signal_type": "BUY",
                                            "entry_price": entry_price,
                                            "stop_loss": stop_loss,
                                            "take_profit": take_profit,
                                            "risk_amount": risk_amount,
                                            "reward_amount": take_profit - entry_price,
                                            "reward_risk_ratio": reward_risk_ratio,
                                            "trend_alignment": trend == "bullish",
                                            "signal_time": idx2,
                                            "confidence": "medium",  # Volume divergence slightly less reliable than RSI
                                            "divergence_type": "bullish_adv_strong"
                                        }
                            elif abs(ad_change_pct) <= equal_tolerance:  # Equal AD low (within tolerance)
                                # Medium Bullish: Lower price low, equal AD low
                                divergences["adv_divergences"]["bullish"]["medium"] = True
                                divergences["details"]["bullish_adv_medium"] = {
                                    "type": "Medium Bullish AD Volume Divergence",
                                    "first_swing": {"date": idx1, "price": price1, "ad": ad1},
                                    "second_swing": {"date": idx2, "price": price2, "ad": ad2}
                                }
                            elif ad2 < ad1 and abs(ad_change_pct) < abs(price_change_pct):  # Lower AD low but less percentage decline
                                # Weak Bullish: Lower price low, lower AD low but less percentage decline
                                divergences["adv_divergences"]["bullish"]["weak"] = True
                                divergences["details"]["bullish_adv_weak"] = {
                                    "type": "Weak Bullish AD Volume Divergence",
                                    "first_swing": {"date": idx1, "price": price1, "ad": ad1},
                                    "second_swing": {"date": idx2, "price": price2, "ad": ad2}
                                }
                        elif price2 > price1:  # Higher low in price (hidden divergence)
                            if ad2 < ad1:  # Lower low in AD
                                # Hidden Bullish: Higher price low, lower AD low
                                divergences["adv_divergences"]["bullish"]["hidden"] = True
                                divergences["details"]["bullish_adv_hidden"] = {
                                    "type": "Hidden Bullish AD Volume Divergence",
                                    "first_swing": {"date": idx1, "price": price1, "ad": ad1},
                                    "second_swing": {"date": idx2, "price": price2, "ad": ad2}
                                }
                
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
                                    stop_loss = price2 + (atr_value * self.STOP_LOSS_ATR_MULT)
                                    risk_amount = stop_loss - entry_price
                                    take_profit = entry_price - (risk_amount * self.RISK_REWARD_RATIO)
                                    
                                    # Find potential support level for take profit optimization
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
                                    
                                    # Get market context
                                    trend = df.loc[idx2, 'trend'] if 'trend' in df.columns else "unknown"
                                    
                                    # Calculate reward:risk ratio
                                    reward_risk_ratio = (entry_price - take_profit) / risk_amount if risk_amount > 0 else 0
                                    
                                    # Only add trade signal if reward:risk ratio meets our criteria
                                    if reward_risk_ratio > 0:
                                        divergences["trade_signals"]["bearish_adv_strong"] = {
                                            "signal_type": "SELL",
                                            "entry_price": entry_price,
                                            "stop_loss": stop_loss,
                                            "take_profit": take_profit,
                                            "risk_amount": risk_amount,
                                            "reward_amount": entry_price - take_profit,
                                            "reward_risk_ratio": reward_risk_ratio,
                                            "trend_alignment": trend == "bearish",
                                            "signal_time": idx2,
                                            "confidence": "medium",  # Volume divergence slightly less reliable than RSI
                                            "divergence_type": "bearish_adv_strong"
                                        }
                            elif abs(ad_change_pct) <= equal_tolerance:  # Equal AD high (within tolerance)
                                # Medium Bearish: Higher price high, equal AD high
                                divergences["adv_divergences"]["bearish"]["medium"] = True
                                divergences["details"]["bearish_adv_medium"] = {
                                    "type": "Medium Bearish AD Volume Divergence",
                                    "first_swing": {"date": idx1, "price": price1, "ad": ad1},
                                    "second_swing": {"date": idx2, "price": price2, "ad": ad2}
                                }
                            elif ad2 > ad1 and abs(ad_change_pct) < abs(price_change_pct):  # Higher AD high but less percentage increase
                                # Weak Bearish: Higher price high, higher AD high but less percentage increase
                                divergences["adv_divergences"]["bearish"]["weak"] = True
                                divergences["details"]["bearish_adv_weak"] = {
                                    "type": "Weak Bearish AD Volume Divergence",
                                    "first_swing": {"date": idx1, "price": price1, "ad": ad1},
                                    "second_swing": {"date": idx2, "price": price2, "ad": ad2}
                                }
                        elif price2 < price1:  # Lower high in price (hidden divergence)
                            if ad2 > ad1:  # Higher high in AD
                                # Hidden Bearish: Lower price high, higher AD high
                                divergences["adv_divergences"]["bearish"]["hidden"] = True
                                divergences["details"]["bearish_adv_hidden"] = {
                                    "type": "Hidden Bearish AD Volume Divergence",
                                    "first_swing": {"date": idx1, "price": price1, "ad": ad1},
                                    "second_swing": {"date": idx2, "price": price2, "ad": ad2}
                                }
            
            return divergences
            
        except Exception as e:
            print(f"Error in detect_divergences for {symbol}: {str(e)}")
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

    def calculate_divergence_metrics(self, df, lookback=50):
        """Calculate divergence metrics for each row in the dataframe"""
        print("Calculating technical indicators for entire dataset...")
        
        # First, calculate all indicators for the entire dataset
        df_with_indicators = self.calculate_indicators(df)
        if df_with_indicators is None:
            print("Failed to calculate indicators")
            return None
        
        print("Detecting price swings...")
        # Detect swings for the entire dataset
        df_with_swings = self.detect_price_swings(df_with_indicators, min_points=2)
        
        print("Processing divergence signals...")
        
        # Initialize lists to store metrics
        divergence_signals = []
        signal_types = []
        entry_prices = []
        stop_losses = []
        take_profits = []
        trade_status = []
        pnl = []
        running_pnl = []
        current_equity = []
        divergence_types = []
        confidence_levels = []
        trend_alignments = []
        
        # Initialize starting equity
        starting_equity = 25000.0
        running_equity = starting_equity
        
        # Calculate metrics for each row
        for i in range(len(df_with_swings)):
            if i < lookback:
                # For early rows, use NaN values for divergence signals but populate indicators
                divergence_signals.append(np.nan)
                signal_types.append(np.nan)
                entry_prices.append(np.nan)
                stop_losses.append(np.nan)
                take_profits.append(np.nan)
                trade_status.append(np.nan)
                pnl.append(np.nan)
                running_pnl.append(np.nan)
                current_equity.append(running_equity)
                divergence_types.append(np.nan)
                confidence_levels.append(np.nan)
                trend_alignments.append(np.nan)
                continue
            
            # Get the lookback window for divergence analysis
            window_end = i + 1  # Include current bar
            window_start = max(0, window_end - lookback)
            window_df = df_with_swings.iloc[window_start:window_end]
            
            if len(window_df) == 0:
                divergence_signals.append(np.nan)
                signal_types.append(np.nan)
                entry_prices.append(np.nan)
                stop_losses.append(np.nan)
                take_profits.append(np.nan)
                trade_status.append(np.nan)
                pnl.append(np.nan)
                running_pnl.append(np.nan)
                current_equity.append(running_equity)
                divergence_types.append(np.nan)
                confidence_levels.append(np.nan)
                trend_alignments.append(np.nan)
                continue
            
            # Detect divergences on the window
            divergence_results = self.detect_divergences(window_df, "BACKTEST")
            
            # Current bar data
            current_bar = df_with_swings.iloc[i]
            current_close = current_bar['close']
            current_high = current_bar['high']
            current_low = current_bar['low']
            
            # Initialize default values
            divergence_signal = None
            signal_type = None
            entry_price = None
            stop_loss = None
            take_profit = None
            trade_status_value = None
            pnl_value = None
            running_pnl_value = None
            divergence_type = None
            confidence_level = None
            trend_alignment = None
            
            # Check if there's currently an open trade
            has_open_trade = False
            open_trade_info = None
            
            # Look back through recent trade statuses to find if there's an open trade
            for j in range(len(trade_status) - 1, -1, -1):
                past_status = trade_status[j]
                if past_status is not None and not pd.isna(past_status):
                    if past_status == "open":
                        has_open_trade = True
                        # Get the trade info
                        open_trade_info = {
                            'entry_price': entry_prices[j],
                            'stop_loss': stop_losses[j],
                            'take_profit': take_profits[j],
                            'signal_type': signal_types[j],
                            'divergence_type': divergence_types[j],
                            'equity_at_entry': current_equity[j]
                        }
                    break
            
            # If we have an open trade, check if current OHLC closes it
            if has_open_trade and open_trade_info is not None:
                entry_p = open_trade_info['entry_price']
                stop_p = open_trade_info['stop_loss']
                target_p = open_trade_info['take_profit']
                sig_type = open_trade_info['signal_type']
                div_type = open_trade_info['divergence_type']
                equity_at_entry = open_trade_info['equity_at_entry']
                
                # Check if stop loss or take profit was hit
                if sig_type == "BUY":  # Long position
                    if current_low <= stop_p:
                        # Stop loss hit
                        trade_status_value = "closed"
                        exit_price = stop_p
                        # Calculate PnL
                        position_size = equity_at_entry * (self.MAX_RISK_PERCENT / 100)
                        shares = position_size / entry_p
                        pnl_value = (exit_price - entry_p) * shares
                        running_equity += pnl_value
                    elif current_high >= target_p:
                        # Take profit hit
                        trade_status_value = "closed"
                        exit_price = target_p
                        # Calculate PnL
                        position_size = equity_at_entry * (self.MAX_RISK_PERCENT / 100)
                        shares = position_size / entry_p
                        pnl_value = (exit_price - entry_p) * shares
                        running_equity += pnl_value
                elif sig_type == "SELL":  # Short position
                    if current_high >= stop_p:
                        # Stop loss hit
                        trade_status_value = "closed"
                        exit_price = stop_p
                        # Calculate PnL
                        position_size = equity_at_entry * (self.MAX_RISK_PERCENT / 100)
                        shares = position_size / entry_p
                        pnl_value = (entry_p - exit_price) * shares
                        running_equity += pnl_value
                    elif current_low <= target_p:
                        # Take profit hit
                        trade_status_value = "closed"
                        exit_price = target_p
                        # Calculate PnL
                        position_size = equity_at_entry * (self.MAX_RISK_PERCENT / 100)
                        shares = position_size / entry_p
                        pnl_value = (entry_p - exit_price) * shares
                        running_equity += pnl_value
                
                # Calculate running PnL for open positions
                if trade_status_value != "closed":
                    position_size = equity_at_entry * (self.MAX_RISK_PERCENT / 100)
                    shares = position_size / entry_p
                    if sig_type == "BUY":
                        running_pnl_value = (current_close - entry_p) * shares
                    else:  # SELL
                        running_pnl_value = (entry_p - current_close) * shares
            
            # Check for new divergence signals if no open trade
            if not has_open_trade and divergence_results and "trade_signals" in divergence_results:
                for signal_name, signal_data in divergence_results["trade_signals"].items():
                    # Only take the first signal found
                    divergence_signal = signal_name
                    signal_type = signal_data['signal_type']
                    entry_price = signal_data['entry_price']
                    stop_loss = signal_data['stop_loss']
                    take_profit = signal_data['take_profit']
                    divergence_type = signal_data['divergence_type']
                    confidence_level = signal_data['confidence']
                    trend_alignment = signal_data['trend_alignment']
                    trade_status_value = "open"
                    break  # Only take the first signal
            
            # Store calculated values
            divergence_signals.append(divergence_signal)
            signal_types.append(signal_type)
            entry_prices.append(entry_price)
            stop_losses.append(stop_loss)
            take_profits.append(take_profit)
            trade_status.append(trade_status_value)
            pnl.append(pnl_value)
            running_pnl.append(running_pnl_value)
            current_equity.append(running_equity)
            divergence_types.append(divergence_type)
            confidence_levels.append(confidence_level)
            trend_alignments.append(trend_alignment)
        
        # Extract indicator values from the pre-calculated dataframe
        print("Extracting indicator values...")
        
        return {
            'divergence_signal': divergence_signals,
            'signal_type': signal_types,
            'entry_price': entry_prices,
            'stop_loss': stop_losses,
            'take_profit': take_profits,
            'trade_status': trade_status,
            'pnl': pnl,
            'running_pnl': running_pnl,
            'current_equity': current_equity,
            'divergence_type': divergence_types,
            'confidence_level': confidence_levels,
            'trend_alignment': trend_alignments,
            'rsi': df_with_swings['rsi'].tolist(),
            'ad': df_with_swings['ad'].tolist(),
            'obv': df_with_swings['obv'].tolist(),
            'macd': df_with_swings['macd'].tolist(),
            'macd_signal': df_with_swings['macd_signal'].tolist(),
            'macd_hist': df_with_swings['macd_hist'].tolist(),
            'ema_short': df_with_swings['ema_short'].tolist(),
            'ema_medium': df_with_swings['ema_medium'].tolist(),
            'ema_long': df_with_swings['ema_long'].tolist(),
            'bb_upper': df_with_swings['bb_upper'].tolist(),
            'bb_middle': df_with_swings['bb_middle'].tolist(),
            'bb_lower': df_with_swings['bb_lower'].tolist(),
            'atr': df_with_swings['atr'].tolist(),
            'trend': df_with_swings['trend'].tolist(),
            'swing_high': df_with_swings['swing_high'].tolist(),
            'swing_low': df_with_swings['swing_low'].tolist(),
            'support': df_with_swings['support'].tolist(),
            'resistance': df_with_swings['resistance'].tolist()
        }
    
    def fetch_and_process_data(self, symbol, period_type="day", period=10, frequency_type="minute", 
                              frequency=5, start_date=None, end_date=None, lookback=50):
        """
        Fetch historical data and add divergence metrics
        
        Args:
            symbol (str): Stock symbol to fetch data for
            period_type (str): Period type (day, month, year, ytd)
            period (int): Number of periods
            frequency_type (str): Frequency type (minute, daily, weekly, monthly)
            frequency (int): Frequency value
            start_date (int): Start date in milliseconds (optional)
            end_date (int): End date in milliseconds (optional)
            lookback (int): Lookback period for divergence calculations
        """
        print(f"Fetching historical data for {symbol}...")
        print(f"Parameters: {period_type}={period}, {frequency_type}={frequency}")
        
        try:
            # Fetch historical data
            data = self.data_handler.fetch_historical_data(
                symbol=symbol,
                periodType=period_type,
                period=period,
                frequencyType=frequency_type, 
                freq=frequency,
                startDate=start_date,
                endDate=end_date,
                needExtendedHoursData=False
            )
            
            if not data or not data.get('candles'):
                print(f"No data received for {symbol}")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(data['candles'])
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Convert datetime to readable format
            df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')
            
            # Sort by datetime
            df = df.sort_values('datetime').reset_index(drop=True)
            
            print(f"Calculating divergence metrics with lookback period of {lookback}...")
            
            # Calculate divergence metrics
            div_metrics = self.calculate_divergence_metrics(df, lookback=lookback)
            
            if div_metrics:
                # Add divergence metrics to dataframe
                for metric_name, metric_values in div_metrics.items():
                    df[metric_name] = metric_values
                
                print(f"Added divergence metrics: {list(div_metrics.keys())}")
            else:
                print("Failed to calculate divergence metrics")
                return None
            
            # Reorder columns for better readability
            base_columns = ['symbol', 'datetime', 'open', 'high', 'low', 'close', 'volume']
            divergence_columns = ['divergence_signal', 'signal_type', 'entry_price', 'stop_loss', 
                                'take_profit', 'trade_status', 'pnl', 'running_pnl', 'current_equity',
                                'divergence_type', 'confidence_level', 'trend_alignment']
            indicator_columns = ['rsi', 'ad', 'obv', 'macd', 'macd_signal', 'macd_hist',
                               'ema_short', 'ema_medium', 'ema_long', 'bb_upper', 'bb_middle', 'bb_lower',
                               'atr', 'trend', 'swing_high', 'swing_low', 'support', 'resistance']
            
            # Ensure all columns exist
            available_columns = [col for col in base_columns + divergence_columns + indicator_columns if col in df.columns]
            df = df[available_columns]
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            freq_str = f"{frequency}{frequency_type}" if frequency != 1 else frequency_type
            filename = f"{self.output_dir}/{symbol}_divergence_backtest_{period}{period_type}_{freq_str}_{timestamp}.csv"
            
            # Save to CSV
            df.to_csv(filename, index=False)
            
            print(f"\nData successfully saved to: {filename}")
            print(f"Total records: {len(df)}")
            print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
            
            # Display first few rows
            print("\nFirst 5 rows:")
            print(df.head())
            
            # Display last few rows
            print("\nLast 5 rows:")
            print(df.tail())
            
            # Display basic statistics
            print(f"\nBasic Statistics for {symbol}:")
            print(f"Average Close Price: ${df['close'].mean():.2f}")
            print(f"Highest Close Price: ${df['close'].max():.2f}")
            print(f"Lowest Close Price: ${df['close'].min():.2f}")
            print(f"Average Daily Volume: {df['volume'].mean():,.0f}")
            
            # Display divergence statistics
            if 'trade_status' in df.columns:
                total_signals = df['divergence_signal'].notna().sum()
                total_trades = df['trade_status'].notna().sum()
                open_trades = (df['trade_status'] == 'open').sum()
                closed_trades = (df['trade_status'] == 'closed').sum()
                
                print(f"\nDivergence Statistics:")
                print(f"Total Divergence Signals: {total_signals}")
                print(f"Total Trade Events: {total_trades}")
                print(f"Trades Opened: {open_trades}")
                print(f"Trades Closed: {closed_trades}")
                
                if 'divergence_type' in df.columns:
                    div_types = df['divergence_type'].dropna().value_counts()
                    if not div_types.empty:
                        print(f"Divergence Types:")
                        for div_type, count in div_types.items():
                            print(f"  {div_type}: {count}")
            
            # Calculate and display performance summary
            self.calculate_performance_summary(filename)
            
            return filename
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Backtest the Stock Divergence Calculator strategy with historical data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backtest 1 year of daily data for divergence strategy
  python3 divergence_backtest.py AAPL
  
  # Backtest 6 months of daily data
  python3 divergence_backtest.py NVDA --period 6 --period-type month
  
  # Backtest 5 days of 15-minute data
  python3 divergence_backtest.py TSLA --period 5 --period-type day --frequency-type minute --frequency 15
  
  # Custom lookback period for divergence calculations
  python3 divergence_backtest.py AAPL --lookback 100
        """
    )
    
    parser.add_argument(
        'ticker',
        type=str,
        help='Stock ticker symbol to backtest (e.g., AAPL, NVDA, TSLA)'
    )
    
    parser.add_argument(
        '--period-type',
        type=str,
        choices=['day', 'month', 'year', 'ytd'],
        default='year',
        help='Period type (default: year)'
    )
    
    parser.add_argument(
        '--period',
        type=int,
        default=1,
        help='Number of periods (default: 1)'
    )
    
    parser.add_argument(
        '--frequency-type',
        type=str,
        choices=['minute', 'daily', 'weekly', 'monthly'],
        default='daily',
        help='Frequency type (default: daily)'
    )
    
    parser.add_argument(
        '--frequency',
        type=int,
        default=1,
        help='Frequency value (default: 1)'
    )
    
    parser.add_argument(
        '--lookback',
        type=int,
        default=10,
        help='Lookback period for divergence calculations (default: 10)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date in YYYY-MM-DD format (optional, for custom date ranges)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date in YYYY-MM-DD format (optional, for custom date ranges)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Convert ticker to uppercase for consistency
    ticker = args.ticker.upper()
    
    # Parse start and end dates if provided
    start_date_ms = None
    end_date_ms = None
    
    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
            start_date_ms = int(start_date.timestamp() * 1000)
        except ValueError:
            print(f"Error: Invalid start date format '{args.start_date}'. Use YYYY-MM-DD format.")
            sys.exit(1)
    
    if args.end_date:
        try:
            end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
            end_date_ms = int(end_date.timestamp() * 1000)
        except ValueError:
            print(f"Error: Invalid end date format '{args.end_date}'. Use YYYY-MM-DD format.")
            sys.exit(1)
    
    print("=" * 80)
    print("STOCK DIVERGENCE CALCULATOR BACKTEST")
    print("=" * 80)
    print(f"Symbol: {ticker}")
    print(f"Period: {args.period} {args.period_type}")
    print(f"Frequency: {args.frequency} {args.frequency_type}")
    print(f"Divergence Lookback: {args.lookback} periods")
    print(f"Risk per Trade: 1.0% of account equity")
    print(f"Risk/Reward Ratio: 2:1")
    if start_date_ms and end_date_ms:
        start_str = datetime.fromtimestamp(start_date_ms/1000).strftime('%Y-%m-%d')
        end_str = datetime.fromtimestamp(end_date_ms/1000).strftime('%Y-%m-%d')
        print(f"Date Range: {start_str} to {end_str}")
    print("=" * 80)
    
    # Create backtester and process data
    backtester = DivergenceBacktester()
    result = backtester.fetch_and_process_data(
        symbol=ticker,
        period_type=args.period_type,
        period=args.period,
        frequency_type=args.frequency_type,
        frequency=args.frequency,
        start_date=start_date_ms,
        end_date=end_date_ms,
        lookback=args.lookback
    )
    
    if result:
        print(f"\n✅ Successfully created divergence backtest CSV: {result}")
        
        # Automatically run the visualizer
        print("\n🎨 Generating visualization...")
        try:
            # Create charts directory if it doesn't exist
            charts_dir = "charts"
            os.makedirs(charts_dir, exist_ok=True)
            
            # Generate output filename for the chart
            base_filename = os.path.splitext(os.path.basename(result))[0]
            chart_filename = f"{charts_dir}/{base_filename}_analysis.png"
            
            # Run the visualizer
            visualizer_cmd = [
                "python3", "divergence_visualization.py", 
                result, 
                "--save", chart_filename
            ]
            
            print(f"Running: {' '.join(visualizer_cmd)}")
            subprocess_result = subprocess.run(visualizer_cmd, capture_output=True, text=True)
            
            if subprocess_result.returncode == 0:
                print(f"✅ Visualization saved to: {chart_filename}")
                print("📊 Visualization output:")
                print(subprocess_result.stdout)
            else:
                print(f"❌ Error generating visualization:")
                print(subprocess_result.stderr)
                
        except Exception as e:
            print(f"❌ Error running visualizer: {str(e)}")
            
    else:
        print("\n❌ Failed to fetch and process data")

if __name__ == "__main__":
    main()

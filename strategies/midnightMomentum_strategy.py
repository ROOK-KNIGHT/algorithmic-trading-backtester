"""
Midnight Momentum Strategy

This script calculates the Profit Potential (Upside) Threshold for a given ticker
using the midnight momentum methodology. It fetches historical data and applies
technical calculations to determine upside potential price points.

Author: Trading System
Date: 2025-01-05
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import json
import time
import signal

# Add the parent directory to the path to import handlers
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from handlers.connection_manager import ensure_valid_tokens
from handlers.historical_data_handler import HistoricalDataHandler
from handlers.order_handler import OrderHandler


class MidnightMomentumStrategy:
    """
    Midnight Momentum Strategy class for calculating upside profit potential thresholds.
    """
    
    def __init__(self, ticker):
        """
        Initialize the strategy with a ticker symbol.
        
        Args:
            ticker (str): Stock ticker symbol
        """
        self.ticker = ticker
        self.data_handler = HistoricalDataHandler()
        self.data = None
        self.order_counter = 0  # Track number of orders placed
        self.orders_placed_today = []  # Track orders placed in current session
        
    def fetch_data(self, period_type="year", period=1, frequency_type="daily", frequency=1):
        """
        Fetch historical data for the ticker using the existing data handler.
        
        Args:
            period_type (str): Period type ('day', 'month', 'year', 'ytd')
            period (int): Number of periods
            frequency_type (str): Frequency type ('minute', 'daily', 'weekly', 'monthly')
            frequency (int): Frequency value
        """
        try:
            print(f"Fetching data for {self.ticker}...")
            
            # Use the existing historical data handler
            raw_data = self.data_handler.fetch_historical_data(
                symbol=self.ticker,
                periodType=period_type,
                period=period,
                frequencyType=frequency_type,
                freq=frequency,
                needExtendedHoursData=False
            )
            
            if not raw_data or 'candles' not in raw_data:
                print(f"No data retrieved for {self.ticker}")
                return False
            
            # Convert to DataFrame
            candles = raw_data['candles']
            self.data = pd.DataFrame(candles)
            
            # Convert datetime column to proper datetime format
            # The historical_data_handler already converts timestamps to datetime strings
            self.data['datetime'] = pd.to_datetime(self.data['datetime'])
            
            # Sort by datetime and reset index
            self.data = self.data.sort_values('datetime').reset_index(drop=True)
            
            # Rename columns to match expected format
            column_mapping = {
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            self.data = self.data.rename(columns=column_mapping)
            
            print(f"Successfully fetched {len(self.data)} data points for {self.ticker}")
            print(f"Date range: {self.data['datetime'].iloc[0]} to {self.data['datetime'].iloc[-1]}")
            return True
                
        except Exception as e:
            print(f"Error fetching data for {self.ticker}: {e}")
            return False
    
    def calculate_basic_metrics(self):
        """
        Calculate basic price metrics needed for threshold calculations.
        """
        if self.data is None or self.data.empty:
            return None
            
        # Create a copy to work with
        result = self.data.copy()
        
        # Calculate previous close (shifted by 1 day)
        result['prev_close'] = result['Close'].shift(1)
        
        # Calculate daily returns
        result['daily_return'] = result['Close'].pct_change()
        
        # Calculate overnight gap (open vs previous close)
        result['overnight_gap'] = (result['Open'] - result['prev_close']) / result['prev_close']
        
        # Calculate intraday return (close vs open)
        result['intraday_return'] = (result['Close'] - result['Open']) / result['Open']
        
        # Binary indicator: did high exceed previous close?
        result['high_above_prev_close'] = (result['High'] > result['prev_close']).astype(int)
        
        # Calculate True Range for volatility
        result['true_range'] = np.maximum(
            result['High'] - result['Low'],
            np.maximum(
                abs(result['High'] - result['prev_close']),
                abs(result['Low'] - result['prev_close'])
            )
        )
        
        # Calculate Average True Range (ATR) - 14 period
        result['atr'] = result['true_range'].rolling(window=14, min_periods=1).mean()
        
        return result
    
    def calculate_profit_potential_thresholds(self, data, confidence_levels=[0.68, 0.90, 0.95, 0.99]):
        """
        Calculate profit potential (upside) thresholds using bootstrap resampling for robust quantile estimation.
        
        Args:
            data: DataFrame with price data
            confidence_levels: List of confidence levels
            
        Returns:
            Dictionary of upside thresholds for each confidence level
        """
        upside_thresholds = {}
        
        # Calculate the percentage gain from previous close to high
        # Only include days where high > prev_close (actual upside potential days)
        upside_mask = data['High'] > data['prev_close']
        high_gains = (data['High'] - data['prev_close']) / data['prev_close'] * 100
        
        # Filter to only include days with actual upside movement
        high_gains = high_gains[upside_mask].dropna()
        
        if len(high_gains) < 30:  # Need minimum sample size
            print(f"Insufficient upside days for threshold calculation: {len(high_gains)} samples (need 30+)")
            return upside_thresholds
        
        # Bootstrap resampling for robust quantile estimation
        n_bootstrap = 1000  # Number of bootstrap samples
        bootstrap_size = min(len(high_gains) * 2, 500)  # Bootstrap sample size (up to 2x original, max 500)
        
        # Generate bootstrap samples
        bootstrap_samples = []
        np.random.seed(42)  # For reproducible results
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            bootstrap_sample = np.random.choice(high_gains.values, size=bootstrap_size, replace=True)
            bootstrap_samples.append(bootstrap_sample)
        
        # Confidence level represents the probability that price will reach or exceed the threshold
        # Higher confidence = more conservative (lower) targets that are hit more frequently
        # Adjusted to provide realistic upside targets across all confidence levels
        quantile_mapping = {
            0.68: 0.32,  # 68% confidence uses 32nd percentile 
            0.90: 0.25,  # 90% confidence uses 25th percentile (more conservative)
            0.95: 0.20,  # 95% confidence uses 20th percentile (very conservative)
            0.99: 0.15   # 99% confidence uses 15th percentile (extremely conservative)
        }
        
        for conf_level in confidence_levels:
            quantile = quantile_mapping.get(conf_level, 0.32)
            
            # Calculate quantile for each bootstrap sample
            bootstrap_quantiles = []
            for sample in bootstrap_samples:
                bootstrap_quantiles.append(np.percentile(sample, quantile * 100))
            
            # Use the median of bootstrap quantiles for robust estimation
            upside_threshold_pct = np.median(bootstrap_quantiles)
            
            # Use actual historical data without artificial minimums
            # Only ensure threshold is not negative (which would be nonsensical for upside targets)
            if upside_threshold_pct < 0:
                upside_threshold_pct = 0.01  # Minimum 0.01% to avoid zero targets
            
            upside_thresholds[f'upside_threshold_{int(conf_level*100)}'] = upside_threshold_pct
        
        return upside_thresholds
    
    def apply_upside_thresholds(self, data, min_window_size=30):
        """
        Apply upside thresholds using expanding windows to avoid look-ahead bias.
        
        Args:
            data: DataFrame with price data
            min_window_size: Minimum window size for calculations
            
        Returns:
            DataFrame with upside threshold columns added
        """
        result = data.copy()
        confidence_levels = [0.68, 0.90, 0.95, 0.99]
        
        # Initialize upside threshold columns
        for conf_level in confidence_levels:
            conf_pct = int(conf_level * 100)
            result[f'upside_threshold_{conf_pct}'] = np.nan
            result[f'upside_threshold_price_{conf_pct}'] = np.nan
        
        # Calculate thresholds using expanding window
        for i in range(min_window_size, len(result)):
            # Use historical data up to (but not including) current day
            historical_data = result.iloc[:i]
            
            # Calculate upside thresholds on historical data only
            upside_thresholds = self.calculate_profit_potential_thresholds(historical_data, confidence_levels)
            
            # Apply thresholds to current day
            current_row = result.iloc[i]
            prev_close = current_row['prev_close']
            
            if pd.notna(prev_close) and upside_thresholds:
                for conf_level in confidence_levels:
                    conf_pct = int(conf_level * 100)
                    threshold_key = f'upside_threshold_{conf_pct}'
                    
                    if threshold_key in upside_thresholds:
                        upside_threshold_gain = upside_thresholds[threshold_key]
                        # Calculate upside threshold price
                        upside_threshold_price = prev_close * (1 + upside_threshold_gain / 100)
                        
                        result.iloc[i, result.columns.get_loc(f'upside_threshold_{conf_pct}')] = upside_threshold_gain
                        result.iloc[i, result.columns.get_loc(f'upside_threshold_price_{conf_pct}')] = upside_threshold_price
        
        return result
    
    def get_last_15_closes_with_thresholds(self):
        """
        Get the last 15 closes with their upside potential thresholds.
        
        Returns:
            pd.DataFrame: DataFrame with last 15 closes and thresholds
        """
        if self.data is None or self.data.empty:
            print("No data available. Please fetch data first.")
            return None
        
        # Calculate basic metrics
        data_with_metrics = self.calculate_basic_metrics()
        
        if data_with_metrics is None:
            print("Failed to calculate basic metrics")
            return None
        
        # Apply upside thresholds
        data_with_thresholds = self.apply_upside_thresholds(data_with_metrics)
        
        # Get last 15 entries with valid threshold data
        valid_data = data_with_thresholds.dropna(subset=['upside_threshold_68'])
        
        if len(valid_data) == 0:
            print("No valid threshold data available")
            return None
        
        last_15 = valid_data.tail(15).copy()
        
        # Calculate potential from previous close (entry point) to threshold
        display_data = pd.DataFrame({
            'Date': last_15['datetime'].dt.strftime('%Y-%m-%d'),
            'Close': last_15['Close'].round(2),
            'Target_68%': last_15['upside_threshold_price_68'].round(2),
            'Target_90%': last_15['upside_threshold_price_90'].round(2),
            'Target_95%': last_15['upside_threshold_price_95'].round(2),
            'Target_99%': last_15['upside_threshold_price_99'].round(2),
            'Potential_68%': ((last_15['upside_threshold_price_68'] - last_15['prev_close']) / last_15['prev_close'] * 100).round(2),
            'Potential_90%': ((last_15['upside_threshold_price_90'] - last_15['prev_close']) / last_15['prev_close'] * 100).round(2),
            'Potential_95%': ((last_15['upside_threshold_price_95'] - last_15['prev_close']) / last_15['prev_close'] * 100).round(2),
            'Potential_99%': ((last_15['upside_threshold_price_99'] - last_15['prev_close']) / last_15['prev_close'] * 100).round(2)
        })
        
        return display_data
    
    def get_next_day_prediction(self):
        """
        Get the next trading day's upside potential thresholds.
        
        Returns:
            dict: Dictionary with next day's predictions
        """
        if self.data is None or self.data.empty:
            print("No data available. Please fetch data first.")
            return None
        
        # Calculate basic metrics
        data_with_metrics = self.calculate_basic_metrics()
        
        if data_with_metrics is None:
            print("Failed to calculate basic metrics")
            return None
        
        # Apply upside thresholds using all available historical data
        data_with_thresholds = self.apply_upside_thresholds(data_with_metrics)
        
        # Get the most recent data point with valid thresholds
        valid_data = data_with_thresholds.dropna(subset=['upside_threshold_68'])
        
        if len(valid_data) == 0:
            print("No valid threshold data available for prediction")
            return None
        
        # Ensure we have enough data for reliable predictions
        if len(data_with_metrics) < 30:
            print(f"Insufficient historical data for reliable prediction: {len(data_with_metrics)} samples (need 30+)")
            return None
        
        # Use all historical data to calculate thresholds for next day
        confidence_levels = [0.68, 0.90, 0.95, 0.99]
        next_day_thresholds = self.calculate_profit_potential_thresholds(data_with_metrics, confidence_levels)
        
        if not next_day_thresholds:
            print("Unable to calculate thresholds for next day prediction")
            return None
        
        # Get the current close price (which becomes tomorrow's "previous close")
        current_close = self.data['Close'].iloc[-1]
        current_date = self.data['datetime'].iloc[-1]
        
        # Calculate next day's target prices
        predictions = {
            'current_date': current_date.strftime('%Y-%m-%d'),
            'current_close': round(current_close, 2),
            'next_day_targets': {},
            'next_day_potentials': {}
        }
        
        for conf_level in confidence_levels:
            conf_pct = int(conf_level * 100)
            threshold_key = f'upside_threshold_{conf_pct}'
            
            if threshold_key in next_day_thresholds:
                threshold_gain = next_day_thresholds[threshold_key]
                target_price = current_close * (1 + threshold_gain / 100)
                potential_pct = threshold_gain
                
                predictions['next_day_targets'][f'{conf_pct}%'] = round(target_price, 2)
                predictions['next_day_potentials'][f'{conf_pct}%'] = round(potential_pct, 2)
        
        return predictions
    
    def is_in_trading_window(self, target_time="12:58", window_minutes=2):
        """
        Check if the current time is within the specified trading window in PST/PDT.
        
        Args:
            target_time (str): Target time in HH:MM format (default "12:58")
            window_minutes (int): Window in minutes around target time (default 2)
            
        Returns:
            dict: Dictionary containing timing information and window status
        """
        try:
            # Define Pacific timezone (handles PST/PDT automatically)
            pacific_tz = pytz.timezone('US/Pacific')
            
            # Get current time in Pacific timezone
            current_time_pacific = datetime.now(pacific_tz)
            
            # Parse target time
            target_hour, target_minute = map(int, target_time.split(':'))
            
            # Create target datetime for today in Pacific timezone
            target_datetime = current_time_pacific.replace(
                hour=target_hour, 
                minute=target_minute, 
                second=0, 
                microsecond=0
            )
            
            # Calculate window boundaries
            window_start = target_datetime - timedelta(minutes=window_minutes)
            window_end = target_datetime + timedelta(minutes=window_minutes)
            
            # Check if current time is within the window
            in_window = window_start <= current_time_pacific <= window_end
            
            # Calculate time until target (can be negative if past target)
            time_until_target = target_datetime - current_time_pacific
            minutes_until_target = int(time_until_target.total_seconds() / 60)
            
            # Determine timezone name (PST or PDT)
            tz_name = current_time_pacific.strftime('%Z')
            
            return {
                'current_time_pacific': current_time_pacific.strftime('%Y-%m-%d %H:%M:%S %Z'),
                'target_time': target_datetime.strftime('%Y-%m-%d %H:%M:%S %Z'),
                'window_start': window_start.strftime('%Y-%m-%d %H:%M:%S %Z'),
                'window_end': window_end.strftime('%Y-%m-%d %H:%M:%S %Z'),
                'in_trading_window': in_window,
                'minutes_until_target': minutes_until_target,
                'timezone': tz_name,
                'window_minutes': window_minutes
            }
            
        except Exception as e:
            print(f"Error checking trading window: {e}")
            return {
                'error': str(e),
                'in_trading_window': False
            }
    
    def check_trading_status(self):
        """
        Check current trading status and display timing information.
        
        Returns:
            bool: True if in trading window, False otherwise
        """
        timing_info = self.is_in_trading_window()
        
        if 'error' in timing_info:
            print(f"❌ Error checking trading status: {timing_info['error']}")
            return False
        
        print(f"\n⏰ === TRADING WINDOW STATUS ===")
        print(f"Current Time: {timing_info['current_time_pacific']}")
        print(f"Target Time: {timing_info['target_time']}")
        print(f"Trading Window: {timing_info['window_start']} to {timing_info['window_end']}")
        print(f"Window Size: ±{timing_info['window_minutes']} minutes")
        
        if timing_info['in_trading_window']:
            print(f"🟢 IN TRADING WINDOW - Ready to execute trades!")
            print(f"⏱️  Time until target: {timing_info['minutes_until_target']} minutes")
        else:
            if timing_info['minutes_until_target'] > 0:
                print(f"🟡 WAITING - {timing_info['minutes_until_target']} minutes until trading window")
            else:
                print(f"🔴 WINDOW CLOSED - Trading window was {abs(timing_info['minutes_until_target'])} minutes ago")
                print(f"⏭️  Next opportunity: Tomorrow at 12:58 {timing_info['timezone']}")
        
        return timing_info['in_trading_window']
    
    def execute_midnight_momentum_trade(self, shares=100):
        """
        Execute midnight momentum trading strategy:
        1. Get account positions
        2. Check if current symbol is in positions
        3. Check if in trading window
        4. Buy at market if no position and in window
        
        Args:
            shares (int): Number of shares to buy (default 100)
            
        Returns:
            dict: Trading execution result
        """
        try:
            print(f"\n🎯 === MIDNIGHT MOMENTUM TRADE EXECUTION ===")
            print(f"Symbol: {self.ticker}")
            print(f"Shares: {shares}")
            print(f"Order Counter: {self.order_counter}")
            
            # Step 0: Check if order already placed today
            if self.order_counter > 0:
                return {
                    'status': 'rejected',
                    'reason': f'Order limit reached. Already placed {self.order_counter} order(s) today.',
                    'orders_placed': self.orders_placed_today,
                    'timestamp': datetime.now()
                }
            
            # Step 1: Check if in trading window
            timing_info = self.is_in_trading_window()
            
            if 'error' in timing_info:
                return {
                    'status': 'error',
                    'reason': f"Error checking trading window: {timing_info['error']}",
                    'timestamp': datetime.now()
                }
            
            if not timing_info['in_trading_window']:
                return {
                    'status': 'rejected',
                    'reason': f"Not in trading window. Current time: {timing_info['current_time_pacific']}, Window: {timing_info['window_start']} to {timing_info['window_end']}",
                    'minutes_until_window': timing_info['minutes_until_target'] if timing_info['minutes_until_target'] > 0 else None,
                    'timestamp': datetime.now()
                }
            
            print(f"✅ In trading window - proceeding with position check")
            
            # Step 2: Initialize order handler and get account positions
            try:
                order_handler = OrderHandler()
                account_info = order_handler.get_account()
                
                if not account_info:
                    return {
                        'status': 'error',
                        'reason': 'Failed to retrieve account information',
                        'timestamp': datetime.now()
                    }
                
                print(f"✅ Connected to account: {order_handler.account_number}")
                
            except Exception as e:
                return {
                    'status': 'error',
                    'reason': f'Failed to initialize order handler: {str(e)}',
                    'timestamp': datetime.now()
                }
            
            # Step 3: Check if current symbol is in positions
            current_position = None
            if 'positions' in account_info:
                for position in account_info['positions']:
                    if position.get('instrument', {}).get('symbol') == self.ticker:
                        current_position = position
                        break
            
            if current_position:
                position_qty = current_position.get('longQuantity', 0) - current_position.get('shortQuantity', 0)
                print(f"📊 Existing position found: {position_qty} shares of {self.ticker}")
                
                return {
                    'status': 'rejected',
                    'reason': f'Already have position in {self.ticker}: {position_qty} shares',
                    'existing_position': position_qty,
                    'timestamp': datetime.now()
                }
            else:
                print(f"✅ No existing position in {self.ticker}")
            
            # Step 4: Get current price for the trade
            if self.data is not None and not self.data.empty:
                current_price = self.data['Close'].iloc[-1]
                print(f"📈 Using last close price: ${current_price:.2f}")
            else:
                print(f"⚠️  No price data available, using market order without price reference")
                current_price = None
            
            # Step 5: Execute buy order
            print(f"🚀 Executing BUY market order for {shares} shares of {self.ticker}")
            
            order_result = order_handler.buy_market(
                symbol=self.ticker,
                shares=shares,
                current_price=current_price
            )
            
            if order_result['status'] == 'submitted':
                # Increment order counter and track order
                self.order_counter += 1
                order_record = {
                    'order_id': order_result.get('order_id', 'N/A'),
                    'symbol': order_result['symbol'],
                    'action': 'BUY',
                    'shares': order_result['shares'],
                    'timestamp': datetime.now().isoformat(),
                    'order_type': 'opening'
                }
                self.orders_placed_today.append(order_record)
                
                print(f"✅ Order submitted successfully!")
                print(f"   Order ID: {order_result.get('order_id', 'N/A')}")
                print(f"   Shares: {order_result['shares']}")
                print(f"   Symbol: {order_result['symbol']}")
                print(f"   Order Counter: {self.order_counter}")
                if order_result.get('dollar_amount'):
                    print(f"   Estimated Cost: ${order_result['dollar_amount']:.2f}")
            else:
                print(f"❌ Order failed: {order_result.get('reason', 'Unknown error')}")
            
            return order_result
                
        except Exception as e:
            error_msg = f"Error executing midnight momentum trade: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                'status': 'error',
                'reason': error_msg,
                'timestamp': datetime.now()
            }
    
    def save_thresholds_to_json(self, filename=None):
        """
        Save price thresholds to JSON file in organized directory structure.
        
        Args:
            filename (str): Optional custom filename. If None, uses ticker and timestamp.
            
        Returns:
            str: Path to saved file
        """
        try:
            # Get next day prediction data
            prediction_data = self.get_next_day_prediction()
            
            if not prediction_data:
                print("❌ No prediction data available to save")
                return None
            
            # Create thresholds directory if it doesn't exist
            thresholds_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'thresholds')
            os.makedirs(thresholds_dir, exist_ok=True)
            
            # Generate filename if not provided
            if filename is None:
            
                filename = f"{self.ticker}_thresholds.json"
            
            filepath = os.path.join(thresholds_dir, filename)
            
            # Prepare data for JSON serialization
            threshold_data = {
                'symbol': self.ticker,
                'generated_at': datetime.now().isoformat(),
                'current_date': prediction_data['current_date'],
                'current_close': prediction_data['current_close'],
                'thresholds': {
                    '68_percent': {
                        'target_price': prediction_data['next_day_targets']['68%'],
                        'potential_percent': prediction_data['next_day_potentials']['68%']
                    },
                    '90_percent': {
                        'target_price': prediction_data['next_day_targets']['90%'],
                        'potential_percent': prediction_data['next_day_potentials']['90%']
                    },
                    '95_percent': {
                        'target_price': prediction_data['next_day_targets']['95%'],
                        'potential_percent': prediction_data['next_day_potentials']['95%']
                    },
                    '99_percent': {
                        'target_price': prediction_data['next_day_targets']['99%'],
                        'potential_percent': prediction_data['next_day_potentials']['99%']
                    }
                }
            }
            
            # Save to JSON file
            with open(filepath, 'w') as f:
                json.dump(threshold_data, f, indent=2)
            
            print(f"💾 Thresholds saved to: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"❌ Error saving thresholds: {str(e)}")
            return None
    
    def calculate_shares(self, method='fixed', amount=100, percentage=10.0):
        """
        Calculate number of shares to trade using fixed amount or percentage of equity.
        
        Args:
            method (str): 'fixed' for fixed dollar amount, 'percentage' for equity percentage
            amount (float): Fixed dollar amount to invest (used when method='fixed')
            percentage (float): Percentage of equity to use (default 10.0%)
            
        Returns:
            int: Number of shares to trade
        """
        try:
            if self.data is None or self.data.empty:
                print("❌ No price data available for share calculation")
                return 0
            
            current_price = self.data['Close'].iloc[-1]
            
            if method == 'fixed':
                shares = int(amount / current_price)
                print(f"📊 Fixed Amount Calculation:")
                print(f"   Investment Amount: ${amount:.2f}")
                print(f"   Current Price: ${current_price:.2f}")
                print(f"   Calculated Shares: {shares}")
                
            elif method == 'percentage':
                # Get real account balance
                try:
                    order_handler = OrderHandler()
                    account_info = order_handler.get_account()
                    
                    if not account_info or 'securitiesAccount' not in account_info:
                        print("❌ Failed to retrieve account information")
                        return 0
                    
                    # Get equity from account
                    balances = account_info['securitiesAccount'].get('currentBalances', {})
                    account_equity = balances.get('equity', 0)
                    
                    if account_equity <= 0:
                        print("❌ No account equity found")
                        return 0
                    
                    investment_amount = account_equity * (percentage / 100)
                    shares = int(investment_amount / current_price)
                    
                    print(f"📊 Percentage-Based Calculation:")
                    print(f"   Account Equity: ${account_equity:.2f}")
                    print(f"   Investment Percentage: {percentage}%")
                    print(f"   Investment Amount: ${investment_amount:.2f}")
                    print(f"   Current Price: ${current_price:.2f}")
                    print(f"   Calculated Shares: {shares}")
                    
                except Exception as e:
                    print(f"❌ Error retrieving account balance: {str(e)}")
                    return 0
                
            else:
                print(f"❌ Invalid method: {method}. Use 'fixed' or 'percentage'")
                return 0
            
            return max(shares, 1)  # Ensure at least 1 share
            
        except Exception as e:
            print(f"❌ Error calculating shares: {str(e)}")
            return 0
    
    def execute_exit_strategy(self, threshold_file=None, confidence_level='90_percent'):
        """
        Execute exit strategy by checking current positions and placing limit orders to close
        positions based on saved thresholds.
        
        Args:
            threshold_file (str): Path to JSON file with saved thresholds
            confidence_level (str): Which confidence level to use ('68_percent', '90_percent', '95_percent', '99_percent')
            
        Returns:
            dict: Exit strategy execution result
        """
        try:
            print(f"\n🎯 === MIDNIGHT MOMENTUM EXIT STRATEGY ===")
            print(f"Symbol: {self.ticker}")
            print(f"Confidence Level: {confidence_level}")
            
            # Step 1: Load thresholds from JSON file
            if threshold_file is None:
                # Look for most recent threshold file for this ticker
                thresholds_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'thresholds')
                
                if not os.path.exists(thresholds_dir):
                    return {
                        'status': 'error',
                        'reason': 'No thresholds directory found',
                        'timestamp': datetime.now()
                    }
                
                # Find most recent threshold file for this ticker
                threshold_files = [f for f in os.listdir(thresholds_dir) if f.startswith(f"{self.ticker}_thresholds_") and f.endswith('.json')]
                
                if not threshold_files:
                    return {
                        'status': 'error',
                        'reason': f'No threshold files found for {self.ticker}',
                        'timestamp': datetime.now()
                    }
                
                # Use most recent file
                threshold_files.sort(reverse=True)
                threshold_file = os.path.join(thresholds_dir, threshold_files[0])
                print(f"📂 Using threshold file: {threshold_files[0]}")
            
            # Load threshold data
            with open(threshold_file, 'r') as f:
                threshold_data = json.load(f)
            
            if threshold_data['symbol'] != self.ticker:
                return {
                    'status': 'error',
                    'reason': f"Threshold file is for {threshold_data['symbol']}, not {self.ticker}",
                    'timestamp': datetime.now()
                }
            
            # Get target price for specified confidence level
            if confidence_level not in threshold_data['thresholds']:
                return {
                    'status': 'error',
                    'reason': f"Confidence level {confidence_level} not found in threshold data",
                    'timestamp': datetime.now()
                }
            
            target_price = threshold_data['thresholds'][confidence_level]['target_price']
            potential_percent = threshold_data['thresholds'][confidence_level]['potential_percent']
            
            print(f"🎯 Target Price: ${target_price:.2f} ({potential_percent}% potential)")
            
            # Step 2: Get current account positions
            try:
                order_handler = OrderHandler()
                account_info = order_handler.get_account()
                
                if not account_info:
                    return {
                        'status': 'error',
                        'reason': 'Failed to retrieve account information',
                        'timestamp': datetime.now()
                    }
                
                print(f"✅ Connected to account: {order_handler.account_number}")
                
            except Exception as e:
                return {
                    'status': 'error',
                    'reason': f'Failed to initialize order handler: {str(e)}',
                    'timestamp': datetime.now()
                }
            
            # Step 3: Check for existing position
            current_position = None
            if 'positions' in account_info:
                for position in account_info['positions']:
                    if position.get('instrument', {}).get('symbol') == self.ticker:
                        current_position = position
                        break
            
            if not current_position:
                return {
                    'status': 'rejected',
                    'reason': f'No existing position found in {self.ticker}',
                    'timestamp': datetime.now()
                }
            
            # Calculate position quantity (long - short)
            long_qty = current_position.get('longQuantity', 0)
            short_qty = current_position.get('shortQuantity', 0)
            net_position = long_qty - short_qty
            
            print(f"📊 Current Position: {net_position} shares of {self.ticker}")
            print(f"   Long: {long_qty}, Short: {short_qty}")
            
            if net_position == 0:
                return {
                    'status': 'rejected',
                    'reason': f'No net position in {self.ticker}',
                    'timestamp': datetime.now()
                }
            
            # Step 4: Determine order action and shares
            if net_position > 0:
                # Long position - place SELL limit order
                action = 'SELL'
                shares_to_trade = net_position
            else:
                # Short position - place BUY_TO_COVER limit order
                action = 'BUY_TO_COVER'
                shares_to_trade = abs(net_position)
            
            print(f"🎯 Exit Strategy: {action} {shares_to_trade} shares at ${target_price:.2f}")
            
            # Step 5: Execute limit order
            print(f"🚀 Executing {action} limit order for {shares_to_trade} shares at ${target_price:.2f}")
            
            if action == 'SELL':
                order_result = order_handler.sell_limit(
                    symbol=self.ticker,
                    shares=shares_to_trade,
                    limit_price=target_price
                )
            else:  # BUY_TO_COVER
                order_result = order_handler.buy_to_cover_limit(
                    symbol=self.ticker,
                    shares=shares_to_trade,
                    limit_price=target_price
                )
            
            if order_result['status'] == 'submitted':
                # Increment order counter and track order
                self.order_counter += 1
                order_record = {
                    'order_id': order_result.get('order_id', 'N/A'),
                    'symbol': order_result['symbol'],
                    'action': action,
                    'shares': order_result['shares'],
                    'limit_price': order_result['limit_price'],
                    'timestamp': datetime.now().isoformat(),
                    'order_type': 'closing'
                }
                self.orders_placed_today.append(order_record)
                
                print(f"✅ Exit order submitted successfully!")
                print(f"   Order ID: {order_result.get('order_id', 'N/A')}")
                print(f"   Action: {order_result['action_type']}")
                print(f"   Shares: {order_result['shares']}")
                print(f"   Limit Price: ${order_result['limit_price']:.2f}")
                print(f"   Order Counter: {self.order_counter}")
                if order_result.get('dollar_amount'):
                    print(f"   Estimated Proceeds: ${order_result['dollar_amount']:.2f}")
            else:
                print(f"❌ Exit order failed: {order_result.get('reason', 'Unknown error')}")
            
            return order_result
                
        except Exception as e:
            error_msg = f"Error executing exit strategy: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                'status': 'error',
                'reason': error_msg,
                'timestamp': datetime.now()
            }
    
    def check_existing_thresholds_for_today(self):
        """
        Check if thresholds for today already exist.
        
        Returns:
            tuple: (exists: bool, filepath: str or None)
        """
        try:
            thresholds_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'thresholds')
            
            if not os.path.exists(thresholds_dir):
                return False, None
            
            # Look for threshold file for this ticker
            threshold_file = os.path.join(thresholds_dir, f"{self.ticker}_thresholds.json")
            
            if not os.path.exists(threshold_file):
                return False, None
            
            # Load and check the generated date
            with open(threshold_file, 'r') as f:
                threshold_data = json.load(f)
            
            # Get today's date
            today = datetime.now().strftime('%Y-%m-%d')
            
            # Check if generated_at is today
            generated_at = threshold_data.get('generated_at', '')
            if generated_at.startswith(today):
                print(f"📋 Found existing thresholds for today: {threshold_file}")
                print(f"   Generated at: {generated_at}")
                return True, threshold_file
            else:
                print(f"📋 Found thresholds but for different date: {generated_at[:10]} (need {today})")
                return False, threshold_file
                
        except Exception as e:
            print(f"⚠️ Error checking existing thresholds: {str(e)}")
            return False, None
    
    def run_continuous_monitor(self, shares=None, method='percentage', percentage=5.0, check_interval=30):
        """
        Continuously monitor for trading window and execute trades automatically.
        
        Args:
            shares (int): Number of shares to trade (if None, will calculate based on method)
            method (str): Share calculation method ('fixed' or 'percentage')
            percentage (float): Percentage of equity to use (default 5.0%)
            check_interval (int): Check interval in seconds (default 30)
        """
        print(f"\n🔄 === CONTINUOUS MIDNIGHT MOMENTUM MONITOR ===")
        print(f"Symbol: {self.ticker}")
        print(f"Share Calculation: {method} ({percentage}% of equity)" if method == 'percentage' else f"Share Calculation: {method} ({shares} shares)")
        print(f"Check Interval: {check_interval} seconds")
        print(f"Target Trading Window: 12:58 PST ±2 minutes")
        print(f"\n🚀 Starting continuous monitoring... (Press Ctrl+C to stop)")
        
        # Set up signal handler for graceful shutdown
        def signal_handler(signum, frame):
            print(f"\n⚠️ Received interrupt signal. Shutting down gracefully...")
            raise KeyboardInterrupt
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Check if thresholds for today already exist
        thresholds_exist, threshold_file = self.check_existing_thresholds_for_today()
        
        if thresholds_exist:
            print(f"✅ Using existing thresholds for today - skipping calculations")
            print(f"📂 Threshold file: {threshold_file}")
        else:
            print(f"📊 No current thresholds found - running calculations...")
            
            # Fetch initial data
            if not self.fetch_data():
                print("❌ Failed to fetch initial data. Exiting.")
                return
            
            # Save thresholds for potential exit strategy
            threshold_file = self.save_thresholds_to_json()
            if threshold_file:
                print(f"💾 Thresholds calculated and saved: {threshold_file}")
            else:
                print("❌ Failed to generate thresholds. Exiting.")
                return
        
        trade_executed = False
        last_status_time = None
        
        try:
            while True:
                current_time = datetime.now()
                
                # Check trading window status
                timing_info = self.is_in_trading_window()
                
                if 'error' in timing_info:
                    print(f"❌ Error checking trading window: {timing_info['error']}")
                    time.sleep(check_interval)
                    continue
                
                # Display status every 5 minutes or when status changes
                show_status = (
                    last_status_time is None or 
                    (current_time - last_status_time).total_seconds() >= 300 or
                    timing_info['in_trading_window']
                )
                
                if show_status:
                    print(f"\n⏰ [{current_time.strftime('%H:%M:%S')}] Trading Window Status:")
                    if timing_info['in_trading_window']:
                        print(f"🟢 IN TRADING WINDOW - Ready to execute!")
                    else:
                        if timing_info['minutes_until_target'] > 0:
                            print(f"🟡 WAITING - {timing_info['minutes_until_target']} minutes until window")
                        else:
                            print(f"🔴 WINDOW CLOSED - {abs(timing_info['minutes_until_target'])} minutes ago")
                    last_status_time = current_time
                
                # Execute trade if in window and not already executed
                if timing_info['in_trading_window'] and not trade_executed:
                    print(f"\n🎯 TRADING WINDOW DETECTED - Executing trade...")
                    
                    # Calculate shares if not provided
                    if shares is None:
                        calculated_shares = self.calculate_shares(method=method, percentage=percentage)
                        if calculated_shares <= 0:
                            print("❌ Failed to calculate shares. Skipping trade.")
                            time.sleep(check_interval)
                            continue
                    else:
                        calculated_shares = shares
                    
                    # Execute the trade
                    trade_result = self.execute_midnight_momentum_trade(shares=calculated_shares)
                    
                    if trade_result['status'] == 'submitted':
                        print(f"✅ Trade executed successfully!")
                        print(f"   Order ID: {trade_result.get('order_id', 'N/A')}")
                        print(f"   Shares: {trade_result['shares']}")
                        print(f"   Symbol: {trade_result['symbol']}")
                        trade_executed = True
                        
                        print(f"\n🎯 Trade completed. Monitor will continue running for status updates.")
                        print(f"💡 Use Ctrl+C to stop monitoring.")
                        
                    elif trade_result['status'] == 'rejected':
                        if 'Already have position' in trade_result.get('reason', ''):
                            print(f"ℹ️ Position already exists - no trade needed")
                            trade_executed = True
                        else:
                            print(f"⚠️ Trade rejected: {trade_result.get('reason', 'Unknown')}")
                    else:
                        print(f"❌ Trade failed: {trade_result.get('reason', 'Unknown error')}")
                
                # Reset trade flag when window closes (for next day)
                if not timing_info['in_trading_window'] and trade_executed:
                    # Check if we've moved to the next day
                    if timing_info['minutes_until_target'] > 720:  # More than 12 hours until next window
                        print(f"\n🔄 New trading day detected. Resetting for next opportunity.")
                        trade_executed = False
                        
                        # Refresh data for new day
                        print(f"📊 Refreshing market data...")
                        if self.fetch_data():
                            # Update thresholds for new day
                            new_threshold_file = self.save_thresholds_to_json()
                            if new_threshold_file:
                                print(f"💾 Updated thresholds saved: {new_threshold_file}")
                
                # Sleep before next check
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print(f"\n⚠️ Monitoring stopped by user")
            print(f"📊 Final Status:")
            print(f"   Trade Executed: {'Yes' if trade_executed else 'No'}")
            if threshold_file:
                print(f"   Threshold File: {threshold_file}")
            print(f"✅ Shutdown complete")
        
        except Exception as e:
            print(f"\n❌ Error in continuous monitoring: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def run_analysis(self):
        """
        Run the complete midnight momentum analysis.
        
        Returns:
            pd.DataFrame: Results of the analysis
        """
        print(f"\n=== Midnight Momentum Strategy Analysis for {self.ticker} ===")
        
        # Fetch data
        if not self.fetch_data():
            return None
        
        # Calculate and display results
        results = self.get_last_15_closes_with_thresholds()
        
        if results is not None:
            print(f"\nLast 15 Daily Closes with Upside Potential Thresholds:")
            print("=" * 120)
            print(results.to_string(index=False))
            
            # Summary statistics
            print(f"\n=== Summary Statistics ===")
            print(f"Average 68% Upside Potential: {results['Potential_68%'].mean():.2f}%")
            print(f"Average 90% Upside Potential: {results['Potential_90%'].mean():.2f}%")
            print(f"Average 95% Upside Potential: {results['Potential_95%'].mean():.2f}%")
            print(f"Average 99% Upside Potential: {results['Potential_99%'].mean():.2f}%")
            
            print(f"\nMaximum 68% Upside Potential: {results['Potential_68%'].max():.2f}%")
            print(f"Minimum 68% Upside Potential: {results['Potential_68%'].min():.2f}%")
            
            
        # Get next day prediction
        next_day_prediction = self.get_next_day_prediction()
        
        if next_day_prediction is not None:
            print(f"\n🎯 === NEXT TRADING DAY PREDICTION ===")
            print(f"Based on close: ${next_day_prediction['current_close']} ({next_day_prediction['current_date']})")
            print(f"68% Confidence Target: ${next_day_prediction['next_day_targets']['68%']} ({next_day_prediction['next_day_potentials']['68%']}% potential)")
            print(f"90% Confidence Target: ${next_day_prediction['next_day_targets']['90%']} ({next_day_prediction['next_day_potentials']['90%']}% potential)")
            print(f"95% Confidence Target: ${next_day_prediction['next_day_targets']['95%']} ({next_day_prediction['next_day_potentials']['95%']}% potential)")
            print(f"99% Confidence Target: ${next_day_prediction['next_day_targets']['99%']} ({next_day_prediction['next_day_potentials']['99%']}% potential)")
            
            # Automatically save thresholds to JSON
            threshold_file = self.save_thresholds_to_json()
            if threshold_file:
                print(f"\n💾 Thresholds automatically saved to: {threshold_file}")
            
        return results


def main():
    """
    Main function to run the strategy analysis and continuous monitoring.
    """
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Midnight Momentum Strategy - Calculate upside profit potential thresholds and monitor for trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 midnightMomentum_strategy.py AAPL
  python3 midnightMomentum_strategy.py NVDA
  python3 midnightMomentum_strategy.py TSLA
        """
    )
    
    parser.add_argument(
        'ticker',
        type=str,
        help='Stock ticker symbol to analyze (e.g., AAPL, NVDA, TSLA)'
    )
    
    parser.add_argument(
        '--analysis-only',
        action='store_true',
        help='Run analysis only without continuous monitoring'
    )
    
    parser.add_argument(
        '--percentage',
        type=float,
        default=5.0,
        help='Percentage of equity to use for position sizing (default: 5.0%%)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Convert ticker to uppercase for consistency
    ticker = args.ticker.upper()
    
    print("=" * 80)
    print("MIDNIGHT MOMENTUM STRATEGY")
    print("Profit Potential (Upside) Threshold Calculator")
    print("=" * 80)
    
    try:
        # Initialize strategy
        strategy = MidnightMomentumStrategy(ticker)
        
        # Check trading window status
        in_trading_window = strategy.check_trading_status()
        
        # Check if thresholds for today already exist
        thresholds_exist, threshold_file = strategy.check_existing_thresholds_for_today()
        
        if thresholds_exist:
            print(f"\n✅ Using existing thresholds for today - skipping analysis")
            print(f"📂 Threshold file: {threshold_file}")
            
            if not args.analysis_only:
                print(f"\n🔄 Starting continuous monitoring...")
                strategy.run_continuous_monitor(method='percentage', percentage=args.percentage)
            else:
                print(f"\n⏳ Analysis complete - use --continuous to start monitoring")
        else:
            print(f"\n📊 No current thresholds found - running analysis...")
            
            # Run analysis
            results = strategy.run_analysis()
            
            if results is not None:
                print(f"\n✅ Analysis completed successfully for {ticker}")
                print(f"📊 Generated upside thresholds for last 15 trading days")
                
                if args.analysis_only:
                    if in_trading_window:
                        print(f"🚀 Ready for midnight momentum trading!")
                    else:
                        print(f"⏳ Analysis ready - run again without --analysis-only to start monitoring")
                else:
                    print(f"\n🔄 Starting continuous monitoring...")
                    strategy.run_continuous_monitor(method='percentage', percentage=args.percentage)
            else:
                print(f"\n❌ Analysis failed for {ticker}")
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

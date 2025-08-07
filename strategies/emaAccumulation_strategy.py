#!/usr/bin/env python3
"""
EMA Accumulation Live Trading Strategy

A real-time trading system that implements the EMA Accumulation strategy on live market data.
The strategy accumulates shares when price is below the 21-EMA and exits at 0.5% profit from average entry price.

Key Features:
- Real-time 1-minute data fetching and analysis
- Dynamic EMA calculation with rolling data window
- Minute-by-minute signal evaluation during trading hours
- Position tracking via account data integration
- Rolling average entry price monitoring
- Automatic profit target calculation and execution
- Comprehensive logging and error handling

Strategy Logic:
- Entry: Buy 1 share every minute when current_price < 21-period EMA
- Exit: Sell entire position when current_price >= (average_entry_price * 1.005) - 0.5% profit target
- No stop loss - pure accumulation strategy
- Only operates during regular trading hours (9:30 AM - 4:00 PM ET)

Author: Trading System
Date: 2025-01-06
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
import threading
from typing import Dict, Any, List, Optional, Tuple

# Add the parent directory to the path to import handlers
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from handlers.connection_manager import ensure_valid_tokens
from handlers.historical_data_handler import HistoricalDataHandler
from handlers.order_handler import OrderHandler


class EMAAccumulationStrategy:
    """
    EMA Accumulation Live Trading Strategy class for real-time market operations.
    """
    
    def __init__(self, symbol, ema_period=21, profit_target=0.005, max_position_size=1000):
        """
        Initialize the EMA Accumulation strategy.
        
        Args:
            symbol (str): Stock symbol to trade
            ema_period (int): EMA period for trend filter (default: 21)
            profit_target (float): Profit target as decimal (default: 0.005 = 0.5%)
            max_position_size (int): Maximum position size in shares (default: 1000)
        """
        self.symbol = symbol.upper()
        self.ema_period = ema_period
        self.profit_target = profit_target
        self.max_position_size = max_position_size
        
        # Initialize handlers
        self.data_handler = HistoricalDataHandler()
        self.order_handler = None  # Initialize when needed
        
        # Data management
        self.price_data = pd.DataFrame()  # Rolling window of price data
        self.current_ema = None
        self.last_update_time = None
        
        # Position tracking
        self.current_position = 0
        self.average_entry_price = 0.0
        self.total_cost_basis = 0.0
        self.profit_target_price = 0.0
        
        # Trading state
        self.is_running = False
        self.orders_today = []
        self.last_signal_time = None
        
        # Logging
        self.setup_logging()
        
    def setup_logging(self):
        """Set up logging for the strategy."""
        import logging
        
        # Create logs directory if it doesn't exist
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure logging
        log_filename = f'{log_dir}/ema_accumulation_{self.symbol}_{datetime.now().strftime("%Y%m%d")}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(f'EMAAccumulation_{self.symbol}')
        self.logger.info(f"EMA Accumulation Strategy initialized for {self.symbol}")
        self.logger.info(f"Parameters: EMA={self.ema_period}, Target={self.profit_target*100:.1f}%, Max Position={self.max_position_size}")
    
    def is_trading_hours(self):
        """
        Check if current time is within regular trading hours (9:30 AM - 4:00 PM ET).
        
        Returns:
            dict: Trading hours status information
        """
        try:
            # Get current time in Eastern timezone
            eastern_tz = pytz.timezone('US/Eastern')
            current_time_et = datetime.now(eastern_tz)
            
            # Define trading hours
            market_open = current_time_et.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = current_time_et.replace(hour=16, minute=0, second=0, microsecond=0)
            
            # Check if it's a weekday (Monday=0, Sunday=6)
            is_weekday = current_time_et.weekday() < 5
            
            # Check if within trading hours
            in_trading_hours = (
                is_weekday and 
                market_open <= current_time_et <= market_close
            )
            
            # Calculate time until market open/close
            if current_time_et < market_open:
                time_until_event = market_open - current_time_et
                next_event = "market_open"
            elif current_time_et > market_close:
                # Calculate next market open (tomorrow or Monday)
                next_open = market_open + timedelta(days=1)
                if next_open.weekday() >= 5:  # Weekend
                    days_to_add = 7 - next_open.weekday()
                    next_open = next_open + timedelta(days=days_to_add)
                time_until_event = next_open - current_time_et
                next_event = "market_open"
            else:
                time_until_event = market_close - current_time_et
                next_event = "market_close"
            
            return {
                'in_trading_hours': in_trading_hours,
                'current_time_et': current_time_et,
                'market_open': market_open,
                'market_close': market_close,
                'is_weekday': is_weekday,
                'time_until_event': time_until_event,
                'next_event': next_event,
                'minutes_until_event': int(time_until_event.total_seconds() / 60)
            }
            
        except Exception as e:
            self.logger.error(f"Error checking trading hours: {e}")
            return {
                'in_trading_hours': False,
                'error': str(e)
            }
    
    def fetch_realtime_data(self, lookback_periods=50):
        """
        Fetch recent 1-minute data for EMA calculation.
        
        Args:
            lookback_periods (int): Number of 1-minute bars to fetch
            
        Returns:
            bool: True if data fetched successfully
        """
        try:
            self.logger.debug(f"Fetching real-time data for {self.symbol}")
            
            # Fetch 1-minute data for the current day
            raw_data = self.data_handler.fetch_historical_data(
                symbol=self.symbol,
                periodType="day",
                period=1,
                frequencyType="minute",
                freq=1,
                needExtendedHoursData=False
            )
            
            if not raw_data or 'candles' not in raw_data:
                self.logger.warning(f"No data retrieved for {self.symbol}")
                return False
            
            # Convert to DataFrame
            candles = raw_data['candles']
            df = pd.DataFrame(candles)
            
            if df.empty:
                self.logger.warning(f"Empty data received for {self.symbol}")
                return False
            
            # Convert datetime and sort
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime').reset_index(drop=True)
            
            # Filter to trading hours only
            df = self.filter_trading_hours(df)
            
            if df.empty:
                self.logger.warning(f"No trading hours data for {self.symbol}")
                return False
            
            # Keep only the most recent data points for EMA calculation
            self.price_data = df.tail(lookback_periods).copy()
            self.last_update_time = datetime.now()
            
            self.logger.debug(f"Updated price data: {len(self.price_data)} bars, latest: {self.price_data['datetime'].iloc[-1]}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error fetching real-time data: {e}")
            return False
    
    def filter_trading_hours(self, df):
        """
        Filter data to include only regular trading hours.
        
        Args:
            df (pd.DataFrame): Raw OHLCV data
            
        Returns:
            pd.DataFrame: Filtered data for trading hours only
        """
        if df.empty:
            return df
        
        try:
            # Convert to Eastern timezone
            df_copy = df.copy()
            
            if df_copy['datetime'].dt.tz is None:
                df_copy['datetime_et'] = df_copy['datetime'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
            else:
                df_copy['datetime_et'] = df_copy['datetime'].dt.tz_convert('US/Eastern')
            
            # Extract hour and minute
            df_copy['hour'] = df_copy['datetime_et'].dt.hour
            df_copy['minute'] = df_copy['datetime_et'].dt.minute
            
            # Filter for trading hours (9:30 AM - 4:00 PM ET)
            trading_hours_mask = (
                ((df_copy['hour'] == 9) & (df_copy['minute'] >= 30)) |
                ((df_copy['hour'] >= 10) & (df_copy['hour'] < 16))
            )
            
            filtered_df = df_copy[trading_hours_mask].copy()
            
            # Remove temporary columns
            if 'datetime_et' in filtered_df.columns:
                filtered_df = filtered_df.drop(['datetime_et', 'hour', 'minute'], axis=1)
            
            return filtered_df.reset_index(drop=True)
            
        except Exception as e:
            self.logger.error(f"Error filtering trading hours: {e}")
            return df
    
    def calculate_ema(self):
        """
        Calculate the current EMA value from price data.
        
        Returns:
            float: Current EMA value or None if insufficient data
        """
        try:
            if self.price_data.empty or len(self.price_data) < self.ema_period:
                self.logger.warning(f"Insufficient data for EMA calculation: {len(self.price_data)} bars (need {self.ema_period})")
                return None
            
            # Calculate EMA using pandas ewm
            ema_series = self.price_data['close'].ewm(span=self.ema_period, adjust=False).mean()
            self.current_ema = ema_series.iloc[-1]
            
            self.logger.debug(f"EMA calculated: {self.current_ema:.2f}")
            return self.current_ema
            
        except Exception as e:
            self.logger.error(f"Error calculating EMA: {e}")
            return None
    
    def get_current_position(self):
        """
        Get current position information from account data.
        
        Returns:
            bool: True if position data retrieved successfully
        """
        try:
            if self.order_handler is None:
                self.order_handler = OrderHandler()
            
            account_info = self.order_handler.get_account()
            
            if not account_info:
                self.logger.error("Failed to retrieve account information")
                return False
            
            # Reset position data
            self.current_position = 0
            self.average_entry_price = 0.0
            self.total_cost_basis = 0.0
            
            # Find position for our symbol
            if 'positions' in account_info:
                for position in account_info['positions']:
                    if position.get('instrument', {}).get('symbol') == self.symbol:
                        long_qty = position.get('longQuantity', 0)
                        short_qty = position.get('shortQuantity', 0)
                        self.current_position = long_qty - short_qty
                        
                        if self.current_position > 0:
                            # Calculate average entry price from account data
                            market_value = position.get('marketValue', 0)
                            if market_value > 0:
                                current_price = self.get_current_price()
                                if current_price:
                                    self.total_cost_basis = self.current_position * (market_value / self.current_position - (current_price - market_value / self.current_position))
                                    self.average_entry_price = self.total_cost_basis / self.current_position
                            
                            # Calculate profit target price
                            self.profit_target_price = self.average_entry_price * (1 + self.profit_target)
                        
                        break
            
            self.logger.debug(f"Position update: {self.current_position} shares, avg entry: ${self.average_entry_price:.2f}, target: ${self.profit_target_price:.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error getting current position: {e}")
            return False
    
    def get_current_price(self):
        """
        Get the most recent price from our data.
        
        Returns:
            float: Current price or None if no data
        """
        try:
            if self.price_data.empty:
                return None
            
            return self.price_data['close'].iloc[-1]
            
        except Exception as e:
            self.logger.error(f"Error getting current price: {e}")
            return None
    
    def should_buy(self):
        """
        Determine if we should place a buy order.
        
        Returns:
            dict: Buy decision with reasoning
        """
        try:
            current_price = self.get_current_price()
            
            if current_price is None:
                return {'should_buy': False, 'reason': 'No current price data'}
            
            if self.current_ema is None:
                return {'should_buy': False, 'reason': 'No EMA data'}
            
            # Check if price is below EMA
            if current_price >= self.current_ema:
                return {
                    'should_buy': False, 
                    'reason': f'Price ${current_price:.2f} >= EMA ${self.current_ema:.2f}'
                }
            
            # Check position size limits
            if self.current_position >= self.max_position_size:
                return {
                    'should_buy': False, 
                    'reason': f'Max position size reached: {self.current_position}/{self.max_position_size}'
                }
            
            # Check if we can afford 1 share (basic check)
            if current_price > 10000:  # Sanity check for extremely high prices
                return {
                    'should_buy': False, 
                    'reason': f'Price too high: ${current_price:.2f}'
                }
            
            return {
                'should_buy': True, 
                'reason': f'Price ${current_price:.2f} < EMA ${self.current_ema:.2f}',
                'current_price': current_price,
                'ema': self.current_ema
            }
            
        except Exception as e:
            self.logger.error(f"Error in should_buy: {e}")
            return {'should_buy': False, 'reason': f'Error: {str(e)}'}
    
    def should_sell(self):
        """
        Determine if we should place a sell order.
        
        Returns:
            dict: Sell decision with reasoning
        """
        try:
            if self.current_position <= 0:
                return {'should_sell': False, 'reason': 'No position to sell'}
            
            current_price = self.get_current_price()
            
            if current_price is None:
                return {'should_sell': False, 'reason': 'No current price data'}
            
            if self.profit_target_price <= 0:
                return {'should_sell': False, 'reason': 'No profit target calculated'}
            
            # Check if we've hit the profit target
            if current_price >= self.profit_target_price:
                profit_pct = ((current_price - self.average_entry_price) / self.average_entry_price) * 100
                return {
                    'should_sell': True,
                    'reason': f'Profit target hit: ${current_price:.2f} >= ${self.profit_target_price:.2f} ({profit_pct:.2f}% profit)',
                    'current_price': current_price,
                    'target_price': self.profit_target_price,
                    'profit_percent': profit_pct
                }
            
            return {
                'should_sell': False,
                'reason': f'Price ${current_price:.2f} < Target ${self.profit_target_price:.2f}',
                'current_price': current_price,
                'target_price': self.profit_target_price
            }
            
        except Exception as e:
            self.logger.error(f"Error in should_sell: {e}")
            return {'should_sell': False, 'reason': f'Error: {str(e)}'}
    
    def execute_buy_signal(self):
        """
        Execute a buy order for 1 share.
        
        Returns:
            dict: Order execution result
        """
        try:
            if self.order_handler is None:
                self.order_handler = OrderHandler()
            
            current_price = self.get_current_price()
            
            self.logger.info(f"Executing BUY order: 1 share of {self.symbol} at ~${current_price:.2f}")
            
            # Place market buy order for 1 share
            order_result = self.order_handler.buy_market(
                symbol=self.symbol,
                shares=1,
                current_price=current_price
            )
            
            if order_result['status'] == 'submitted':
                # Track the order
                order_record = {
                    'timestamp': datetime.now().isoformat(),
                    'action': 'BUY',
                    'symbol': self.symbol,
                    'shares': 1,
                    'order_id': order_result.get('order_id', 'N/A'),
                    'estimated_price': current_price
                }
                self.orders_today.append(order_record)
                self.last_signal_time = datetime.now()
                
                self.logger.info(f"✅ BUY order submitted: Order ID {order_result.get('order_id', 'N/A')}")
            else:
                self.logger.error(f"❌ BUY order failed: {order_result.get('reason', 'Unknown error')}")
            
            return order_result
            
        except Exception as e:
            error_msg = f"Error executing buy order: {str(e)}"
            self.logger.error(error_msg)
            return {
                'status': 'error',
                'reason': error_msg
            }
    
    def execute_sell_signal(self):
        """
        Execute a sell order for entire position.
        
        Returns:
            dict: Order execution result
        """
        try:
            if self.order_handler is None:
                self.order_handler = OrderHandler()
            
            if self.current_position <= 0:
                return {
                    'status': 'error',
                    'reason': 'No position to sell'
                }
            
            current_price = self.get_current_price()
            
            self.logger.info(f"Executing SELL order: {self.current_position} shares of {self.symbol} at ~${current_price:.2f}")
            
            # Place market sell order for entire position
            order_result = self.order_handler.sell_market(
                symbol=self.symbol,
                shares=self.current_position,
                current_price=current_price
            )
            
            if order_result['status'] == 'submitted':
                # Calculate estimated profit
                estimated_profit = (current_price - self.average_entry_price) * self.current_position
                profit_pct = ((current_price - self.average_entry_price) / self.average_entry_price) * 100
                
                # Track the order
                order_record = {
                    'timestamp': datetime.now().isoformat(),
                    'action': 'SELL',
                    'symbol': self.symbol,
                    'shares': self.current_position,
                    'order_id': order_result.get('order_id', 'N/A'),
                    'estimated_price': current_price,
                    'estimated_profit': estimated_profit,
                    'profit_percent': profit_pct
                }
                self.orders_today.append(order_record)
                self.last_signal_time = datetime.now()
                
                self.logger.info(f"✅ SELL order submitted: Order ID {order_result.get('order_id', 'N/A')}")
                self.logger.info(f"💰 Estimated profit: ${estimated_profit:.2f} ({profit_pct:.2f}%)")
            else:
                self.logger.error(f"❌ SELL order failed: {order_result.get('reason', 'Unknown error')}")
            
            return order_result
            
        except Exception as e:
            error_msg = f"Error executing sell order: {str(e)}"
            self.logger.error(error_msg)
            return {
                'status': 'error',
                'reason': error_msg
            }
    
    def wait_for_next_minute(self):
        """
        Wait until the next minute boundary for synchronized execution.
        """
        try:
            current_time = datetime.now()
            next_minute = current_time.replace(second=0, microsecond=0) + timedelta(minutes=1)
            wait_seconds = (next_minute - current_time).total_seconds()
            
            if wait_seconds > 0:
                self.logger.debug(f"Waiting {wait_seconds:.1f} seconds until next minute")
                time.sleep(wait_seconds)
            
        except Exception as e:
            self.logger.error(f"Error in wait_for_next_minute: {e}")
            time.sleep(60)  # Fallback: wait 1 minute
    
    def print_status(self):
        """Print current strategy status."""
        try:
            current_price = self.get_current_price()
            trading_status = self.is_trading_hours()
            
            print(f"\n📊 === EMA ACCUMULATION STRATEGY STATUS ===")
            print(f"Symbol: {self.symbol}")
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Trading Hours: {'🟢 ACTIVE' if trading_status['in_trading_hours'] else '🔴 CLOSED'}")
            
            if current_price:
                print(f"Current Price: ${current_price:.2f}")
            
            if self.current_ema:
                print(f"21-EMA: ${self.current_ema:.2f}")
                if current_price:
                    ema_diff = ((current_price - self.current_ema) / self.current_ema) * 100
                    print(f"Price vs EMA: {ema_diff:+.2f}%")
            
            print(f"Position: {self.current_position} shares")
            if self.current_position > 0:
                print(f"Average Entry: ${self.average_entry_price:.2f}")
                print(f"Profit Target: ${self.profit_target_price:.2f}")
                if current_price:
                    unrealized_pnl = (current_price - self.average_entry_price) * self.current_position
                    unrealized_pct = ((current_price - self.average_entry_price) / self.average_entry_price) * 100
                    print(f"Unrealized P&L: ${unrealized_pnl:.2f} ({unrealized_pct:+.2f}%)")
            
            print(f"Orders Today: {len(self.orders_today)}")
            print(f"Data Points: {len(self.price_data)}")
            print("=" * 50)
            
        except Exception as e:
            self.logger.error(f"Error printing status: {e}")
    
    def run_continuous_monitor(self, status_interval=300):
        """
        Run the continuous monitoring loop.
        
        Args:
            status_interval (int): Seconds between status updates (default: 300 = 5 minutes)
        """
        self.logger.info(f"🚀 Starting EMA Accumulation Strategy for {self.symbol}")
        self.logger.info(f"Parameters: EMA={self.ema_period}, Target={self.profit_target*100:.1f}%, Max Position={self.max_position_size}")
        
        self.is_running = True
        last_status_time = None
        
        # Set up signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            self.logger.info("Received shutdown signal")
            self.is_running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            while self.is_running:
                current_time = datetime.now()
                
                # Check if we're in trading hours
                trading_status = self.is_trading_hours()
                
                if not trading_status['in_trading_hours']:
                    if trading_status.get('minutes_until_event', 0) > 60:
                        self.logger.info(f"Market closed. Next event: {trading_status['next_event']} in {trading_status['minutes_until_event']} minutes")
                        time.sleep(300)  # Check every 5 minutes when market is closed
                        continue
                    else:
                        self.logger.info(f"Market opening soon. Waiting {trading_status['minutes_until_event']} minutes")
                        time.sleep(60)  # Check every minute when close to market open
                        continue
                
                # We're in trading hours - execute strategy
                try:
                    # Step 1: Fetch latest data
                    if not self.fetch_realtime_data():
                        self.logger.warning("Failed to fetch data, retrying in 1 minute")
                        time.sleep(60)
                        continue
                    
                    # Step 2: Calculate EMA
                    if self.calculate_ema() is None:
                        self.logger.warning("Failed to calculate EMA, retrying in 1 minute")
                        time.sleep(60)
                        continue
                    
                    # Step 3: Get current position
                    if not self.get_current_position():
                        self.logger.warning("Failed to get position data, retrying in 1 minute")
                        time.sleep(60)
                        continue
                    
                    # Step 4: Check for sell signal first (exit before entry)
                    sell_decision = self.should_sell()
                    if sell_decision['should_sell']:
                        self.logger.info(f"🔴 SELL SIGNAL: {sell_decision['reason']}")
                        sell_result = self.execute_sell_signal()
                        
                        if sell_result['status'] == 'submitted':
                            self.logger.info("✅ Sell order executed successfully")
                        else:
                            self.logger.error(f"❌ Sell order failed: {sell_result.get('reason', 'Unknown')}")
                    
                    # Step 5: Check for buy signal
                    else:
                        buy_decision = self.should_buy()
                        if buy_decision['should_buy']:
                            self.logger.info(f"🟢 BUY SIGNAL: {buy_decision['reason']}")
                            buy_result = self.execute_buy_signal()
                            
                            if buy_result['status'] == 'submitted':
                                self.logger.info("✅ Buy order executed successfully")
                            else:
                                self.logger.error(f"❌ Buy order failed: {buy_result.get('reason', 'Unknown')}")
                        else:
                            self.logger.debug(f"No buy signal: {buy_decision['reason']}")
                    
                    # Step 6: Print status periodically
                    if (last_status_time is None or 
                        (current_time - last_status_time).total_seconds() >= status_interval):
                        self.print_status()
                        last_status_time = current_time
                    
                except Exception as e:
                    self.logger.error(f"Error in strategy loop: {e}")
                
                # Wait until next minute for synchronized execution
                self.wait_for_next_minute()
        
        except KeyboardInterrupt:
            self.logger.info("Strategy interrupted by user")
        except Exception as e:
            self.logger.error(f"Fatal error in continuous monitor: {e}")
        finally:
            self.is_running = False
            self.logger.info("EMA Accumulation Strategy stopped")
            
            # Print final summary
            if self.orders_today:
                self.logger.info(f"📊 Final Summary: {len(self.orders_today)} orders executed today")
                for order in self.orders_today:
                    self.logger.info(f"  {order['action']} {order['shares']} shares at {order['timestamp']}")
    
    def run_analysis_mode(self):
        """
        Run in analysis mode - fetch data and show current status without trading.
        """
        self.logger.info(f"📊 Running EMA Accumulation Analysis for {self.symbol}")
        
        try:
            # Check trading hours
            trading_status = self.is_trading_hours()
            print(f"Trading Hours: {'🟢 ACTIVE' if trading_status['in_trading_hours'] else '🔴 CLOSED'}")
            
            # Fetch data
            if self.fetch_realtime_data():
                print(f"✅ Data fetched: {len(self.price_data)} bars")
                
                # Calculate EMA
                if self.calculate_ema():
                    print(f"✅ EMA calculated: ${self.current_ema:.2f}")
                    
                    # Get position
                    if self.get_current_position():
                        print(f"✅ Position data retrieved")
                        
                        # Show current status
                        self.print_status()
                        
                        # Show signals
                        buy_decision = self.should_buy()
                        sell_decision = self.should_sell()
                        
                        print(f"\n🔍 SIGNAL ANALYSIS:")
                        print(f"Buy Signal: {'🟢 YES' if buy_decision['should_buy'] else '🔴 NO'} - {buy_decision['reason']}")
                        print(f"Sell Signal: {'🟢 YES' if sell_decision['should_sell'] else '🔴 NO'} - {sell_decision['reason']}")
                        
                        return True
                    else:
                        print(f"❌ Failed to get position data")
                else:
                    print(f"❌ Failed to calculate EMA")
            else:
                print(f"❌ Failed to fetch data")
                
        except Exception as e:
            self.logger.error(f"Error in analysis mode: {e}")
            print(f"❌ Analysis failed: {str(e)}")
            
        return False


def main():
    """
    Main function to run the EMA Accumulation live trading strategy.
    """
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="EMA Accumulation Live Trading Strategy - Real-time accumulation trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run live trading strategy for AAPL
  python3 emaAccumulation_strategy.py AAPL
  
  # Run analysis mode only (no trading)
  python3 emaAccumulation_strategy.py AAPL --analysis-only
  
  # Run with custom parameters
  python3 emaAccumulation_strategy.py NVDA --ema-period 50 --profit-target 0.01 --max-position 500
        """
    )
    
    parser.add_argument(
        'symbol',
        type=str,
        help='Stock symbol to trade (e.g., AAPL, NVDA, TSLA)'
    )
    
    parser.add_argument(
        '--analysis-only',
        action='store_true',
        help='Run analysis mode only without live trading'
    )
    
    parser.add_argument(
        '--ema-period',
        type=int,
        default=21,
        help='EMA period for trend filter (default: 21)'
    )
    
    parser.add_argument(
        '--profit-target',
        type=float,
        default=0.005,
        help='Profit target as decimal (default: 0.005 = 0.5%%)'
    )
    
    parser.add_argument(
        '--max-position',
        type=int,
        default=1000,
        help='Maximum position size in shares (default: 1000)'
    )
    
    parser.add_argument(
        '--status-interval',
        type=int,
        default=300,
        help='Status update interval in seconds (default: 300 = 5 minutes)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Convert symbol to uppercase for consistency
    symbol = args.symbol.upper()
    
    print("=" * 80)
    print("EMA ACCUMULATION LIVE TRADING STRATEGY")
    print("Real-Time Market Operations")
    print("=" * 80)
    print(f"Symbol: {symbol}")
    print(f"EMA Period: {args.ema_period}")
    print(f"Profit Target: {args.profit_target*100:.1f}%")
    print(f"Max Position: {args.max_position} shares")
    print(f"Mode: {'Analysis Only' if args.analysis_only else 'Live Trading'}")
    print("=" * 80)
    
    try:
        # Initialize strategy
        strategy = EMAAccumulationStrategy(
            symbol=symbol,
            ema_period=args.ema_period,
            profit_target=args.profit_target,
            max_position_size=args.max_position
        )
        
        if args.analysis_only:
            # Run analysis mode
            print(f"\n📊 Running analysis mode for {symbol}...")
            success = strategy.run_analysis_mode()
            
            if success:
                print(f"\n✅ Analysis completed successfully for {symbol}")
            else:
                print(f"\n❌ Analysis failed for {symbol}")
        else:
            # Run live trading
            print(f"\n🚀 Starting live trading for {symbol}...")
            print(f"⚠️  WARNING: This will place real orders with real money!")
            print(f"💡 Press Ctrl+C to stop the strategy safely")
            
            # Add a confirmation prompt for live trading
            try:
                confirmation = input(f"\nConfirm live trading for {symbol}? (yes/no): ").lower().strip()
                if confirmation not in ['yes', 'y']:
                    print("Live trading cancelled by user")
                    return
            except KeyboardInterrupt:
                print("\nLive trading cancelled by user")
                return
            
            strategy.run_continuous_monitor(status_interval=args.status_interval)
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Strategy interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

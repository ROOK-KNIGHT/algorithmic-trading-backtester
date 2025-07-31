#!/usr/bin/env python3
"""
DEMA-SMA Crossover Trading Strategy
Uses DEMA (Double Exponential Moving Average) and SMA (Simple Moving Average) to identify trading opportunities
"""
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import pytz
import argparse
import sys
from typing import Dict, Any, List, Tuple, Optional

# Import our modular components
from modules import SchwabapiAuth, SchwabapiAccount, SchwabapiMarketData, SchwabapiTrading

class DEMASMACrossoverStrategy:
    def __init__(self, symbol, lengthDEMA=15, lengthSMA_OPEN=30, auto_mode=False, shares=42, direction='both'):
        """
        Initialize the DEMA-SMA Crossover Trading Strategy.
        
        Args:
            symbol: Stock symbol to trade
            lengthDEMA: Length of DEMA calculation period
            lengthSMA_OPEN: Length of SMA calculation period for open price
            auto_mode: Whether to execute trades automatically
            shares: Number of shares to trade (default: 42)
            direction: Trading direction ('long', 'short', or 'both')
        """
        self.symbol = symbol.upper()
        self.lengthDEMA = lengthDEMA
        self.lengthSMA_OPEN = lengthSMA_OPEN
        self.auto_mode = auto_mode
        self.shares = shares
        self.direction = direction
        self.orders_placed = 0  # Counter to prevent duplicate orders
        
        # Initialize market timezones
        self.market_timezone = pytz.timezone('America/New_York')  # NYSE/NASDAQ market timezone
        self.pacific_timezone = pytz.timezone('America/Los_Angeles')
        
        # Initialize our API modules
        print(f"Initializing API connection for {symbol}...")
        self.auth = SchwabapiAuth()
        self.account = SchwabapiAccount(self.auth)
        self.market = SchwabapiMarketData(self.auth)
        self.trading = SchwabapiTrading(self.auth, self.account)
        
        # Position tracking
        self.current_position = "FLAT"  # Can be "FLAT", "LONG", or "SHORT"
        self.position_size = 0
        self.entry_price_long = 0
        self.entry_price_short = 0
        self.entry_time = None
        self.position_id = None
        
        # Strategy specific data
        self.prev_distance = None
        
        # Sync position at startup
        self.sync_current_positions()
    
    def sync_current_positions(self) -> bool:
        """Get current positions and sync internal state for our symbol."""
        try:
            # Use the trading module to get position for our symbol
            position = self.trading.sync_positions(self.symbol)
            
            # Check if position exists and update our tracking variables
            position_size = abs(position.get('quantity', 0))
            
            if position_size <= 0:
                self.current_position = "FLAT"
                self.position_size = 0
                self.entry_price_long = 0
                self.entry_price_short = 0
                self.entry_time = None
                return True
            
            # Set position type based on quantity
            if position.get('quantity', 0) > 0:
                self.current_position = "LONG"
                self.position_size = position_size
                self.entry_price_long = position.get('average_price', 0)
            else:
                self.current_position = "SHORT"
                self.position_size = position_size
                self.entry_price_short = position.get('average_price', 0)
            
            return True
        except Exception as e:
            print(f"Error syncing positions: {e}")
            return False
    
    def calculate_dema(self, prices, length):
        """Calculate Double Exponential Moving Average."""
        ema1 = prices.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        dema = (2 * ema1) - ema2
        return dema
    
    def calculate_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate DEMA-SMA metrics for the strategy."""
        if len(df) < max(self.lengthDEMA, self.lengthSMA_OPEN) + 5:  # Need minimum data
            return None
            
        # Calculate the moving averages
        close_prices = df['close']
        open_prices = df['open']
        
        # Calculate DEMA for close prices
        dema_close = self.calculate_dema(close_prices, self.lengthDEMA)
        
        # Calculate SMA for open prices
        sma_open = open_prices.rolling(window=self.lengthSMA_OPEN).mean()
        
        # Calculate distance between DEMA and SMA
        distance = dema_close - sma_open
        
        # Get current values
        current_price = df['close'].iloc[-1]
        current_dema = dema_close.iloc[-1]
        current_sma = sma_open.iloc[-1]
        current_distance = distance.iloc[-1]
        
        # Check if we have previous distance value
        if self.prev_distance is None:
            self.prev_distance = distance.iloc[-2] if len(distance) > 1 else current_distance
        
        # Entry conditions for long and short positions
        long_entry_condition = (current_dema < current_sma) and (self.prev_distance < current_distance)
        short_entry_condition = (current_dema > current_sma)
        
        # Exit conditions for long and short positions
        long_exit_condition = False
        short_exit_condition = False
        
        if len(dema_close) > 1 and len(sma_open) > 1:
            prev_dema = dema_close.iloc[-2]
            prev_sma = sma_open.iloc[-2]
            
            # Check for crossovers (DEMA crosses below SMA for long exit)
            long_exit_condition = (prev_dema >= prev_sma) and (current_dema < current_sma)
            
            # Check for crossovers (SMA crosses above DEMA for short exit)
            short_exit_condition = (prev_sma <= prev_dema) and (current_sma > current_dema)
        
        # Save the current distance for next iteration
        self.prev_distance = current_distance
        
        # Determine trading signals
        trading_signal = None
        signal_direction = None
        
        # Check if we should enter a long position
        if long_entry_condition and self.direction in ['long', 'both']:
            trading_signal = f"LONG Signal @ {datetime.now(self.pacific_timezone).strftime('%H:%M:%S')}"
            signal_direction = "LONG"
        
        # Check if we should enter a short position
        elif short_entry_condition and self.direction in ['short', 'both']:
            trading_signal = f"SHORT Signal @ {datetime.now(self.pacific_timezone).strftime('%H:%M:%S')}"
            signal_direction = "SHORT"
        
        # Check if we should exit a long position
        elif long_exit_condition and self.current_position == "LONG":
            trading_signal = f"EXIT LONG Signal @ {datetime.now(self.pacific_timezone).strftime('%H:%M:%S')}"
            signal_direction = "EXIT_LONG"
        
        # Check if we should exit a short position
        elif short_exit_condition and self.current_position == "SHORT":
            trading_signal = f"EXIT SHORT Signal @ {datetime.now(self.pacific_timezone).strftime('%H:%M:%S')}"
            signal_direction = "EXIT_SHORT"
        
        return {
            'dema': current_dema,
            'sma': current_sma,
            'distance': current_distance,
            'trading_signal': trading_signal,
            'signal_direction': signal_direction,
            'long_entry_condition': long_entry_condition,
            'short_entry_condition': short_entry_condition,
            'long_exit_condition': long_exit_condition,
            'short_exit_condition': short_exit_condition,
            'current_price': current_price
        }
    
    def open_position(self, direction: str) -> Tuple[bool, str, Optional[Dict]]:
        """Open a new position."""
        # Use our trading module to open the position
        quantity = self.shares  # Use the configured shares value
        
        if direction == "LONG":
            success, message, result = self.trading.place_market_order(
                self.symbol, quantity, "BUY"
            )
        else:  # SHORT
            success, message, result = self.trading.place_market_order(
                self.symbol, quantity, "SELL_SHORT"
            )
        
        # If successful, update our position tracking
        if success:
            self.current_position = direction
            self.position_size = quantity
            if direction == "LONG":
                self.entry_price_long = result.get('execution_price', 0)
            else:
                self.entry_price_short = result.get('execution_price', 0)
            self.entry_time = datetime.now(self.pacific_timezone)
            
        return success, message, result
    
    def close_position(self) -> Tuple[bool, str]:
        """Close current position without opening a new one."""
        # Use our trading module to close the position
        success, message = self.trading.close_position(self.symbol)
        
        # If successful, update our position tracking
        if success:
            self.current_position = "FLAT"
            self.position_size = 0
            self.entry_price_long = 0
            self.entry_price_short = 0
            self.entry_time = None
            
        return success, message
    
    def update_data(self):
        """Update price data, calculate metrics, and check for signals."""
        # Get price data using our market data module
        try:
            df = self.market.get_intraday_data(self.symbol, days_back=1, minute_interval=1)
            if df is None or df.empty:
                print("No price data available")
                return
            
            current_price = df['close'].iloc[-1]
            price_change = df['close'].iloc[-1] - df['close'].iloc[0]
            price_change_pct = (price_change / df['close'].iloc[0]) * 100
            change_symbol = '▲' if price_change >= 0 else '▼'
            volume = df['volume'].iloc[-1]
            
            # Calculate strategy metrics
            metrics = self.calculate_metrics(df)
            
            # Get account data and sync positions
            self.sync_current_positions()
            account_data = self.account.get_account_data()
            
            # Clear screen
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Print price and strategy update
            # Get current system time
            current_time = datetime.now(self.pacific_timezone)
            print(f"{self.symbol}: ${current_price:.2f} {change_symbol} ${abs(price_change):.2f} ({abs(price_change_pct):.2f}%) Vol: {volume:,.0f} | {current_time.strftime('%H:%M:%S')} PST")
            print(f"Strategy: DEMA-SMA CROSSOVER | Current Position: {self.current_position} | Size: {self.position_size}")
            print("-" * 80)
            
            if metrics:
                print("\nStrategy Metrics:")
                print(f"DEMA({self.lengthDEMA}): ${metrics['dema']:.2f}")
                print(f"SMA({self.lengthSMA_OPEN}) of Open: ${metrics['sma']:.2f}")
                print(f"Distance: {metrics['distance']:.4f}")
                
                # Check for trading signals
                if metrics['trading_signal']:
                    signal_direction = metrics['signal_direction']
                    print("\n🚨 TRADING SIGNAL 🚨")
                    print(f"{metrics['trading_signal']} - Price: ${current_price:.2f}")
                    
                    # Check if we need to take action
                    take_action = False
                    action_msg = ""
                    
                    # If signal is to exit current position
                    if signal_direction in ["EXIT_LONG", "EXIT_SHORT"]:
                        take_action = True
                        action_msg = f"EXIT SIGNAL: Close {self.current_position} position"
                    
                    # If we're flat and get a new position signal
                    elif self.current_position == "FLAT" and signal_direction in ["LONG", "SHORT"]:
                        take_action = True
                        action_msg = f"NEW SIGNAL: Open {signal_direction} position"
                    
                    # Handle order execution based on signal
                    if take_action:
                        print(f"\n{action_msg}")
                        if self.auto_mode:
                            print("Auto-trading enabled - Executing trade...")
                            
                            if signal_direction in ["EXIT_LONG", "EXIT_SHORT"]:
                                success, message = self.close_position()
                            else:  # Open new position
                                success, message, result = self.open_position(signal_direction)
                                
                            if success:
                                print(f"SUCCESS: {message}")
                            else:
                                print(f"ERROR: {message}")
                        else:
                            # Manual mode
                            print("Would you like to execute this trade? (y/n)")
                            user_input = input().lower()
                            if user_input == 'y':
                                if signal_direction in ["EXIT_LONG", "EXIT_SHORT"]:
                                    success, message = self.close_position()
                                else:  # Open new position
                                    success, message, result = self.open_position(signal_direction)
                                    
                                if success:
                                    print(f"SUCCESS: {message}")
                                else:
                                    print(f"ERROR: {message}")
                
                print("-" * 80)
            
                
            # Print account data
            if account_data:
                print("\nAccount Information:")
                print("-" * 80)
                for account in account_data:
                    print(f"\nAccount: {account['account_id']} ({account['type']})")
                    print(f"Day Trader: {'Yes' if account['is_day_trader'] else 'No'}")
                    print(f"Round Trips: {account['round_trips']}")
                    
                    # Print balances
                    balances = account['balances']
                    print("\nBalances:")
                    print(f"Cash: ${balances['cash']:,.2f}")
                    print(f"Equity: ${balances['equity']:,.2f}")
                    print(f"Buying Power: ${balances['buying_power']:,.2f}")
                    print(f"Day Trading Buying Power: ${balances['day_trading_buying_power']:,.2f}")
                    print(f"Liquidation Value: ${balances['liquidation_value']:,.2f}")
                    
                    # Print positions with P&L
                    if account['positions']:
                        print("\nPositions:")
                        for pos in account['positions']:
                            position_type = "LONG" if pos['quantity'] > 0 else "SHORT"
                            print(f"\n{position_type} {pos['symbol']}: {abs(pos['quantity']):,.0f} shares")
                            print(f"Market Value: ${abs(pos['market_value']):,.2f}")
                            print(f"Average Price: ${pos['average_price']:,.2f}")
                            pnl = pos['current_day_profit_loss']
                            pnl_pct = pos['current_day_profit_loss_pct']
                            print(f"P&L: {'▲' if pnl >= 0 else '▼'} ${abs(pnl):,.2f} ({abs(pnl_pct):.2f}%)")
                    else:
                        print("\nNo open positions")
                    
                    # Print calls if any
                    if balances['maintenance_call'] > 0 or balances['reg_t_call'] > 0:
                        print("\nCalls:")
                        if balances['maintenance_call'] > 0:
                            print(f"Maintenance Call: ${balances['maintenance_call']:,.2f}")
                        if balances['reg_t_call'] > 0:
                            print(f"Reg T Call: ${balances['reg_t_call']:,.2f}")
            
            print("\n" + "-" * 80)
            
            # Display position entry prices if we have an open position
            if self.current_position != "FLAT":
                if self.current_position == "LONG":
                    entry_price = self.entry_price_long
                else:
                    entry_price = self.entry_price_short
                    
                profit_loss = 0
                if entry_price > 0:
                    if self.current_position == "LONG":
                        profit_loss = (current_price - entry_price) / entry_price * 100
                    else:
                        profit_loss = (entry_price - current_price) / entry_price * 100
                        
                print(f"Current {self.current_position} Position:")
                print(f"Entry Price: ${entry_price:.2f}")
                print(f"Current Price: ${current_price:.2f}")
                print(f"P&L: {'▲' if profit_loss >= 0 else '▼'} {abs(profit_loss):.2f}%")
                
                if self.entry_time:
                    time_in_trade = current_time - self.entry_time
                    hours, remainder = divmod(time_in_trade.total_seconds(), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    print(f"Time in Trade: {int(hours)}h {int(minutes)}m {int(seconds)}s")
            
            print("-" * 80)
        except Exception as e:
            print(f"Error updating data: {e}")
            
    def run(self):
        """Main execution loop for the strategy."""
        print(f"\nInitializing {self.symbol} DEMA-SMA Crossover Strategy...")
        print("Will run during market hours (9:30 AM - 4:00 PM ET / 6:30 AM - 1:00 PM PT)")
        print(f"Strategy Parameters:")
        print(f"  DEMA Length: {self.lengthDEMA}")
        print(f"  SMA Open Length: {self.lengthSMA_OPEN}")
        print(f"  Direction: {self.direction.upper()}")
        print(f"  Auto Trading: {'ENABLED' if self.auto_mode else 'DISABLED'}")
        print(f"  Position Size: {self.shares} shares")
        print("Press Ctrl+C to stop at any time")
        
        try:
            # Sync positions at startup
            self.sync_current_positions()
            print(f"Current position: {self.current_position} | Size: {self.position_size}")
            
            while True:
                # Check if market is open
                if not self.market.is_market_open():
                    # If we were running and market closed
                    print("\nMarket is now closed. Waiting for next market open...")
                    self.wait_for_market_open()
                
                # Get time until market closes
                close_time = self.market.time_until_market_close()
                hours, remainder = divmod(close_time.total_seconds(), 3600)
                minutes, seconds = divmod(remainder, 60)
                
                # Update data and display time until close
                self.update_data()
                print(f"Time until market close: {int(hours)}h {int(minutes)}m {int(seconds)}s (Pacific Time)")
                
                # Sleep for a short time before next update
                time.sleep(0.25)  # Update every 0.25 seconds
        except KeyboardInterrupt:
            print("\nStopping tracker...")
            
    def wait_for_market_open(self):
        """Wait until the market opens with periodic updates."""
        if self.market.is_market_open():
            print(f"Market is already open. Starting {self.symbol} tracker...")
            return
        
        wait_time = self.market.time_until_market_open()
        hours, remainder = divmod(wait_time.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Get market open time in both Eastern and Pacific time
        market_open_et = datetime.now(self.market_timezone) + wait_time
        market_open_pt = market_open_et.astimezone(self.pacific_timezone)
        
        # Display times in both timezones
        print(f"Market is closed. Waiting for market to open in {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"Expected market open: {market_open_et.strftime('%Y-%m-%d %H:%M:%S')} ET / {market_open_pt.strftime('%Y-%m-%d %H:%M:%S')} PT")
        
        # Wait with updates every minute
        remaining_seconds = wait_time.total_seconds()
        update_interval = 60  # Update wait message every minute
        
        while remaining_seconds > 0:
            sleep_time = min(update_interval, remaining_seconds)
            time.sleep(sleep_time)
            remaining_seconds -= sleep_time
            
            if remaining_seconds > 0:
                hours, remainder = divmod(remaining_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                print(f"Market opens in {int(hours)}h {int(minutes)}m {int(seconds)}s (Pacific Time)")
        
        print("Market is now open! Starting tracker...")


def validate_symbol(symbol):
    """Basic symbol validation."""
    if not symbol or not isinstance(symbol, str):
        return False
    # Remove any whitespace and check if we have a non-empty string
    symbol = symbol.strip()
    if not symbol:
        return False
    # Check for invalid characters
    invalid_chars = set('<>{}[]~`')
    if any(char in invalid_chars for char in symbol):
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DEMA-SMA Crossover Trading Strategy')
    parser.add_argument('symbol', type=str, help='Stock symbol to track (e.g. AAPL, MSFT, NVDA)')
    parser.add_argument('--dema', type=int, default=15, help='Length of DEMA calculation period (default: 15)')
    parser.add_argument('--sma', type=int, default=30, help='Length of SMA calculation period for open price (default: 30)')
    parser.add_argument('--auto', type=str, choices=['yes', 'no'], default='no',
                      help='Enable auto-trading mode (yes/no)')
    parser.add_argument('--shares', type=int, default=42, help='Number of shares to trade (default: 42)')
    parser.add_argument('--direction', type=str, choices=['long', 'short', 'both'], default='both',
                      help='Trading direction (long/short/both). Default is both')
    
    args = parser.parse_args()
    symbol = args.symbol.upper()
    
    if not validate_symbol(symbol):
        print(f"Error: Invalid symbol '{symbol}'")
        sys.exit(1)
    
    # Print header with strategy parameters
    print("\n" + "=" * 80)
    print(f"DEMA-SMA CROSSOVER TRADING STRATEGY - {symbol}")
    print("=" * 80)
    print(f"PARAMETERS:")
    print(f"  DEMA Length: {args.dema}")
    print(f"  SMA Open Length: {args.sma}")
    print(f"  Direction: {args.direction.upper()}")
    print(f"  Auto Trading: {'ENABLED' if args.auto == 'yes' else 'DISABLED'}")
    print(f"  Position Size: {args.shares} shares")
    print("=" * 80 + "\n")
    
    trader = DEMASMACrossoverStrategy(
        symbol=symbol, 
        lengthDEMA=args.dema,
        lengthSMA_OPEN=args.sma,
        auto_mode=args.auto == 'yes', 
        shares=args.shares, 
        direction=args.direction
    )
    trader.run()

#!/usr/bin/env python3
"""
Backtest script for the DEMA-SMA Crossover Trading Strategy
Uses historical data to test DEMA-SMA crossover signals
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import subprocess

# Add parent directory to path to import from handlers
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from handlers.historical_data_handler import HistoricalDataHandler

class DEMASMACrossoverBacktester:
    def __init__(self):
        self.data_handler = HistoricalDataHandler()
        self.output_dir = 'historical_data'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Strategy parameters
        self.DEMA_LENGTH = 15
        self.SMA_OPEN_LENGTH = 30
        
        # Trading strategy parameters
        self.RISK_REWARD_RATIO = 2.0
        self.MAX_RISK_PERCENT = 2.0  # 2% risk per trade
        self.STOP_LOSS_PERCENT = 1.5  # 1.5% stop loss
        self.EARLY_TAKE_PROFIT_PERCENT = 0.5  # 0.5% early take profit
        self.DISTANCE_STD_LOOKBACK = 20  # Lookback period for calculating distance standard deviation
        self.DISTANCE_STD_MULTIPLIER = 2.0  # Standard deviation multiplier for take profit
    
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
            
            # Account metrics (starting with $25,000, using 2% per trade = $500)
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
            print("\n📊 DEMA-SMA CROSSOVER STRATEGY PERFORMANCE SUMMARY")
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
            
            # Signal type analysis
            if 'signal_type' in trades.columns:
                print(f"\n🔍 SIGNAL TYPE ANALYSIS")
                signal_performance = trades.groupby('signal_type').agg({
                    'pnl': ['count', 'sum', 'mean']
                }).round(2)
                
                for signal_type in signal_performance.index:
                    count = signal_performance.loc[signal_type, ('pnl', 'count')]
                    total_pnl = signal_performance.loc[signal_type, ('pnl', 'sum')]
                    avg_pnl = signal_performance.loc[signal_type, ('pnl', 'mean')]
                    print(f"   {signal_type}: {count} trades, ${total_pnl:.2f} total, ${avg_pnl:.2f} avg")
            
            # Additional insights
            if total_trades > 0:
                print(f"\n🔍 ADDITIONAL INSIGHTS")
                print(f"   Risk per Trade: ${risk_per_trade:.2f} ({self.MAX_RISK_PERCENT:.1f}% of capital)")
                print(f"   Risk/Reward Ratio Target: {self.RISK_REWARD_RATIO}:1")
                print(f"   Stop Loss: {self.STOP_LOSS_PERCENT}%")
                
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
    
    def calculate_dema(self, prices, length):
        """Calculate Double Exponential Moving Average."""
        ema1 = prices.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        dema = (2 * ema1) - ema2
        return dema
    
    def calculate_indicators(self, df):
        """Calculate DEMA and SMA indicators"""
        if df is None or len(df) < max(self.DEMA_LENGTH, self.SMA_OPEN_LENGTH) + 5:
            return None
        
        df_indicators = df.copy()
        
        # Convert data to proper type
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df_indicators[col] = df_indicators[col].astype(float)
        
        # Calculate DEMA for close prices
        df_indicators['dema'] = self.calculate_dema(df_indicators['close'], self.DEMA_LENGTH)
        
        # Calculate SMA for open prices
        df_indicators['sma_open'] = df_indicators['open'].rolling(window=self.SMA_OPEN_LENGTH).mean()
        
        # Calculate distance between DEMA and SMA
        df_indicators['distance'] = df_indicators['dema'] - df_indicators['sma_open']
        
        # Calculate rolling standard deviation of distance for take profit logic
        df_indicators['distance_std'] = df_indicators['distance'].rolling(window=self.DISTANCE_STD_LOOKBACK).std()
        
        # Calculate dynamic take profit threshold (2 standard deviations above mean)
        df_indicators['distance_mean'] = df_indicators['distance'].rolling(window=self.DISTANCE_STD_LOOKBACK).mean()
        df_indicators['take_profit_threshold'] = df_indicators['distance_mean'] + (df_indicators['distance_std'] * self.DISTANCE_STD_MULTIPLIER)
        
        return df_indicators

    def calculate_crossover_metrics(self, df, lookback=50):
        """Calculate DEMA-SMA crossover metrics for each row in the dataframe"""
        print("Calculating technical indicators for entire dataset...")
        
        # First, calculate all indicators for the entire dataset
        df_with_indicators = self.calculate_indicators(df)
        if df_with_indicators is None:
            print("Failed to calculate indicators")
            return None
        
        print("Processing crossover signals...")
        
        # Initialize lists to store metrics
        crossover_signals = []
        signal_types = []
        entry_prices = []
        stop_losses = []
        take_profits = []
        trade_status = []
        pnl = []
        running_pnl = []
        current_equity = []
        long_entry_conditions = []
        short_entry_conditions = []
        long_exit_conditions = []
        short_exit_conditions = []
        
        # Initialize starting equity
        starting_equity = 25000.0
        running_equity = starting_equity
        
        # Track previous distance for signal detection
        prev_distance = None
        
        # Calculate metrics for each row
        for i in range(len(df_with_indicators)):
            if i < max(self.DEMA_LENGTH, self.SMA_OPEN_LENGTH) + 5:
                # For early rows, use NaN values for crossover signals but populate indicators
                crossover_signals.append(np.nan)
                signal_types.append(np.nan)
                entry_prices.append(np.nan)
                stop_losses.append(np.nan)
                take_profits.append(np.nan)
                trade_status.append(np.nan)
                pnl.append(np.nan)
                running_pnl.append(np.nan)
                current_equity.append(running_equity)
                long_entry_conditions.append(np.nan)
                short_entry_conditions.append(np.nan)
                long_exit_conditions.append(np.nan)
                short_exit_conditions.append(np.nan)
                continue
            
            # Current bar data
            current_bar = df_with_indicators.iloc[i]
            current_close = current_bar['close']
            current_high = current_bar['high']
            current_low = current_bar['low']
            current_dema = current_bar['dema']
            current_sma = current_bar['sma_open']
            current_distance = current_bar['distance']
            
            # Get previous values for crossover detection
            if i > 0:
                prev_bar = df_with_indicators.iloc[i-1]
                prev_dema = prev_bar['dema']
                prev_sma = prev_bar['sma_open']
                if prev_distance is None:
                    prev_distance = prev_bar['distance']
            else:
                prev_dema = current_dema
                prev_sma = current_sma
                if prev_distance is None:
                    prev_distance = current_distance
            
            # Calculate entry and exit conditions based on strategy logic
            # Long entry: DEMA < SMA AND distance is increasing (momentum building)
            long_entry_condition = (current_dema < current_sma) and (prev_distance < current_distance)
            
            # Short entry: DEMA > SMA
            short_entry_condition = (current_dema > current_sma)
            
            # Long exit: DEMA crosses below SMA (was above, now below)
            long_exit_condition = (prev_dema >= prev_sma) and (current_dema < current_sma)
            
            # Short exit: SMA crosses above DEMA (SMA was below, now above)
            short_exit_condition = (prev_sma <= prev_dema) and (current_sma > current_dema)
            
            # Initialize default values
            crossover_signal = None
            signal_type = None
            entry_price = None
            stop_loss = None
            take_profit = None
            trade_status_value = None
            pnl_value = None
            running_pnl_value = None
            
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
                            'equity_at_entry': current_equity[j]
                        }
                    break
            
            # If we have an open trade, check if current OHLC closes it
            if has_open_trade and open_trade_info is not None:
                entry_p = open_trade_info['entry_price']
                stop_p = open_trade_info['stop_loss']
                target_p = open_trade_info['take_profit']
                sig_type = open_trade_info['signal_type']
                equity_at_entry = open_trade_info['equity_at_entry']
                
                # Get current take profit threshold based on standard deviation
                current_take_profit_threshold = current_bar.get('take_profit_threshold', np.nan)
                
                # Calculate early take profit level (0.5% gain)
                early_take_profit_long = entry_p * (1 + self.EARLY_TAKE_PROFIT_PERCENT / 100)
                
                # Check if stop loss or take profit was hit
                if sig_type == "LONG":  # Long position
                    if current_low <= stop_p:
                        # Stop loss hit
                        trade_status_value = "closed"
                        exit_price = stop_p
                        # Calculate PnL
                        position_size = equity_at_entry * (self.MAX_RISK_PERCENT / 100)
                        shares = position_size / entry_p
                        pnl_value = (exit_price - entry_p) * shares
                        running_equity += pnl_value
                    elif current_high >= early_take_profit_long:
                        # Early take profit hit (0.5% gain)
                        trade_status_value = "closed"
                        exit_price = early_take_profit_long
                        # Calculate PnL
                        position_size = equity_at_entry * (self.MAX_RISK_PERCENT / 100)
                        shares = position_size / entry_p
                        pnl_value = (exit_price - entry_p) * shares
                        running_equity += pnl_value
                    elif not pd.isna(current_take_profit_threshold) and current_distance >= current_take_profit_threshold:
                        # Dynamic take profit hit (DEMA is 2 std devs above SMA)
                        trade_status_value = "closed"
                        exit_price = current_close
                        # Calculate PnL
                        position_size = equity_at_entry * (self.MAX_RISK_PERCENT / 100)
                        shares = position_size / entry_p
                        pnl_value = (exit_price - entry_p) * shares
                        running_equity += pnl_value
                    elif current_high >= target_p:
                        # Traditional take profit hit (fallback)
                        trade_status_value = "closed"
                        exit_price = target_p
                        # Calculate PnL
                        position_size = equity_at_entry * (self.MAX_RISK_PERCENT / 100)
                        shares = position_size / entry_p
                        pnl_value = (exit_price - entry_p) * shares
                        running_equity += pnl_value
                    # Check for exit signal
                    elif long_exit_condition:
                        # Exit signal hit
                        trade_status_value = "closed"
                        exit_price = current_close
                        # Calculate PnL
                        position_size = equity_at_entry * (self.MAX_RISK_PERCENT / 100)
                        shares = position_size / entry_p
                        pnl_value = (exit_price - entry_p) * shares
                        running_equity += pnl_value
                elif sig_type == "SHORT":  # Short position
                    # Calculate early take profit level for short (0.5% gain)
                    early_take_profit_short = entry_p * (1 - self.EARLY_TAKE_PROFIT_PERCENT / 100)
                    
                    if current_high >= stop_p:
                        # Stop loss hit
                        trade_status_value = "closed"
                        exit_price = stop_p
                        # Calculate PnL
                        position_size = equity_at_entry * (self.MAX_RISK_PERCENT / 100)
                        shares = position_size / entry_p
                        pnl_value = (entry_p - exit_price) * shares
                        running_equity += pnl_value
                    elif current_low <= early_take_profit_short:
                        # Early take profit hit (0.5% gain for short)
                        trade_status_value = "closed"
                        exit_price = early_take_profit_short
                        # Calculate PnL
                        position_size = equity_at_entry * (self.MAX_RISK_PERCENT / 100)
                        shares = position_size / entry_p
                        pnl_value = (entry_p - exit_price) * shares
                        running_equity += pnl_value
                    elif not pd.isna(current_take_profit_threshold) and current_distance <= -current_take_profit_threshold:
                        # Dynamic take profit hit (DEMA is 2 std devs below SMA for short)
                        trade_status_value = "closed"
                        exit_price = current_close
                        # Calculate PnL
                        position_size = equity_at_entry * (self.MAX_RISK_PERCENT / 100)
                        shares = position_size / entry_p
                        pnl_value = (entry_p - exit_price) * shares
                        running_equity += pnl_value
                    elif current_low <= target_p:
                        # Traditional take profit hit (fallback)
                        trade_status_value = "closed"
                        exit_price = target_p
                        # Calculate PnL
                        position_size = equity_at_entry * (self.MAX_RISK_PERCENT / 100)
                        shares = position_size / entry_p
                        pnl_value = (entry_p - exit_price) * shares
                        running_equity += pnl_value
                    # Check for exit signal
                    elif short_exit_condition:
                        # Exit signal hit
                        trade_status_value = "closed"
                        exit_price = current_close
                        # Calculate PnL
                        position_size = equity_at_entry * (self.MAX_RISK_PERCENT / 100)
                        shares = position_size / entry_p
                        pnl_value = (entry_p - exit_price) * shares
                        running_equity += pnl_value
                
                # Calculate running PnL for open positions
                if trade_status_value != "closed":
                    position_size = equity_at_entry * (self.MAX_RISK_PERCENT / 100)
                    shares = position_size / entry_p
                    if sig_type == "LONG":
                        running_pnl_value = (current_close - entry_p) * shares
                    else:  # SHORT
                        running_pnl_value = (entry_p - current_close) * shares
            
            # Check for new crossover signals if no open trade
            # Add cooldown period - don't enter new trade immediately after closing one
            cooldown_periods = 3  # Wait 3 bars after closing before new entry
            recently_closed = False
            
            # Check if we recently closed a trade
            if len(trade_status) >= cooldown_periods:
                for j in range(max(0, len(trade_status) - cooldown_periods), len(trade_status)):
                    if trade_status[j] == "closed":
                        recently_closed = True
                        break
            
            if not has_open_trade and not recently_closed:
                if long_entry_condition:
                    crossover_signal = "LONG_ENTRY"
                    signal_type = "LONG"
                    entry_price = current_close
                    stop_loss = current_close * (1 - self.STOP_LOSS_PERCENT / 100)
                    risk_amount = entry_price - stop_loss
                    take_profit = entry_price + (risk_amount * self.RISK_REWARD_RATIO)
                    trade_status_value = "open"
                elif short_entry_condition:
                    crossover_signal = "SHORT_ENTRY"
                    signal_type = "SHORT"
                    entry_price = current_close
                    stop_loss = current_close * (1 + self.STOP_LOSS_PERCENT / 100)
                    risk_amount = stop_loss - entry_price
                    take_profit = entry_price - (risk_amount * self.RISK_REWARD_RATIO)
                    trade_status_value = "open"
            
            # Update previous distance for next iteration
            prev_distance = current_distance
            
            # Store calculated values
            crossover_signals.append(crossover_signal)
            signal_types.append(signal_type)
            entry_prices.append(entry_price)
            stop_losses.append(stop_loss)
            take_profits.append(take_profit)
            trade_status.append(trade_status_value)
            pnl.append(pnl_value)
            running_pnl.append(running_pnl_value)
            current_equity.append(running_equity)
            long_entry_conditions.append(long_entry_condition)
            short_entry_conditions.append(short_entry_condition)
            long_exit_conditions.append(long_exit_condition)
            short_exit_conditions.append(short_exit_condition)
        
        # Extract indicator values from the pre-calculated dataframe
        print("Extracting indicator values...")
        
        return {
            'crossover_signal': crossover_signals,
            'signal_type': signal_types,
            'entry_price': entry_prices,
            'stop_loss': stop_losses,
            'take_profit': take_profits,
            'trade_status': trade_status,
            'pnl': pnl,
            'running_pnl': running_pnl,
            'current_equity': current_equity,
            'long_entry_condition': long_entry_conditions,
            'short_entry_condition': short_entry_conditions,
            'long_exit_condition': long_exit_conditions,
            'short_exit_condition': short_exit_conditions,
            'dema': df_with_indicators['dema'].tolist(),
            'sma_open': df_with_indicators['sma_open'].tolist(),
            'distance': df_with_indicators['distance'].tolist()
        }
    
    def fetch_and_process_data(self, symbol, period_type="day", period=10, frequency_type="minute", 
                              frequency=5, start_date=None, end_date=None, lookback=50):
        """
        Fetch historical data and add DEMA-SMA crossover metrics
        
        Args:
            symbol (str): Stock symbol to fetch data for
            period_type (str): Period type (day, month, year, ytd)
            period (int): Number of periods
            frequency_type (str): Frequency type (minute, daily, weekly, monthly)
            frequency (int): Frequency value
            start_date (int): Start date in milliseconds (optional)
            end_date (int): End date in milliseconds (optional)
            lookback (int): Lookback period for crossover calculations
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
            
            print(f"Calculating crossover metrics with lookback period of {lookback}...")
            
            # Calculate crossover metrics
            crossover_metrics = self.calculate_crossover_metrics(df, lookback=lookback)
            
            if crossover_metrics:
                # Add crossover metrics to dataframe
                for metric_name, metric_values in crossover_metrics.items():
                    df[metric_name] = metric_values
                
                print(f"Added crossover metrics: {list(crossover_metrics.keys())}")
            else:
                print("Failed to calculate crossover metrics")
                return None
            
            # Reorder columns for better readability
            base_columns = ['symbol', 'datetime', 'open', 'high', 'low', 'close', 'volume']
            crossover_columns = ['crossover_signal', 'signal_type', 'entry_price', 'stop_loss', 
                                'take_profit', 'trade_status', 'pnl', 'running_pnl', 'current_equity',
                                'long_entry_condition', 'short_entry_condition', 'long_exit_condition', 'short_exit_condition']
            indicator_columns = ['dema', 'sma_open', 'distance', 'distance_std', 'distance_mean', 'take_profit_threshold']
            
            # Ensure all columns exist
            available_columns = [col for col in base_columns + crossover_columns + indicator_columns if col in df.columns]
            df = df[available_columns]
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            freq_str = f"{frequency}{frequency_type}" if frequency != 1 else frequency_type
            filename = f"{self.output_dir}/{symbol}_dema_sma_backtest_{period}{period_type}_{freq_str}_{timestamp}.csv"
            
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
            
            # Display crossover statistics
            if 'trade_status' in df.columns:
                total_signals = df['crossover_signal'].notna().sum()
                total_trades = df['trade_status'].notna().sum()
                open_trades = (df['trade_status'] == 'open').sum()
                closed_trades = (df['trade_status'] == 'closed').sum()
                
                print(f"\nCrossover Statistics:")
                print(f"Total Crossover Signals: {total_signals}")
                print(f"Total Trade Events: {total_trades}")
                print(f"Trades Opened: {open_trades}")
                print(f"Trades Closed: {closed_trades}")
                
                if 'signal_type' in df.columns:
                    signal_types = df['signal_type'].dropna().value_counts()
                    if not signal_types.empty:
                        print(f"Signal Types:")
                        for signal_type, count in signal_types.items():
                            print(f"  {signal_type}: {count}")
            
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
        description="Backtest the DEMA-SMA Crossover strategy with historical data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backtest 1 year of daily data for DEMA-SMA crossover strategy
  python3 dema_sma_crossover_backtest.py AAPL
  
  # Backtest 6 months of daily data
  python3 dema_sma_crossover_backtest.py NVDA --period 6 --period-type month
  
  # Backtest 5 days of 15-minute data
  python3 dema_sma_crossover_backtest.py TSLA --period 5 --period-type day --frequency-type minute --frequency 15
  
  # Custom lookback period for crossover calculations
  python3 dema_sma_crossover_backtest.py AAPL --lookback 100
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
        default='day',
        help='Period type (default: day)'
    )
    
    parser.add_argument(
        '--period',
        type=int,
        default=10,
        help='Number of periods (default: 1)'
    )
    
    parser.add_argument(
        '--frequency-type',
        type=str,
        choices=['minute', 'daily', 'weekly', 'monthly'],
        default='minute',
        help='Frequency type (default: minute)'
    )
    
    parser.add_argument(
        '--frequency',
        type=int,
        default=5,
        help='Frequency value (default: 5)'
    )
    
    parser.add_argument(
        '--lookback',
        type=int,
        default=50,
        help='Lookback period for crossover calculations (default: 50)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
        help='Start date in YYYY-MM-DD format (default: one year ago)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='End date in YYYY-MM-DD format (default: current date)'
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
    
    # Check if user provided custom start/end dates
    print("DEMA-SMA CROSSOVER STRATEGY BACKTEST")
    print("=" * 80)
    print(f"Symbol: {ticker}")
    print(f"Period: {args.period} {args.period_type}")
    print(f"Frequency: {args.frequency} {args.frequency_type}")
    print(f"Crossover Lookback: {args.lookback} periods")
    print(f"Risk per Trade: {2.0}% of account equity")
    print(f"Risk/Reward Ratio: 2:1")
    print(f"Stop Loss: 1.5%")
    if start_date_ms and end_date_ms:
        start_str = datetime.fromtimestamp(start_date_ms/1000).strftime('%Y-%m-%d')
        end_str = datetime.fromtimestamp(end_date_ms/1000).strftime('%Y-%m-%d')
        print(f"Date Range: {start_str} to {end_str}")
    print("=" * 80)
    
    # Create backtester and process data
    backtester = DEMASMACrossoverBacktester()
    
    # Check if user provided any custom period/frequency arguments
    # If only ticker is provided, use 1-year date range as default
    # If any period/frequency args are provided, use period-based approach
    
    # Get the original command line arguments to see what was actually provided
    import sys
    provided_args = sys.argv[1:]  # Skip script name
    
    # Check if any period/frequency arguments were explicitly provided
    period_args_provided = any(arg in provided_args for arg in [
        '--period', '--period-type', '--frequency', '--frequency-type'
    ])
    
    # Check if custom dates were provided
    date_args_provided = any(arg in provided_args for arg in [
        '--start-date', '--end-date'
    ])
    
    if date_args_provided or (not period_args_provided and len(provided_args) == 1):
        # Use 1-year date range when:
        # 1. Custom dates are explicitly provided, OR
        # 2. Only ticker is provided (no period/frequency args)
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
    else:
        # Use period-based parameters when period/frequency args are provided
        result = backtester.fetch_and_process_data(
            symbol=ticker,
            period_type=args.period_type,
            period=args.period,
            frequency_type=args.frequency_type,
            frequency=args.frequency,
            start_date=None,
            end_date=None,
            lookback=args.lookback
        )
    
    if result:
        print(f"\n✅ Successfully created DEMA-SMA crossover backtest CSV: {result}")
        
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
                "python3", "visualizers/dema_sma_visualization.py", 
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

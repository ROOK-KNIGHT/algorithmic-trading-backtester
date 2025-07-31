#!/usr/bin/env python3
"""
Script to fetch historical data and add volatility metrics to create a comprehensive CSV
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from historical_data_handler import HistoricalDataHandler
import time
import subprocess

class VolatilityDataFetcher:
    def __init__(self):
        self.data_handler = HistoricalDataHandler()
        self.output_dir = 'historical_data'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def calculate_band_stability(self, band_values):
        """Calculate if a volatility band has been stable over the period"""
        if len(band_values) < 2:
            return "stable"  # Not enough data to determine instability
        
        # Define stability threshold (0.19% change per bar)
        stability_threshold = 0.0019
        
        # Calculate percentage changes for the band relative to itself
        band_changes = []
        for i in range(1, len(band_values)):
            prev_value = band_values[i-1]
            curr_value = band_values[i]
            if prev_value != 0:  # Avoid division by zero
                pct_change = abs((curr_value - prev_value) / prev_value)
                band_changes.append(pct_change)
        
        # Check each change against threshold
        unstable_changes = [change for change in band_changes if change >= stability_threshold]
        
        if unstable_changes:
            return "not stable"
        else:
            return "stable"
    
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
            
            # Account metrics (starting with $25,000, using 10% per trade = $2,500)
            initial_capital = 25000.0
            risk_per_trade = 2500.0
            
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
            # Convert datetime column to datetime if it's a string
            if df['datetime'].dtype == 'object':
                df['datetime'] = pd.to_datetime(df['datetime'])
            
            date_range_start = df['datetime'].min().strftime('%Y-%m-%d %H:%M:%S')
            date_range_end = df['datetime'].max().strftime('%Y-%m-%d %H:%M:%S')
            
            # Print performance summary
            print("\n📊 PERFORMANCE SUMMARY")
            print("=" * 50)
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
            
            # Additional insights
            if total_trades > 0:
                avg_trade_duration = "N/A"  # Would need timestamp analysis for this
                print(f"\n🔍 ADDITIONAL INSIGHTS")
                print(f"   Risk per Trade: ${risk_per_trade:.2f} ({(risk_per_trade/initial_capital)*100:.1f}% of capital)")
                
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
            
            print("=" * 50)
            
        except Exception as e:
            print(f"Error calculating performance summary: {str(e)}")
    
    def calculate_volatility_metrics(self, df, lookback=20):
        """
        Calculate volatility metrics for each row in the dataframe
        Based on the calculate_volatility_metrics function from exceedence_strategy.py
        """
        if len(df) < lookback:
            return None
            
        # Initialize lists to store metrics
        high_bands = []
        low_bands = []
        high_exceedances = []
        low_exceedances = []
        distances_to_high = []
        distances_to_low = []
        positions_in_range = []
        close_above_high_band = []
        close_below_low_band = []
        close_vs_high_band = []
        close_vs_low_band = []
        momentum_signals = []
        target_prices = []
        trade_status = []
        entry_prices = []
        pnl = []
        running_pnl = []
        current_equity = []
        band_stability = []
        trade_validation = []
        
        # Initialize starting equity
        starting_equity = 25000.0
        running_equity = starting_equity
        
        # Calculate metrics for each row (starting from lookback index)
        for i in range(len(df)):
            if i < lookback:
                # For early rows, use NaN values
                high_bands.append(np.nan)
                low_bands.append(np.nan)
                high_exceedances.append(np.nan)
                low_exceedances.append(np.nan)
                distances_to_high.append(np.nan)
                distances_to_low.append(np.nan)
                positions_in_range.append(np.nan)
                close_above_high_band.append(np.nan)
                close_below_low_band.append(np.nan)
                close_vs_high_band.append(np.nan)
                close_vs_low_band.append(np.nan)
                momentum_signals.append(np.nan)
                target_prices.append(np.nan)
                trade_status.append(np.nan)
                entry_prices.append(np.nan)
                pnl.append(np.nan)
                running_pnl.append(np.nan)
                current_equity.append(running_equity)
                band_stability.append(np.nan)
                continue
            
            # Get the lookback window (excluding current bar for calculation)
            window_end = i  # Current bar index
            window_start = max(0, window_end - lookback)
            window_df = df.iloc[window_start:window_end]  # Exclude current bar
            
            if len(window_df) == 0:
                high_bands.append(np.nan)
                low_bands.append(np.nan)
                high_exceedances.append(np.nan)
                low_exceedances.append(np.nan)
                distances_to_high.append(np.nan)
                distances_to_low.append(np.nan)
                positions_in_range.append(np.nan)
                close_above_high_band.append(np.nan)
                close_below_low_band.append(np.nan)
                close_vs_high_band.append(np.nan)
                close_vs_low_band.append(np.nan)
                momentum_signals.append(np.nan)
                target_prices.append(np.nan)
                trade_status.append(np.nan)
                continue
            
            # Use previous bar's close for calculations (if available)
            if i > 0:
                prev_close = df.iloc[i-1]['close']
            else:
                prev_close = df.iloc[i]['close']
            
            # Calculate volatility components
            highside_vol = window_df['high'] - window_df['close']
            lowside_vol = window_df['low'] - window_df['close']
            
            mean_highside = highside_vol.mean()
            mean_lowside = lowside_vol.mean()
            std_highside = highside_vol.std()
            std_lowside = lowside_vol.std()
            
            # Handle NaN values
            if pd.isna(mean_highside) or pd.isna(mean_lowside) or pd.isna(std_highside) or pd.isna(std_lowside):
                high_bands.append(np.nan)
                low_bands.append(np.nan)
                high_exceedances.append(np.nan)
                low_exceedances.append(np.nan)
                distances_to_high.append(np.nan)
                distances_to_low.append(np.nan)
                positions_in_range.append(np.nan)
                close_above_high_band.append(np.nan)
                close_below_low_band.append(np.nan)
                close_vs_high_band.append(np.nan)
                close_vs_low_band.append(np.nan)
                momentum_signals.append(np.nan)
                target_prices.append(np.nan)
                continue
            
            # Calculate volatility bands based on previous bar's close
            high_side_limit = prev_close + (std_highside + mean_highside)
            low_side_limit = prev_close - (std_lowside - mean_lowside)
            
            # Current bar data
            current_bar = df.iloc[i]
            current_high = current_bar['high']
            current_low = current_bar['low']
            current_close = current_bar['close']
            
            # Calculate exceedances for current bar
            high_exceedance = max(0, current_high - high_side_limit)
            low_exceedance = max(0, low_side_limit - current_low)
            
            # Calculate close price comparisons with bands
            is_close_above_high = current_close > high_side_limit
            is_close_below_low = current_close < low_side_limit
            close_vs_high_diff = current_close - high_side_limit  # Positive if above, negative if below
            close_vs_low_diff = current_close - low_side_limit    # Positive if above, negative if below
            
            # Calculate relative distances and levels
            band_range = high_side_limit - low_side_limit
            
            if band_range > 0:
                # Calculate distance from each band as percentage
                distance_to_high = ((high_side_limit - current_close) / band_range) * 100
                distance_to_low = ((current_close - low_side_limit) / band_range) * 100
                
                # Calculate position within band range as percentage (0% = at lower band, 100% = at upper band)
                position_in_range = ((current_close - low_side_limit) / band_range) * 100
                
                # Calculate momentum signals based on position in range
                # Momentum strategy: LONG at top of range (>=99%), SHORT at bottom of range (<=1%)
                if position_in_range >= 99:
                    momentum_signal = "LONG"
                elif position_in_range <= 1:
                    momentum_signal = "SHORT"
                else:
                    momentum_signal = None
            else:
                distance_to_high = np.nan
                distance_to_low = np.nan
                position_in_range = np.nan
                momentum_signal = None
            
            # Calculate target price based on momentum signal (will be set later if position opens)
            target_price = None
            
            # Calculate trade status
            trade_status_value = None
            
            # Check if there's currently an open trade by looking at recent trade statuses
            has_open_trade = False
            open_trade_target = None
            
            # Look back through recent trade statuses to find if there's an open trade
            # We need to find the most recent non-None trade status
            for j in range(len(trade_status) - 1, -1, -1):
                past_status = trade_status[j]
                if past_status is not None and not pd.isna(past_status):
                    if past_status == "open":
                        has_open_trade = True
                        # Find the corresponding target price
                        if j < len(target_prices):
                            open_trade_target = target_prices[j]
                    break  # We found the most recent trade status, so we're done
            
            # Calculate band stability using recent high band values (needed for breakeven logic)
            stability_value = "stable"  # Default value
            if len(high_bands) >= 10:  # Need at least 10 values for stability analysis
                recent_high_bands = high_bands[-10:]  # Use last 10 high band values
                stability_value = self.calculate_band_stability(recent_high_bands)
            
            # If we have an open trade, check if current OHLC closes it
            if has_open_trade and open_trade_target is not None:
                if current_low <= open_trade_target <= current_high:
                    trade_status_value = "closed"
            
            # Check for breakeven close conditions:
            # If position is open, band stable, and trade signal same as current open position
            if has_open_trade and trade_status_value is None:
                # Get the momentum signal from the most recent open trade
                open_momentum_signal = None
                for j in range(len(trade_status) - 1, -1, -1):
                    if trade_status[j] == "open":
                        if j < len(momentum_signals):
                            open_momentum_signal = momentum_signals[j]
                        break
                
                # Check if conditions are met for breakeven close
                if (stability_value == "stable" and 
                    momentum_signal is not None and 
                    open_momentum_signal is not None and 
                    momentum_signal == open_momentum_signal):
                    trade_status_value = "closed"  # Close at breakeven
            
            # Only allow opening a new trade if there are no currently open trades and we have a momentum signal
            if trade_status_value is None and momentum_signal is not None and not has_open_trade:
                trade_status_value = "open"
                # Calculate target price only when position is opened
                if momentum_signal == "LONG":
                    # For LONG signals, target is close price + 0.135%
                    target_price = current_close * (1 + 0.01)
                else:  # SHORT
                    # For SHORT signals, target is close price - 0.135%
                    target_price = current_close * (1 - 0.01)
            
            # Calculate entry price - use close price when position is opened
            entry_price = None
            if trade_status_value == "open":
                entry_price = current_close
            
            # Calculate PnL when trade is closed
            # Using compounding returns - 10% of current equity per trade
            pnl_value = None
            if trade_status_value == "closed":
                # Find the entry price and momentum signal from the most recent open trade
                open_entry_price = None
                open_momentum_signal = None
                open_trade_equity = None
                
                # Look back to find the corresponding open trade
                for j in range(len(trade_status) - 1, -1, -1):
                    if trade_status[j] == "open":
                        if j < len(entry_prices) and j < len(momentum_signals) and j < len(current_equity):
                            open_entry_price = entry_prices[j]
                            open_momentum_signal = momentum_signals[j]
                            open_trade_equity = current_equity[j]
                        break
                
                if open_entry_price is not None and open_momentum_signal is not None and open_trade_equity is not None:
                    # Calculate number of shares with 10% of equity at time of trade opening
                    position_size = open_trade_equity * 0.10
                    shares = position_size / open_entry_price
                    
                    # Calculate PnL based on trade direction
                    if open_momentum_signal == "LONG":
                        # For LONG: PnL = (exit_price - entry_price) * shares
                        # Exit price is the target price (already hit since trade is closed)
                        exit_price = open_trade_target if open_trade_target is not None else current_close
                        pnl_value = (exit_price - open_entry_price) * shares
                    elif open_momentum_signal == "SHORT":
                        # For SHORT: PnL = (entry_price - exit_price) * shares
                        exit_price = open_trade_target if open_trade_target is not None else current_close
                        pnl_value = (open_entry_price - exit_price) * shares
                    
                    # Update running equity with the PnL
                    running_equity += pnl_value
            
            # Calculate running PnL for open trades (unrealized P&L)
            running_pnl_value = None
            if has_open_trade:
                # Find the entry price and momentum signal from the most recent open trade
                open_entry_price = None
                open_momentum_signal = None
                open_trade_equity = None
                
                # Look back to find the corresponding open trade
                for j in range(len(trade_status) - 1, -1, -1):
                    if trade_status[j] == "open":
                        if j < len(entry_prices) and j < len(momentum_signals) and j < len(current_equity):
                            open_entry_price = entry_prices[j]
                            open_momentum_signal = momentum_signals[j]
                            open_trade_equity = current_equity[j]
                        break
                
                if open_entry_price is not None and open_momentum_signal is not None and open_trade_equity is not None:
                    # Calculate number of shares with 10% of equity at time of trade opening
                    position_size = open_trade_equity * 0.10
                    shares = position_size / open_entry_price
                    
                    # Calculate unrealized PnL based on current close price and trade direction
                    if open_momentum_signal == "LONG":
                        # For LONG: Running PnL = (current_close - entry_price) * shares
                        running_pnl_value = (current_close - open_entry_price) * shares
                    elif open_momentum_signal == "SHORT":
                        # For SHORT: Running PnL = (entry_price - current_close) * shares
                        running_pnl_value = (open_entry_price - current_close) * shares
            
            # Validate trade status sequence
            validation_status = "VALID"
            if trade_status_value is not None:
                # Check if this violates the open->close->open pattern
                if len(trade_status) > 0:
                    # Get the last non-None trade status
                    last_trade_status = None
                    for j in range(len(trade_status) - 1, -1, -1):
                        if trade_status[j] is not None and not pd.isna(trade_status[j]):
                            last_trade_status = trade_status[j]
                            break
                    
                    # Check for invalid sequences
                    if last_trade_status == "open" and trade_status_value == "open":
                        validation_status = "ERROR: Two consecutive 'open' statuses"
                        row_info = f"Row {i}: datetime={df.iloc[i]['datetime']}"
                        raise ValueError(f"Trade validation failed - {validation_status} at {row_info}")
                    elif last_trade_status == "closed" and trade_status_value == "closed":
                        validation_status = "ERROR: Two consecutive 'closed' statuses"
                        row_info = f"Row {i}: datetime={df.iloc[i]['datetime']}"
                        raise ValueError(f"Trade validation failed - {validation_status} at {row_info}")
            
            # Store calculated values
            high_bands.append(high_side_limit)
            low_bands.append(low_side_limit)
            high_exceedances.append(high_exceedance)
            low_exceedances.append(low_exceedance)
            distances_to_high.append(distance_to_high)
            distances_to_low.append(distance_to_low)
            positions_in_range.append(position_in_range)
            close_above_high_band.append(is_close_above_high)
            close_below_low_band.append(is_close_below_low)
            close_vs_high_band.append(close_vs_high_diff)
            close_vs_low_band.append(close_vs_low_diff)
            momentum_signals.append(momentum_signal)
            target_prices.append(target_price)
            trade_status.append(trade_status_value)
            entry_prices.append(entry_price)
            pnl.append(pnl_value)
            running_pnl.append(running_pnl_value)
            current_equity.append(running_equity)
            band_stability.append(stability_value)
        
        return {
            'high_band': high_bands,
            'low_band': low_bands,
            'high_exceedance': high_exceedances,
            'low_exceedance': low_exceedances,
            'distance_to_high': distances_to_high,
            'distance_to_low': distances_to_low,
            'position_in_range': positions_in_range,
            'close_above_high_band': close_above_high_band,
            'close_below_low_band': close_below_low_band,
            'close_vs_high_band': close_vs_high_band,
            'close_vs_low_band': close_vs_low_band,
            'momentum_signal': momentum_signals,
            'target_price': target_prices,
            'trade_status': trade_status,
            'entry_price': entry_prices,
            'pnl': pnl,
            'running_pnl': running_pnl,
            'current_equity': current_equity,
            'band_stability': band_stability
        }
    
    def fetch_and_process_data(self, symbol, period_type="year", period=1, frequency_type="daily", 
                              frequency=1, start_date=None, end_date=None, lookback=20):
        """
        Fetch historical data and add volatility metrics
        
        Args:
            symbol (str): Stock symbol to fetch data for
            period_type (str): Period type (day, month, year, ytd)
            period (int): Number of periods
            frequency_type (str): Frequency type (minute, daily, weekly, monthly)
            frequency (int): Frequency value
            start_date (int): Start date in milliseconds (optional)
            end_date (int): End date in milliseconds (optional)
            lookback (int): Lookback period for volatility calculations
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
            
            print(f"Calculating volatility metrics with lookback period of {lookback}...")
            
            # Calculate volatility metrics
            vol_metrics = self.calculate_volatility_metrics(df, lookback=lookback)
            
            if vol_metrics:
                # Add volatility metrics to dataframe
                for metric_name, metric_values in vol_metrics.items():
                    df[metric_name] = metric_values
                
                print(f"Added volatility metrics: {list(vol_metrics.keys())}")
            else:
                print("Failed to calculate volatility metrics")
                return None
            
            # Reorder columns for better readability
            base_columns = ['symbol', 'datetime', 'open', 'high', 'low', 'close', 'volume']
            volatility_columns = ['high_band', 'low_band', 'high_exceedance', 'low_exceedance', 
                                'distance_to_high', 'distance_to_low', 'position_in_range',
                                'close_above_high_band', 'close_below_low_band', 
                                'close_vs_high_band', 'close_vs_low_band', 'momentum_signal', 'target_price', 'trade_status', 'entry_price', 'pnl', 'running_pnl', 'current_equity', 'band_stability']
            
            # Ensure all columns exist
            available_columns = [col for col in base_columns + volatility_columns if col in df.columns]
            df = df[available_columns]
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            freq_str = f"{frequency}{frequency_type}" if frequency != 1 else frequency_type
            filename = f"{self.output_dir}/{symbol}_volatility_data_{period}{period_type}_{freq_str}_{timestamp}.csv"
            
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
            
            # Display volatility statistics (excluding NaN values)
            if 'high_band' in df.columns:
                valid_high_bands = df['high_band'].dropna()
                valid_low_bands = df['low_band'].dropna()
                valid_exceedances = df['high_exceedance'].dropna() + df['low_exceedance'].dropna()
                
                if len(valid_high_bands) > 0:
                    print(f"\nVolatility Statistics:")
                    print(f"Average High Band: ${valid_high_bands.mean():.2f}")
                    print(f"Average Low Band: ${valid_low_bands.mean():.2f}")
                    print(f"Average Band Width: ${(valid_high_bands - valid_low_bands).mean():.2f}")
                    print(f"Total Exceedances: {(valid_exceedances > 0).sum()}")
            
            # Calculate and display performance summary
            self.calculate_performance_summary(filename)
            
            return filename
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Fetch historical data with volatility metrics and save as CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch 1 year of daily data with volatility metrics
  python3 volatility_data_fetcher.py AAPL
  
  # Fetch 6 months of daily data
  python3 volatility_data_fetcher.py NVDA --period 6 --period-type month
  
  # Fetch 5 days of 5-minute data
  python3 volatility_data_fetcher.py TSLA --period 5 --period-type day --frequency-type minute --frequency 5
  
  # Custom lookback period for volatility calculations
  python3 volatility_data_fetcher.py AAPL --lookback 50
        """
    )
    
    parser.add_argument(
        'ticker',
        type=str,
        help='Stock ticker symbol to fetch data for (e.g., AAPL, NVDA, TSLA)'
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
        default=20,
        help='Lookback period for volatility calculations (default: 20)'
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
    
    # If no dates provided but we want 10 years of data, calculate them
    if not args.start_date and not args.end_date and args.period_type == 'day' and args.frequency_type == 'minute':
        end_date = datetime.now()
        start_date = end_date - timedelta(days=10*365)  # Approximately 10 years
        start_date_ms = int(start_date.timestamp() * 1000)
        end_date_ms = int(end_date.timestamp() * 1000)
        print(f"Auto-calculated date range for 10 years: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    print("=" * 80)
    print("HISTORICAL DATA WITH VOLATILITY METRICS FETCHER")
    print("=" * 80)
    print(f"Symbol: {ticker}")
    print(f"Period: {args.period} {args.period_type}")
    print(f"Frequency: {args.frequency} {args.frequency_type}")
    print(f"Volatility Lookback: {args.lookback} periods")
    if start_date_ms and end_date_ms:
        start_str = datetime.fromtimestamp(start_date_ms/1000).strftime('%Y-%m-%d')
        end_str = datetime.fromtimestamp(end_date_ms/1000).strftime('%Y-%m-%d')
        print(f"Date Range: {start_str} to {end_str}")
    print("=" * 80)
    
    # Create fetcher and process data
    fetcher = VolatilityDataFetcher()
    result = fetcher.fetch_and_process_data(
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
        print(f"\n✅ Successfully created volatility data CSV: {result}")
        
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
                "python3", "exceedence_visualization.py", 
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

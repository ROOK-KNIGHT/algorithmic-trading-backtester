#!/usr/bin/env python3
"""
Open Momentum Backtester

A comprehensive backtesting system for the Open Momentum strategy that operates on 5-minute data.
The strategy buys when current open > previous open and sells when current open < previous open.

Key Features:
- 5-minute intraday momentum analysis
- Regular trading hours filtering (9:30 AM - 4:00 PM ET)
- Comprehensive performance metrics
- CSV output with full trade history
- Automatic visualization generation

Strategy Logic:
- Entry: Buy when current_5min_open > previous_5min_open AND price > 21-period EMA
- Exit: Sell after 5 minutes (1 bar) - time-based exit only
- EMA Filter: Only enter positions when price is above the 21-period EMA (trend filter)
- Time-based exit: Maximum hold time of 5 minutes
- Position sizing: 10% of current equity per trade

Author: Trading System
Date: 2025-01-06
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import argparse
import time
import subprocess
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from handlers.historical_data_handler import HistoricalDataHandler

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")


class OpenMomentumBacktester:
    """
    Open Momentum Strategy Backtester for 5-minute data
    """
    
    def __init__(self, initial_capital=25000.0, position_size_pct=10.0):
        """
        Initialize the Open Momentum Backtester
        
        Args:
            initial_capital (float): Starting capital amount
            position_size_pct (float): Percentage of equity to use per trade
        """
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct / 100.0  # Convert to decimal
        self.data_handler = HistoricalDataHandler()
        
        # Create output directories
        self.output_dir = 'historical_data'
        self.charts_dir = 'charts'
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)
        
    def fetch_data(self, symbol, period_type="day", period=10, frequency_type="minute", frequency=5):
        """
        Fetch 5-minute historical data for the symbol
        
        Args:
            symbol (str): Stock symbol
            period_type (str): Period type ('day', 'month', 'year') - For 5-min data, must be 'day'
            period (int): Number of periods - For 'day': 1, 2, 3, 4, 5, 10
            frequency_type (str): Frequency type ('minute', 'daily', etc.) - For 5-min data, must be 'minute'
            frequency (int): Frequency value (5 for 5-minute bars) - Valid: 1, 5, 10, 15, 30
            
        Returns:
            pd.DataFrame: Historical OHLCV data
        """
        try:
            # Validate parameters according to Schwab API documentation
            if frequency_type == "minute" and period_type != "day":
                print(f"⚠️  Warning: For minute data, periodType must be 'day'. Adjusting from '{period_type}' to 'day'")
                period_type = "day"
            
            if period_type == "day" and period not in [1, 2, 3, 4, 5, 10]:
                print(f"⚠️  Warning: For periodType='day', period must be 1,2,3,4,5,10. Adjusting from {period} to 10")
                period = 10
            
            if frequency_type == "minute" and frequency not in [1, 5, 10, 15, 30]:
                print(f"⚠️  Warning: For frequencyType='minute', frequency must be 1,5,10,15,30. Using {frequency}")
            
            print(f"Fetching 5-minute data for {symbol}...")
            print(f"API Parameters: periodType={period_type}, period={period}, frequencyType={frequency_type}, frequency={frequency}")
            
            # Fetch data using existing handler
            raw_data = self.data_handler.fetch_historical_data(
                symbol=symbol,
                periodType=period_type,
                period=period,
                frequencyType=frequency_type,
                freq=frequency,
                needExtendedHoursData=False  # Regular trading hours only
            )
            
            if not raw_data or 'candles' not in raw_data:
                print(f"No data retrieved for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            candles = raw_data['candles']
            df = pd.DataFrame(candles)
            
            # Convert datetime
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Sort by datetime and reset index
            df = df.sort_values('datetime').reset_index(drop=True)
            
            # Add symbol column
            df['symbol'] = symbol
            
            print(f"Successfully fetched {len(df)} 5-minute bars for {symbol}")
            print(f"Date range: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def filter_trading_hours(self, df):
        """
        Filter data to include only regular trading hours (9:30 AM - 4:00 PM ET)
        
        Args:
            df (pd.DataFrame): Raw OHLCV data
            
        Returns:
            pd.DataFrame: Filtered data for trading hours only
        """
        if df.empty:
            return df
        
        # Create a copy to work with
        df_copy = df.copy()
        
        try:
            # Check if datetime is already timezone-aware
            if df_copy['datetime'].dt.tz is None:
                # Assume UTC if no timezone info
                df_copy['datetime_et'] = df_copy['datetime'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
            else:
                # Convert existing timezone to Eastern
                df_copy['datetime_et'] = df_copy['datetime'].dt.tz_convert('US/Eastern')
            
            # Extract hour and minute
            df_copy['hour'] = df_copy['datetime_et'].dt.hour
            df_copy['minute'] = df_copy['datetime_et'].dt.minute
            
            # Debug: Print some sample times to understand the data
            print(f"Sample times (first 5 bars):")
            for i in range(min(5, len(df_copy))):
                print(f"  {df_copy.iloc[i]['datetime']} -> {df_copy.iloc[i]['datetime_et']} (Hour: {df_copy.iloc[i]['hour']}, Min: {df_copy.iloc[i]['minute']})")
            
            # Filter for regular trading hours (9:30 AM - 4:00 PM ET)
            # 9:30 AM = hour 9, minute >= 30 OR hour >= 10
            # 4:00 PM = hour < 16
            trading_hours_mask = (
                ((df_copy['hour'] == 9) & (df_copy['minute'] >= 30)) |
                ((df_copy['hour'] >= 10) & (df_copy['hour'] < 16))
            )
            
            filtered_df = df_copy[trading_hours_mask].copy()
            
            # Remove the temporary columns
            if 'datetime_et' in filtered_df.columns:
                filtered_df = filtered_df.drop(['datetime_et', 'hour', 'minute'], axis=1)
            
            print(f"Filtered to trading hours: {len(filtered_df)} bars (from {len(df)} total)")
            
            # If no data after filtering, let's be more permissive and just use all data
            if len(filtered_df) == 0:
                print("⚠️  No data found in standard trading hours. Using all available data.")
                return df.reset_index(drop=True)
            
            return filtered_df.reset_index(drop=True)
            
        except Exception as e:
            print(f"⚠️  Error in trading hours filtering: {e}")
            print("Using all available data without filtering.")
            return df.reset_index(drop=True)
    
    def calculate_signals_and_performance(self, df):
        """
        Calculate Open Momentum signals and performance metrics
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            pd.DataFrame: Data with signals and performance metrics added
        """
        if df.empty or len(df) < 2:
            return df
        
        result = df.copy()
        
        # Calculate previous open
        result['prev_open'] = result['open'].shift(1)
        
        # Calculate open momentum
        result['open_momentum'] = result['open'] - result['prev_open']
        
        # Calculate 21-period EMA
        result['ema_21'] = result['close'].ewm(span=21, adjust=False).mean()
        
        # Initialize signal and position tracking columns
        result['signal'] = None
        result['position'] = 0  # 0 = flat, 1 = long
        result['entry_price'] = np.nan
        result['exit_price'] = np.nan
        result['trade_pnl'] = np.nan
        result['cumulative_pnl'] = 0.0
        result['equity'] = self.initial_capital
        result['shares'] = np.nan
        result['position_value'] = np.nan
        result['unrealized_pnl'] = np.nan
        
        # Track current state
        current_equity = self.initial_capital
        current_position = 0
        entry_price = None
        entry_bar_index = None  # Track when position was opened
        shares = 0
        cumulative_pnl = 0.0
        
        for i in range(1, len(result)):  # Start from index 1 since we need previous open
            current_open = result.iloc[i]['open']
            prev_open = result.iloc[i]['prev_open']
            close_price = result.iloc[i]['close']
            prev_close = result.iloc[i - 1]['close'] 
            
            if pd.isna(prev_open):
                # Update running totals even if no signal
                result.iloc[i, result.columns.get_loc('cumulative_pnl')] = cumulative_pnl
                result.iloc[i, result.columns.get_loc('equity')] = current_equity
                result.iloc[i, result.columns.get_loc('position')] = current_position
                continue
            
            signal = None
            
            # Get current EMA value
            current_ema = result.iloc[i]['ema_21']
            
            # Signal logic with EMA filter
            if (current_open > prev_close and 
                current_position == 0 and 
                pd.notna(current_ema) and 
                current_open > current_ema):
                # Buy signal - enter long position (price above EMA)
                signal = 'BUY'
                current_position = 1
                entry_price = current_open
                entry_bar_index = i  # Track when position was opened
                position_size = current_equity * self.position_size_pct
                shares = position_size / entry_price
                
                result.iloc[i, result.columns.get_loc('entry_price')] = entry_price
                result.iloc[i, result.columns.get_loc('shares')] = shares
                
            elif current_position == 1 and entry_bar_index is not None:
                # Check exit conditions for open position
                bars_held = i - entry_bar_index
                
                # Exit condition: Time-based exit after 1 bar (5 minutes)
                if bars_held >= 6:
                    # Sell signal - exit long position
                    signal = 'SELL'
                    current_position = 0
                    exit_price = close_price
                    
                    if entry_price is not None and shares > 0:
                        trade_pnl = (exit_price - entry_price) * shares
                        cumulative_pnl += trade_pnl
                        current_equity += trade_pnl
                        
                        result.iloc[i, result.columns.get_loc('exit_price')] = exit_price
                        result.iloc[i, result.columns.get_loc('trade_pnl')] = trade_pnl
                    
                    # Reset position
                    entry_price = None
                    entry_bar_index = None
                    shares = 0
            
            # Update columns
            result.iloc[i, result.columns.get_loc('signal')] = signal
            result.iloc[i, result.columns.get_loc('position')] = current_position
            result.iloc[i, result.columns.get_loc('cumulative_pnl')] = cumulative_pnl
            result.iloc[i, result.columns.get_loc('equity')] = current_equity
            
            # Calculate current position value and unrealized P&L for open positions
            if current_position == 1 and entry_price is not None and shares > 0:
                position_value = shares * close_price
                unrealized_pnl = (close_price - entry_price) * shares
                result.iloc[i, result.columns.get_loc('position_value')] = position_value
                result.iloc[i, result.columns.get_loc('unrealized_pnl')] = unrealized_pnl
        
        return result
    
    def calculate_performance_summary(self, df, symbol):
        """
        Calculate comprehensive performance summary
        
        Args:
            df (pd.DataFrame): Data with performance metrics
            symbol (str): Stock symbol
            
        Returns:
            dict: Performance summary statistics
        """
        if df.empty:
            return {}
        
        # Get completed trades
        trades = df[df['trade_pnl'].notna()].copy()
        
        if len(trades) == 0:
            return {
                'symbol': symbol,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'final_equity': self.initial_capital,
                'roi': 0.0,
                'data_period': {
                    'start_date': df['datetime'].iloc[0] if not df.empty else None,
                    'end_date': df['datetime'].iloc[-1] if not df.empty else None,
                    'total_bars': len(df)
                }
            }
        
        # Basic trade statistics
        total_trades = len(trades)
        winning_trades = len(trades[trades['trade_pnl'] > 0])
        losing_trades = len(trades[trades['trade_pnl'] < 0])
        breakeven_trades = len(trades[trades['trade_pnl'] == 0])
        
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0.0
        
        total_pnl = trades['trade_pnl'].sum()
        avg_pnl = trades['trade_pnl'].mean()
        max_win = trades['trade_pnl'].max()
        max_loss = trades['trade_pnl'].min()
        
        final_equity = self.initial_capital + total_pnl
        roi = (total_pnl / self.initial_capital) * 100
        
        # Calculate profit factor
        gross_profit = trades[trades['trade_pnl'] > 0]['trade_pnl'].sum()
        gross_loss = abs(trades[trades['trade_pnl'] < 0]['trade_pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate Sharpe ratio
        returns = trades['trade_pnl'] / self.initial_capital
        avg_return = returns.mean()
        return_std = returns.std()
        sharpe_ratio = avg_return / return_std if return_std > 0 else 0
        
        # Calculate maximum drawdown
        equity_curve = df['equity'].dropna()
        if len(equity_curve) > 0:
            running_max = equity_curve.expanding().max()
            drawdown = equity_curve - running_max
            max_drawdown = drawdown.min()
            max_drawdown_pct = (max_drawdown / self.initial_capital) * 100
        else:
            max_drawdown = 0
            max_drawdown_pct = 0
        
        # Calculate holding periods (in 5-minute bars)
        buy_signals = df[df['signal'] == 'BUY'].index
        sell_signals = df[df['signal'] == 'SELL'].index
        
        holding_periods = []
        for sell_idx in sell_signals:
            # Find the most recent buy signal before this sell
            buy_idx = None
            for buy in reversed(buy_signals):
                if buy < sell_idx:
                    buy_idx = buy
                    break
            
            if buy_idx is not None:
                holding_period = sell_idx - buy_idx
                holding_periods.append(holding_period)
        
        avg_holding_period = np.mean(holding_periods) if holding_periods else 0
        
        # Calculate trades per day
        if not df.empty:
            trading_days = (df['datetime'].iloc[-1] - df['datetime'].iloc[0]).days
            trades_per_day = total_trades / max(trading_days, 1)
        else:
            trades_per_day = 0
        
        return {
            'symbol': symbol,
            'data_period': {
                'start_date': df['datetime'].iloc[0],
                'end_date': df['datetime'].iloc[-1],
                'total_bars': len(df),
                'trading_days': trading_days if not df.empty else 0
            },
            'trade_statistics': {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'breakeven_trades': breakeven_trades,
                'win_rate': win_rate,
                'trades_per_day': trades_per_day
            },
            'performance_metrics': {
                'initial_capital': self.initial_capital,
                'final_equity': final_equity,
                'total_pnl': total_pnl,
                'roi': roi,
                'avg_pnl': avg_pnl,
                'max_win': max_win,
                'max_loss': max_loss,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown_pct
            },
            'timing_statistics': {
                'avg_holding_period_bars': avg_holding_period,
                'avg_holding_period_minutes': avg_holding_period * 5,
                'position_size_pct': self.position_size_pct * 100
            }
        }
    
    def save_results_to_csv(self, df, symbol):
        """
        Save backtest results to CSV file
        
        Args:
            df (pd.DataFrame): Results dataframe
            symbol (str): Stock symbol
            
        Returns:
            str: Path to saved CSV file
        """
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'{self.output_dir}/{symbol}_openMomentum_backtest_{timestamp}.csv'
            
            # Prepare data for CSV (ensure proper formatting)
            csv_data = df.copy()
            
            # Format datetime for better readability
            csv_data['datetime'] = csv_data['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Round numeric columns to reasonable precision
            numeric_columns = csv_data.select_dtypes(include=[np.number]).columns
            csv_data[numeric_columns] = csv_data[numeric_columns].round(6)
            
            # Save to CSV
            csv_data.to_csv(filename, index=False)
            
            print(f"Results saved to: {filename}")
            print(f"CSV contains {len(csv_data)} rows and {len(csv_data.columns)} columns")
            
            return filename
            
        except Exception as e:
            print(f"Error saving CSV for {symbol}: {e}")
            return ""
    
    def create_performance_visualization(self, df, summary, symbol):
        """
        Create comprehensive performance visualization
        
        Args:
            df (pd.DataFrame): Results dataframe
            summary (dict): Performance summary
            symbol (str): Stock symbol
            
        Returns:
            str: Path to saved chart
        """
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Open Momentum Strategy - {symbol}', fontsize=16, fontweight='bold')
            
            # Plot 1: Price chart with signals
            ax1 = axes[0, 0]
            ax1.plot(df.index, df['close'], label='Close Price', alpha=0.7, linewidth=1)
            
            # Mark buy and sell signals
            buy_signals = df[df['signal'] == 'BUY']
            sell_signals = df[df['signal'] == 'SELL']
            
            if not buy_signals.empty:
                ax1.scatter(buy_signals.index, buy_signals['close'], 
                           color='green', marker='^', s=50, label='Buy Signal', alpha=0.8)
            
            if not sell_signals.empty:
                ax1.scatter(sell_signals.index, sell_signals['close'], 
                           color='red', marker='v', s=50, label='Sell Signal', alpha=0.8)
            
            ax1.set_title('Price Action with Trading Signals')
            ax1.set_ylabel('Price ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Equity curve
            ax2 = axes[0, 1]
            ax2.plot(df.index, df['equity'], label='Equity Curve', color='blue', linewidth=2)
            ax2.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
            ax2.set_title('Equity Curve')
            ax2.set_ylabel('Equity ($)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Open momentum histogram
            ax3 = axes[1, 0]
            momentum_data = df['open_momentum'].dropna()
            if not momentum_data.empty:
                ax3.hist(momentum_data, bins=50, alpha=0.7, color='purple', edgecolor='black')
                ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero Line')
                ax3.set_title('Open Momentum Distribution')
                ax3.set_xlabel('Open Momentum ($)')
                ax3.set_ylabel('Frequency')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # Plot 4: Performance statistics
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            # Create performance statistics text
            stats_text = f"""
PERFORMANCE SUMMARY

Total Trades: {summary['trade_statistics']['total_trades']}
Win Rate: {summary['trade_statistics']['win_rate']:.1f}%
Trades/Day: {summary['trade_statistics']['trades_per_day']:.1f}

Initial Capital: ${summary['performance_metrics']['initial_capital']:,.0f}
Final Equity: ${summary['performance_metrics']['final_equity']:,.0f}
Total P&L: ${summary['performance_metrics']['total_pnl']:,.0f}
ROI: {summary['performance_metrics']['roi']:.2f}%

Profit Factor: {summary['performance_metrics']['profit_factor']:.2f}
Sharpe Ratio: {summary['performance_metrics']['sharpe_ratio']:.3f}
Max Drawdown: {summary['performance_metrics']['max_drawdown_pct']:.2f}%

Avg Hold Time: {summary['timing_statistics']['avg_holding_period_minutes']:.1f} min
Position Size: {summary['timing_statistics']['position_size_pct']:.1f}%

Data Period: {summary['data_period']['start_date'].strftime('%Y-%m-%d')} to 
{summary['data_period']['end_date'].strftime('%Y-%m-%d')}
Total Bars: {summary['data_period']['total_bars']:,}
            """
            
            ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            
            # Save chart
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            chart_filename = f'{self.charts_dir}/{symbol}_openMomentum_analysis_{timestamp}.png'
            plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Chart saved to: {chart_filename}")
            return chart_filename
            
        except Exception as e:
            print(f"Error creating visualization for {symbol}: {e}")
            return ""
    
    def print_performance_summary(self, summary):
        """
        Print formatted performance summary
        
        Args:
            summary (dict): Performance summary statistics
        """
        symbol = summary['symbol']
        
        print(f"\n📊 OPEN MOMENTUM STRATEGY RESULTS: {symbol}")
        print("=" * 70)
        
        if summary['trade_statistics']['total_trades'] == 0:
            print("No trades executed during the backtest period.")
            return
        
        # Data period info
        print(f"📅 DATA PERIOD")
        print(f"   Start Date: {summary['data_period']['start_date'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   End Date: {summary['data_period']['end_date'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Total 5-min Bars: {summary['data_period']['total_bars']:,}")
        print(f"   Trading Days: {summary['data_period']['trading_days']}")
        
        # Trade statistics
        trade_stats = summary['trade_statistics']
        print(f"\n📈 TRADE STATISTICS")
        print(f"   Total Trades: {trade_stats['total_trades']}")
        print(f"   Winning Trades: {trade_stats['winning_trades']}")
        print(f"   Losing Trades: {trade_stats['losing_trades']}")
        print(f"   Breakeven Trades: {trade_stats['breakeven_trades']}")
        print(f"   Win Rate: {trade_stats['win_rate']:.1f}%")
        print(f"   Trades per Day: {trade_stats['trades_per_day']:.1f}")
        
        # Performance metrics
        perf_metrics = summary['performance_metrics']
        print(f"\n💰 PERFORMANCE METRICS")
        print(f"   Initial Capital: ${perf_metrics['initial_capital']:,.2f}")
        print(f"   Final Equity: ${perf_metrics['final_equity']:,.2f}")
        print(f"   Total P&L: ${perf_metrics['total_pnl']:,.2f}")
        print(f"   ROI: {perf_metrics['roi']:.2f}%")
        print(f"   Average P&L per Trade: ${perf_metrics['avg_pnl']:.2f}")
        print(f"   Largest Win: ${perf_metrics['max_win']:.2f}")
        print(f"   Largest Loss: ${perf_metrics['max_loss']:.2f}")
        print(f"   Gross Profit: ${perf_metrics['gross_profit']:.2f}")
        print(f"   Gross Loss: ${perf_metrics['gross_loss']:.2f}")
        print(f"   Profit Factor: {perf_metrics['profit_factor']:.2f}")
        print(f"   Sharpe Ratio: {perf_metrics['sharpe_ratio']:.3f}")
        print(f"   Maximum Drawdown: ${perf_metrics['max_drawdown']:.2f} ({perf_metrics['max_drawdown_pct']:.2f}%)")
        
        # Timing statistics
        timing_stats = summary['timing_statistics']
        print(f"\n⏱️ TIMING STATISTICS")
        print(f"   Average Holding Period: {timing_stats['avg_holding_period_bars']:.1f} bars")
        print(f"   Average Holding Period: {timing_stats['avg_holding_period_minutes']:.1f} minutes")
        print(f"   Position Size: {timing_stats['position_size_pct']:.1f}% of equity")
        
        # Strategy insights
        print(f"\n🔍 STRATEGY INSIGHTS")
        if trade_stats['win_rate'] > 50:
            print(f"   ✅ Positive win rate strategy")
        else:
            print(f"   ⚠️  Win rate below 50%")
        
        if perf_metrics['roi'] > 0:
            print(f"   ✅ Profitable strategy")
        else:
            print(f"   ❌ Unprofitable strategy")
        
        if perf_metrics['profit_factor'] > 1.5:
            print(f"   ✅ Strong profit factor")
        elif perf_metrics['profit_factor'] > 1.0:
            print(f"   ⚠️  Moderate profit factor")
        else:
            print(f"   ❌ Poor profit factor")
        
        if timing_stats['avg_holding_period_minutes'] < 30:
            print(f"   ⚡ Very short-term strategy (avg {timing_stats['avg_holding_period_minutes']:.1f} min holds)")
        elif timing_stats['avg_holding_period_minutes'] < 120:
            print(f"   🕐 Short-term strategy (avg {timing_stats['avg_holding_period_minutes']:.1f} min holds)")
        else:
            print(f"   🕑 Medium-term strategy (avg {timing_stats['avg_holding_period_minutes']:.1f} min holds)")
        
        print("=" * 70)
    
    def run_backtest(self, symbol, period_type="day", period=10):
        """
        Run complete backtest for the Open Momentum strategy
        
        Args:
            symbol (str): Stock symbol to backtest
            period_type (str): Period type for data fetching - For 5-min data, should be 'day'
            period (int): Number of periods - For 'day': 1, 2, 3, 4, 5, 10
            
        Returns:
            tuple: (DataFrame with results, dict with summary, str csv_path, str chart_path)
        """
        print(f"\n🚀 === OPEN MOMENTUM STRATEGY BACKTEST ===")
        print(f"Symbol: {symbol}")
        print(f"Strategy: Buy when open > prev_open, Sell when open < prev_open")
        print(f"Data: 5-minute bars, {period} {period_type}")
        print(f"Position Size: {self.position_size_pct*100:.1f}% of equity")
        print("=" * 60)
        
        # Fetch data with proper API parameters for 5-minute data
        df = self.fetch_data(symbol, period_type=period_type, period=period, frequency_type="minute", frequency=5)
        
        if df.empty:
            print("❌ No data available for backtesting")
            return pd.DataFrame(), {}, "", ""
        
        # Filter to trading hours
        df = self.filter_trading_hours(df)
        
        if df.empty:
            print("❌ No trading hours data available")
            return pd.DataFrame(), {}, "", ""
        
        # Calculate signals and performance
        print("📊 Calculating signals and performance...")
        df = self.calculate_signals_and_performance(df)
        
        # Generate performance summary
        print("📈 Generating performance summary...")
        summary = self.calculate_performance_summary(df, symbol)
        
        # Save results to CSV
        print("💾 Saving results to CSV...")
        csv_path = self.save_results_to_csv(df, symbol)
        
        # Create visualization
        print("🎨 Creating performance visualization...")
        chart_path = self.create_performance_visualization(df, summary, symbol)
        
        return df, summary, csv_path, chart_path


def main():
    """
    Main function for running the Open Momentum backtester
    """
    parser = argparse.ArgumentParser(
        description="Open Momentum Backtester - 5-minute momentum strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run backtest on AAPL with default settings (6 months, 10% position size)
  python3 openMomentum_backtest.py AAPL
  
  # Run backtest on NVDA with 3 months of data
  python3 openMomentum_backtest.py NVDA --period 3 --period-type month
  
  # Run backtest on TSLA with 5% position size
  python3 openMomentum_backtest.py TSLA --position-size 5.0
  
  # Run backtest with 1 year of data and custom capital
  python3 openMomentum_backtest.py AAPL --period 1 --period-type year --initial-capital 50000
        """
    )
    
    parser.add_argument(
        'symbol',
        type=str,
        help='Stock symbol to backtest (e.g., AAPL, NVDA, TSLA)'
    )
    
    parser.add_argument(
        '--period-type',
        type=str,
        choices=['day', 'month', 'year'],
        default='month',
        help='Period type for data fetching (default: month)'
    )
    
    parser.add_argument(
        '--period',
        type=int,
        default=6,
        help='Number of periods to fetch (default: 6)'
    )
    
    parser.add_argument(
        '--position-size',
        type=float,
        default=10.0,
        help='Position size as percentage of equity (default: 10.0)'
    )
    
    parser.add_argument(
        '--initial-capital',
        type=float,
        default=25000.0,
        help='Initial capital amount (default: 25000.0)'
    )
    
    args = parser.parse_args()
    
    # Convert symbol to uppercase
    symbol = args.symbol.upper()
    
    print("=" * 80)
    print("OPEN MOMENTUM BACKTESTER")
    print("5-Minute Momentum Strategy")
    print("=" * 80)
    print(f"Symbol: {symbol}")
    print(f"Period: {args.period} {args.period_type}")
    print(f"Position Size: {args.position_size}% of equity")
    print(f"Initial Capital: ${args.initial_capital:,.2f}")
    print("=" * 80)
    
    try:
        # Initialize backtester
        backtester = OpenMomentumBacktester(
            initial_capital=args.initial_capital,
            position_size_pct=args.position_size
        )
        
        # Run backtest
        results_df, summary, csv_path, chart_path = backtester.run_backtest(
            symbol=symbol,
            period_type=args.period_type,
            period=args.period
        )
        
        if not results_df.empty and summary:
            # Print performance summary
            backtester.print_performance_summary(summary)
            
            print(f"\n✅ BACKTEST COMPLETED SUCCESSFULLY")
            print(f"📊 Processed {len(results_df):,} 5-minute bars")
            print(f"💾 Results saved to: {csv_path}")
            print(f"🎨 Chart saved to: {chart_path}")
            
            # Show key metrics
            if summary['trade_statistics']['total_trades'] > 0:
                print(f"\n🎯 KEY RESULTS:")
                print(f"   Total Trades: {summary['trade_statistics']['total_trades']}")
                print(f"   Win Rate: {summary['trade_statistics']['win_rate']:.1f}%")
                print(f"   ROI: {summary['performance_metrics']['roi']:.2f}%")
                print(f"   Profit Factor: {summary['performance_metrics']['profit_factor']:.2f}")
            
        else:
            print(f"\n❌ Backtest failed for {symbol}")
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Backtest interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during backtest: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

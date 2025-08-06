#!/usr/bin/env python3
"""
EMA Accumulation Backtester

A comprehensive backtesting system for the EMA Accumulation strategy that operates on 1-minute data.
The strategy accumulates shares when price is below the 21-EMA and exits at 0.5% profit from average entry price.

Key Features:
- 1-minute intraday accumulation analysis
- Regular trading hours filtering (9:30 AM - 4:00 PM ET)
- Rolling average position price tracking
- Rolling PnL tracking since position opened
- Comprehensive performance metrics
- CSV output with full trade history
- Automatic visualization generation

Strategy Logic:
- Entry: Buy 1 share every minute when current_price < 21-period EMA
- Exit: Sell entire position when current_price >= (average_entry_price * 1.005) - 0.5% profit target
- No stop loss
- Accumulation: Continuously add shares while price remains below EMA
- Position tracking: Track rolling average entry price and unrealized PnL

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
from visualizers.emaAccumulation_visualization import EMAAccumulationVisualizer

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")


class EMAAccumulationBacktester:
    """
    EMA Accumulation Strategy Backtester for 1-minute data
    """
    
    def __init__(self, initial_capital=25000.0, ema_period=21, profit_target=0.005):
        """
        Initialize the EMA Accumulation Backtester
        
        Args:
            initial_capital (float): Starting capital amount
            ema_period (int): EMA period for trend filter
            profit_target (float): Profit target as decimal (0.005 = 0.5%)
        """
        self.initial_capital = initial_capital
        self.ema_period = ema_period
        self.profit_target = profit_target
        self.data_handler = HistoricalDataHandler()
        
        # Create output directories
        self.output_dir = 'historical_data'
        self.charts_dir = 'charts'
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)
        
    def fetch_data(self, symbol, period_type="day", period=10, frequency_type="minute", frequency=1):
        """
        Fetch 1-minute historical data for the symbol
        
        Args:
            symbol (str): Stock symbol
            period_type (str): Period type ('day', 'month', 'year') - For 1-min data, must be 'day'
            period (int): Number of periods - For 'day': 1, 2, 3, 4, 5, 10
            frequency_type (str): Frequency type ('minute', 'daily', etc.) - For 1-min data, must be 'minute'
            frequency (int): Frequency value (1 for 1-minute bars) - Valid: 1, 5, 10, 15, 30
            
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
            
            print(f"Fetching 1-minute data for {symbol}...")
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
            
            print(f"Successfully fetched {len(df)} 1-minute bars for {symbol}")
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
        Calculate EMA Accumulation signals and performance metrics
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            pd.DataFrame: Data with signals and performance metrics added
        """
        if df.empty or len(df) < self.ema_period:
            return df
        
        result = df.copy()
        
        # Calculate EMA
        result['ema_21'] = result['close'].ewm(span=self.ema_period, adjust=False).mean()
        
        # Initialize tracking columns
        result['signal'] = None
        result['position_shares'] = 0  # Current position size in shares
        result['avg_entry_price'] = np.nan  # Rolling average entry price
        result['total_cost'] = 0.0  # Total cost basis of position
        result['position_value'] = 0.0  # Current market value of position
        result['unrealized_pnl'] = 0.0  # Unrealized P&L since position opened
        result['rolling_pnl'] = 0.0  # Rolling P&L since position was opened
        result['trade_pnl'] = np.nan  # Realized P&L when position is closed
        result['cumulative_pnl'] = 0.0  # Cumulative realized P&L
        result['equity'] = self.initial_capital  # Current equity
        result['cash'] = self.initial_capital  # Available cash
        
        # Track current state
        current_shares = 0
        total_cost = 0.0
        avg_entry_price = 0.0
        cumulative_pnl = 0.0
        current_cash = self.initial_capital
        position_start_pnl = 0.0  # Track PnL from when position was first opened
        
        for i in range(self.ema_period, len(result)):  # Start after EMA calculation period
            current_price = result.iloc[i]['close']
            current_ema = result.iloc[i]['ema_21']
            
            if pd.isna(current_ema):
                continue
            
            signal = None
            
            # Strategy Logic: Buy 1 share when price < EMA, Sell all when profit target hit
            if current_price < current_ema and current_cash >= current_price:
                # BUY SIGNAL: Accumulate 1 share
                signal = 'BUY'
                shares_to_buy = 1
                cost = shares_to_buy * current_price
                
                # Check if this is the start of a new position
                if current_shares == 0:
                    position_start_pnl = cumulative_pnl  # Reset rolling PnL tracker
                
                # Update position
                current_shares += shares_to_buy
                total_cost += cost
                current_cash -= cost
                avg_entry_price = total_cost / current_shares
                
            elif current_shares > 0:
                # Check exit condition: 0.5% profit from average entry price
                profit_target_price = avg_entry_price * (1 + self.profit_target)
                
                if current_price >= profit_target_price:
                    # SELL SIGNAL: Exit entire position
                    signal = 'SELL'
                    
                    # Calculate trade P&L
                    position_value = current_shares * current_price
                    trade_pnl = position_value - total_cost
                    cumulative_pnl += trade_pnl
                    current_cash += position_value
                    
                    result.iloc[i, result.columns.get_loc('trade_pnl')] = trade_pnl
                    
                    # Reset position
                    current_shares = 0
                    total_cost = 0.0
                    avg_entry_price = 0.0
            
            # Update all tracking columns
            result.iloc[i, result.columns.get_loc('signal')] = signal
            result.iloc[i, result.columns.get_loc('position_shares')] = current_shares
            result.iloc[i, result.columns.get_loc('total_cost')] = total_cost
            result.iloc[i, result.columns.get_loc('cash')] = current_cash
            result.iloc[i, result.columns.get_loc('cumulative_pnl')] = cumulative_pnl
            
            if current_shares > 0:
                result.iloc[i, result.columns.get_loc('avg_entry_price')] = avg_entry_price
                position_value = current_shares * current_price
                unrealized_pnl = position_value - total_cost
                rolling_pnl = cumulative_pnl - position_start_pnl + unrealized_pnl
                
                result.iloc[i, result.columns.get_loc('position_value')] = position_value
                result.iloc[i, result.columns.get_loc('unrealized_pnl')] = unrealized_pnl
                result.iloc[i, result.columns.get_loc('rolling_pnl')] = rolling_pnl
            
            # Update equity (cash + position value)
            current_equity = current_cash + (current_shares * current_price)
            result.iloc[i, result.columns.get_loc('equity')] = current_equity
        
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
        
        # Calculate accumulation statistics
        buy_signals = df[df['signal'] == 'BUY']
        sell_signals = df[df['signal'] == 'SELL']
        
        # Calculate position sizes for each completed trade
        position_sizes = []
        max_position_sizes = []
        
        # For each sell signal, find the maximum position size during that position period
        for sell_idx in sell_signals.index:
            # Look backwards from sell signal to find when position started
            position_start_idx = None
            for i in range(sell_idx - 1, -1, -1):
                if i == 0 or df.iloc[i-1]['position_shares'] == 0:
                    position_start_idx = i
                    break
            
            if position_start_idx is not None:
                # Get the maximum position size during this position period
                position_period = df.iloc[position_start_idx:sell_idx+1]
                max_pos_size = position_period['position_shares'].max()
                if max_pos_size > 0:
                    max_position_sizes.append(max_pos_size)
                    position_sizes.append(max_pos_size)  # For backward compatibility
        
        # Also get the overall maximum position size from the entire dataset
        overall_max_position = df['position_shares'].max()
        
        avg_position_size = np.mean(position_sizes) if position_sizes else 0
        max_position_size = overall_max_position if overall_max_position > 0 else 0
        
        # Calculate average holding periods (in 1-minute bars)
        holding_periods = []
        for sell_idx in sell_signals.index:
            # Find when this position was first opened
            # Look backwards for the first BUY signal where position was 0 before
            for i in range(sell_idx - 1, -1, -1):
                if i == 0 or df.iloc[i-1]['position_shares'] == 0:
                    holding_period = sell_idx - i
                    holding_periods.append(holding_period)
                    break
        
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
            'accumulation_statistics': {
                'avg_position_size_shares': avg_position_size,
                'max_position_size_shares': max_position_size,
                'avg_holding_period_bars': avg_holding_period,
                'avg_holding_period_minutes': avg_holding_period,
                'total_buy_signals': len(buy_signals),
                'ema_period': self.ema_period,
                'profit_target_pct': self.profit_target * 100
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
            filename = f'{self.output_dir}/{symbol}_emaAccumulation_backtest_{timestamp}.csv'
            
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
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle(f'EMA Accumulation Strategy - {symbol}', fontsize=16, fontweight='bold')
            
            # Plot 1: Price chart with EMA and signals
            ax1 = axes[0, 0]
            ax1.plot(df.index, df['close'], label='Close Price', alpha=0.7, linewidth=1, color='blue')
            ax1.plot(df.index, df['ema_21'], label=f'{self.ema_period}-EMA', alpha=0.8, linewidth=1, color='orange')
            
            # Mark buy and sell signals
            buy_signals = df[df['signal'] == 'BUY']
            sell_signals = df[df['signal'] == 'SELL']
            
            if not buy_signals.empty:
                ax1.scatter(buy_signals.index, buy_signals['close'], 
                           color='green', marker='^', s=30, label='Buy Signal', alpha=0.6)
            
            if not sell_signals.empty:
                ax1.scatter(sell_signals.index, sell_signals['close'], 
                           color='red', marker='v', s=50, label='Sell Signal', alpha=0.8)
            
            ax1.set_title('Price Action with EMA and Trading Signals')
            ax1.set_ylabel('Price ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Position size over time
            ax2 = axes[0, 1]
            ax2.plot(df.index, df['position_shares'], label='Position Size (Shares)', color='purple', linewidth=1)
            ax2.fill_between(df.index, df['position_shares'], alpha=0.3, color='purple')
            ax2.set_title('Position Size Accumulation')
            ax2.set_ylabel('Shares Held')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Rolling average entry price vs current price
            ax3 = axes[0, 2]
            ax3.plot(df.index, df['close'], label='Current Price', alpha=0.7, linewidth=1, color='blue')
            ax3.plot(df.index, df['avg_entry_price'], label='Avg Entry Price', alpha=0.8, linewidth=1, color='red')
            ax3.set_title('Current Price vs Average Entry Price')
            ax3.set_ylabel('Price ($)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Rolling PnL since position opened
            ax4 = axes[1, 0]
            ax4.plot(df.index, df['rolling_pnl'], label='Rolling PnL', color='green', linewidth=1)
            ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax4.fill_between(df.index, df['rolling_pnl'], alpha=0.3, color='green')
            ax4.set_title('Rolling PnL Since Position Opened')
            ax4.set_ylabel('PnL ($)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Plot 5: Equity curve
            ax5 = axes[1, 1]
            ax5.plot(df.index, df['equity'], label='Equity Curve', color='blue', linewidth=2)
            ax5.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
            ax5.set_title('Equity Curve')
            ax5.set_ylabel('Equity ($)')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            # Plot 6: Performance statistics
            ax6 = axes[1, 2]
            ax6.axis('off')
            
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

ACCUMULATION STATS
Avg Position Size: {summary['accumulation_statistics']['avg_position_size_shares']:.1f} shares
Max Position Size: {summary['accumulation_statistics']['max_position_size_shares']:.0f} shares
Avg Hold Time: {summary['accumulation_statistics']['avg_holding_period_minutes']:.1f} min
Total Buy Signals: {summary['accumulation_statistics']['total_buy_signals']}

EMA Period: {summary['accumulation_statistics']['ema_period']}
Profit Target: {summary['accumulation_statistics']['profit_target_pct']:.1f}%

Data Period: {summary['data_period']['start_date'].strftime('%Y-%m-%d')} to 
{summary['data_period']['end_date'].strftime('%Y-%m-%d')}
Total Bars: {summary['data_period']['total_bars']:,}
            """
            
            ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=9,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            
            # Save chart
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            chart_filename = f'{self.charts_dir}/{symbol}_emaAccumulation_analysis_{timestamp}.png'
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
        
        print(f"\n📊 EMA ACCUMULATION STRATEGY RESULTS: {symbol}")
        print("=" * 70)
        
        if summary['trade_statistics']['total_trades'] == 0:
            print("No trades executed during the backtest period.")
            return
        
        # Data period info
        print(f"📅 DATA PERIOD")
        print(f"   Start Date: {summary['data_period']['start_date'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   End Date: {summary['data_period']['end_date'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Total 1-min Bars: {summary['data_period']['total_bars']:,}")
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
        
        # Accumulation statistics
        accum_stats = summary['accumulation_statistics']
        print(f"\n📊 ACCUMULATION STATISTICS")
        print(f"   Average Position Size: {accum_stats['avg_position_size_shares']:.1f} shares")
        print(f"   Maximum Position Size: {accum_stats['max_position_size_shares']:.0f} shares")
        print(f"   Average Holding Period: {accum_stats['avg_holding_period_bars']:.1f} bars")
        print(f"   Average Holding Period: {accum_stats['avg_holding_period_minutes']:.1f} minutes")
        print(f"   Total Buy Signals: {accum_stats['total_buy_signals']}")
        print(f"   EMA Period: {accum_stats['ema_period']}")
        print(f"   Profit Target: {accum_stats['profit_target_pct']:.1f}%")
        
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
        
        if accum_stats['avg_holding_period_minutes'] < 30:
            print(f"   ⚡ Very short-term strategy (avg {accum_stats['avg_holding_period_minutes']:.1f} min holds)")
        elif accum_stats['avg_holding_period_minutes'] < 120:
            print(f"   🕐 Short-term strategy (avg {accum_stats['avg_holding_period_minutes']:.1f} min holds)")
        else:
            print(f"   🕑 Medium-term strategy (avg {accum_stats['avg_holding_period_minutes']:.1f} min holds)")
        
        print("=" * 70)
    
    def run_backtest(self, symbol, period_type="day", period=10):
        """
        Run complete backtest for the EMA Accumulation strategy
        
        Args:
            symbol (str): Stock symbol to backtest
            period_type (str): Period type for data fetching - For 1-min data, should be 'day'
            period (int): Number of periods - For 'day': 1, 2, 3, 4, 5, 10
            
        Returns:
            tuple: (DataFrame with results, dict with summary, str csv_path, str chart_path)
        """
        print(f"\n🚀 === EMA ACCUMULATION STRATEGY BACKTEST ===")
        print(f"Symbol: {symbol}")
        print(f"Strategy: Buy 1 share when price < {self.ema_period}-EMA, Exit at {self.profit_target*100:.1f}% profit")
        print(f"Data: 1-minute bars, {period} {period_type}")
        print(f"No stop loss, accumulation strategy")
        print("=" * 60)
        
        # Fetch data with proper API parameters for 1-minute data
        df = self.fetch_data(symbol, period_type=period_type, period=period, frequency_type="minute", frequency=1)
        
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
        
        # Generate automatic visualizations using the dedicated visualizer
        if csv_path:
            self._generate_automatic_visualizations(csv_path, symbol)
        
        return df, summary, csv_path, chart_path
    
    def _generate_automatic_visualizations(self, csv_filename: str, symbol: str):
        """
        Automatically generate visualizations after saving CSV data
        
        Args:
            csv_filename: Path to the saved CSV file
            symbol: Stock symbol
        """
        try:
            print(f"\n🎨 Generating automatic visualizations for {symbol}...")
            
            # Initialize the visualizer with the CSV file
            visualizer = EMAAccumulationVisualizer(csv_filename)
            
            # Generate comprehensive visualization
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            comprehensive_chart_path = f'charts/{symbol}_emaAccumulation_comprehensive_visualization.png'
            
            print(f"   📊 Creating comprehensive visualization...")
            visualizer.create_comprehensive_visualization(save_path=comprehensive_chart_path)
            
            # Generate simple visualization
            simple_chart_path = f'charts/{symbol}_emaAccumulation_simple_visualization.png'
            
            print(f"   📈 Creating simple visualization...")
            visualizer.create_simple_chart(save_path=simple_chart_path)
            
            print(f"✅ Visualizations completed successfully!")
            print(f"   📊 Comprehensive chart: {comprehensive_chart_path}")
            print(f"   📈 Simple chart: {simple_chart_path}")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not generate automatic visualizations: {str(e)}")
            print(f"   You can manually run: python3 visualizers/emaAccumulation_visualization.py {csv_filename}")


def main():
    """
    Main function for running the EMA Accumulation backtester
    """
    parser = argparse.ArgumentParser(
        description="EMA Accumulation Backtester - 1-minute accumulation strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run backtest on AAPL with default settings (10 days, 0.5% profit target)
  python3 ema_accumulation_backtest.py AAPL
  
  # Run backtest on NVDA with 5 days of data
  python3 ema_accumulation_backtest.py NVDA --period 5
  
  # Run backtest on TSLA with 1% profit target
  python3 ema_accumulation_backtest.py TSLA --profit-target 0.01
  
  # Run backtest with custom EMA period and capital
  python3 ema_accumulation_backtest.py AAPL --ema-period 50 --initial-capital 50000
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
        choices=['day'],
        default='day',
        help='Period type for data fetching (default: day) - For 1-min data, must be day'
    )
    
    parser.add_argument(
        '--period',
        type=int,
        default=10,
        help='Number of periods to fetch (default: 10 days)'
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
        help='Profit target as decimal (default: 0.005 = 0.5%)'
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
    print("EMA ACCUMULATION BACKTESTER")
    print("1-Minute Accumulation Strategy")
    print("=" * 80)
    print(f"Symbol: {symbol}")
    print(f"Period: {args.period} {args.period_type}")
    print(f"EMA Period: {args.ema_period}")
    print(f"Profit Target: {args.profit_target*100:.1f}%")
    print(f"Initial Capital: ${args.initial_capital:,.2f}")
    print("=" * 80)
    
    try:
        # Initialize backtester
        backtester = EMAAccumulationBacktester(
            initial_capital=args.initial_capital,
            ema_period=args.ema_period,
            profit_target=args.profit_target
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
            print(f"📊 Processed {len(results_df):,} 1-minute bars")
            print(f"💾 Results saved to: {csv_path}")
            print(f"🎨 Chart saved to: {chart_path}")
            
            # Show key metrics
            if summary['trade_statistics']['total_trades'] > 0:
                print(f"\n🎯 KEY RESULTS:")
                print(f"   Total Trades: {summary['trade_statistics']['total_trades']}")
                print(f"   Win Rate: {summary['trade_statistics']['win_rate']:.1f}%")
                print(f"   ROI: {summary['performance_metrics']['roi']:.2f}%")
                print(f"   Profit Factor: {summary['performance_metrics']['profit_factor']:.2f}")
                print(f"   Avg Position Size: {summary['accumulation_statistics']['avg_position_size_shares']:.1f} shares")
            
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

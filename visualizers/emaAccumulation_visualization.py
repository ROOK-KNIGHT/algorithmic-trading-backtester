#!/usr/bin/env python3
"""
Visualization script for EMA Accumulation Strategy Results
Creates comprehensive charts showing price action, EMA, accumulation, and performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import datetime
import argparse
import os

class EMAAccumulationVisualizer:
    def __init__(self, csv_file_path):
        """Initialize the visualizer with backtest results"""
        self.csv_file_path = csv_file_path
        self.df = None
        self.load_data()
        
    def load_data(self):
        """Load and prepare the backtest data"""
        try:
            self.df = pd.read_csv(self.csv_file_path)
            self.df['datetime'] = pd.to_datetime(self.df['datetime'])
            self.df = self.df.sort_values('datetime').reset_index(drop=True)
            
            # Extract symbol and timeframe from filename for titles
            filename = os.path.basename(self.csv_file_path)
            parts = filename.split('_')
            self.symbol = parts[0] if len(parts) > 0 else 'Unknown'
            
            print(f"Loaded {len(self.df)} records for {self.symbol}")
            print(f"Date range: {self.df['datetime'].min()} to {self.df['datetime'].max()}")
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def format_x_axis(self, ax, datetime_data):
        """Format x-axis with intelligent scaling based on data range"""
        if len(datetime_data) == 0:
            return
            
        # Calculate time span
        time_span = datetime_data.max() - datetime_data.min()
        total_days = time_span.total_seconds() / (24 * 3600)
        
        # Choose appropriate formatting based on time span
        if total_days <= 1:  # Less than 1 day - show hours and minutes
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, int(total_days * 24 / 8))))
        elif total_days <= 7:  # Less than 1 week - show day and hour
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(6, int(total_days * 24 / 10))))
        elif total_days <= 30:  # Less than 1 month - show day
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, int(total_days / 10))))
        elif total_days <= 365:  # Less than 1 year - show month/day
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=max(1, int(total_days / 60))))
        else:  # More than 1 year - show month/year
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, int(total_days / 365))))
        
        # Rotate labels for better readability
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Reduce number of ticks if too many
        max_ticks = 10
        current_ticks = len(ax.get_xticklabels())
        if current_ticks > max_ticks:
            # Thin out the ticks
            nth_tick = max(1, current_ticks // max_ticks)
            for i, label in enumerate(ax.get_xticklabels()):
                if i % nth_tick != 0:
                    label.set_visible(False)

    def calculate_performance_metrics(self):
        """Calculate key performance metrics"""
        # Filter for completed trades
        trades = self.df[self.df['trade_pnl'].notna() & (self.df['trade_pnl'] != 0)]
        
        if len(trades) == 0:
            return {}
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = len(trades[trades['trade_pnl'] > 0])
        losing_trades = len(trades[trades['trade_pnl'] < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # PnL metrics
        total_pnl = trades['trade_pnl'].sum()
        avg_pnl = trades['trade_pnl'].mean()
        max_win = trades['trade_pnl'].max()
        max_loss = trades['trade_pnl'].min()
        
        # Account metrics
        initial_capital = self.df['equity'].iloc[0] if 'equity' in self.df.columns else 25000.0
        final_capital = self.df['equity'].iloc[-1] if 'equity' in self.df.columns else initial_capital + total_pnl
        roi = (total_pnl / initial_capital) * 100
        
        # Drawdown calculation
        equity_curve = self.df['equity'].dropna()
        if len(equity_curve) > 0:
            running_max = equity_curve.expanding().max()
            drawdown = equity_curve - running_max
            max_drawdown = drawdown.min()
            max_drawdown_pct = (max_drawdown / initial_capital) * 100
        else:
            max_drawdown = 0
            max_drawdown_pct = 0
        
        # Profit factor
        gross_profit = trades[trades['trade_pnl'] > 0]['trade_pnl'].sum()
        gross_loss = abs(trades[trades['trade_pnl'] < 0]['trade_pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Accumulation metrics
        buy_signals = self.df[self.df['signal'] == 'BUY']
        max_position = self.df['position_shares'].max() if 'position_shares' in self.df.columns else 0
        avg_position = self.df[self.df['position_shares'] > 0]['position_shares'].mean() if 'position_shares' in self.df.columns else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'max_win': max_win,
            'max_loss': max_loss,
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'roi': roi,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'total_buy_signals': len(buy_signals),
            'max_position_size': max_position,
            'avg_position_size': avg_position
        }
    
    def create_price_chart(self, ax):
        """Create the main price chart with EMA and signals"""
        # Plot close price
        ax.plot(self.df['datetime'], self.df['close'], 
               label='Close Price', color='blue', linewidth=1, alpha=0.8)
        
        # Plot 21-period EMA if available
        if 'ema_21' in self.df.columns:
            valid_ema = self.df['ema_21'].notna()
            if valid_ema.any():
                ax.plot(self.df.loc[valid_ema, 'datetime'], 
                       self.df.loc[valid_ema, 'ema_21'], 
                       label='21-period EMA', 
                       color='orange', 
                       linewidth=2, 
                       alpha=0.8)
        
        # Plot trade signals
        buy_signals = self.df[self.df['signal'] == 'BUY']
        sell_signals = self.df[self.df['signal'] == 'SELL']
        
        if not buy_signals.empty:
            ax.scatter(buy_signals['datetime'], buy_signals['close'], 
                      color='lime', marker='^', s=30, label='Buy Signal', zorder=10, 
                      edgecolors='darkgreen', linewidth=0.5, alpha=0.7)
        
        if not sell_signals.empty:
            ax.scatter(sell_signals['datetime'], sell_signals['close'], 
                      color='red', marker='v', s=80, label='Sell Signal', zorder=10,
                      edgecolors='darkred', linewidth=1)
        
        # Plot average entry price when position is open
        if 'avg_entry_price' in self.df.columns:
            position_data = self.df[self.df['position_shares'] > 0]
            if not position_data.empty:
                ax.plot(position_data['datetime'], position_data['avg_entry_price'], 
                       label='Avg Entry Price', color='red', linewidth=1.5, 
                       alpha=0.8, linestyle='--')
        
        ax.set_title(f'{self.symbol} - EMA Accumulation Strategy (1-min bars)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis with better scaling
        self.format_x_axis(ax, self.df['datetime'])
    
    def create_position_chart(self, ax):
        """Create chart showing position size accumulation over time"""
        if 'position_shares' not in self.df.columns:
            ax.text(0.5, 0.5, 'Position data not available', 
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title('Position Size (Not Available)', fontsize=12, fontweight='bold')
            ax.axis('off')
            return
        
        # Plot position size
        ax.plot(self.df['datetime'], self.df['position_shares'], 
               color='purple', linewidth=1.5, alpha=0.8, label='Position Size (Shares)')
        
        # Fill area under the curve
        ax.fill_between(self.df['datetime'], self.df['position_shares'], 
                       alpha=0.3, color='purple')
        
        # Highlight buy signals on position chart
        buy_signals = self.df[self.df['signal'] == 'BUY']
        if not buy_signals.empty:
            ax.scatter(buy_signals['datetime'], buy_signals['position_shares'], 
                      color='lime', marker='^', s=40, alpha=0.8, zorder=5, 
                      label='Buy Signal', edgecolors='darkgreen')
        
        # Highlight sell signals
        sell_signals = self.df[self.df['signal'] == 'SELL']
        if not sell_signals.empty:
            ax.scatter(sell_signals['datetime'], sell_signals['position_shares'], 
                      color='red', marker='v', s=60, alpha=0.8, zorder=5, 
                      label='Sell Signal', edgecolors='darkred')
        
        ax.set_title('Position Size Accumulation', fontsize=12, fontweight='bold')
        ax.set_ylabel('Shares Held', fontsize=10)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis with better scaling
        self.format_x_axis(ax, self.df['datetime'])
    
    def create_rolling_pnl_chart(self, ax):
        """Create chart showing rolling PnL since position opened"""
        if 'rolling_pnl' not in self.df.columns:
            ax.text(0.5, 0.5, 'Rolling PnL data not available', 
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title('Rolling PnL (Not Available)', fontsize=12, fontweight='bold')
            ax.axis('off')
            return
        
        # Plot rolling PnL
        ax.plot(self.df['datetime'], self.df['rolling_pnl'], 
               color='green', linewidth=1.5, alpha=0.8, label='Rolling PnL')
        
        # Fill areas above and below zero
        ax.fill_between(self.df['datetime'], self.df['rolling_pnl'], 0, 
                       where=(self.df['rolling_pnl'] >= 0), color='green', alpha=0.2, label='Profit')
        ax.fill_between(self.df['datetime'], self.df['rolling_pnl'], 0, 
                       where=(self.df['rolling_pnl'] < 0), color='red', alpha=0.2, label='Loss')
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Highlight sell signals (realized P&L)
        sell_signals = self.df[self.df['signal'] == 'SELL']
        if not sell_signals.empty:
            ax.scatter(sell_signals['datetime'], sell_signals['rolling_pnl'], 
                      color='red', marker='v', s=80, alpha=0.8, zorder=5, 
                      label='Position Closed', edgecolors='darkred')
        
        ax.set_title('Rolling PnL Since Position Opened', fontsize=12, fontweight='bold')
        ax.set_ylabel('PnL ($)', fontsize=10)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis to show currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}'))
        
        # Format x-axis with better scaling
        self.format_x_axis(ax, self.df['datetime'])
    
    def create_price_vs_ema_chart(self, ax):
        """Create chart showing price relative to EMA"""
        if 'ema_21' not in self.df.columns:
            ax.text(0.5, 0.5, 'EMA data not available', 
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title('Price vs EMA Analysis (Not Available)', fontsize=12, fontweight='bold')
            ax.axis('off')
            return
        
        # Calculate price relative to EMA
        valid_data = (self.df['ema_21'].notna()) & (self.df['close'].notna())
        
        if valid_data.any():
            price_ema_diff = ((self.df['close'] - self.df['ema_21']) / self.df['ema_21'] * 100)
            datetime_data = self.df.loc[valid_data, 'datetime']
            diff_data = price_ema_diff.loc[valid_data]
            
            # Plot price relative to EMA
            ax.plot(datetime_data, diff_data, color='orange', linewidth=1.5, alpha=0.8, label='Price vs EMA (%)')
            
            # Fill areas above and below zero
            ax.fill_between(datetime_data, diff_data, 0, 
                           where=(diff_data >= 0), color='green', alpha=0.2, label='Above EMA')
            ax.fill_between(datetime_data, diff_data, 0, 
                           where=(diff_data < 0), color='red', alpha=0.2, label='Below EMA (Buy Zone)')
            
            # Add zero line
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Highlight buy signals
            buy_signals = self.df[self.df['signal'] == 'BUY']
            if not buy_signals.empty:
                buy_diff = price_ema_diff.loc[buy_signals.index]
                ax.scatter(buy_signals['datetime'], buy_diff, 
                          color='lime', marker='^', s=40, alpha=0.8, zorder=5, 
                          label='Buy Signal', edgecolors='darkgreen')
        
        ax.set_title('Price Relative to 21-period EMA', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price vs EMA (%)', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis with better scaling
        self.format_x_axis(ax, datetime_data)
    
    def create_equity_curve(self, ax):
        """Create equity curve chart"""
        equity_data = self.df['equity'].dropna()
        
        if len(equity_data) > 0:
            datetime_data = self.df.loc[equity_data.index, 'datetime']
            
            # Plot equity curve
            ax.plot(datetime_data, equity_data, color='darkblue', linewidth=2, label='Portfolio Value')
            
            # Add starting capital line
            initial_capital = equity_data.iloc[0]
            ax.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.7, label='Starting Capital')
            
            # Highlight drawdown periods
            running_max = equity_data.expanding().max()
            drawdown = equity_data - running_max
            
            # Fill drawdown areas
            ax.fill_between(datetime_data, equity_data, running_max, 
                           where=(drawdown < 0), color='red', alpha=0.2, label='Drawdown')
            
            # Mark trade points
            trade_points = self.df[self.df['trade_pnl'].notna() & (self.df['trade_pnl'] != 0)]
            if not trade_points.empty:
                profitable_trades = trade_points[trade_points['trade_pnl'] > 0]
                losing_trades = trade_points[trade_points['trade_pnl'] < 0]
                
                if not profitable_trades.empty:
                    ax.scatter(profitable_trades['datetime'], profitable_trades['equity'], 
                              color='green', marker='o', s=40, alpha=0.8, zorder=5, label='Profitable Trade')
                
                if not losing_trades.empty:
                    ax.scatter(losing_trades['datetime'], losing_trades['equity'], 
                              color='red', marker='o', s=40, alpha=0.8, zorder=5, label='Losing Trade')
        
        ax.set_title('Portfolio Equity Curve', fontsize=12, fontweight='bold')
        ax.set_ylabel('Portfolio Value ($)', fontsize=10)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis to show currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Format x-axis with better scaling
        self.format_x_axis(ax, datetime_data)
    
    def create_performance_summary(self, ax):
        """Create performance metrics summary"""
        metrics = self.calculate_performance_metrics()
        
        if not metrics:
            ax.text(0.5, 0.5, 'No completed trades found', 
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title('Performance Summary', fontsize=12, fontweight='bold')
            ax.axis('off')
            return
        
        # Create text summary
        summary_text = f"""
PERFORMANCE SUMMARY

Total Trades: {metrics['total_trades']}
Winning Trades: {metrics['winning_trades']}
Losing Trades: {metrics['losing_trades']}
Win Rate: {metrics['win_rate']:.1f}%

Total P&L: ${metrics['total_pnl']:.2f}
Average P&L: ${metrics['avg_pnl']:.2f}
Largest Win: ${metrics['max_win']:.2f}
Largest Loss: ${metrics['max_loss']:.2f}

Initial Capital: ${metrics['initial_capital']:,.0f}
Final Capital: ${metrics['final_capital']:,.0f}
ROI: {metrics['roi']:.2f}%

Max Drawdown: ${metrics['max_drawdown']:.2f}
Max Drawdown %: {metrics['max_drawdown_pct']:.2f}%
Profit Factor: {metrics['profit_factor']:.2f}

ACCUMULATION STATS
Total Buy Signals: {metrics['total_buy_signals']}
Max Position Size: {metrics['max_position_size']:.0f} shares
Avg Position Size: {metrics['avg_position_size']:.1f} shares

Strategy: EMA Accumulation
Entry: Price < 21-EMA (1 share/min)
Exit: 0.5% profit target
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        ax.set_title('Performance Summary', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    def create_trade_distribution(self, ax):
        """Create trade P&L distribution chart"""
        trades = self.df[self.df['trade_pnl'].notna() & (self.df['trade_pnl'] != 0)]
        
        if len(trades) == 0:
            ax.text(0.5, 0.5, 'No completed trades found', 
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title('Trade P&L Distribution', fontsize=12, fontweight='bold')
            ax.axis('off')
            return
        
        # Create histogram of P&L
        pnl_data = trades['trade_pnl']
        
        # Separate winning and losing trades
        winning_pnl = pnl_data[pnl_data > 0]
        losing_pnl = pnl_data[pnl_data < 0]
        
        bins = min(20, len(pnl_data) // 2) if len(pnl_data) > 10 else 10
        
        if len(winning_pnl) > 0:
            ax.hist(winning_pnl, bins=bins, alpha=0.7, color='green', label=f'Wins ({len(winning_pnl)})', edgecolor='black')
        
        if len(losing_pnl) > 0:
            ax.hist(losing_pnl, bins=bins, alpha=0.7, color='red', label=f'Losses ({len(losing_pnl)})', edgecolor='black')
        
        # Add vertical line at zero
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        
        # Add mean line
        mean_pnl = pnl_data.mean()
        ax.axvline(x=mean_pnl, color='blue', linestyle='-', alpha=0.7, label=f'Mean: ${mean_pnl:.2f}')
        
        ax.set_title('Trade P&L Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('P&L ($)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}'))
    
    def create_accumulation_analysis(self, ax):
        """Create analysis of accumulation patterns"""
        if 'position_shares' not in self.df.columns:
            ax.text(0.5, 0.5, 'Position data not available', 
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title('Accumulation Analysis', fontsize=12, fontweight='bold')
            ax.axis('off')
            return
        
        # Get position size distribution
        position_sizes = self.df[self.df['position_shares'] > 0]['position_shares']
        
        if len(position_sizes) == 0:
            ax.text(0.5, 0.5, 'No position data found', 
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title('Accumulation Analysis', fontsize=12, fontweight='bold')
            ax.axis('off')
            return
        
        # Create histogram of position sizes
        bins = min(50, len(position_sizes.unique()))
        ax.hist(position_sizes, bins=bins, alpha=0.7, color='purple', edgecolor='black')
        
        # Add statistics
        mean_pos = position_sizes.mean()
        max_pos = position_sizes.max()
        ax.axvline(x=mean_pos, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_pos:.1f} shares')
        ax.axvline(x=max_pos, color='orange', linestyle='--', alpha=0.8, label=f'Max: {max_pos:.0f} shares')
        
        ax.set_title('Position Size Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Position Size (Shares)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    def create_comprehensive_visualization(self, save_path=None):
        """Create comprehensive visualization with all charts"""
        # Set up the figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Create a grid layout
        gs = fig.add_gridspec(4, 3, height_ratios=[2.5, 1.5, 1.5, 1.5], width_ratios=[2, 1, 1], 
                             hspace=0.4, wspace=0.3)
        
        # Main price chart (spans top row)
        ax1 = fig.add_subplot(gs[0, :])
        self.create_price_chart(ax1)
        
        # Position size chart (left, second row)
        ax2 = fig.add_subplot(gs[1, 0])
        self.create_position_chart(ax2)
        
        # Rolling PnL chart (middle, second row)
        ax3 = fig.add_subplot(gs[1, 1])
        self.create_rolling_pnl_chart(ax3)
        
        # Price vs EMA chart (right, second row)
        ax4 = fig.add_subplot(gs[1, 2])
        self.create_price_vs_ema_chart(ax4)
        
        # Equity curve (spans left two columns, third row)
        ax5 = fig.add_subplot(gs[2, :2])
        self.create_equity_curve(ax5)
        
        # Performance summary (right, third row)
        ax6 = fig.add_subplot(gs[2, 2])
        self.create_performance_summary(ax6)
        
        # Trade distribution (left, bottom row)
        ax7 = fig.add_subplot(gs[3, 0])
        self.create_trade_distribution(ax7)
        
        # Accumulation analysis (right two columns, bottom row)
        ax8 = fig.add_subplot(gs[3, 1:])
        self.create_accumulation_analysis(ax8)
        
        # Add overall title
        fig.suptitle(f'EMA Accumulation Strategy Analysis - {self.symbol}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comprehensive visualization saved to: {save_path}")
        else:
            plt.show()
        
        return fig
    
    def create_simple_chart(self, save_path=None):
        """Create a simpler chart focusing on key metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), height_ratios=[2, 1])
        
        # Price chart with EMA and signals
        self.create_price_chart(axes[0, 0])
        
        # Position accumulation
        self.create_position_chart(axes[0, 1])
        
        # Rolling PnL
        self.create_rolling_pnl_chart(axes[1, 0])
        
        # Equity curve
        self.create_equity_curve(axes[1, 1])
        
        # Add overall title
        fig.suptitle(f'EMA Accumulation Strategy - {self.symbol}', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Simple chart saved to: {save_path}")
        else:
            plt.show()
        
        return fig

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Visualize EMA Accumulation backtest results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create comprehensive visualization
  python3 emaAccumulation_visualization.py historical_data/AAPL_emaAccumulation_backtest_20250806_124316.csv
  
  # Create simple chart
  python3 emaAccumulation_visualization.py historical_data/AAPL_emaAccumulation_backtest_20250806_124316.csv --simple
  
  # Save to file
  python3 emaAccumulation_visualization.py historical_data/AAPL_emaAccumulation_backtest_20250806_124316.csv --save charts/aapl_emaAccumulation_analysis.png
        """
    )
    
    parser.add_argument(
        'csv_file',
        type=str,
        help='Path to the backtest results CSV file'
    )
    
    parser.add_argument(
        '--simple',
        action='store_true',
        help='Create a simple chart with key metrics only'
    )
    
    parser.add_argument(
        '--save',
        type=str,
        help='Save chart to specified file path (e.g., charts/analysis.png)'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for saved images (default: 300)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.csv_file):
        print(f"Error: File '{args.csv_file}' not found.")
        return
    
    print("=" * 80)
    print("EMA ACCUMULATION STRATEGY VISUALIZATION")
    print("=" * 80)
    print(f"Loading data from: {args.csv_file}")
    
    try:
        # Create visualizer
        visualizer = EMAAccumulationVisualizer(args.csv_file)
        
        # Create output directory if saving
        if args.save:
            output_dir = os.path.dirname(args.save)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
        
        # Create visualization
        if args.simple:
            print("Creating simple visualization...")
            fig = visualizer.create_simple_chart(save_path=args.save)
        else:
            print("Creating comprehensive visualization...")
            fig = visualizer.create_comprehensive_visualization(save_path=args.save)
        
        if not args.save:
            print("Displaying interactive chart...")
            plt.show()
        
        print("✅ Visualization completed successfully!")
        
    except Exception as e:
        print(f"❌ Error creating visualization: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()

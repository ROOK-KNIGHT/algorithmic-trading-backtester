#!/usr/bin/env python3
"""
Visualization script for Exceedence Backtest Results
Creates comprehensive charts showing price action, volatility bands, momentum signals, and performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import datetime
import argparse
import os
from matplotlib.patches import Rectangle

class ExceedenceVisualizer:
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
            self.timeframe = f"{parts[3]}_{parts[4]}" if len(parts) > 4 else 'Unknown'
            
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
        trades = self.df[self.df['pnl'].notna() & (self.df['pnl'] != 0)]
        
        if len(trades) == 0:
            return {}
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = len(trades[trades['pnl'] > 0])
        losing_trades = len(trades[trades['pnl'] < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # PnL metrics
        total_pnl = trades['pnl'].sum()
        avg_pnl = trades['pnl'].mean()
        max_win = trades['pnl'].max()
        max_loss = trades['pnl'].min()
        
        # Account metrics
        initial_capital = 25000.0
        final_capital = initial_capital + total_pnl
        roi = (total_pnl / initial_capital) * 100
        
        # Drawdown calculation
        equity_curve = self.df['current_equity'].dropna()
        if len(equity_curve) > 0:
            running_max = equity_curve.expanding().max()
            drawdown = equity_curve - running_max
            max_drawdown = drawdown.min()
            max_drawdown_pct = (max_drawdown / initial_capital) * 100
        else:
            max_drawdown = 0
            max_drawdown_pct = 0
        
        # Profit factor
        gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
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
            'gross_loss': gross_loss
        }
    
    def create_candlestick_chart(self, ax):
        """Create candlestick chart with OHLC data"""
        # Calculate candle width based on time interval
        time_diff = (self.df['datetime'].iloc[1] - self.df['datetime'].iloc[0]).total_seconds() / 86400  # in days
        candle_width = time_diff * 0.8  # 80% of the time interval
        
        # Create candlesticks
        for i, row in self.df.iterrows():
            date = mdates.date2num(row['datetime'])
            open_price = row['open']
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']
            
            # Determine candle color
            if close_price >= open_price:
                # Green/bullish candle
                color = 'green'
                body_bottom = open_price
                body_top = close_price
            else:
                # Red/bearish candle
                color = 'red'
                body_bottom = close_price
                body_top = open_price
            
            # Draw the wick (high-low line)
            ax.plot([date, date], [low_price, high_price], color='black', linewidth=0.8, alpha=0.8)
            
            # Draw the body (rectangle)
            body_height = abs(close_price - open_price)
            if body_height > 0:
                rect = Rectangle((date - candle_width/2, body_bottom), candle_width, body_height,
                               facecolor=color, edgecolor='black', alpha=0.8, linewidth=0.5)
                ax.add_patch(rect)
            else:
                # Doji candle (open == close)
                ax.plot([date - candle_width/2, date + candle_width/2], [close_price, close_price], 
                       color='black', linewidth=1)
        
        return ax
    
    def create_price_chart(self, ax):
        """Create the main price chart with candlesticks and volatility bands"""
        # Create candlestick chart
        self.create_candlestick_chart(ax)
        
        # Plot volatility bands
        if 'high_band' in self.df.columns and 'low_band' in self.df.columns:
            valid_bands = self.df['high_band'].notna() & self.df['low_band'].notna()
            if valid_bands.any():
                ax.plot(self.df.loc[valid_bands, 'datetime'], self.df.loc[valid_bands, 'high_band'], 
                       color='red', linewidth=2, alpha=0.8, label='High Volatility Band')
                ax.plot(self.df.loc[valid_bands, 'datetime'], self.df.loc[valid_bands, 'low_band'], 
                       color='blue', linewidth=2, alpha=0.8, label='Low Volatility Band')
                
                # Fill the band area
                ax.fill_between(self.df.loc[valid_bands, 'datetime'], 
                               self.df.loc[valid_bands, 'high_band'], 
                               self.df.loc[valid_bands, 'low_band'], 
                               alpha=0.1, color='gray', label='Volatility Range')
        
        # Plot momentum signals
        long_signals = self.df[self.df['momentum_signal'] == 'LONG']
        short_signals = self.df[self.df['momentum_signal'] == 'SHORT']
        
        if not long_signals.empty:
            ax.scatter(long_signals['datetime'], long_signals['close'], 
                      color='lime', marker='^', s=200, label='Long Signal', zorder=10, 
                      edgecolors='darkgreen', linewidth=2)
        
        if not short_signals.empty:
            ax.scatter(short_signals['datetime'], short_signals['close'], 
                      color='red', marker='v', s=200, label='Short Signal', zorder=10,
                      edgecolors='darkred', linewidth=2)
        
        # Plot trade exits (closed positions)
        closed_trades = self.df[self.df['trade_status'] == 'closed']
        if not closed_trades.empty:
            # Color exits based on profit/loss
            profitable_exits = closed_trades[closed_trades['pnl'] > 0]
            losing_exits = closed_trades[closed_trades['pnl'] < 0]
            
            if not profitable_exits.empty:
                ax.scatter(profitable_exits['datetime'], profitable_exits['close'], 
                          color='lime', marker='X', s=150, label='Profitable Exit', zorder=10,
                          edgecolors='darkgreen', linewidth=1)
            
            if not losing_exits.empty:
                ax.scatter(losing_exits['datetime'], losing_exits['close'], 
                          color='red', marker='X', s=150, label='Loss Exit', zorder=10,
                          edgecolors='darkred', linewidth=1)
        
        ax.set_title(f'{self.symbol} - Exceedence Strategy ({self.timeframe})', fontsize=14, fontweight='bold')
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.legend(loc='upper left', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis with better scaling
        self.format_x_axis(ax, self.df['datetime'])
    
    def create_position_chart(self, ax):
        """Create chart showing position in volatility range"""
        if 'position_in_range' not in self.df.columns:
            ax.text(0.5, 0.5, 'Position data not available', 
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title('Position in Range (Not Available)', fontsize=12, fontweight='bold')
            ax.axis('off')
            return
        
        valid_position = self.df['position_in_range'].notna()
        
        if valid_position.any():
            position_data = self.df.loc[valid_position, 'position_in_range']
            datetime_data = self.df.loc[valid_position, 'datetime']
            
            # Plot position line
            ax.plot(datetime_data, position_data, color='purple', linewidth=1.5, alpha=0.8)
            
            # Add reference lines
            ax.axhline(y=99, color='red', linestyle='--', alpha=0.7, label='Long Signal (99%)')
            ax.axhline(y=1, color='blue', linestyle='--', alpha=0.7, label='Short Signal (1%)')
            ax.axhline(y=50, color='gray', linestyle='-', alpha=0.5, label='Midpoint (50%)')
            
            # Fill extreme areas
            ax.fill_between(datetime_data, 99, 100, alpha=0.2, color='red', label='Long Zone')
            ax.fill_between(datetime_data, 0, 1, alpha=0.2, color='blue', label='Short Zone')
            
            # Highlight momentum signals
            long_signals = self.df[self.df['momentum_signal'] == 'LONG']
            short_signals = self.df[self.df['momentum_signal'] == 'SHORT']
            
            if not long_signals.empty:
                long_positions = long_signals['position_in_range'].dropna()
                if not long_positions.empty:
                    ax.scatter(long_signals.loc[long_positions.index, 'datetime'], long_positions, 
                              color='lime', marker='^', s=100, zorder=8, 
                              edgecolors='darkgreen', linewidth=1)
            
            if not short_signals.empty:
                short_positions = short_signals['position_in_range'].dropna()
                if not short_positions.empty:
                    ax.scatter(short_signals.loc[short_positions.index, 'datetime'], short_positions, 
                              color='red', marker='v', s=100, zorder=8,
                              edgecolors='darkred', linewidth=1)
        
        ax.set_title('Position in Volatility Range', fontsize=12, fontweight='bold')
        ax.set_ylabel('Position (%)', fontsize=10)
        ax.set_ylim(0, 100)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis with better scaling
        self.format_x_axis(ax, datetime_data)
    
    def create_exceedance_chart(self, ax):
        """Create chart showing high and low exceedances"""
        if 'high_exceedance' not in self.df.columns or 'low_exceedance' not in self.df.columns:
            ax.text(0.5, 0.5, 'Exceedance data not available', 
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title('Exceedances (Not Available)', fontsize=12, fontweight='bold')
            ax.axis('off')
            return
        
        valid_exceedance = self.df['high_exceedance'].notna() & self.df['low_exceedance'].notna()
        
        if valid_exceedance.any():
            high_exc_data = self.df.loc[valid_exceedance, 'high_exceedance']
            low_exc_data = self.df.loc[valid_exceedance, 'low_exceedance']
            datetime_data = self.df.loc[valid_exceedance, 'datetime']
            
            # Plot exceedances as bar chart
            width = 0.4
            x_pos = np.arange(len(datetime_data))
            
            # Only plot every nth point to avoid overcrowding
            step = max(1, len(datetime_data) // 50)  # Show max 50 bars
            indices = range(0, len(datetime_data), step)
            
            ax.bar([x_pos[i] - width/2 for i in indices], 
                   [high_exc_data.iloc[i] for i in indices], 
                   width, label='High Exceedance', color='red', alpha=0.7)
            ax.bar([x_pos[i] + width/2 for i in indices], 
                   [-low_exc_data.iloc[i] for i in indices],  # Negative for visual separation
                   width, label='Low Exceedance', color='blue', alpha=0.7)
            
            # Set x-axis labels
            ax.set_xticks([x_pos[i] for i in indices[::max(1, len(indices)//10)]])
            ax.set_xticklabels([datetime_data.iloc[i].strftime('%m/%d') for i in indices[::max(1, len(indices)//10)]], 
                              rotation=45)
        
        ax.set_title('Volatility Band Exceedances', fontsize=12, fontweight='bold')
        ax.set_ylabel('Exceedance Amount ($)', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    def create_equity_curve(self, ax):
        """Create equity curve chart"""
        equity_data = self.df['current_equity'].dropna()
        
        if len(equity_data) > 0:
            datetime_data = self.df.loc[equity_data.index, 'datetime']
            
            # Plot equity curve
            ax.plot(datetime_data, equity_data, color='darkblue', linewidth=2, label='Portfolio Value')
            
            # Add starting capital line
            ax.axhline(y=25000, color='gray', linestyle='--', alpha=0.7, label='Starting Capital')
            
            # Highlight drawdown periods
            running_max = equity_data.expanding().max()
            drawdown = equity_data - running_max
            
            # Fill drawdown areas
            ax.fill_between(datetime_data, equity_data, running_max, 
                           where=(drawdown < 0), color='red', alpha=0.2, label='Drawdown')
            
            # Mark trade points
            trade_points = self.df[self.df['pnl'].notna() & (self.df['pnl'] != 0)]
            if not trade_points.empty:
                profitable_trades = trade_points[trade_points['pnl'] > 0]
                losing_trades = trade_points[trade_points['pnl'] < 0]
                
                if not profitable_trades.empty:
                    ax.scatter(profitable_trades['datetime'], profitable_trades['current_equity'], 
                              color='green', marker='o', s=30, alpha=0.7, zorder=5)
                
                if not losing_trades.empty:
                    ax.scatter(losing_trades['datetime'], losing_trades['current_equity'], 
                              color='red', marker='o', s=30, alpha=0.7, zorder=5)
        
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

Gross Profit: ${metrics['gross_profit']:.2f}
Gross Loss: ${metrics['gross_loss']:.2f}
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        ax.set_title('Performance Summary', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    def create_trade_distribution(self, ax):
        """Create trade P&L distribution chart"""
        trades = self.df[self.df['pnl'].notna() & (self.df['pnl'] != 0)]
        
        if len(trades) == 0:
            ax.text(0.5, 0.5, 'No completed trades found', 
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title('Trade P&L Distribution', fontsize=12, fontweight='bold')
            ax.axis('off')
            return
        
        # Create histogram of P&L
        pnl_data = trades['pnl']
        
        # Separate winning and losing trades
        winning_pnl = pnl_data[pnl_data > 0]
        losing_pnl = pnl_data[pnl_data < 0]
        
        bins = 20
        
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
    
    def create_signal_analysis(self, ax):
        """Create momentum signal analysis chart"""
        if 'momentum_signal' not in self.df.columns:
            ax.text(0.5, 0.5, 'Signal data not available', 
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title('Signal Analysis', fontsize=12, fontweight='bold')
            ax.axis('off')
            return
        
        # Get completed trades with momentum signals
        trades = self.df[self.df['pnl'].notna() & (self.df['pnl'] != 0) & self.df['momentum_signal'].notna()]
        
        if len(trades) == 0:
            ax.text(0.5, 0.5, 'No completed trades with signals found', 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title('Signal Analysis', fontsize=12, fontweight='bold')
            ax.axis('off')
            return
        
        # Group by signal type
        signal_analysis = trades.groupby('momentum_signal').agg({
            'pnl': ['count', 'sum', 'mean']
        }).round(2)
        
        signal_types = signal_analysis.index.tolist()
        trade_counts = signal_analysis[('pnl', 'count')].tolist()
        total_pnls = signal_analysis[('pnl', 'sum')].tolist()
        avg_pnls = signal_analysis[('pnl', 'mean')].tolist()
        
        # Create bar chart
        x_pos = np.arange(len(signal_types))
        
        # Color bars based on profitability
        colors = ['green' if pnl > 0 else 'red' for pnl in total_pnls]
        
        bars = ax.bar(x_pos, total_pnls, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for i, (bar, count, total_pnl, avg_pnl) in enumerate(zip(bars, trade_counts, total_pnls, avg_pnls)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (max(total_pnls) * 0.01),
                   f'${total_pnl:.0f}\n({count} trades)',
                   ha='center', va='bottom', fontsize=8)
        
        # Customize chart
        ax.set_xlabel('Signal Type', fontsize=10)
        ax.set_ylabel('Total P&L ($)', fontsize=10)
        ax.set_title('Performance by Signal Type', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(signal_types)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Format y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}'))
    
    def create_comprehensive_visualization(self, save_path=None):
        """Create comprehensive visualization with all charts"""
        # Set up the figure with subplots
        fig = plt.figure(figsize=(20, 18))
        
        # Create a grid layout
        gs = fig.add_gridspec(5, 3, height_ratios=[3, 1.5, 1.5, 2, 1.5], width_ratios=[2, 1, 1], 
                             hspace=0.4, wspace=0.3)
        
        # Main price chart (spans top row)
        ax1 = fig.add_subplot(gs[0, :])
        self.create_price_chart(ax1)
        
        # Position in range chart (spans second row)
        ax2 = fig.add_subplot(gs[1, :])
        self.create_position_chart(ax2)
        
        # Exceedance chart (spans third row)
        ax3 = fig.add_subplot(gs[2, :])
        self.create_exceedance_chart(ax3)
        
        # Equity curve (bottom left, spans 2 columns)
        ax4 = fig.add_subplot(gs[3, :2])
        self.create_equity_curve(ax4)
        
        # Performance summary (bottom right)
        ax5 = fig.add_subplot(gs[3, 2])
        self.create_performance_summary(ax5)
        
        # Trade distribution (bottom row, left)
        ax6 = fig.add_subplot(gs[4, :2])
        self.create_trade_distribution(ax6)
        
        # Signal analysis (bottom row, right)
        ax7 = fig.add_subplot(gs[4, 2])
        self.create_signal_analysis(ax7)
        
        # Add overall title
        fig.suptitle(f'Exceedence Strategy Analysis - {self.symbol}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        else:
            plt.show()
        
        return fig
    
    def create_simple_chart(self, save_path=None):
        """Create a simpler chart focusing on price and signals"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), height_ratios=[3, 1, 1])
        
        # Price chart
        self.create_price_chart(ax1)
        
        # Position in range chart
        self.create_position_chart(ax2)
        
        # Equity curve
        self.create_equity_curve(ax3)
        
        # Add overall title
        fig.suptitle(f'Exceedence Strategy - {self.symbol}', 
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
        description="Visualize Exceedence backtest results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create comprehensive visualization
  python3 exceedence_visualization.py historical_data/AAPL_volatility_data_1year_daily_20250729_123631.csv
  
  # Create simple chart
  python3 exceedence_visualization.py historical_data/AAPL_volatility_data_1year_daily_20250729_123631.csv --simple
  
  # Save to file
  python3 exceedence_visualization.py historical_data/AAPL_volatility_data_1year_daily_20250729_123631.csv --save charts/aapl_exceedence_analysis.png
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
        help='Create a simple chart with just price, position, and equity curve'
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
    print("EXCEEDENCE STRATEGY VISUALIZATION")
    print("=" * 80)
    print(f"Loading data from: {args.csv_file}")
    
    try:
        # Create visualizer
        visualizer = ExceedenceVisualizer(args.csv_file)
        
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

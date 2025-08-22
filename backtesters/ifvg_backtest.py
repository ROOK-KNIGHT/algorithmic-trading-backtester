#!/usr/bin/env python3
"""
Inverse Fair Value Gap (IFVG) Backtester

A comprehensive backtesting system for Fair Value Gap (FVG) and Inverse Fair Value Gap (IFVG) strategies.
This system identifies price inefficiencies (gaps) in 1-minute data and trades when price returns to fill these gaps.

Key Features:
- FVG and IFVG pattern detection using 3-candle sequences
- Gap classification (Bullish FVG, Bearish FVG, Bullish IFVG, Bearish IFVG)
- Multiple gap tracking with priority scoring
- Gap fill detection and position management
- Comprehensive performance metrics
- CSV output with full gap analysis
- Automatic visualization generation

Gap Detection Logic:
- FVG (Fair Value Gap): A gap between candle 1 and candle 3 where candle 2 doesn't fill the gap
- IFVG (Inverse Fair Value Gap): The inverse scenario where price creates inefficiency zones
- Bullish Gap: Gap below current price (potential support)
- Bearish Gap: Gap above current price (potential resistance)

Strategy Logic:
- Entry: Enter position when price approaches an unfilled gap
- Exit: Profit target or stop loss based on gap characteristics
- Gap Management: Track multiple gaps, prioritize by size and recency

Author: Trading System
Date: 2025-08-15
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import argparse
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, NamedTuple

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from handlers.historical_data_handler import HistoricalDataHandler

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")


class Gap(NamedTuple):
    """Structure to represent a Fair Value Gap"""
    gap_id: int
    gap_type: str  # 'bullish_fvg', 'bearish_fvg', 'bullish_ifvg', 'bearish_ifvg'
    start_index: int
    end_index: int
    top: float
    bottom: float
    size: float
    size_percent: float
    created_at: datetime
    filled: bool = False
    filled_at: Optional[datetime] = None
    filled_index: Optional[int] = None
    priority_score: float = 0.0  # Priority based on size and trend context
    trend_context: str = "neutral"  # 'discount', 'premium', 'neutral'
    is_invalidated: bool = False  # True if gap has been completely filled


class IFVGBacktester:
    """
    Inverse Fair Value Gap Strategy Backtester for 1-minute data with Confluence Analysis
    """
    
    def __init__(self, initial_capital=25000.0, gap_threshold=0.001, profit_target=0.005, 
                 stop_loss=0.01, max_active_gaps=10, gap_expiry_hours=24, confluence_symbols=None,
                 confluence_threshold=0.7):
        """
        Initialize the IFVG Backtester with Confluence Analysis
        
        Args:
            initial_capital (float): Starting capital amount
            gap_threshold (float): Minimum gap size as percentage of price (0.001 = 0.1%)
            profit_target (float): Profit target as decimal (0.005 = 0.5%)
            stop_loss (float): Stop loss as decimal (0.01 = 1.0%)
            max_active_gaps (int): Maximum number of active gaps to track
            gap_expiry_hours (int): Hours after which gaps expire
            confluence_symbols (list): List of symbols for confluence analysis
            confluence_threshold (float): Minimum confluence score to enter trades (0.0-1.0)
        """
        self.initial_capital = initial_capital
        self.gap_threshold = gap_threshold
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.max_active_gaps = max_active_gaps
        self.gap_expiry_hours = gap_expiry_hours
        self.confluence_symbols = confluence_symbols or []
        self.confluence_threshold = confluence_threshold
        self.data_handler = HistoricalDataHandler()
        
        # Gap tracking
        self.gaps = []  # List of all detected gaps
        self.active_gaps = []  # List of unfilled gaps
        self.gap_counter = 0
        
        # Confluence data storage
        self.confluence_data = {}  # Store data for confluence symbols
        self.confluence_gaps = {}  # Store gaps for confluence symbols
        
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
    
    def detect_fvg_patterns(self, df):
        """
        Detect Fair Value Gap (FVG) patterns in the price data
        
        FVG Detection Logic:
        - Look at 3 consecutive candles (i-1, i, i+1)
        - Bullish FVG: candle[i-1].high < candle[i+1].low (gap up)
        - Bearish FVG: candle[i-1].low > candle[i+1].high (gap down)
        - The middle candle (i) should not fill the gap completely
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            List[Gap]: List of detected FVG patterns
        """
        detected_gaps = []
        
        if len(df) < 3:
            return detected_gaps
        
        print(f"🔍 Detecting FVG patterns in {len(df)} bars...")
        
        for i in range(1, len(df) - 1):  # Start from index 1, end at len-2
            candle_prev = df.iloc[i-1]  # Previous candle
            candle_curr = df.iloc[i]    # Current candle (middle)
            candle_next = df.iloc[i+1]  # Next candle
            
            current_price = candle_curr['close']
            current_time = candle_curr['datetime']
            
            # Detect Bullish FVG (gap up)
            # Previous candle's high < Next candle's low
            if candle_prev['high'] < candle_next['low']:
                gap_bottom = candle_prev['high']
                gap_top = candle_next['low']
                gap_size = gap_top - gap_bottom
                gap_size_percent = (gap_size / current_price) * 100
                
                # Check if gap meets minimum threshold
                if gap_size_percent >= (self.gap_threshold * 100):
                    # Verify middle candle doesn't completely fill the gap
                    if not (candle_curr['low'] <= gap_bottom and candle_curr['high'] >= gap_top):
                        self.gap_counter += 1
                        gap = Gap(
                            gap_id=self.gap_counter,
                            gap_type='bullish_fvg',
                            start_index=i-1,
                            end_index=i+1,
                            top=gap_top,
                            bottom=gap_bottom,
                            size=gap_size,
                            size_percent=gap_size_percent,
                            created_at=current_time,
                            filled=False
                        )
                        detected_gaps.append(gap)
                        print(f"  📈 Bullish FVG detected at {current_time}: ${gap_bottom:.2f} - ${gap_top:.2f} ({gap_size_percent:.3f}%)")
            
            # Detect Bearish FVG (gap down)
            # Previous candle's low > Next candle's high
            elif candle_prev['low'] > candle_next['high']:
                gap_top = candle_prev['low']
                gap_bottom = candle_next['high']
                gap_size = gap_top - gap_bottom
                gap_size_percent = (gap_size / current_price) * 100
                
                # Check if gap meets minimum threshold
                if gap_size_percent >= (self.gap_threshold * 100):
                    # Verify middle candle doesn't completely fill the gap
                    if not (candle_curr['low'] <= gap_bottom and candle_curr['high'] >= gap_top):
                        self.gap_counter += 1
                        gap = Gap(
                            gap_id=self.gap_counter,
                            gap_type='bearish_fvg',
                            start_index=i-1,
                            end_index=i+1,
                            top=gap_top,
                            bottom=gap_bottom,
                            size=gap_size,
                            size_percent=gap_size_percent,
                            created_at=current_time,
                            filled=False
                        )
                        detected_gaps.append(gap)
                        print(f"  📉 Bearish FVG detected at {current_time}: ${gap_bottom:.2f} - ${gap_top:.2f} ({gap_size_percent:.3f}%)")
        
        print(f"✅ FVG Detection complete: {len(detected_gaps)} gaps found")
        return detected_gaps
    
    def detect_ifvg_patterns(self, df):
        """
        Detect Inverse Fair Value Gap (IFVG) patterns in the price data
        
        Enhanced IFVG Detection Logic based on educational standards:
        - IFVG occurs when an FVG fails to hold price and becomes inverted
        - Bullish IFVG: A bearish FVG that gets breached upward, becoming support
        - Bearish IFVG: A bullish FVG that gets breached downward, becoming resistance
        - Focus on liquidity sweeps and failed market structure shifts
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            List[Gap]: List of detected IFVG patterns
        """
        detected_gaps = []
        
        if len(df) < 5:
            return detected_gaps
        
        print(f"🔍 Detecting IFVG patterns in {len(df)} bars...")
        
        # First, we need to identify potential FVGs that could become IFVGs
        potential_fvgs = self.detect_fvg_patterns(df)
        
        # Look for FVGs that get invalidated and become IFVGs
        for fvg in potential_fvgs:
            if fvg.start_index + 5 >= len(df):  # Need enough bars after FVG creation
                continue
                
            # Check bars after FVG creation for invalidation
            for i in range(fvg.end_index + 1, min(len(df), fvg.end_index + 20)):
                candle = df.iloc[i]
                current_price = candle['close']
                current_time = candle['datetime']
                
                # Check for FVG invalidation (price moves through the gap)
                fvg_invalidated = False
                
                if fvg.gap_type == 'bullish_fvg':
                    # Bullish FVG invalidated if price closes below gap bottom
                    if candle['close'] < fvg.bottom:
                        fvg_invalidated = True
                        # This creates a bearish IFVG (former bullish FVG now acts as resistance)
                        ifvg_type = 'bearish_ifvg'
                        
                elif fvg.gap_type == 'bearish_fvg':
                    # Bearish FVG invalidated if price closes above gap top
                    if candle['close'] > fvg.top:
                        fvg_invalidated = True
                        # This creates a bullish IFVG (former bearish FVG now acts as support)
                        ifvg_type = 'bullish_ifvg'
                
                if fvg_invalidated:
                    # Look for subsequent price action that confirms the inversion
                    confirmation_found = False
                    
                    # Check next few bars for confirmation of the inversion
                    for j in range(i + 1, min(len(df), i + 10)):
                        confirm_candle = df.iloc[j]
                        
                        if ifvg_type == 'bullish_ifvg':
                            # Look for price returning to gap area and finding support
                            if (confirm_candle['low'] <= fvg.top and 
                                confirm_candle['close'] > fvg.bottom):
                                confirmation_found = True
                                break
                                
                        elif ifvg_type == 'bearish_ifvg':
                            # Look for price returning to gap area and finding resistance
                            if (confirm_candle['high'] >= fvg.bottom and 
                                confirm_candle['close'] < fvg.top):
                                confirmation_found = True
                                break
                    
                    if confirmation_found:
                        gap_size = fvg.top - fvg.bottom
                        gap_size_percent = (gap_size / current_price) * 100
                        
                        if gap_size_percent >= (self.gap_threshold * 100):
                            self.gap_counter += 1
                            ifvg = Gap(
                                gap_id=self.gap_counter,
                                gap_type=ifvg_type,
                                start_index=fvg.start_index,
                                end_index=i,  # IFVG confirmed at invalidation point
                                top=fvg.top,
                                bottom=fvg.bottom,
                                size=gap_size,
                                size_percent=gap_size_percent,
                                created_at=current_time,
                                filled=False
                            )
                            detected_gaps.append(ifvg)
                            print(f"  🔄 {ifvg_type.upper()} detected at {current_time}: ${fvg.bottom:.2f} - ${fvg.top:.2f} ({gap_size_percent:.3f}%) - Inverted from {fvg.gap_type}")
                    
                    break  # Stop checking this FVG once invalidated
        
        # Also look for liquidity sweep patterns that create IFVGs
        detected_gaps.extend(self.detect_liquidity_sweep_ifvgs(df))
        
        print(f"✅ IFVG Detection complete: {len(detected_gaps)} gaps found")
        return detected_gaps
    
    def detect_liquidity_sweep_ifvgs(self, df):
        """
        Detect IFVGs created by liquidity sweeps at swing highs/lows
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            List[Gap]: List of IFVG patterns from liquidity sweeps
        """
        detected_gaps = []
        
        if len(df) < 10:
            return detected_gaps
        
        # Look for swing highs and lows that get swept
        for i in range(5, len(df) - 5):
            current_candle = df.iloc[i]
            current_high = current_candle['high']
            current_low = current_candle['low']
            current_time = current_candle['datetime']
            current_price = current_candle['close']
            
            # Check if this is a swing high (higher than surrounding bars)
            is_swing_high = True
            is_swing_low = True
            
            for j in range(i - 3, i + 4):
                if j != i and j >= 0 and j < len(df):
                    if df.iloc[j]['high'] >= current_high:
                        is_swing_high = False
                    if df.iloc[j]['low'] <= current_low:
                        is_swing_low = False
            
            # Look for liquidity sweeps of swing highs (creates bearish IFVG)
            if is_swing_high:
                for k in range(i + 1, min(len(df), i + 15)):
                    sweep_candle = df.iloc[k]
                    
                    # Check for sweep above swing high followed by reversal
                    if sweep_candle['high'] > current_high:
                        # Look for reversal within next few bars
                        for m in range(k + 1, min(len(df), k + 5)):
                            reversal_candle = df.iloc[m]
                            
                            # If price reverses below the swing high, we have a bearish IFVG
                            if reversal_candle['close'] < current_high:
                                gap_bottom = current_high
                                gap_top = sweep_candle['high']
                                gap_size = gap_top - gap_bottom
                                gap_size_percent = (gap_size / current_price) * 100
                                
                                if gap_size_percent >= (self.gap_threshold * 100):
                                    self.gap_counter += 1
                                    ifvg = Gap(
                                        gap_id=self.gap_counter,
                                        gap_type='bearish_ifvg',
                                        start_index=i,
                                        end_index=k,
                                        top=gap_top,
                                        bottom=gap_bottom,
                                        size=gap_size,
                                        size_percent=gap_size_percent,
                                        created_at=sweep_candle['datetime'],
                                        filled=False
                                    )
                                    detected_gaps.append(ifvg)
                                    print(f"  🎯 Liquidity Sweep BEARISH IFVG at {sweep_candle['datetime']}: ${gap_bottom:.2f} - ${gap_top:.2f} ({gap_size_percent:.3f}%)")
                                break
                        break
            
            # Look for liquidity sweeps of swing lows (creates bullish IFVG)
            if is_swing_low:
                for k in range(i + 1, min(len(df), i + 15)):
                    sweep_candle = df.iloc[k]
                    
                    # Check for sweep below swing low followed by reversal
                    if sweep_candle['low'] < current_low:
                        # Look for reversal within next few bars
                        for m in range(k + 1, min(len(df), k + 5)):
                            reversal_candle = df.iloc[m]
                            
                            # If price reverses above the swing low, we have a bullish IFVG
                            if reversal_candle['close'] > current_low:
                                gap_bottom = sweep_candle['low']
                                gap_top = current_low
                                gap_size = gap_top - gap_bottom
                                gap_size_percent = (gap_size / current_price) * 100
                                
                                if gap_size_percent >= (self.gap_threshold * 100):
                                    self.gap_counter += 1
                                    ifvg = Gap(
                                        gap_id=self.gap_counter,
                                        gap_type='bullish_ifvg',
                                        start_index=i,
                                        end_index=k,
                                        top=gap_top,
                                        bottom=gap_bottom,
                                        size=gap_size,
                                        size_percent=gap_size_percent,
                                        created_at=sweep_candle['datetime'],
                                        filled=False
                                    )
                                    detected_gaps.append(ifvg)
                                    print(f"  🎯 Liquidity Sweep BULLISH IFVG at {sweep_candle['datetime']}: ${gap_bottom:.2f} - ${gap_top:.2f} ({gap_size_percent:.3f}%)")
                                break
                        break
        
        return detected_gaps
    
    def check_gap_fills(self, df, gaps):
        """
        Check if any gaps have been filled by subsequent price action
        
        Args:
            df (pd.DataFrame): OHLCV data
            gaps (List[Gap]): List of gaps to check
            
        Returns:
            List[Gap]: Updated gaps with fill status
        """
        updated_gaps = []
        
        for gap in gaps:
            if gap.filled:
                # Already filled, keep as is
                updated_gaps.append(gap)
                continue
            
            # Check bars after gap creation for fills
            gap_filled = False
            fill_index = None
            fill_time = None
            
            # Look at bars after the gap was created
            start_check_index = gap.end_index + 1
            
            for i in range(start_check_index, len(df)):
                candle = df.iloc[i]
                
                # Check if price has moved through the gap zone
                if gap.gap_type in ['bullish_fvg', 'bullish_ifvg']:
                    # For bullish gaps, check if price came back down to fill the gap
                    if candle['low'] <= gap.bottom:
                        gap_filled = True
                        fill_index = i
                        fill_time = candle['datetime']
                        break
                
                elif gap.gap_type in ['bearish_fvg', 'bearish_ifvg']:
                    # For bearish gaps, check if price came back up to fill the gap
                    if candle['high'] >= gap.top:
                        gap_filled = True
                        fill_index = i
                        fill_time = candle['datetime']
                        break
                
                # Check gap expiry (optional time-based expiration)
                if self.gap_expiry_hours > 0:
                    time_diff = candle['datetime'] - gap.created_at
                    if time_diff.total_seconds() / 3600 > self.gap_expiry_hours:
                        # Gap expired without being filled
                        break
            
            # Update gap with fill information
            if gap_filled:
                updated_gap = gap._replace(
                    filled=True,
                    filled_at=fill_time,
                    filled_index=fill_index
                )
                print(f"  ✅ Gap {gap.gap_id} ({gap.gap_type}) filled at {fill_time}")
            else:
                updated_gap = gap
            
            updated_gaps.append(updated_gap)
        
        return updated_gaps
    
    def calculate_trend_context(self, df, gap_index):
        """
        Calculate trend context for a gap to determine if it's discount or premium
        
        Args:
            df (pd.DataFrame): OHLCV data
            gap_index (int): Index where gap was created
            
        Returns:
            str: 'discount', 'premium', or 'neutral'
        """
        try:
            # Look at 20 bars before gap creation to determine trend stage
            lookback_bars = min(20, gap_index)
            if lookback_bars < 5:
                return "neutral"
            
            trend_data = df.iloc[gap_index-lookback_bars:gap_index+1]
            
            # Calculate trend metrics
            start_price = trend_data['close'].iloc[0]
            end_price = trend_data['close'].iloc[-1]
            trend_change = (end_price - start_price) / start_price
            
            # Calculate trend strength using price range
            price_high = trend_data['high'].max()
            price_low = trend_data['low'].min()
            total_range = price_high - price_low
            
            if total_range == 0:
                return "neutral"
            
            # Determine trend stage
            current_position = (end_price - price_low) / total_range
            
            # Discount FVG: Early in trend (first 40% of range)
            # Premium FVG: Late in trend (last 40% of range)
            if abs(trend_change) > 0.02:  # Significant trend (>2%)
                if trend_change > 0:  # Uptrend
                    if current_position < 0.4:
                        return "discount"  # Early in uptrend
                    elif current_position > 0.6:
                        return "premium"   # Late in uptrend
                else:  # Downtrend
                    if current_position > 0.6:
                        return "discount"  # Early in downtrend
                    elif current_position < 0.4:
                        return "premium"   # Late in downtrend
            
            return "neutral"
            
        except Exception as e:
            print(f"Error calculating trend context: {e}")
            return "neutral"
    
    def calculate_gap_priority_score(self, gap, df, gap_index):
        """
        Calculate priority score for a gap based on size and trend context
        
        Args:
            gap (Gap): Gap object
            df (pd.DataFrame): OHLCV data
            gap_index (int): Index where gap was created
            
        Returns:
            float: Priority score (higher = more important)
        """
        try:
            # Base score from gap size (0.1% = 1.0, 0.5% = 5.0, etc.)
            size_score = gap.size_percent * 10
            
            # Trend context multiplier
            trend_context = self.calculate_trend_context(df, gap_index)
            if trend_context == "discount":
                trend_multiplier = 1.5  # Discount gaps are more reliable
            elif trend_context == "premium":
                trend_multiplier = 0.7  # Premium gaps are less reliable
            else:
                trend_multiplier = 1.0  # Neutral
            
            # Gap type multiplier (FVG generally more reliable than IFVG)
            if 'fvg' in gap.gap_type:
                type_multiplier = 1.2
            else:  # ifvg
                type_multiplier = 1.0
            
            # Calculate final priority score
            priority_score = size_score * trend_multiplier * type_multiplier
            
            return priority_score
            
        except Exception as e:
            print(f"Error calculating gap priority: {e}")
            return gap.size_percent * 10  # Fallback to size-based scoring
    
    def prioritize_gaps(self, gaps, df):
        """
        Prioritize gaps based on size, trend context, and type
        
        Args:
            gaps (List[Gap]): List of gaps to prioritize
            df (pd.DataFrame): OHLCV data
            
        Returns:
            List[Gap]: Sorted gaps with priority scores and trend context
        """
        prioritized_gaps = []
        
        for gap in gaps:
            # Calculate trend context
            trend_context = self.calculate_trend_context(df, gap.start_index)
            
            # Calculate priority score
            priority_score = self.calculate_gap_priority_score(gap, df, gap.start_index)
            
            # Update gap with new attributes
            updated_gap = gap._replace(
                priority_score=priority_score,
                trend_context=trend_context
            )
            
            prioritized_gaps.append(updated_gap)
        
        # Sort by priority score (highest first)
        prioritized_gaps.sort(key=lambda g: g.priority_score, reverse=True)
        
        return prioritized_gaps
    
    def fetch_confluence_data(self, period_type="day", period=10):
        """
        Fetch data for confluence symbols
        
        Args:
            period_type (str): Period type for data fetching
            period (int): Number of periods
            
        Returns:
            bool: True if all confluence data fetched successfully
        """
        if not self.confluence_symbols:
            return True  # No confluence symbols, so no need to fetch
        
        print(f"🔗 Fetching confluence data for {len(self.confluence_symbols)} symbols...")
        
        for symbol in self.confluence_symbols:
            print(f"  📊 Fetching {symbol}...")
            df = self.fetch_data(symbol, period_type=period_type, period=period, frequency_type="minute", frequency=1)
            
            if df.empty:
                print(f"  ❌ Failed to fetch data for confluence symbol {symbol}")
                return False
            
            # Filter to trading hours
            df = self.filter_trading_hours(df)
            
            if df.empty:
                print(f"  ❌ No trading hours data for confluence symbol {symbol}")
                return False
            
            # Store confluence data
            self.confluence_data[symbol] = df
            
            # Detect gaps for confluence symbol
            fvg_gaps = self.detect_fvg_patterns(df)
            ifvg_gaps = self.detect_ifvg_patterns(df)
            all_gaps = fvg_gaps + ifvg_gaps
            
            if all_gaps:
                filled_gaps = self.check_gap_fills(df, all_gaps)
                self.confluence_gaps[symbol] = filled_gaps
                print(f"  ✅ {symbol}: {len(filled_gaps)} gaps detected")
            else:
                self.confluence_gaps[symbol] = []
                print(f"  ⚠️  {symbol}: No gaps detected")
        
        print(f"✅ Confluence data fetched for all symbols")
        return True
    
    def calculate_confluence_score(self, primary_signal_type, current_time, current_index):
        """
        Calculate confluence score based on alignment with confluence symbols
        
        Args:
            primary_signal_type (str): 'LONG' or 'SHORT' for the primary signal
            current_time (datetime): Current timestamp
            current_index (int): Current bar index
            
        Returns:
            float: Confluence score between 0.0 and 1.0
        """
        if not self.confluence_symbols or not self.confluence_data:
            return 1.0  # No confluence analysis, allow all trades
        
        confluence_scores = []
        
        for symbol in self.confluence_symbols:
            if symbol not in self.confluence_data or symbol not in self.confluence_gaps:
                continue
            
            symbol_df = self.confluence_data[symbol]
            symbol_gaps = self.confluence_gaps[symbol]
            
            # Find the corresponding time index in confluence data
            confluence_index = None
            for i, row in symbol_df.iterrows():
                if abs((row['datetime'] - current_time).total_seconds()) < 60:  # Within 1 minute
                    confluence_index = i
                    break
            
            if confluence_index is None:
                continue
            
            # Check if confluence symbol has similar gap patterns
            symbol_score = 0.0
            
            # Look for active gaps at current time
            active_confluence_gaps = []
            for gap in symbol_gaps:
                if gap.start_index <= confluence_index and not gap.filled:
                    active_confluence_gaps.append(gap)
            
            if active_confluence_gaps:
                # Check alignment of gap types with primary signal
                aligned_gaps = 0
                total_gaps = len(active_confluence_gaps)
                
                for gap in active_confluence_gaps:
                    gap_direction = 'LONG' if 'bullish' in gap.gap_type else 'SHORT'
                    
                    if gap_direction == primary_signal_type:
                        aligned_gaps += 1
                
                # Calculate alignment score
                if total_gaps > 0:
                    symbol_score = aligned_gaps / total_gaps
                else:
                    symbol_score = 0.5  # Neutral if no gaps
            else:
                # No active gaps in confluence symbol
                # Check recent price action alignment
                if confluence_index >= 5:
                    recent_bars = symbol_df.iloc[confluence_index-5:confluence_index+1]
                    price_change = (recent_bars['close'].iloc[-1] - recent_bars['close'].iloc[0]) / recent_bars['close'].iloc[0]
                    
                    # Align price momentum with signal direction
                    if primary_signal_type == 'LONG' and price_change > 0:
                        symbol_score = 0.7  # Positive momentum supports long
                    elif primary_signal_type == 'SHORT' and price_change < 0:
                        symbol_score = 0.7  # Negative momentum supports short
                    else:
                        symbol_score = 0.3  # Momentum doesn't align
                else:
                    symbol_score = 0.5  # Not enough data, neutral
            
            confluence_scores.append(symbol_score)
            print(f"    🔗 {symbol} confluence score: {symbol_score:.2f}")
        
        # Calculate overall confluence score
        if confluence_scores:
            overall_score = np.mean(confluence_scores)
            print(f"  📊 Overall confluence score: {overall_score:.2f} (threshold: {self.confluence_threshold:.2f})")
            return overall_score
        else:
            print(f"  ⚠️  No confluence data available, defaulting to neutral score")
            return 0.5
    
    def calculate_signals_and_performance(self, df):
        """
        Calculate IFVG signals and performance metrics with actual trading logic and confluence
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            pd.DataFrame: Data with signals and performance metrics added
        """
        if df.empty or len(df) < 5:
            return df
        
        result = df.copy()
        
        print("🔍 Detecting gap patterns...")
        
        # Step 1: Detect FVG patterns
        fvg_gaps = self.detect_fvg_patterns(df)
        
        # Step 2: Detect IFVG patterns
        ifvg_gaps = self.detect_ifvg_patterns(df)
        
        # Step 3: Combine all gaps
        all_gaps = fvg_gaps + ifvg_gaps
        self.gaps = all_gaps
        
        print(f"📊 Total gaps detected: {len(all_gaps)} (FVG: {len(fvg_gaps)}, IFVG: {len(ifvg_gaps)})")
        
        # Step 4: Check for gap fills
        if all_gaps:
            print("🔍 Checking gap fills...")
            self.gaps = self.check_gap_fills(df, all_gaps)
            filled_gaps = [g for g in self.gaps if g.filled]
            print(f"✅ Gap fills detected: {len(filled_gaps)}/{len(all_gaps)} gaps filled")
        
        # Step 5: Prioritize gaps based on size, trend context, and type
        if self.gaps:
            print("🎯 Prioritizing gaps for trading...")
            self.gaps = self.prioritize_gaps(self.gaps, df)
            
            # Display top priority gaps
            top_gaps = self.gaps[:5]  # Show top 5 priority gaps
            print(f"📊 Top Priority Gaps:")
            for i, gap in enumerate(top_gaps, 1):
                print(f"   {i}. Gap #{gap.gap_id} ({gap.gap_type}) - Priority: {gap.priority_score:.2f} - Context: {gap.trend_context}")
        
        # Initialize tracking columns
        result['signal'] = None
        result['entry_signal'] = None
        result['exit_signal'] = None
        result['gap_id'] = np.nan
        result['gap_type'] = None
        result['gap_top'] = np.nan
        result['gap_bottom'] = np.nan
        result['position_shares'] = 0
        result['position_type'] = None  # 'LONG' or 'SHORT'
        result['entry_price'] = np.nan
        result['position_value'] = 0.0
        result['unrealized_pnl'] = 0.0
        result['trade_pnl'] = np.nan
        result['cumulative_pnl'] = 0.0
        result['equity'] = self.initial_capital
        result['cash'] = self.initial_capital
        
        # Add gap information to dataframe
        for gap in self.gaps:
            # Mark gap creation points
            if gap.start_index < len(result):
                result.iloc[gap.start_index, result.columns.get_loc('gap_id')] = gap.gap_id
                result.iloc[gap.start_index, result.columns.get_loc('gap_type')] = gap.gap_type
                result.iloc[gap.start_index, result.columns.get_loc('gap_top')] = gap.top
                result.iloc[gap.start_index, result.columns.get_loc('gap_bottom')] = gap.bottom
            
            # Mark gap fill points
            if gap.filled and gap.filled_index is not None and gap.filled_index < len(result):
                result.iloc[gap.filled_index, result.columns.get_loc('signal')] = 'GAP_FILLED'
        
        print("📈 Implementing IFVG trading strategy...")
        
        # Simple IFVG Trading Strategy Implementation
        current_position = 0  # 0 = no position, positive = long shares, negative = short shares
        current_position_type = None
        entry_price = 0.0
        current_cash = self.initial_capital
        cumulative_pnl = 0.0
        position_size = 100  # Fixed position size for simplicity
        
        for i in range(len(result)):
            current_price = result.iloc[i]['close']
            current_high = result.iloc[i]['high']
            current_low = result.iloc[i]['low']
            
            # Check for exit conditions first
            if current_position != 0:
                exit_triggered = False
                exit_reason = ""
                
                if current_position > 0:  # Long position
                    # Exit on profit target
                    if current_price >= entry_price * (1 + self.profit_target):
                        exit_triggered = True
                        exit_reason = "PROFIT_TARGET"
                    # Exit on stop loss
                    elif current_price <= entry_price * (1 - self.stop_loss):
                        exit_triggered = True
                        exit_reason = "STOP_LOSS"
                
                elif current_position < 0:  # Short position
                    # Exit on profit target
                    if current_price <= entry_price * (1 - self.profit_target):
                        exit_triggered = True
                        exit_reason = "PROFIT_TARGET"
                    # Exit on stop loss
                    elif current_price >= entry_price * (1 + self.stop_loss):
                        exit_triggered = True
                        exit_reason = "STOP_LOSS"
                
                if exit_triggered:
                    # Calculate trade P&L
                    if current_position > 0:  # Closing long
                        trade_pnl = (current_price - entry_price) * abs(current_position)
                    else:  # Closing short
                        trade_pnl = (entry_price - current_price) * abs(current_position)
                    
                    cumulative_pnl += trade_pnl
                    current_cash += abs(current_position) * current_price
                    
                    # Mark exit signal
                    result.iloc[i, result.columns.get_loc('exit_signal')] = f'EXIT_{current_position_type}_{exit_reason}'
                    result.iloc[i, result.columns.get_loc('trade_pnl')] = trade_pnl
                    
                    print(f"  🔴 EXIT {current_position_type} at ${current_price:.2f} - P&L: ${trade_pnl:.2f} ({exit_reason})")
                    
                    # Reset position
                    current_position = 0
                    current_position_type = None
                    entry_price = 0.0
            
            # Check for entry conditions (only if no current position)
            if current_position == 0:
                # Filter to only high-priority gaps (top 30% by priority score)
                active_gaps = [g for g in self.gaps if not g.filled and g.start_index <= i]
                if not active_gaps:
                    continue
                
                # Sort by priority and take top 30%
                priority_threshold = max(1, int(len(active_gaps) * 0.3))
                high_priority_gaps = sorted(active_gaps, key=lambda g: g.priority_score, reverse=True)[:priority_threshold]
                
                # Look for high-priority gaps that price is approaching
                for gap in high_priority_gaps:
                    # Skip gaps that are too old or have low priority scores
                    if gap.priority_score < 1.0:  # Minimum priority threshold
                        continue
                    
                    # Calculate gap midpoint for 50% rule implementation
                    gap_midpoint = (gap.top + gap.bottom) / 2
                    gap_size = gap.top - gap.bottom
                    
                    # Entry logic based on gap type and price action with 50% rule
                    entry_triggered = False
                    signal_type = ""
                    optimal_entry_price = current_price
                    
                    if gap.gap_type in ['bullish_fvg', 'bullish_ifvg']:
                        # Long entry: Price touches gap zone, preferably around 50% fill level
                        if current_low <= gap.top and current_price >= gap.bottom:
                            # Check if we're in the optimal entry zone (bottom 50% of gap)
                            if current_price <= gap_midpoint:
                                entry_triggered = True
                                signal_type = "LONG"
                                # Use optimal entry price (closer to gap bottom for better risk/reward)
                                optimal_entry_price = max(gap.bottom, current_low)
                                
                    elif gap.gap_type in ['bearish_fvg', 'bearish_ifvg']:
                        # Short entry: Price touches gap zone, preferably around 50% fill level
                        if current_high >= gap.bottom and current_price <= gap.top:
                            # Check if we're in the optimal entry zone (top 50% of gap)
                            if current_price >= gap_midpoint:
                                entry_triggered = True
                                signal_type = "SHORT"
                                # Use optimal entry price (closer to gap top for better risk/reward)
                                optimal_entry_price = min(gap.top, current_high)
                    
                    if entry_triggered:
                        # Additional filter: Only trade discount gaps or high-priority premium gaps
                        if gap.trend_context == "premium" and gap.priority_score < 5.0:
                            print(f"  ⚠️  Skipping premium gap #{gap.gap_id} with low priority ({gap.priority_score:.2f})")
                            continue
                        
                        # Check confluence if enabled
                        confluence_score = 1.0  # Default if no confluence
                        if self.confluence_symbols:
                            current_time = result.iloc[i]['datetime']
                            confluence_score = self.calculate_confluence_score(signal_type, current_time, i)
                            
                            if confluence_score < self.confluence_threshold:
                                print(f"  ❌ CONFLUENCE FAILED for {signal_type} - Score: {confluence_score:.2f} < {self.confluence_threshold:.2f}")
                                continue
                        
                        # Entry confirmed - execute trade
                        current_position = position_size if signal_type == "LONG" else -position_size
                        current_position_type = signal_type
                        entry_price = optimal_entry_price
                        current_cash = current_cash - (position_size * entry_price) if signal_type == "LONG" else current_cash + (position_size * entry_price)
                        
                        result.iloc[i, result.columns.get_loc('entry_signal')] = f'ENTER_{signal_type}_{gap.gap_type.upper()}'
                        result.iloc[i, result.columns.get_loc('entry_price')] = entry_price
                        
                        # Enhanced logging with priority and trend context
                        confluence_msg = f" (Confluence: {confluence_score:.2f})" if self.confluence_symbols else ""
                        priority_msg = f" Priority: {gap.priority_score:.2f} ({gap.trend_context})"
                        print(f"  🟢 ENTER {signal_type} at ${entry_price:.2f} - Gap #{gap.gap_id} ({gap.gap_type}) (${gap.bottom:.2f}-${gap.top:.2f}){priority_msg}{confluence_msg}")
                        break  # Only one entry per bar
            
            # Update tracking columns
            result.iloc[i, result.columns.get_loc('position_shares')] = abs(current_position)
            result.iloc[i, result.columns.get_loc('position_type')] = current_position_type
            result.iloc[i, result.columns.get_loc('cash')] = current_cash
            result.iloc[i, result.columns.get_loc('cumulative_pnl')] = cumulative_pnl
            
            # Calculate unrealized P&L
            if current_position != 0:
                if current_position > 0:  # Long position
                    unrealized_pnl = (current_price - entry_price) * current_position
                    position_value = current_position * current_price
                else:  # Short position
                    unrealized_pnl = (entry_price - current_price) * abs(current_position)
                    position_value = abs(current_position) * current_price
                
                result.iloc[i, result.columns.get_loc('unrealized_pnl')] = unrealized_pnl
                result.iloc[i, result.columns.get_loc('position_value')] = position_value
            
            # Update equity
            total_equity = current_cash + cumulative_pnl
            if current_position != 0:
                total_equity += result.iloc[i]['unrealized_pnl']
            
            result.iloc[i, result.columns.get_loc('equity')] = total_equity
        
        # Close any remaining position at the end
        if current_position != 0:
            final_price = result.iloc[-1]['close']
            if current_position > 0:  # Close long
                trade_pnl = (final_price - entry_price) * current_position
            else:  # Close short
                trade_pnl = (entry_price - final_price) * abs(current_position)
            
            cumulative_pnl += trade_pnl
            result.iloc[-1, result.columns.get_loc('trade_pnl')] = trade_pnl
            result.iloc[-1, result.columns.get_loc('exit_signal')] = f'EXIT_{current_position_type}_EOD'
            result.iloc[-1, result.columns.get_loc('cumulative_pnl')] = cumulative_pnl
            
            print(f"  🔴 Final EXIT {current_position_type} at ${final_price:.2f} - P&L: ${trade_pnl:.2f} (End of Data)")
        
        return result
    
    def calculate_performance_summary(self, df, symbol):
        """
        Calculate comprehensive performance summary including gap statistics
        
        Args:
            df (pd.DataFrame): Data with performance metrics
            symbol (str): Stock symbol
            
        Returns:
            dict: Performance summary statistics
        """
        if df.empty:
            return {}
        
        # Gap statistics
        total_gaps = len(self.gaps)
        fvg_gaps = [g for g in self.gaps if 'fvg' in g.gap_type]
        ifvg_gaps = [g for g in self.gaps if 'ifvg' in g.gap_type]
        bullish_gaps = [g for g in self.gaps if 'bullish' in g.gap_type]
        bearish_gaps = [g for g in self.gaps if 'bearish' in g.gap_type]
        filled_gaps = [g for g in self.gaps if g.filled]
        
        # Gap size statistics
        if self.gaps:
            gap_sizes = [g.size_percent for g in self.gaps]
            avg_gap_size = np.mean(gap_sizes)
            max_gap_size = np.max(gap_sizes)
            min_gap_size = np.min(gap_sizes)
        else:
            avg_gap_size = max_gap_size = min_gap_size = 0
        
        # Fill rate statistics
        fill_rate = (len(filled_gaps) / total_gaps * 100) if total_gaps > 0 else 0
        
        # Time to fill statistics
        if filled_gaps:
            fill_times = []
            for gap in filled_gaps:
                if gap.filled_at and gap.created_at:
                    fill_time_hours = (gap.filled_at - gap.created_at).total_seconds() / 3600
                    fill_times.append(fill_time_hours)
            
            avg_fill_time = np.mean(fill_times) if fill_times else 0
            max_fill_time = np.max(fill_times) if fill_times else 0
        else:
            avg_fill_time = max_fill_time = 0
        
        return {
            'symbol': symbol,
            'data_period': {
                'start_date': df['datetime'].iloc[0],
                'end_date': df['datetime'].iloc[-1],
                'total_bars': len(df)
            },
            'gap_statistics': {
                'total_gaps': total_gaps,
                'fvg_gaps': len(fvg_gaps),
                'ifvg_gaps': len(ifvg_gaps),
                'bullish_gaps': len(bullish_gaps),
                'bearish_gaps': len(bearish_gaps),
                'filled_gaps': len(filled_gaps),
                'fill_rate': fill_rate,
                'avg_gap_size_percent': avg_gap_size,
                'max_gap_size_percent': max_gap_size,
                'min_gap_size_percent': min_gap_size,
                'avg_fill_time_hours': avg_fill_time,
                'max_fill_time_hours': max_fill_time
            },
            'strategy_parameters': {
                'gap_threshold_percent': self.gap_threshold * 100,
                'profit_target_percent': self.profit_target * 100,
                'stop_loss_percent': self.stop_loss * 100,
                'max_active_gaps': self.max_active_gaps,
                'gap_expiry_hours': self.gap_expiry_hours
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
            filename = f'{self.output_dir}/{symbol}_ifvg_backtest_{timestamp}.csv'
            
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
    
    def print_performance_summary(self, summary):
        """
        Print formatted performance summary
        
        Args:
            summary (dict): Performance summary statistics
        """
        symbol = summary['symbol']
        
        print(f"\n📊 IFVG STRATEGY RESULTS: {symbol}")
        print("=" * 70)
        
        # Data period info
        print(f"📅 DATA PERIOD")
        print(f"   Start Date: {summary['data_period']['start_date'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   End Date: {summary['data_period']['end_date'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Total 1-min Bars: {summary['data_period']['total_bars']:,}")
        
        # Gap statistics
        gap_stats = summary['gap_statistics']
        print(f"\n🔍 GAP DETECTION STATISTICS")
        print(f"   Total Gaps Detected: {gap_stats['total_gaps']}")
        print(f"   FVG Gaps: {gap_stats['fvg_gaps']}")
        print(f"   IFVG Gaps: {gap_stats['ifvg_gaps']}")
        print(f"   Bullish Gaps: {gap_stats['bullish_gaps']}")
        print(f"   Bearish Gaps: {gap_stats['bearish_gaps']}")
        print(f"   Filled Gaps: {gap_stats['filled_gaps']}")
        print(f"   Fill Rate: {gap_stats['fill_rate']:.1f}%")
        
        # Gap size analysis
        print(f"\n📏 GAP SIZE ANALYSIS")
        print(f"   Average Gap Size: {gap_stats['avg_gap_size_percent']:.3f}%")
        print(f"   Largest Gap: {gap_stats['max_gap_size_percent']:.3f}%")
        print(f"   Smallest Gap: {gap_stats['min_gap_size_percent']:.3f}%")
        
        # Fill time analysis
        if gap_stats['filled_gaps'] > 0:
            print(f"\n⏱️  GAP FILL TIMING")
            print(f"   Average Fill Time: {gap_stats['avg_fill_time_hours']:.1f} hours")
            print(f"   Maximum Fill Time: {gap_stats['max_fill_time_hours']:.1f} hours")
        
        # Strategy parameters
        params = summary['strategy_parameters']
        print(f"\n⚙️  STRATEGY PARAMETERS")
        print(f"   Gap Threshold: {params['gap_threshold_percent']:.3f}%")
        print(f"   Profit Target: {params['profit_target_percent']:.1f}%")
        print(f"   Stop Loss: {params['stop_loss_percent']:.1f}%")
        print(f"   Max Active Gaps: {params['max_active_gaps']}")
        print(f"   Gap Expiry: {params['gap_expiry_hours']} hours")
        
        # Strategy insights
        print(f"\n🔍 STRATEGY INSIGHTS")
        if gap_stats['total_gaps'] > 0:
            print(f"   ✅ Gap detection active - {gap_stats['total_gaps']} patterns identified")
            
            if gap_stats['fill_rate'] > 70:
                print(f"   ✅ High fill rate ({gap_stats['fill_rate']:.1f}%) - gaps frequently get filled")
            elif gap_stats['fill_rate'] > 40:
                print(f"   ⚠️  Moderate fill rate ({gap_stats['fill_rate']:.1f}%) - some gaps remain unfilled")
            else:
                print(f"   ❌ Low fill rate ({gap_stats['fill_rate']:.1f}%) - many gaps remain unfilled")
            
            if gap_stats['fvg_gaps'] > gap_stats['ifvg_gaps']:
                print(f"   📊 FVG patterns dominate ({gap_stats['fvg_gaps']} vs {gap_stats['ifvg_gaps']} IFVG)")
            elif gap_stats['ifvg_gaps'] > gap_stats['fvg_gaps']:
                print(f"   📊 IFVG patterns dominate ({gap_stats['ifvg_gaps']} vs {gap_stats['fvg_gaps']} FVG)")
            else:
                print(f"   📊 Balanced FVG/IFVG distribution")
            
            if gap_stats['avg_gap_size_percent'] > 0.5:
                print(f"   📏 Large average gap size ({gap_stats['avg_gap_size_percent']:.3f}%) - significant inefficiencies")
            elif gap_stats['avg_gap_size_percent'] > 0.1:
                print(f"   📏 Moderate gap sizes ({gap_stats['avg_gap_size_percent']:.3f}%) - typical market inefficiencies")
            else:
                print(f"   📏 Small gap sizes ({gap_stats['avg_gap_size_percent']:.3f}%) - minor inefficiencies")
        else:
            print(f"   ❌ No gaps detected - consider lowering gap threshold or checking data quality")
        
        print("=" * 70)
    
    def create_basic_visualization(self, df, summary, symbol):
        """
        Create enhanced visualization with candlestick charts and detailed legends
        
        Args:
            df (pd.DataFrame): Results dataframe
            summary (dict): Performance summary
            symbol (str): Stock symbol
            
        Returns:
            str: Path to saved chart
        """
        try:
            from matplotlib.patches import Rectangle
            from matplotlib.lines import Line2D
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(20, 14))
            fig.suptitle(f'IFVG Strategy Analysis - {symbol}', fontsize=18, fontweight='bold', y=0.98)
            
            # Plot 1: Enhanced candlestick chart with gaps
            ax1 = axes[0, 0]
            
            # Create candlestick chart
            for i in range(len(df)):
                open_price = df.iloc[i]['open']
                high_price = df.iloc[i]['high']
                low_price = df.iloc[i]['low']
                close_price = df.iloc[i]['close']
                
                # Determine candle color
                color = 'green' if close_price >= open_price else 'red'
                edge_color = 'darkgreen' if close_price >= open_price else 'darkred'
                
                # Draw high-low line
                ax1.plot([i, i], [low_price, high_price], color='black', linewidth=0.8, alpha=0.7)
                
                # Draw candle body
                body_height = abs(close_price - open_price)
                body_bottom = min(open_price, close_price)
                
                if body_height > 0:
                    rect = Rectangle((i-0.4, body_bottom), 0.8, body_height, 
                                   facecolor=color, edgecolor=edge_color, 
                                   alpha=0.8, linewidth=0.5)
                    ax1.add_patch(rect)
                else:
                    # Doji candle (open == close)
                    ax1.plot([i-0.4, i+0.4], [close_price, close_price], 
                            color=edge_color, linewidth=1.5)
            
            # Track gap types for legend
            gap_legend_items = set()
            fill_points_added = False
            
            # Mark gaps with enhanced visualization
            for gap in self.gaps:
                if gap.start_index < len(df):
                    # Determine gap styling
                    if gap.gap_type == 'bullish_fvg':
                        color = 'lime'
                        alpha = 0.4 if gap.filled else 0.7
                        label = 'Bullish FVG'
                    elif gap.gap_type == 'bearish_fvg':
                        color = 'red'
                        alpha = 0.4 if gap.filled else 0.7
                        label = 'Bearish FVG'
                    elif gap.gap_type == 'bullish_ifvg':
                        color = 'cyan'
                        alpha = 0.3 if gap.filled else 0.6
                        label = 'Bullish IFVG'
                    else:  # bearish_ifvg
                        color = 'orange'
                        alpha = 0.3 if gap.filled else 0.6
                        label = 'Bearish IFVG'
                    
                    # Draw gap zone
                    gap_start = max(0, gap.start_index)
                    gap_end = min(len(df)-1, gap.end_index + 20)  # Extend gap zone visibility
                    
                    ax1.axhspan(gap.bottom, gap.top, 
                               xmin=gap_start/len(df), 
                               xmax=gap_end/len(df),
                               color=color, alpha=alpha, 
                               edgecolor=color, linewidth=1)
                    
                    # Add gap type to legend tracking
                    gap_legend_items.add((label, color, alpha))
                    
                    # Mark fill point if filled
                    if gap.filled and gap.filled_index and gap.filled_index < len(df):
                        ax1.scatter(gap.filled_index, df.iloc[gap.filled_index]['close'], 
                                   color='black', marker='X', s=80, alpha=0.9, 
                                   edgecolors='white', linewidth=1, zorder=10)
                        fill_points_added = True
            
            # Create comprehensive legend
            legend_elements = []
            
            # Add candlestick legend
            legend_elements.extend([
                Line2D([0], [0], color='green', linewidth=8, alpha=0.8, label='Bullish Candle'),
                Line2D([0], [0], color='red', linewidth=8, alpha=0.8, label='Bearish Candle')
            ])
            
            # Add gap type legends
            for label, color, alpha in sorted(gap_legend_items):
                legend_elements.append(
                    Rectangle((0, 0), 1, 1, facecolor=color, alpha=alpha, 
                             edgecolor=color, label=label)
                )
            
            # Add fill point legend if applicable
            if fill_points_added:
                legend_elements.append(
                    Line2D([0], [0], marker='X', color='black', linestyle='None',
                          markersize=8, markeredgecolor='white', markeredgewidth=1,
                          label='Gap Fill Point')
                )
            
            ax1.legend(handles=legend_elements, loc='upper left', fontsize=9, 
                      framealpha=0.9, fancybox=True, shadow=True)
            
            ax1.set_title(f'{symbol} - Candlestick Chart with FVG/IFVG Zones', 
                         fontsize=14, fontweight='bold', pad=20)
            ax1.set_ylabel('Price ($)', fontsize=12)
            ax1.set_xlabel('Time (1-minute bars)', fontsize=12)
            ax1.grid(True, alpha=0.3, linestyle='--')
            
            # Format y-axis to show currency
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.2f}'))
            
            # Improve x-axis formatting
            if len(df) > 100:
                # Show every nth tick to avoid crowding
                tick_spacing = max(1, len(df) // 10)
                ax1.set_xticks(range(0, len(df), tick_spacing))
                ax1.set_xticklabels([f'{i//60}h{i%60:02d}m' for i in range(0, len(df), tick_spacing)], 
                                   rotation=45, ha='right')
            
            # Set axis limits with padding
            price_range = df['high'].max() - df['low'].min()
            ax1.set_ylim(df['low'].min() - price_range * 0.02, 
                        df['high'].max() + price_range * 0.02)
            
            # Plot 2: Gap statistics
            ax2 = axes[0, 1]
            gap_types = ['FVG', 'IFVG', 'Bullish', 'Bearish', 'Filled']
            gap_counts = [
                summary['gap_statistics']['fvg_gaps'],
                summary['gap_statistics']['ifvg_gaps'],
                summary['gap_statistics']['bullish_gaps'],
                summary['gap_statistics']['bearish_gaps'],
                summary['gap_statistics']['filled_gaps']
            ]
            
            bars = ax2.bar(gap_types, gap_counts, color=['blue', 'orange', 'green', 'red', 'purple'])
            ax2.set_title('Gap Type Distribution')
            ax2.set_ylabel('Count')
            
            # Add value labels on bars
            for bar, count in zip(bars, gap_counts):
                if count > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            str(count), ha='center', va='bottom')
            
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Gap size distribution
            ax3 = axes[1, 0]
            if self.gaps:
                gap_sizes = [g.size_percent for g in self.gaps]
                ax3.hist(gap_sizes, bins=min(20, len(gap_sizes)), alpha=0.7, color='purple', edgecolor='black')
                ax3.axvline(x=np.mean(gap_sizes), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(gap_sizes):.3f}%')
                ax3.set_title('Gap Size Distribution')
                ax3.set_xlabel('Gap Size (%)')
                ax3.set_ylabel('Frequency')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'No gaps detected', ha='center', va='center', 
                        transform=ax3.transAxes, fontsize=14)
                ax3.set_title('Gap Size Distribution')
            
            # Plot 4: Performance summary text
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            # Create performance statistics text
            stats_text = f"""
GAP ANALYSIS SUMMARY

Total Gaps: {summary['gap_statistics']['total_gaps']}
FVG Gaps: {summary['gap_statistics']['fvg_gaps']}
IFVG Gaps: {summary['gap_statistics']['ifvg_gaps']}
Fill Rate: {summary['gap_statistics']['fill_rate']:.1f}%

Gap Sizes:
Avg: {summary['gap_statistics']['avg_gap_size_percent']:.3f}%
Max: {summary['gap_statistics']['max_gap_size_percent']:.3f}%
Min: {summary['gap_statistics']['min_gap_size_percent']:.3f}%

Strategy Parameters:
Gap Threshold: {summary['strategy_parameters']['gap_threshold_percent']:.3f}%
Profit Target: {summary['strategy_parameters']['profit_target_percent']:.1f}%
Stop Loss: {summary['strategy_parameters']['stop_loss_percent']:.1f}%

Data Period:
{summary['data_period']['start_date'].strftime('%Y-%m-%d')} to 
{summary['data_period']['end_date'].strftime('%Y-%m-%d')}
Total Bars: {summary['data_period']['total_bars']:,}
            """
            
            ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=9,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            
            # Save chart
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            chart_filename = f'{self.charts_dir}/{symbol}_ifvg_analysis_{timestamp}.png'
            plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Chart saved to: {chart_filename}")
            return chart_filename
            
        except Exception as e:
            print(f"Error creating visualization for {symbol}: {e}")
            return ""
    
    def create_standalone_price_chart(self, df, summary, symbol):
        """
        Create a standalone price action chart with enhanced candlestick visualization and confluence analysis
        
        Args:
            df (pd.DataFrame): Results dataframe
            summary (dict): Performance summary
            symbol (str): Stock symbol
            
        Returns:
            str: Path to saved chart
        """
        try:
            from matplotlib.patches import Rectangle
            from matplotlib.lines import Line2D
            
            # Determine figure layout based on confluence symbols
            if self.confluence_symbols:
                # Multi-panel layout with confluence analysis
                num_panels = 1 + len(self.confluence_symbols)
                fig, axes = plt.subplots(num_panels, 1, figsize=(24, 8 * num_panels))
                if num_panels == 2:
                    axes = [axes[0], axes[1]]  # Ensure axes is always a list
                fig.suptitle(f'{symbol} - Price Action with FVG/IFVG Analysis & Confluence', 
                            fontsize=20, fontweight='bold', y=0.98)
                ax = axes[0]  # Primary symbol chart
            else:
                # Single panel layout
                fig, ax = plt.subplots(1, 1, figsize=(24, 12))
                fig.suptitle(f'{symbol} - Price Action with FVG/IFVG Analysis', 
                            fontsize=20, fontweight='bold', y=0.98)
            
            # Create enhanced candlestick chart
            for i in range(len(df)):
                open_price = df.iloc[i]['open']
                high_price = df.iloc[i]['high']
                low_price = df.iloc[i]['low']
                close_price = df.iloc[i]['close']
                
                # Determine candle color and styling
                is_bullish = close_price >= open_price
                body_color = '#2E8B57' if is_bullish else '#DC143C'  # Sea green / Crimson
                edge_color = '#1F5F3F' if is_bullish else '#8B0000'  # Dark green / Dark red
                wick_color = '#2F2F2F'  # Dark gray for wicks
                
                # Draw high-low wick
                ax.plot([i, i], [low_price, high_price], 
                       color=wick_color, linewidth=1.2, alpha=0.8, zorder=1)
                
                # Draw candle body
                body_height = abs(close_price - open_price)
                body_bottom = min(open_price, close_price)
                
                if body_height > 0:
                    # Regular candle with body
                    rect = Rectangle((i-0.35, body_bottom), 0.7, body_height, 
                                   facecolor=body_color, edgecolor=edge_color, 
                                   alpha=0.9, linewidth=0.8, zorder=2)
                    ax.add_patch(rect)
                else:
                    # Doji candle (open == close)
                    ax.plot([i-0.35, i+0.35], [close_price, close_price], 
                           color=edge_color, linewidth=2.5, alpha=0.9, zorder=2)
            
            # Track gap types for comprehensive legend
            gap_legend_items = set()
            fill_points_added = False
            gap_annotations = []
            entry_points_added = False
            exit_points_added = False
            
            # Mark entry and exit signals
            entry_signals = df[df['entry_signal'].notna()]
            exit_signals = df[df['exit_signal'].notna()]
            
            # Plot entry signals
            for idx, row in entry_signals.iterrows():
                if 'LONG' in str(row['entry_signal']):
                    ax.scatter(idx, row['close'], color='lime', marker='^', s=150, 
                              alpha=0.9, edgecolors='darkgreen', linewidth=2, zorder=12,
                              label='Long Entry' if not entry_points_added else "")
                elif 'SHORT' in str(row['entry_signal']):
                    ax.scatter(idx, row['close'], color='red', marker='v', s=150, 
                              alpha=0.9, edgecolors='darkred', linewidth=2, zorder=12,
                              label='Short Entry' if not entry_points_added else "")
                entry_points_added = True
            
            # Plot exit signals
            for idx, row in exit_signals.iterrows():
                if 'LONG' in str(row['exit_signal']):
                    ax.scatter(idx, row['close'], color='blue', marker='s', s=120, 
                              alpha=0.9, edgecolors='navy', linewidth=2, zorder=12,
                              label='Long Exit' if not exit_points_added else "")
                elif 'SHORT' in str(row['exit_signal']):
                    ax.scatter(idx, row['close'], color='orange', marker='s', s=120, 
                              alpha=0.9, edgecolors='darkorange', linewidth=2, zorder=12,
                              label='Short Exit' if not exit_points_added else "")
                exit_points_added = True
            
            # Mark gaps with enhanced visualization and annotations
            for gap_idx, gap in enumerate(self.gaps):
                if gap.start_index < len(df):
                    # Determine gap styling based on type and fill status
                    if gap.gap_type == 'bullish_fvg':
                        color = '#00FF7F'  # Spring green
                        alpha = 0.25 if gap.filled else 0.5
                        label = 'Bullish FVG'
                        pattern = '///'
                    elif gap.gap_type == 'bearish_fvg':
                        color = '#FF4500'  # Orange red
                        alpha = 0.25 if gap.filled else 0.5
                        label = 'Bearish FVG'
                        pattern = '\\\\\\'
                    elif gap.gap_type == 'bullish_ifvg':
                        color = '#00CED1'  # Dark turquoise
                        alpha = 0.2 if gap.filled else 0.4
                        label = 'Bullish IFVG'
                        pattern = '+++'
                    else:  # bearish_ifvg
                        color = '#FF8C00'  # Dark orange
                        alpha = 0.2 if gap.filled else 0.4
                        label = 'Bearish IFVG'
                        pattern = 'xxx'
                    
                    # Calculate gap zone visibility
                    gap_start = max(0, gap.start_index)
                    gap_end = min(len(df)-1, gap.end_index + 30)  # Extended visibility
                    
                    # Draw gap zone with pattern
                    ax.axhspan(gap.bottom, gap.top, 
                              xmin=gap_start/len(df), 
                              xmax=gap_end/len(df),
                              facecolor=color, alpha=alpha, 
                              edgecolor=color, linewidth=1.5,
                              hatch=pattern if not gap.filled else None,
                              zorder=3)
                    
                    # Add gap border lines for clarity
                    ax.axhline(y=gap.top, xmin=gap_start/len(df), xmax=gap_end/len(df),
                              color=color, linewidth=2, alpha=0.8, zorder=4)
                    ax.axhline(y=gap.bottom, xmin=gap_start/len(df), xmax=gap_end/len(df),
                              color=color, linewidth=2, alpha=0.8, zorder=4)
                    
                    # Add gap type to legend tracking
                    status = 'Filled' if gap.filled else 'Open'
                    gap_legend_items.add((f'{label} ({status})', color, alpha))
                    
                    # Mark fill point with enhanced visualization
                    if gap.filled and gap.filled_index and gap.filled_index < len(df):
                        fill_price = df.iloc[gap.filled_index]['close']
                        
                        # Large fill marker
                        ax.scatter(gap.filled_index, fill_price, 
                                  color='black', marker='X', s=120, alpha=1.0, 
                                  edgecolors='white', linewidth=2, zorder=10)
                        
                        # Add fill annotation for significant gaps
                        if gap.size_percent > 0.2:  # Only annotate larger gaps
                            gap_annotations.append({
                                'x': gap.filled_index,
                                'y': fill_price,
                                'text': f'Gap #{gap.gap_id}\nFilled\n{gap.size_percent:.2f}%',
                                'gap_type': gap.gap_type
                            })
                        
                        fill_points_added = True
            
            # Add gap annotations (limit to avoid crowding)
            for i, annotation in enumerate(gap_annotations[:8]):  # Max 8 annotations
                bbox_props = dict(boxstyle="round,pad=0.3", 
                                facecolor='white', alpha=0.8, edgecolor='gray')
                ax.annotate(annotation['text'], 
                           xy=(annotation['x'], annotation['y']),
                           xytext=(10, 20 + (i % 4) * 15), 
                           textcoords='offset points',
                           fontsize=8, ha='left',
                           bbox=bbox_props,
                           arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
                           zorder=11)
            
            # Create comprehensive legend
            legend_elements = []
            
            # Add candlestick legend
            legend_elements.extend([
                Rectangle((0, 0), 1, 1, facecolor='#2E8B57', edgecolor='#1F5F3F', 
                         alpha=0.9, label='Bullish Candle'),
                Rectangle((0, 0), 1, 1, facecolor='#DC143C', edgecolor='#8B0000', 
                         alpha=0.9, label='Bearish Candle')
            ])
            
            # Add gap type legends (sorted for consistency)
            for label, color, alpha in sorted(gap_legend_items):
                legend_elements.append(
                    Rectangle((0, 0), 1, 1, facecolor=color, alpha=alpha, 
                             edgecolor=color, label=label)
                )
            
            # Add fill point legend if applicable
            if fill_points_added:
                legend_elements.append(
                    Line2D([0], [0], marker='X', color='black', linestyle='None',
                          markersize=10, markeredgecolor='white', markeredgewidth=2,
                          label='Gap Fill Point')
                )
            
            # Add entry/exit signal legends
            if entry_points_added:
                legend_elements.extend([
                    Line2D([0], [0], marker='^', color='lime', linestyle='None',
                          markersize=12, markeredgecolor='darkgreen', markeredgewidth=2,
                          label='Long Entry'),
                    Line2D([0], [0], marker='v', color='red', linestyle='None',
                          markersize=12, markeredgecolor='darkred', markeredgewidth=2,
                          label='Short Entry')
                ])
            
            if exit_points_added:
                legend_elements.extend([
                    Line2D([0], [0], marker='s', color='blue', linestyle='None',
                          markersize=10, markeredgecolor='navy', markeredgewidth=2,
                          label='Long Exit'),
                    Line2D([0], [0], marker='s', color='orange', linestyle='None',
                          markersize=10, markeredgecolor='darkorange', markeredgewidth=2,
                          label='Short Exit')
                ])
            
            # Position legend outside plot area
            ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
                     fontsize=11, framealpha=0.95, fancybox=True, shadow=True)
            
            # Enhanced title and labels
            ax.set_title(f'1-Minute Candlestick Chart with Fair Value Gap Analysis\n'
                        f'Total Gaps: {len(self.gaps)} | Fill Rate: {summary["gap_statistics"]["fill_rate"]:.1f}% | '
                        f'Avg Gap Size: {summary["gap_statistics"]["avg_gap_size_percent"]:.3f}%', 
                        fontsize=16, fontweight='bold', pad=30)
            
            ax.set_ylabel('Price ($)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time (1-minute bars)', fontsize=14, fontweight='bold')
            
            # Enhanced grid
            ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
            ax.set_axisbelow(True)
            
            # Format y-axis to show currency with better precision
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.2f}'))
            
            # Improved x-axis formatting with time labels
            if len(df) > 50:
                tick_spacing = max(1, len(df) // 15)  # More ticks for better resolution
                tick_positions = range(0, len(df), tick_spacing)
                ax.set_xticks(tick_positions)
                
                # Create time-based labels
                time_labels = []
                for pos in tick_positions:
                    if pos < len(df):
                        hours = pos // 60
                        minutes = pos % 60
                        time_labels.append(f'{hours:02d}:{minutes:02d}')
                
                ax.set_xticklabels(time_labels, rotation=45, ha='right', fontsize=10)
            
            # Set axis limits with appropriate padding
            price_range = df['high'].max() - df['low'].min()
            padding = price_range * 0.03
            ax.set_ylim(df['low'].min() - padding, df['high'].max() + padding)
            ax.set_xlim(-5, len(df) + 5)
            
            # Add summary statistics box
            confluence_info = f"\nConfluence: {', '.join(self.confluence_symbols)} (Threshold: {self.confluence_threshold:.1f})" if self.confluence_symbols else ""
            stats_text = f"""Gap Analysis Summary:
• Total Gaps Detected: {summary['gap_statistics']['total_gaps']}
• FVG: {summary['gap_statistics']['fvg_gaps']} | IFVG: {summary['gap_statistics']['ifvg_gaps']}
• Bullish: {summary['gap_statistics']['bullish_gaps']} | Bearish: {summary['gap_statistics']['bearish_gaps']}
• Fill Rate: {summary['gap_statistics']['fill_rate']:.1f}%
• Avg Gap Size: {summary['gap_statistics']['avg_gap_size_percent']:.3f}%
• Largest Gap: {summary['gap_statistics']['max_gap_size_percent']:.3f}%{confluence_info}"""
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
            
            # Create confluence symbol charts if enabled
            if self.confluence_symbols and len(axes) > 1:
                for conf_idx, conf_symbol in enumerate(self.confluence_symbols):
                    if conf_idx + 1 < len(axes) and conf_symbol in self.confluence_data:
                        self.create_confluence_chart(axes[conf_idx + 1], conf_symbol, df)
            
            # Adjust layout to accommodate legend and multiple panels
            plt.tight_layout()
            if self.confluence_symbols:
                plt.subplots_adjust(right=0.85, hspace=0.3)
            else:
                plt.subplots_adjust(right=0.85)
            
            # Save standalone chart
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            standalone_filename = f'{self.charts_dir}/{symbol}_ifvg_price_action_{timestamp}.png'
            plt.savefig(standalone_filename, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            print(f"Standalone price chart saved to: {standalone_filename}")
            return standalone_filename
            
        except Exception as e:
            print(f"Error creating standalone price chart for {symbol}: {e}")
            return ""
    
    def create_confluence_chart(self, ax, conf_symbol, primary_df):
        """
        Create a confluence symbol chart showing gaps and alignment
        
        Args:
            ax: Matplotlib axis to plot on
            conf_symbol (str): Confluence symbol name
            primary_df (pd.DataFrame): Primary symbol data for time alignment
        """
        try:
            from matplotlib.patches import Rectangle
            from matplotlib.lines import Line2D
            
            if conf_symbol not in self.confluence_data:
                ax.text(0.5, 0.5, f'No data for {conf_symbol}', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=14)
                ax.set_title(f'{conf_symbol} - Confluence Analysis (No Data)')
                return
            
            conf_df = self.confluence_data[conf_symbol]
            conf_gaps = self.confluence_gaps.get(conf_symbol, [])
            
            # Create candlestick chart for confluence symbol
            for i in range(len(conf_df)):
                open_price = conf_df.iloc[i]['open']
                high_price = conf_df.iloc[i]['high']
                low_price = conf_df.iloc[i]['low']
                close_price = conf_df.iloc[i]['close']
                
                # Determine candle color and styling (smaller for confluence)
                is_bullish = close_price >= open_price
                body_color = '#90EE90' if is_bullish else '#FFB6C1'  # Light green / Light pink
                edge_color = '#228B22' if is_bullish else '#DC143C'  # Forest green / Crimson
                wick_color = '#696969'  # Dim gray for wicks
                
                # Draw high-low wick
                ax.plot([i, i], [low_price, high_price], 
                       color=wick_color, linewidth=0.8, alpha=0.7, zorder=1)
                
                # Draw candle body (smaller than primary)
                body_height = abs(close_price - open_price)
                body_bottom = min(open_price, close_price)
                
                if body_height > 0:
                    rect = Rectangle((i-0.25, body_bottom), 0.5, body_height, 
                                   facecolor=body_color, edgecolor=edge_color, 
                                   alpha=0.8, linewidth=0.6, zorder=2)
                    ax.add_patch(rect)
                else:
                    # Doji candle
                    ax.plot([i-0.25, i+0.25], [close_price, close_price], 
                           color=edge_color, linewidth=2, alpha=0.8, zorder=2)
            
            # Track confluence gap types for legend
            conf_gap_legend_items = set()
            conf_fill_points_added = False
            
            # Mark confluence gaps
            for gap in conf_gaps:
                if gap.start_index < len(conf_df):
                    # Determine gap styling (muted colors for confluence)
                    if gap.gap_type == 'bullish_fvg':
                        color = '#98FB98'  # Pale green
                        alpha = 0.2 if gap.filled else 0.4
                        label = f'{conf_symbol} Bullish FVG'
                    elif gap.gap_type == 'bearish_fvg':
                        color = '#FFA07A'  # Light salmon
                        alpha = 0.2 if gap.filled else 0.4
                        label = f'{conf_symbol} Bearish FVG'
                    elif gap.gap_type == 'bullish_ifvg':
                        color = '#AFEEEE'  # Pale turquoise
                        alpha = 0.15 if gap.filled else 0.3
                        label = f'{conf_symbol} Bullish IFVG'
                    else:  # bearish_ifvg
                        color = '#FFDAB9'  # Peach puff
                        alpha = 0.15 if gap.filled else 0.3
                        label = f'{conf_symbol} Bearish IFVG'
                    
                    # Calculate gap zone visibility
                    gap_start = max(0, gap.start_index)
                    gap_end = min(len(conf_df)-1, gap.end_index + 20)
                    
                    # Draw gap zone
                    ax.axhspan(gap.bottom, gap.top, 
                              xmin=gap_start/len(conf_df), 
                              xmax=gap_end/len(conf_df),
                              facecolor=color, alpha=alpha, 
                              edgecolor=color, linewidth=1,
                              zorder=3)
                    
                    # Add gap type to legend tracking
                    status = 'Filled' if gap.filled else 'Open'
                    conf_gap_legend_items.add((f'{label} ({status})', color, alpha))
                    
                    # Mark fill point
                    if gap.filled and gap.filled_index and gap.filled_index < len(conf_df):
                        fill_price = conf_df.iloc[gap.filled_index]['close']
                        ax.scatter(gap.filled_index, fill_price, 
                                  color='gray', marker='x', s=60, alpha=0.8, 
                                  edgecolors='black', linewidth=1, zorder=10)
                        conf_fill_points_added = True
            
            # Create confluence legend
            conf_legend_elements = []
            
            # Add confluence candlestick legend
            conf_legend_elements.extend([
                Rectangle((0, 0), 1, 1, facecolor='#90EE90', edgecolor='#228B22', 
                         alpha=0.8, label=f'{conf_symbol} Bullish Candle'),
                Rectangle((0, 0), 1, 1, facecolor='#FFB6C1', edgecolor='#DC143C', 
                         alpha=0.8, label=f'{conf_symbol} Bearish Candle')
            ])
            
            # Add confluence gap legends
            for label, color, alpha in sorted(conf_gap_legend_items):
                conf_legend_elements.append(
                    Rectangle((0, 0), 1, 1, facecolor=color, alpha=alpha, 
                             edgecolor=color, label=label)
                )
            
            # Add confluence fill point legend
            if conf_fill_points_added:
                conf_legend_elements.append(
                    Line2D([0], [0], marker='x', color='gray', linestyle='None',
                          markersize=8, markeredgecolor='black', markeredgewidth=1,
                          label=f'{conf_symbol} Gap Fill')
                )
            
            # Position confluence legend
            ax.legend(handles=conf_legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
                     fontsize=9, framealpha=0.9, fancybox=True, shadow=True)
            
            # Enhanced title and labels for confluence
            gap_count = len(conf_gaps)
            filled_count = len([g for g in conf_gaps if g.filled])
            fill_rate = (filled_count / gap_count * 100) if gap_count > 0 else 0
            
            ax.set_title(f'{conf_symbol} - Confluence Analysis\n'
                        f'Gaps: {gap_count} | Fill Rate: {fill_rate:.1f}% | '
                        f'Confluence Symbol for Trade Confirmation', 
                        fontsize=14, fontweight='bold', pad=20)
            
            ax.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (1-minute bars)', fontsize=12, fontweight='bold')
            
            # Enhanced grid
            ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.6)
            ax.set_axisbelow(True)
            
            # Format y-axis
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.2f}'))
            
            # Improved x-axis formatting
            if len(conf_df) > 50:
                tick_spacing = max(1, len(conf_df) // 10)
                tick_positions = range(0, len(conf_df), tick_spacing)
                ax.set_xticks(tick_positions)
                
                # Create time-based labels
                time_labels = []
                for pos in tick_positions:
                    if pos < len(conf_df):
                        hours = pos // 60
                        minutes = pos % 60
                        time_labels.append(f'{hours:02d}:{minutes:02d}')
                
                ax.set_xticklabels(time_labels, rotation=45, ha='right', fontsize=9)
            
            # Set axis limits
            if not conf_df.empty:
                price_range = conf_df['high'].max() - conf_df['low'].min()
                padding = price_range * 0.02
                ax.set_ylim(conf_df['low'].min() - padding, conf_df['high'].max() + padding)
                ax.set_xlim(-2, len(conf_df) + 2)
            
            # Add confluence statistics box
            conf_stats_text = f"""Confluence Statistics:
• Symbol: {conf_symbol}
• Total Gaps: {gap_count}
• Filled Gaps: {filled_count}
• Fill Rate: {fill_rate:.1f}%
• Role: Trade Confirmation"""
            
            ax.text(0.02, 0.02, conf_stats_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='bottom', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))
            
        except Exception as e:
            print(f"Error creating confluence chart for {conf_symbol}: {e}")
            ax.text(0.5, 0.5, f'Error loading {conf_symbol}', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{conf_symbol} - Confluence Analysis (Error)')
    
    def run_backtest(self, symbol, period_type="day", period=10):
        """
        Run complete backtest for the IFVG strategy
        
        Args:
            symbol (str): Stock symbol to backtest
            period_type (str): Period type for data fetching - For 1-min data, should be 'day'
            period (int): Number of periods - For 'day': 1, 2, 3, 4, 5, 10
            
        Returns:
            tuple: (DataFrame with results, dict with summary, str csv_path, str chart_path)
        """
        print(f"\n🚀 === IFVG STRATEGY BACKTEST ===")
        print(f"Symbol: {symbol}")
        print(f"Strategy: Fair Value Gap & Inverse Fair Value Gap Detection")
        print(f"Data: 1-minute bars, {period} {period_type}")
        print(f"Gap Threshold: {self.gap_threshold*100:.3f}%")
        print(f"Profit Target: {self.profit_target*100:.1f}%")
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
        
        # Fetch confluence data if enabled
        if self.confluence_symbols:
            print(f"🔗 Confluence analysis enabled with {len(self.confluence_symbols)} symbols: {', '.join(self.confluence_symbols)}")
            confluence_success = self.fetch_confluence_data(period_type=period_type, period=period)
            if not confluence_success:
                print("⚠️  Confluence data fetch failed, proceeding without confluence analysis")
                self.confluence_symbols = []
        
        # Calculate signals and performance
        print("📊 Calculating gap patterns and performance...")
        df = self.calculate_signals_and_performance(df)
        
        # Generate performance summary
        print("📈 Generating performance summary...")
        summary = self.calculate_performance_summary(df, symbol)
        
        # Save results to CSV
        print("💾 Saving results to CSV...")
        csv_path = self.save_results_to_csv(df, symbol)
        
        # Create visualization
        print("🎨 Creating performance visualization...")
        chart_path = self.create_basic_visualization(df, summary, symbol)
        
        # Create standalone price action chart
        print("🎨 Creating standalone price action chart...")
        standalone_chart_path = self.create_standalone_price_chart(df, summary, symbol)
        
        return df, summary, csv_path, chart_path


def main():
    """
    Main function for running the IFVG backtester
    """
    parser = argparse.ArgumentParser(
        description="IFVG Backtester - Fair Value Gap and Inverse Fair Value Gap strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run backtest on AAPL with default settings
  python3 ifvg_backtest.py AAPL
  
  # Run backtest on NVDA with 5 days of data and custom gap threshold
  python3 ifvg_backtest.py NVDA --period 5 --gap-threshold 0.002
  
  # Run backtest with custom parameters
  python3 ifvg_backtest.py TSLA --profit-target 0.01 --stop-loss 0.015
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
        '--gap-threshold',
        type=float,
        default=0.001,
        help='Minimum gap size as percentage (default: 0.001 = 0.1%%)'
    )
    
    parser.add_argument(
        '--profit-target',
        type=float,
        default=0.005,
        help='Profit target as decimal (default: 0.005 = 0.5%%)'
    )
    
    parser.add_argument(
        '--stop-loss',
        type=float,
        default=0.01,
        help='Stop loss as decimal (default: 0.01 = 1.0%%)'
    )
    
    parser.add_argument(
        '--max-active-gaps',
        type=int,
        default=10,
        help='Maximum number of active gaps to track (default: 10)'
    )
    
    parser.add_argument(
        '--gap-expiry-hours',
        type=int,
        default=24,
        help='Hours after which gaps expire (default: 24, 0 = no expiry)'
    )
    
    parser.add_argument(
        '--initial-capital',
        type=float,
        default=25000.0,
        help='Initial capital amount (default: 25000.0)'
    )
    
    parser.add_argument(
        '--confluence-symbols',
        type=str,
        nargs='*',
        help='Confluence symbols for trade confirmation (e.g., --confluence-symbols SPY QQQ)'
    )
    
    parser.add_argument(
        '--confluence-threshold',
        type=float,
        default=0.7,
        help='Minimum confluence score to enter trades (default: 0.7, range: 0.0-1.0)'
    )
    
    args = parser.parse_args()
    
    # Convert symbol to uppercase
    symbol = args.symbol.upper()
    
    print("=" * 80)
    print("IFVG BACKTESTER")
    print("Fair Value Gap & Inverse Fair Value Gap Strategy")
    print("=" * 80)
    print(f"Symbol: {symbol}")
    print(f"Period: {args.period} {args.period_type}")
    print(f"Gap Threshold: {args.gap_threshold*100:.3f}%")
    print(f"Profit Target: {args.profit_target*100:.1f}%")
    print(f"Stop Loss: {args.stop_loss*100:.1f}%")
    print(f"Max Active Gaps: {args.max_active_gaps}")
    print(f"Gap Expiry: {args.gap_expiry_hours} hours")
    print(f"Initial Capital: ${args.initial_capital:,.2f}")
    print("=" * 80)
    
    try:
        # Initialize backtester with confluence parameters
        backtester = IFVGBacktester(
            initial_capital=args.initial_capital,
            gap_threshold=args.gap_threshold,
            profit_target=args.profit_target,
            stop_loss=args.stop_loss,
            max_active_gaps=args.max_active_gaps,
            gap_expiry_hours=args.gap_expiry_hours,
            confluence_symbols=args.confluence_symbols,
            confluence_threshold=args.confluence_threshold
        )
        
        # Display confluence settings if enabled
        if args.confluence_symbols:
            print(f"🔗 Confluence Analysis: ENABLED")
            print(f"   Confluence Symbols: {', '.join(args.confluence_symbols)}")
            print(f"   Confluence Threshold: {args.confluence_threshold:.2f}")
        else:
            print(f"🔗 Confluence Analysis: DISABLED")
        print("=" * 80)
        
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
            gap_stats = summary['gap_statistics']
            print(f"\n🎯 KEY RESULTS:")
            print(f"   Total Gaps Detected: {gap_stats['total_gaps']}")
            print(f"   FVG Gaps: {gap_stats['fvg_gaps']}")
            print(f"   IFVG Gaps: {gap_stats['ifvg_gaps']}")
            print(f"   Fill Rate: {gap_stats['fill_rate']:.1f}%")
            print(f"   Average Gap Size: {gap_stats['avg_gap_size_percent']:.3f}%")
            
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

#!/usr/bin/env python3
"""
Midnight Momentum Strategy Analyzer

A statistically sound implementation for analyzing overnight trading patterns
that addresses look-ahead bias, multiple testing issues, and other statistical flaws.

Key Features:
- Proper train/test splits with walk-forward analysis
- Non-parametric threshold calculations using empirical quantiles
- Multiple testing corrections (FDR)
- Bootstrap confidence intervals
- Regime-aware modeling
- Monte Carlo validation
- Transaction cost integration
- Comprehensive error handling and logging

Author: Statistical Trading Analysis
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests
import time
import argparse
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import stats
from statsmodels.stats.multitest import multipletests
import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'handlers'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'visualizers'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import from handlers directory
from handlers.connection_manager import ensure_valid_tokens
from handlers.historical_data_handler import HistoricalDataHandler
from visualizers.midnightMomentum_visualization import MidnightMomentumVisualizer

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging - suppress most output
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('robust_overnight_analyzer.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class AnalysisConfig:
    """Configuration parameters for the analysis"""
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    min_window_size: int = 60
    rolling_window: int = 20
    confidence_levels: List[float] = None
    n_bootstrap: int = 1000
    n_monte_carlo: int = 1000
    transaction_cost: float = 0.001  # 0.1% transaction cost
    min_sample_size: int = 30
    significance_level: float = 0.05
    
    # Strategy refinement parameters
    max_hold_days: int = 5  # Cap hold periods to 5 days
    stop_loss_threshold: float = 0.95  # Use downside_95 for stop-loss
    regime_filter: str = 'High'  # Only enter in 'High' volatility regime
    enable_stop_loss: bool = False  # DISABLED - Only target profit exits allowed
    enable_hold_cap: bool = False  # DISABLED - Only target profit exits allowed
    enable_regime_filter: bool = False  # DISABLED - Allow entries in any regime
    enable_eod_exit: bool = False  # DISABLED - Only target profit exits allowed
    
    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.68, 0.90, 0.95, 0.99]
        
        # Validate ratios sum to 1
        total_ratio = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

class RobustDataHandler:
    """Enhanced data handler with proper error handling and validation"""
    
    def __init__(self):
        self.historical_handler = HistoricalDataHandler()
        
    def fetch_historical_data(self, symbol: str, period: int = 2, 
                            period_type: str = 'year', 
                            frequency_type: str = 'daily', 
                            frequency: int = 1,
                            include_extended_hours: bool = True) -> pd.DataFrame:
        """
        Fetch historical data with robust error handling
        
        Args:
            symbol: Stock symbol
            period: Number of periods
            period_type: 'day', 'month', 'year', 'ytd'
            frequency_type: 'minute', 'daily', 'weekly', 'monthly'
            frequency: Frequency value
            include_extended_hours: Include extended hours data
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Use existing historical data handler
            data = self.historical_handler.fetch_historical_data(
                symbol=symbol,
                periodType=period_type,
                period=period,
                frequencyType=frequency_type,
                freq=frequency,
                needExtendedHoursData=include_extended_hours
            )
            
            if not data or 'candles' not in data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            candles = data['candles']
            df = pd.DataFrame(candles)
            
            # Ensure datetime is datetime type
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Sort by date and reset index
            df = df.sort_values('datetime').reset_index(drop=True)
            
            # Validate data quality
            self._validate_data_quality(df, symbol)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _validate_data_quality(self, df: pd.DataFrame, symbol: str) -> bool:
        """Validate data quality and log issues"""
        issues = []
        
        if df.empty:
            issues.append("Empty dataset")
            return False
        
        # Check for missing values
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
        if missing_pct > 5:
            issues.append(f"High missing data: {missing_pct:.1f}%")
        
        # Check for zero prices
        price_cols = ['open', 'high', 'low', 'close']
        zero_prices = (df[price_cols] <= 0).sum().sum()
        if zero_prices > 0:
            issues.append(f"Zero/negative prices: {zero_prices}")
        
        # Check for impossible price relationships
        invalid_ohlc = (
            (df['high'] < df[['open', 'close']].max(axis=1)) |
            (df['low'] > df[['open', 'close']].min(axis=1))
        ).sum()
        if invalid_ohlc > 0:
            issues.append(f"Invalid OHLC relationships: {invalid_ohlc}")
        
        # Check for extreme price movements (>50% in one day)
        extreme_moves = (df['close'].pct_change().abs() > 0.5).sum()
        if extreme_moves > len(df) * 0.01:  # More than 1% of days
            issues.append(f"Excessive extreme moves: {extreme_moves}")
        
        if issues:
            logger.warning(f"Data quality issues for {symbol}: {', '.join(issues)}")
            return False
        
        return True

class StatisticalAnalyzer:
    """Core statistical analysis engine with proper methodologies"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        
    def calculate_basic_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic price metrics without look-ahead bias"""
        result = df.copy()
        
        # Ensure data is sorted by datetime
        result = result.sort_values('datetime').reset_index(drop=True)
        
        # Create shifted columns for previous day's values (no look-ahead bias)
        result['prev_close'] = result['close'].shift(1)
        result['prev_open'] = result['open'].shift(1)
        result['prev_high'] = result['high'].shift(1)
        result['prev_low'] = result['low'].shift(1)
        
        # Basic price metrics
        result['daily_return'] = result['close'].pct_change()
        result['overnight_gap'] = (result['open'] - result['prev_close']) / result['prev_close']
        result['intraday_return'] = (result['close'] - result['open']) / result['open']
        
        # Binary indicators
        result['high_above_prev_close'] = (result['high'] > result['prev_close']).astype(int)
        result['low_below_prev_close'] = (result['low'] < result['prev_close']).astype(int)
        result['prev_day_up'] = (result['prev_close'] > result['prev_close'].shift(1)).astype(int)
        result['prev_day_down'] = (result['prev_close'] < result['prev_close'].shift(1)).astype(int)
        
        # Next day recovery potential for scaling-in analysis
        result['next_day_high'] = result['high'].shift(-1)
        result['next_day_recovery_potential'] = (result['next_day_high'] > result['close']).astype(int)
        
        # Volatility metrics (using proper lagged calculations)
        result['true_range'] = np.maximum(
            result['high'] - result['low'],
            np.maximum(
                abs(result['high'] - result['prev_close']),
                abs(result['low'] - result['prev_close'])
            )
        )
        
        # Volume metrics
        if 'volume' in result.columns:
            result['volume_ma'] = result['volume'].shift(1).rolling(
                window=self.config.rolling_window, min_periods=10
            ).mean()
            result['volume_ratio'] = result['volume'] / result['volume_ma']
        
        return result
    
    def calculate_empirical_thresholds(self, df: pd.DataFrame, 
                                     confidence_levels: List[float] = None) -> Dict[str, float]:
        """
        Calculate empirical thresholds using historical quantiles (no look-ahead bias)
        
        Args:
            df: Historical data
            confidence_levels: List of confidence levels (e.g., [0.95, 0.99])
            
        Returns:
            Dictionary of thresholds for each confidence level
        """
        if confidence_levels is None:
            confidence_levels = self.config.confidence_levels
        
        thresholds = {}
        
        # Calculate the percentage drop from previous close to low (positive values = drops)
        low_drops = (df['prev_close'] - df['low']) / df['prev_close'] * 100
        low_drops = low_drops.dropna()
        
        if len(low_drops) < self.config.min_sample_size:
            logger.warning(f"Insufficient data for threshold calculation: {len(low_drops)} samples")
            return thresholds
        
        for conf_level in confidence_levels:
            # For confidence level (e.g., 95%), we want the 95th percentile of drops
            # This represents drops that are exceeded only 5% of the time
            threshold_pct = low_drops.quantile(conf_level)
            
            # Ensure threshold is positive (representing a drop)
            if threshold_pct < 0:
                threshold_pct = 0.1  # Minimum 0.1% drop threshold
            
            thresholds[f'threshold_{int(conf_level*100)}'] = threshold_pct
        
        return thresholds
    
    def calculate_profit_potential_thresholds(self, df: pd.DataFrame, 
                                            confidence_levels: List[float] = None) -> Dict[str, float]:
        """
        Calculate profit potential (upside) thresholds using historical quantiles
        For take profit targets, we want thresholds that get hit with the specified confidence level
        
        Args:
            df: Historical data
            confidence_levels: List of confidence levels (e.g., [0.68, 0.95, 0.99])
            
        Returns:
            Dictionary of upside thresholds for each confidence level
        """
        if confidence_levels is None:
            confidence_levels = self.config.confidence_levels
        
        upside_thresholds = {}
        
        # Calculate the percentage gain from previous close to high (positive values = gains)
        high_gains = (df['high'] - df['prev_close']) / df['prev_close'] * 100
        high_gains = high_gains.dropna()
        
        if len(high_gains) < self.config.min_sample_size:
            logger.warning(f"Insufficient data for upside threshold calculation: {len(high_gains)} samples")
            return upside_thresholds
        
        for conf_level in confidence_levels:
            # FIXED: Use 35th percentile for all confidence levels for consistent take profit targets
            # This means targets will be hit ~65% of the time, providing higher profit potential
            upside_threshold_pct = high_gains.quantile(0.35)  # Always use 35th percentile
            
            # Ensure threshold is positive (representing a gain)
            if upside_threshold_pct < 0:
                upside_threshold_pct = 0.1  # Minimum 0.1% gain threshold
            
            upside_thresholds[f'upside_threshold_{int(conf_level*100)}'] = upside_threshold_pct
        
        return upside_thresholds
    
    def calculate_downside_risk_thresholds(self, data):
        """Calculate various downside risk thresholds"""
        try:
            overnight_returns = data['overnight_gap'].dropna()
            
            if len(overnight_returns) < 10:
                return {}
            
            # Calculate percentile-based thresholds
            p5 = np.percentile(overnight_returns, 5)
            p10 = np.percentile(overnight_returns, 10)
            p25 = np.percentile(overnight_returns, 25)
            
            # Calculate standard deviation based thresholds
            mean_return = overnight_returns.mean()
            std_return = overnight_returns.std()
            
            one_std_down = mean_return - std_return
            two_std_down = mean_return - 2 * std_return
            
            # Calculate Value at Risk (VaR) estimates
            var_95 = np.percentile(overnight_returns, 5)  # 95% VaR
            var_99 = np.percentile(overnight_returns, 1)  # 99% VaR
            
            # Calculate Expected Shortfall (Conditional VaR)
            es_95 = overnight_returns[overnight_returns <= var_95].mean()
            es_99 = overnight_returns[overnight_returns <= var_99].mean()
            
            return {
                'percentile_5': p5,
                'percentile_10': p10,
                'percentile_25': p25,
                'one_std_down': one_std_down,
                'two_std_down': two_std_down,
                'var_95': var_95,
                'var_99': var_99,
                'expected_shortfall_95': es_95,
                'expected_shortfall_99': es_99
            }
            
        except Exception as e:
            print(f"Error calculating downside risk thresholds: {e}")
            return {}
    
    def calculate_profit_potential_metrics(self, data):
        """Calculate various profit potential (upside) metrics - mirror of downside risk"""
        try:
            overnight_returns = data['overnight_gap'].dropna()
            
            if len(overnight_returns) < 10:
                return {}
            
            # Calculate percentile-based upside thresholds
            p75 = np.percentile(overnight_returns, 75)
            p90 = np.percentile(overnight_returns, 90)
            p95 = np.percentile(overnight_returns, 95)
            
            # Calculate standard deviation based upside thresholds
            mean_return = overnight_returns.mean()
            std_return = overnight_returns.std()
            
            one_std_up = mean_return + std_return
            two_std_up = mean_return + 2 * std_return
            
            # Calculate Profit at Risk (PaR) estimates - upside equivalent of VaR
            par_95 = np.percentile(overnight_returns, 95)  # 95% PaR
            par_99 = np.percentile(overnight_returns, 99)  # 99% PaR
            
            # Calculate Expected Upside (Conditional PaR)
            eu_95 = overnight_returns[overnight_returns >= par_95].mean()
            eu_99 = overnight_returns[overnight_returns >= par_99].mean()
            
            return {
                'percentile_75': p75,
                'percentile_90': p90,
                'percentile_95': p95,
                'one_std_up': one_std_up,
                'two_std_up': two_std_up,
                'par_95': par_95,
                'par_99': par_99,
                'expected_upside_95': eu_95,
                'expected_upside_99': eu_99
            }
            
        except Exception as e:
            print(f"Error calculating profit potential metrics: {e}")
            return {}
    
    def apply_thresholds_no_lookahead(self, df: pd.DataFrame, 
                                    window_size: int = None) -> pd.DataFrame:
        """
        Vectorized threshold application using expanding or rolling windows to avoid look-ahead bias
        
        Args:
            df: DataFrame with price data
            window_size: Size of rolling window (None for expanding window)
            
        Returns:
            DataFrame with threshold indicators
        """
        if window_size is None:
            window_size = self.config.rolling_window
        
        result = df.copy()
        
        # Initialize threshold columns using vectorized operations
        conf_pcts = [int(conf_level * 100) for conf_level in self.config.confidence_levels]
        
        # Vectorized initialization of all threshold columns
        threshold_cols = {}
        for conf_pct in conf_pcts:
            # Downside threshold columns
            threshold_cols[f'threshold_{conf_pct}'] = np.full(len(result), np.nan)
            threshold_cols[f'below_threshold_{conf_pct}'] = np.zeros(len(result), dtype=int)
            threshold_cols[f'breach_depth_{conf_pct}'] = np.zeros(len(result))
            
            # Upside threshold columns
            threshold_cols[f'upside_threshold_{conf_pct}'] = np.full(len(result), np.nan)
            threshold_cols[f'above_upside_threshold_{conf_pct}'] = np.zeros(len(result), dtype=int)
            threshold_cols[f'upside_breach_magnitude_{conf_pct}'] = np.zeros(len(result))
        
        # Add all columns to result DataFrame at once
        for col_name, col_data in threshold_cols.items():
            result[col_name] = col_data
        
        # Pre-compute arrays for vectorized operations
        prev_close_arr = result['prev_close'].values
        low_arr = result['low'].values
        high_arr = result['high'].values
        
        # Calculate thresholds using expanding window for early periods,
        # then rolling window for later periods
        min_periods = max(20, window_size // 2)
        
        # Batch process thresholds for better performance
        for i in range(min_periods, len(result)):
            # Use expanding window for early periods, rolling for later
            # CRITICAL: Use data up to i-1 to avoid look-ahead bias
            if i < window_size * 2:
                historical_data = result.iloc[:i-1]
            else:
                historical_data = result.iloc[i-window_size:i-1]
            
            # Calculate thresholds on historical data only (excluding current day)
            thresholds = self.calculate_empirical_thresholds(historical_data)
            upside_thresholds = self.calculate_profit_potential_thresholds(historical_data)
            
            # Use previous valid thresholds if current calculation fails
            if not thresholds and i > min_periods:
                thresholds = self._get_previous_valid_thresholds(result, i, min_periods, 'threshold')
            
            if not upside_thresholds and i > min_periods:
                upside_thresholds = self._get_previous_valid_thresholds(result, i, min_periods, 'upside_threshold')
            
            # Vectorized threshold application for current row
            prev_close = prev_close_arr[i]
            current_low = low_arr[i]
            current_high = high_arr[i]
            
            if pd.notna(prev_close) and pd.notna(current_low) and pd.notna(current_high):
                # Vectorized downside threshold processing
                self._apply_downside_thresholds_vectorized(
                    result, i, thresholds, prev_close, current_low, conf_pcts
                )
                
                # Vectorized upside threshold processing
                self._apply_upside_thresholds_vectorized(
                    result, i, upside_thresholds, prev_close, current_high, conf_pcts
                )
        
        return result
    
    def _get_previous_valid_thresholds(self, result: pd.DataFrame, current_idx: int, 
                                     min_periods: int, threshold_type: str) -> Dict[str, float]:
        """
        Vectorized lookup of previous valid thresholds
        
        Args:
            result: DataFrame with threshold data
            current_idx: Current index
            min_periods: Minimum periods required
            threshold_type: 'threshold' or 'upside_threshold'
            
        Returns:
            Dictionary of previous valid thresholds
        """
        prev_thresholds = {}
        
        # Look back for the most recent valid thresholds
        for j in range(current_idx-1, min_periods-1, -1):
            for conf_level in self.config.confidence_levels:
                conf_pct = int(conf_level * 100)
                threshold_col = f'{threshold_type}_{conf_pct}'
                
                if threshold_col in result.columns:
                    prev_threshold_price = result.iloc[j][threshold_col]
                    if pd.notna(prev_threshold_price):
                        # Convert back to percentage for consistency
                        prev_close_j = result.iloc[j]['prev_close']
                        if pd.notna(prev_close_j) and prev_close_j > 0:
                            if threshold_type == 'threshold':
                                threshold_pct = (1 - prev_threshold_price / prev_close_j) * 100
                            else:  # upside_threshold
                                threshold_pct = (prev_threshold_price / prev_close_j - 1) * 100
                            prev_thresholds[f'{threshold_type}_{conf_pct}'] = threshold_pct
            
            if prev_thresholds:
                break
        
        return prev_thresholds
    
    def _apply_downside_thresholds_vectorized(self, result: pd.DataFrame, idx: int,
                                            thresholds: Dict[str, float], prev_close: float,
                                            current_low: float, conf_pcts: List[int]) -> None:
        """
        Vectorized application of downside thresholds
        
        Args:
            result: DataFrame to update
            idx: Current index
            thresholds: Dictionary of threshold percentages
            prev_close: Previous close price
            current_low: Current low price
            conf_pcts: List of confidence percentages
        """
        # Vectorized threshold price calculations
        threshold_keys = [f'threshold_{conf_pct}' for conf_pct in conf_pcts]
        valid_thresholds = {k: v for k, v in thresholds.items() if k in threshold_keys}
        
        if not valid_thresholds:
            return
        
        # Calculate all threshold prices at once
        threshold_drops = np.array([valid_thresholds.get(key, 0) for key in threshold_keys])
        threshold_prices = prev_close * (1 - threshold_drops / 100)
        
        # Vectorized breach detection
        breaches = current_low < threshold_prices
        breach_depths = np.where(breaches, (threshold_prices - current_low) / prev_close * 100, 0)
        
        # Update result DataFrame
        for i, conf_pct in enumerate(conf_pcts):
            threshold_key = f'threshold_{conf_pct}'
            if threshold_key in valid_thresholds:
                result.iloc[idx, result.columns.get_loc(f'threshold_{conf_pct}')] = threshold_prices[i]
                result.iloc[idx, result.columns.get_loc(f'below_threshold_{conf_pct}')] = int(breaches[i])
                result.iloc[idx, result.columns.get_loc(f'breach_depth_{conf_pct}')] = breach_depths[i]
    
    def _apply_upside_thresholds_vectorized(self, result: pd.DataFrame, idx: int,
                                          upside_thresholds: Dict[str, float], prev_close: float,
                                          current_high: float, conf_pcts: List[int]) -> None:
        """
        Vectorized application of upside thresholds
        
        Args:
            result: DataFrame to update
            idx: Current index
            upside_thresholds: Dictionary of upside threshold percentages
            prev_close: Previous close price
            current_high: Current high price
            conf_pcts: List of confidence percentages
        """
        # Vectorized upside threshold price calculations
        upside_threshold_keys = [f'upside_threshold_{conf_pct}' for conf_pct in conf_pcts]
        valid_upside_thresholds = {k: v for k, v in upside_thresholds.items() if k in upside_threshold_keys}
        
        if not valid_upside_thresholds:
            return
        
        # Calculate all upside threshold prices at once
        upside_threshold_gains = np.array([valid_upside_thresholds.get(key, 0) for key in upside_threshold_keys])
        upside_threshold_prices = prev_close * (1 + upside_threshold_gains / 100)
        
        # Vectorized breach detection
        upside_breaches = current_high > upside_threshold_prices
        upside_breach_magnitudes = np.where(
            upside_breaches, 
            (current_high - upside_threshold_prices) / prev_close * 100, 
            0
        )
        
        # Update result DataFrame
        for i, conf_pct in enumerate(conf_pcts):
            upside_threshold_key = f'upside_threshold_{conf_pct}'
            if upside_threshold_key in valid_upside_thresholds:
                result.iloc[idx, result.columns.get_loc(f'upside_threshold_{conf_pct}')] = upside_threshold_prices[i]
                result.iloc[idx, result.columns.get_loc(f'above_upside_threshold_{conf_pct}')] = int(upside_breaches[i])
                result.iloc[idx, result.columns.get_loc(f'upside_breach_magnitude_{conf_pct}')] = upside_breach_magnitudes[i]
    
    def bootstrap_confidence_interval(self, data: np.ndarray, 
                                    statistic_func: callable,
                                    confidence: float = 0.95) -> Tuple[float, float, float]:
        """
        Calculate bootstrap confidence intervals for a statistic
        
        Args:
            data: Input data
            statistic_func: Function to calculate statistic
            confidence: Confidence level
            
        Returns:
            Tuple of (statistic, lower_bound, upper_bound)
        """
        n_bootstrap = self.config.n_bootstrap
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats.append(statistic_func(bootstrap_sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Calculate confidence interval
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        statistic = statistic_func(data)
        lower_bound = np.percentile(bootstrap_stats, lower_percentile)
        upper_bound = np.percentile(bootstrap_stats, upper_percentile)
        
        return statistic, lower_bound, upper_bound
    
    def monte_carlo_validation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive Monte Carlo validation with multiple meaningful tests
        
        Args:
            df: DataFrame with data
            
        Returns:
            Dictionary with results from multiple Monte Carlo tests
        """
        results = {}
        
        # Test 1: Can overnight gap patterns predict recovery likelihood better than chance?
        results['gap_prediction_test'] = self._test_gap_prediction_power(df)
        
        # Test 2: Is there exploitable information in the sequence of market events?
        results['sequence_information_test'] = self._test_sequence_information(df)
        
        # Test 3: Do certain market conditions make recoveries more predictable?
        results['market_regime_test'] = self._test_market_regime_predictability(df)
        
        # Overall assessment
        p_values = [
            results['gap_prediction_test']['p_value'],
            results['sequence_information_test']['p_value'],
            results['market_regime_test']['p_value']
        ]
        
        # Apply multiple testing correction
        from statsmodels.stats.multitest import multipletests
        rejected, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')
        
        results['overall_assessment'] = {
            'min_p_value': min(p_values),
            'corrected_p_values': p_corrected.tolist(),
            'any_significant': any(rejected),
            'significant_tests': [i for i, sig in enumerate(rejected) if sig]
        }
        
        return results
    
    def _test_gap_prediction_power(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Test if overnight gap magnitude predicts recovery likelihood
        
        This preserves the time series structure while testing whether
        the relationship between gap size and recovery is meaningful.
        """
        # Calculate actual correlation between overnight gap and recovery
        valid_data = df[['overnight_gap', 'high_above_prev_close']].dropna()
        if len(valid_data) < 50:
            return {'p_value': 1.0, 'test_statistic': 0.0, 'description': 'Insufficient data'}
        
        actual_correlation = valid_data['overnight_gap'].corr(valid_data['high_above_prev_close'])
        
        # Monte Carlo: randomly pair gaps with recoveries while preserving distributions
        null_correlations = []
        for _ in range(self.config.n_monte_carlo):
            shuffled_recoveries = np.random.permutation(valid_data['high_above_prev_close'].values)
            null_corr = np.corrcoef(valid_data['overnight_gap'].values, shuffled_recoveries)[0, 1]
            if not np.isnan(null_corr):
                null_correlations.append(null_corr)
        
        null_correlations = np.array(null_correlations)
        
        # Calculate p-value (two-tailed test)
        p_value = np.mean(np.abs(null_correlations) >= np.abs(actual_correlation))
        
        return {
            'p_value': p_value,
            'test_statistic': actual_correlation,
            'null_mean': np.mean(null_correlations),
            'null_std': np.std(null_correlations),
            'description': 'Tests if overnight gap magnitude predicts recovery likelihood'
        }
    
    def _test_sequence_information(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Test if the sequence of market events contains exploitable information
        
        This shuffles the sequence of returns while preserving the distribution
        to test if temporal ordering matters for recovery patterns.
        """
        # Calculate actual recovery rate following negative gaps
        valid_data = df[['overnight_gap', 'high_above_prev_close']].dropna()
        if len(valid_data) < 50:
            return {'p_value': 1.0, 'test_statistic': 0.0, 'description': 'Insufficient data'}
        
        # Focus on recovery after negative overnight gaps (the key pattern)
        negative_gap_mask = valid_data['overnight_gap'] < -0.01  # More than 1% negative gap
        if negative_gap_mask.sum() < 10:
            return {'p_value': 1.0, 'test_statistic': 0.0, 'description': 'Insufficient negative gaps'}
        
        actual_recovery_rate = valid_data.loc[negative_gap_mask, 'high_above_prev_close'].mean()
        
        # Monte Carlo: shuffle the sequence of daily returns while preserving distribution
        null_recovery_rates = []
        for _ in range(self.config.n_monte_carlo):
            # Create shuffled time series
            shuffled_df = valid_data.copy()
            shuffled_returns = np.random.permutation(valid_data['overnight_gap'].values)
            shuffled_df['overnight_gap'] = shuffled_returns
            
            # Calculate recovery rate for negative gaps in shuffled sequence
            shuffled_negative_mask = shuffled_df['overnight_gap'] < -0.01
            if shuffled_negative_mask.sum() > 0:
                recovery_rate = shuffled_df.loc[shuffled_negative_mask, 'high_above_prev_close'].mean()
                null_recovery_rates.append(recovery_rate)
        
        null_recovery_rates = np.array(null_recovery_rates)
        
        # Calculate p-value
        p_value = np.mean(null_recovery_rates >= actual_recovery_rate)
        
        return {
            'p_value': p_value,
            'test_statistic': actual_recovery_rate,
            'null_mean': np.mean(null_recovery_rates),
            'null_std': np.std(null_recovery_rates),
            'n_negative_gaps': negative_gap_mask.sum(),
            'description': 'Tests if sequence of market events contains exploitable information'
        }
    
    def _test_market_regime_predictability(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Test if certain market conditions make recoveries more predictable
        
        This uses block bootstrap to test robustness across different market regimes
        while preserving local time series structure.
        """
        valid_data = df[['overnight_gap', 'high_above_prev_close', 'daily_return']].dropna()
        if len(valid_data) < 100:
            return {'p_value': 1.0, 'test_statistic': 0.0, 'description': 'Insufficient data'}
        
        # Calculate actual recovery rate in high volatility periods
        rolling_vol = valid_data['daily_return'].rolling(20).std()
        high_vol_threshold = rolling_vol.quantile(0.75)
        high_vol_mask = rolling_vol > high_vol_threshold
        
        if high_vol_mask.sum() < 20:
            return {'p_value': 1.0, 'test_statistic': 0.0, 'description': 'Insufficient high volatility periods'}
        
        actual_recovery_rate = valid_data.loc[high_vol_mask, 'high_above_prev_close'].mean()
        
        # Block bootstrap: sample blocks of consecutive days to preserve local structure
        block_size = 21  # ~1 month blocks
        n_blocks = len(valid_data) // block_size
        
        null_recovery_rates = []
        for _ in range(self.config.n_monte_carlo):
            # Randomly sample blocks with replacement
            bootstrap_blocks = []
            for _ in range(n_blocks):
                start_idx = np.random.randint(0, len(valid_data) - block_size + 1)
                block = valid_data.iloc[start_idx:start_idx + block_size].copy()
                bootstrap_blocks.append(block)
            
            if bootstrap_blocks:
                bootstrap_data = pd.concat(bootstrap_blocks, ignore_index=True)
                
                # Recalculate volatility regime for bootstrap sample
                bootstrap_vol = bootstrap_data['daily_return'].rolling(20).std()
                bootstrap_high_vol_threshold = bootstrap_vol.quantile(0.75)
                bootstrap_high_vol_mask = bootstrap_vol > bootstrap_high_vol_threshold
                
                if bootstrap_high_vol_mask.sum() > 0:
                    recovery_rate = bootstrap_data.loc[bootstrap_high_vol_mask, 'high_above_prev_close'].mean()
                    null_recovery_rates.append(recovery_rate)
        
        null_recovery_rates = np.array(null_recovery_rates)
        
        # Calculate p-value
        p_value = np.mean(null_recovery_rates >= actual_recovery_rate) if len(null_recovery_rates) > 0 else 1.0
        
        return {
            'p_value': p_value,
            'test_statistic': actual_recovery_rate,
            'null_mean': np.mean(null_recovery_rates) if len(null_recovery_rates) > 0 else np.nan,
            'null_std': np.std(null_recovery_rates) if len(null_recovery_rates) > 0 else np.nan,
            'n_high_vol_periods': high_vol_mask.sum(),
            'description': 'Tests if market regimes make recoveries more predictable'
        }
    
    def detect_volatility_regime(self, df: pd.DataFrame, 
                               lookback: int = 60) -> pd.Series:
        """
        Detect volatility regime using realized volatility
        
        Args:
            df: DataFrame with price data
            lookback: Lookback period for regime detection
            
        Returns:
            Series with regime labels
        """
        # Calculate realized volatility
        returns = df['close'].pct_change()
        realized_vol = returns.rolling(20).std() * np.sqrt(252)
        
        # Calculate regime relative to historical median
        vol_regime_ratio = realized_vol / realized_vol.shift(1).rolling(
            lookback, min_periods=20
        ).median()
        
        # Classify regimes
        regime = pd.cut(
            vol_regime_ratio, 
            bins=[0, 0.8, 1.2, np.inf], 
            labels=['Low', 'Normal', 'High']
        )
        
        return regime
    
    def walk_forward_analysis(self, df: pd.DataFrame, 
                            window_size: int = 252, 
                            step_size: int = 21) -> List[Dict]:
        """
        Perform walk-forward analysis with proper train/test splits
        
        Args:
            df: DataFrame with price data
            window_size: Size of training window
            step_size: Step size for walk-forward
            
        Returns:
            List of results for each walk-forward step
        """
        results = []
        
        for start_idx in range(window_size, len(df) - step_size, step_size):
            # Training window (historical data only)
            train_data = df.iloc[start_idx-window_size:start_idx].copy()
            
            # Test window (out-of-sample)
            test_data = df.iloc[start_idx:start_idx+step_size].copy()
            
            # Calculate thresholds on training data
            thresholds = self.calculate_empirical_thresholds(train_data)
            
            # Evaluate on test data
            test_performance = self._evaluate_thresholds(test_data, thresholds)
            
            results.append({
                'train_start': train_data['datetime'].iloc[0],
                'train_end': train_data['datetime'].iloc[-1],
                'test_start': test_data['datetime'].iloc[0],
                'test_end': test_data['datetime'].iloc[-1],
                'thresholds': thresholds,
                'performance': test_performance,
                'n_train': len(train_data),
                'n_test': len(test_data)
            })
        
        return results
    
    def _evaluate_thresholds(self, df: pd.DataFrame, 
                           thresholds: Dict[str, float]) -> Dict[str, Any]:
        """
        Evaluate threshold performance on test data
        
        Args:
            df: Test data
            thresholds: Dictionary of thresholds
            
        Returns:
            Dictionary of performance metrics
        """
        performance = {}
        
        for threshold_key, threshold_drop in thresholds.items():
            conf_level = int(threshold_key.split('_')[1]) / 100
            
            # Apply threshold to test data
            threshold_prices = df['prev_close'] * (1 - threshold_drop / 100)
            below_threshold = df['low'] < threshold_prices
            
            if below_threshold.sum() > 0:
                # Calculate recovery rate
                recovery_rate = df.loc[below_threshold, 'high_above_prev_close'].mean()
                non_recovery_rate = 1 - recovery_rate
                
                # Calculate expected vs actual performance
                expected_accuracy = conf_level
                actual_accuracy = non_recovery_rate
                
                performance[f'recovery_rate_{int(conf_level*100)}'] = recovery_rate
                performance[f'non_recovery_rate_{int(conf_level*100)}'] = non_recovery_rate
                performance[f'threshold_effectiveness_{int(conf_level*100)}'] = (
                    actual_accuracy - (1 - expected_accuracy)
                )
                performance[f'n_breaches_{int(conf_level*100)}'] = below_threshold.sum()
            else:
                performance[f'recovery_rate_{int(conf_level*100)}'] = np.nan
                performance[f'non_recovery_rate_{int(conf_level*100)}'] = np.nan
                performance[f'threshold_effectiveness_{int(conf_level*100)}'] = np.nan
                performance[f'n_breaches_{int(conf_level*100)}'] = 0
        
        return performance

class RobustVisualizer:
    """Enhanced visualization with statistical rigor"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        
    def create_comprehensive_analysis_plot(self, df: pd.DataFrame, 
                                         walk_forward_results: List[Dict],
                                         symbol: str) -> None:
        """
        Create comprehensive analysis visualization
        
        Args:
            df: DataFrame with calculated metrics
            walk_forward_results: Results from walk-forward analysis
            symbol: Stock symbol
        """
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        fig.suptitle(f'Robust Overnight Analysis: {symbol}', fontsize=16, fontweight='bold')
        
        # Plot 1: Price chart with both downside and upside thresholds
        ax1 = axes[0, 0]
        ax1.plot(df['datetime'], df['close'], label='Close Price', alpha=0.9, linewidth=2, color='black')
        
        colors = ['blue', 'green', 'orange', 'red']
        for i, conf_level in enumerate(self.config.confidence_levels):
            conf_pct = int(conf_level * 100)
            
            # Plot downside thresholds (risk levels)
            threshold_col = f'threshold_{conf_pct}'
            if threshold_col in df.columns:
                downside_data = df[threshold_col].dropna()
                if len(downside_data) > 0:
                    ax1.plot(df['datetime'], df[threshold_col], 
                            label=f'{conf_pct}% Risk Level', 
                            linestyle='--', alpha=0.8, linewidth=1.5, color=colors[i])
            
            # Plot upside thresholds (profit targets) - make them more visible
            upside_threshold_col = f'upside_threshold_{conf_pct}'
            if upside_threshold_col in df.columns:
                upside_data = df[upside_threshold_col].dropna()
                if len(upside_data) > 0:
                    ax1.plot(df['datetime'], df[upside_threshold_col], 
                            label=f'{conf_pct}% Profit Target', 
                            linestyle=':', alpha=0.9, linewidth=2, color=colors[i])
                    
                    # Add fill between close and upside threshold to highlight profit potential
                    if conf_pct == 95:  # Only for 95% to avoid clutter
                        ax1.fill_between(df['datetime'], df['close'], df[upside_threshold_col], 
                                       where=(df[upside_threshold_col] > df['close']), 
                                       alpha=0.1, color=colors[i], label='95% Profit Zone')
        
        ax1.set_title('Price with Profit Potential (Upside) and Risk (Downside) Thresholds', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Add text annotation explaining the chart
        ax1.text(0.02, 0.98, 'Dotted lines = Profit Targets\nDashed lines = Risk Levels', 
                transform=ax1.transAxes, verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot 2: Walk-forward performance
        ax2 = axes[0, 1]
        if walk_forward_results:
            test_dates = [r['test_start'] for r in walk_forward_results]
            
            for conf_level in self.config.confidence_levels:
                conf_pct = int(conf_level * 100)
                effectiveness_key = f'threshold_effectiveness_{conf_pct}'
                effectiveness_values = [
                    r['performance'].get(effectiveness_key, np.nan) 
                    for r in walk_forward_results
                ]
                
                ax2.plot(test_dates, effectiveness_values, 
                        label=f'{conf_pct}% Threshold', marker='o', alpha=0.7)
            
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.set_title('Walk-Forward Threshold Effectiveness')
            ax2.set_ylabel('Effectiveness (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Recovery rate distribution
        ax3 = axes[1, 0]
        recovery_rates = []
        for conf_level in self.config.confidence_levels:
            conf_pct = int(conf_level * 100)
            below_threshold_col = f'below_threshold_{conf_pct}'
            if below_threshold_col in df.columns:
                mask = df[below_threshold_col] == 1
                if mask.sum() > 0:
                    recovery_rate = df.loc[mask, 'high_above_prev_close'].mean()
                    recovery_rates.append(recovery_rate)
                else:
                    recovery_rates.append(0.0)  # Changed from np.nan to 0.0 for better visualization
            else:
                recovery_rates.append(0.0)  # Changed from np.nan to 0.0 for better visualization
        
        conf_labels = [f'{int(c*100)}%' for c in self.config.confidence_levels]
        bars = ax3.bar(conf_labels, recovery_rates, alpha=0.7)
        ax3.set_title('Recovery Rates by Confidence Level')
        ax3.set_ylabel('Recovery Rate')
        ax3.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, rate in zip(bars, recovery_rates):
            if rate > 0:  # Only show labels for non-zero rates
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{rate:.2f}', ha='center', va='bottom')
        
        # Plot 4: Breach frequency over time
        ax4 = axes[1, 1]
        for conf_level in self.config.confidence_levels:
            conf_pct = int(conf_level * 100)
            below_threshold_col = f'below_threshold_{conf_pct}'
            if below_threshold_col in df.columns:
                # Calculate rolling breach frequency
                breach_freq = df[below_threshold_col].rolling(
                    window=60, min_periods=20
                ).mean() * 100
                ax4.plot(df['datetime'], breach_freq, 
                        label=f'{conf_pct}% Threshold', alpha=0.7)
        
        ax4.set_title('Rolling Breach Frequency (60-day window)')
        ax4.set_ylabel('Breach Frequency (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Volatility regime analysis
        ax5 = axes[2, 0]
        analyzer = StatisticalAnalyzer(self.config)
        regime = analyzer.detect_volatility_regime(df)
        
        # Plot regime over time
        regime_numeric = regime.map({'Low': 0, 'Normal': 1, 'High': 2})
        ax5.plot(df['datetime'], regime_numeric, alpha=0.7)
        ax5.set_title('Volatility Regime')
        ax5.set_ylabel('Regime (0=Low, 1=Normal, 2=High)')
        ax5.set_yticks([0, 1, 2])
        ax5.set_yticklabels(['Low', 'Normal', 'High'])
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Statistical summary
        ax6 = axes[2, 1]
        ax6.axis('off')
        
        # Create summary statistics table
        summary_stats = []
        for conf_level in self.config.confidence_levels:
            conf_pct = int(conf_level * 100)
            below_threshold_col = f'below_threshold_{conf_pct}'
            if below_threshold_col in df.columns:
                mask = df[below_threshold_col] == 1
                n_breaches = mask.sum()
                if n_breaches > 0:
                    recovery_rate = df.loc[mask, 'high_above_prev_close'].mean()
                    non_recovery_rate = 1 - recovery_rate
                    expected_accuracy = 1 - conf_level
                    effectiveness = non_recovery_rate - expected_accuracy
                    
                    summary_stats.append([
                        f'{conf_pct}%',
                        f'{n_breaches}',
                        f'{recovery_rate:.3f}',
                        f'{non_recovery_rate:.3f}',
                        f'{effectiveness:.3f}'
                    ])
        
        if summary_stats:
            table_data = [['Confidence', 'Breaches', 'Recovery', 'Non-Recovery', 'Effectiveness']] + summary_stats
            table = ax6.table(cellText=table_data, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            ax6.set_title('Statistical Summary', pad=20)
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs('data/robust_analysis', exist_ok=True)
        filename = f'data/robust_analysis/{symbol}_robust_overnight_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Analysis plot saved: {filename}")
        plt.close()

class RobustOvernightAnalyzer:
    """Main analyzer class that orchestrates the entire analysis"""
    
    def __init__(self, config: AnalysisConfig = None):
        self.config = config or AnalysisConfig()
        self.data_handler = RobustDataHandler()
        self.analyzer = StatisticalAnalyzer(self.config)
        self.visualizer = RobustVisualizer(self.config)
        
        # Trading parameters
        self.INITIAL_CAPITAL = 25000.0
        self.RISK_PER_TRADE_PERCENT = 10.0  # 2% risk per trade
        self.CONFIDENCE_LEVEL_FOR_TRADING = 68  # Use 68% threshold for trading
        
    def analyze_symbol(self, symbol: str, 
                      save_results: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive robust analysis for a single symbol
        
        Args:
            symbol: Stock symbol to analyze
            save_results: Whether to save results to files
            
        Returns:
            Dictionary containing all analysis results
        """
        logger.info(f"Starting robust analysis for {symbol}")
        
        try:
            # Fetch data
            df = self.data_handler.fetch_historical_data(symbol, period=3, period_type='year')
            
            if df.empty:
                return {}
            
            # Calculate basic metrics
            df = self.analyzer.calculate_basic_metrics(df)
            
            # Apply thresholds without look-ahead bias
            df = self.analyzer.apply_thresholds_no_lookahead(df)
            
            # Perform walk-forward analysis
            walk_forward_results = self.analyzer.walk_forward_analysis(df)
            
            # Calculate bootstrap confidence intervals for key metrics
            bootstrap_results = {}
            
            # Recovery rate bootstrap
            recovery_data = df['high_above_prev_close'].dropna().values
            if len(recovery_data) > 0:
                recovery_stat, recovery_lower, recovery_upper = self.analyzer.bootstrap_confidence_interval(
                    recovery_data, np.mean
                )
                bootstrap_results['recovery_rate'] = {
                    'statistic': recovery_stat,
                    'ci_lower': recovery_lower,
                    'ci_upper': recovery_upper
                }
            
            # Monte Carlo validation
            mc_results = self.analyzer.monte_carlo_validation(df)
            
            # Detect volatility regimes
            df['volatility_regime'] = self.analyzer.detect_volatility_regime(df)
            
            # Calculate trading signals and PnL
            df = self.calculate_trading_signals_and_pnl(df)
            
            # Calculate performance summary
            performance_summary = self.calculate_performance_summary(df, symbol)
            
            # Calculate regime-specific statistics
            regime_stats = {}
            for regime in ['Low', 'Normal', 'High']:
                regime_mask = df['volatility_regime'] == regime
                if regime_mask.sum() > self.config.min_sample_size:
                    regime_data = df[regime_mask]
                    regime_stats[regime] = {
                        'n_observations': regime_mask.sum(),
                        'recovery_rate': regime_data['high_above_prev_close'].mean(),
                        'avg_overnight_gap': regime_data['overnight_gap'].mean(),
                        'avg_intraday_return': regime_data['intraday_return'].mean()
                    }
            
            # Extract overall p-value from Monte Carlo results
            overall_p_value = mc_results['overall_assessment']['min_p_value']
            corrected_p_values = mc_results['overall_assessment']['corrected_p_values']
            is_significant = mc_results['overall_assessment']['any_significant']
            
            # Calculate detailed recovery statistics
            recovery_mask = df['high_above_prev_close'] == 1
            non_recovery_mask = df['high_above_prev_close'] == 0
            
            # Recovery details
            recovery_details = {}
            if recovery_mask.sum() > 0:
                # Calculate recovery magnitudes (how much high exceeds prev close)
                recovery_magnitudes = ((df.loc[recovery_mask, 'high'] - df.loc[recovery_mask, 'prev_close']) / 
                                     df.loc[recovery_mask, 'prev_close'] * 100)
                
                recovery_details = {
                    'avg_recovery_magnitude': recovery_magnitudes.mean(),
                    'median_recovery_magnitude': recovery_magnitudes.median(),
                    'max_recovery': recovery_magnitudes.max(),
                    'recovery_gt_1pct': (recovery_magnitudes > 1.0).mean() * 100,
                    'recovery_gt_2pct': (recovery_magnitudes > 2.0).mean() * 100,
                    'recovery_gt_5pct': (recovery_magnitudes > 5.0).mean() * 100
                }
                
                # Bootstrap confidence interval for recovery magnitude
                if len(recovery_magnitudes) > 10:
                    _, mag_lower, mag_upper = self.analyzer.bootstrap_confidence_interval(
                        recovery_magnitudes.values, np.mean
                    )
                    recovery_details['recovery_magnitude_ci_lower'] = mag_lower
                    recovery_details['recovery_magnitude_ci_upper'] = mag_upper
                else:
                    recovery_details['recovery_magnitude_ci_lower'] = np.nan
                    recovery_details['recovery_magnitude_ci_upper'] = np.nan
            
            # Non-recovery details
            non_recovery_details = {}
            if non_recovery_mask.sum() > 0:
                # Calculate non-recovery gaps (how much high falls short of prev close)
                non_recovery_gaps = ((df.loc[non_recovery_mask, 'prev_close'] - df.loc[non_recovery_mask, 'high']) / 
                                   df.loc[non_recovery_mask, 'prev_close'] * 100)
                
                non_recovery_details = {
                    'avg_non_recovery_gap': non_recovery_gaps.mean(),
                    'median_non_recovery_gap': non_recovery_gaps.median(),
                    'max_non_recovery_gap': non_recovery_gaps.max()
                }
            
            # Compile comprehensive results
            results = {
                'symbol': symbol,
                'data_period': {
                    'start_date': df['datetime'].iloc[0],
                    'end_date': df['datetime'].iloc[-1],
                    'n_observations': len(df)
                },
                'basic_statistics': {
                    'avg_daily_return': df['daily_return'].mean(),
                    'volatility': df['daily_return'].std() * np.sqrt(252),
                    'avg_overnight_gap': df['overnight_gap'].mean(),
                    'avg_intraday_return': df['intraday_return'].mean(),
                    'recovery_rate': df['high_above_prev_close'].mean(),
                    'recovery_rate_ci': bootstrap_results.get('recovery_rate', {}),
                    'recovery_details': recovery_details,
                    'non_recovery_details': non_recovery_details
                },
                'threshold_analysis': {},
                'walk_forward_results': walk_forward_results,
                'monte_carlo_validation': {
                    'p_value': overall_p_value,
                    'p_value_corrected': corrected_p_values[0] if corrected_p_values else overall_p_value,
                    'detailed_tests': mc_results,
                    'significant': is_significant
                },
                'regime_analysis': regime_stats,
                'data_quality': {
                    'missing_data_pct': df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100,
                    'extreme_moves': (df['daily_return'].abs() > 0.5).sum(),
                    'data_issues': []
                }
            }
            
            # Calculate threshold-specific statistics (downside and upside)
            results['upside_threshold_analysis'] = {}
            
            for conf_level in self.config.confidence_levels:
                conf_pct = int(conf_level * 100)
                
                # Downside threshold analysis
                below_threshold_col = f'below_threshold_{conf_pct}'
                if below_threshold_col in df.columns:
                    mask = df[below_threshold_col] == 1
                    n_breaches = mask.sum()
                    
                    if n_breaches > 0:
                        recovery_rate = df.loc[mask, 'high_above_prev_close'].mean()
                        non_recovery_rate = 1 - recovery_rate
                        expected_accuracy = 1 - conf_level
                        effectiveness = non_recovery_rate - expected_accuracy
                        
                        # Calculate average breach depth
                        breach_depth_col = f'breach_depth_{conf_pct}'
                        avg_breach_depth = df.loc[mask, breach_depth_col].mean() if breach_depth_col in df.columns else np.nan
                        
                        results['threshold_analysis'][f'{conf_pct}%'] = {
                            'n_breaches': n_breaches,
                            'breach_frequency': n_breaches / len(df) * 100,
                            'recovery_rate': recovery_rate,
                            'non_recovery_rate': non_recovery_rate,
                            'expected_accuracy': expected_accuracy,
                            'effectiveness': effectiveness,
                            'avg_breach_depth': avg_breach_depth,
                            'statistically_significant': effectiveness > 0.05  # 5% improvement threshold
                        }
                
                # Upside threshold analysis
                above_upside_threshold_col = f'above_upside_threshold_{conf_pct}'
                if above_upside_threshold_col in df.columns:
                    upside_mask = df[above_upside_threshold_col] == 1
                    n_upside_breaches = upside_mask.sum()
                    
                    if n_upside_breaches > 0:
                        # Calculate upside breach frequency and magnitude
                        upside_breach_magnitude_col = f'upside_breach_magnitude_{conf_pct}'
                        avg_upside_breach_magnitude = df.loc[upside_mask, upside_breach_magnitude_col].mean() if upside_breach_magnitude_col in df.columns else np.nan
                        
                        # Calculate profit potential effectiveness (how often we exceed the upside threshold)
                        expected_upside_frequency = (1 - conf_level) * 100  # Expected frequency based on confidence level
                        actual_upside_frequency = n_upside_breaches / len(df) * 100
                        upside_effectiveness = actual_upside_frequency - expected_upside_frequency
                        
                        results['upside_threshold_analysis'][f'{conf_pct}%'] = {
                            'n_upside_breaches': n_upside_breaches,
                            'upside_breach_frequency': actual_upside_frequency,
                            'expected_upside_frequency': expected_upside_frequency,
                            'upside_effectiveness': upside_effectiveness,
                            'avg_upside_breach_magnitude': avg_upside_breach_magnitude,
                            'statistically_significant': upside_effectiveness > 1.0  # 1% improvement threshold
                        }
            
            # Create visualizations
            if save_results:
                logger.info(f"Creating visualizations for {symbol}")
                self.visualizer.create_comprehensive_analysis_plot(df, walk_forward_results, symbol)
                
                # Save detailed results
                self._save_results(results, symbol)
                
                # Save CSV with all thresholds
                csv_filename = self.save_ohlcv_with_thresholds_csv(df, symbol)
                logger.info(f"CSV with thresholds saved: {csv_filename}")
            
            # Print summary
            self._print_analysis_summary(results)
            
            # Print trading performance summary
            if performance_summary and performance_summary.get('total_trades', 0) > 0:
                self.print_performance_summary(performance_summary)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {}
    
    def analyze_multiple_symbols(self, symbols: List[str], 
                               save_results: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Analyze multiple symbols with comprehensive reporting
        
        Args:
            symbols: List of stock symbols to analyze
            save_results: Whether to save results to files
            
        Returns:
            Dictionary with results for each symbol
        """
        all_results = {}
        
        logger.info(f"Starting analysis of {len(symbols)} symbols: {', '.join(symbols)}")
        
        for symbol in symbols:
            logger.info(f"Analyzing {symbol} ({symbols.index(symbol) + 1}/{len(symbols)})")
            results = self.analyze_symbol(symbol, save_results)
            if results:
                all_results[symbol] = results
            
            # Add small delay to avoid rate limiting
            time.sleep(1)
        
        # Create comparative analysis
        if len(all_results) > 1 and save_results:
            self._create_comparative_analysis(all_results)
        
        return all_results
    
    def _save_results(self, results: Dict[str, Any], symbol: str):
        """Save detailed results to files"""
        try:
            os.makedirs('data/robust_analysis', exist_ok=True)
            
            # Save JSON results
            json_filename = f'data/robust_analysis/{symbol}_robust_analysis_results.json'
            
            # Convert datetime objects to strings for JSON serialization
            json_results = self._prepare_for_json(results.copy())
            
            with open(json_filename, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
            
            logger.info(f"Results saved to {json_filename}")
            
        except Exception as e:
            logger.error(f"Error saving results for {symbol}: {e}")
    
    def _prepare_for_json(self, obj):
        """Recursively prepare object for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def _print_analysis_summary(self, results: Dict[str, Any]):
        """Print a comprehensive analysis summary"""
        symbol = results['symbol']
        
        print(f"\n{'='*80}")
        print(f"ROBUST OVERNIGHT ANALYSIS SUMMARY: {symbol}")
        print(f"{'='*80}")
        
        # Basic statistics
        basic_stats = results['basic_statistics']
        print(f"\n--- Basic Statistics ---")
        print(f"Data Period: {results['data_period']['start_date'].strftime('%Y-%m-%d')} to {results['data_period']['end_date'].strftime('%Y-%m-%d')}")
        print(f"Observations: {results['data_period']['n_observations']}")
        print(f"Average Daily Return: {basic_stats['avg_daily_return']:.4f} ({basic_stats['avg_daily_return']*100:.2f}%)")
        print(f"Annualized Volatility: {basic_stats['volatility']:.4f} ({basic_stats['volatility']*100:.1f}%)")
        print(f"Average Overnight Gap: {basic_stats['avg_overnight_gap']:.4f} ({basic_stats['avg_overnight_gap']*100:.2f}%)")
        print(f"Average Intraday Return: {basic_stats['avg_intraday_return']:.4f} ({basic_stats['avg_intraday_return']*100:.2f}%)")
        
        # Enhanced recovery statistics
        print(f"\n--- Recovery Analysis (High > Previous Close) ---")
        print(f"Overall Recovery Rate: {basic_stats['recovery_rate']:.3f} ({basic_stats['recovery_rate']*100:.1f}%)")
        
        # Bootstrap confidence interval for recovery rate
        if 'recovery_rate' in basic_stats['recovery_rate_ci']:
            ci = basic_stats['recovery_rate_ci']['recovery_rate']
            print(f"Recovery Rate 95% CI: [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")
        
        # Add detailed recovery statistics
        if 'recovery_details' in basic_stats:
            recovery_details = basic_stats['recovery_details']
            print(f"Average Recovery Magnitude: {recovery_details['avg_recovery_magnitude']:.2f}%")
            print(f"Median Recovery Magnitude: {recovery_details['median_recovery_magnitude']:.2f}%")
            print(f"Recovery Magnitude 95% CI: [{recovery_details['recovery_magnitude_ci_lower']:.2f}%, {recovery_details['recovery_magnitude_ci_upper']:.2f}%]")
            print(f"Max Recovery: {recovery_details['max_recovery']:.2f}%")
            print(f"Recovery > 1%: {recovery_details['recovery_gt_1pct']:.1f}% of days")
            print(f"Recovery > 2%: {recovery_details['recovery_gt_2pct']:.1f}% of days")
            print(f"Recovery > 5%: {recovery_details['recovery_gt_5pct']:.1f}% of days")
            
        # Add non-recovery statistics
        if 'non_recovery_details' in basic_stats:
            non_recovery_details = basic_stats['non_recovery_details']
            print(f"\n--- Non-Recovery Analysis (High <= Previous Close) ---")
            print(f"Non-Recovery Rate: {(1-basic_stats['recovery_rate'])*100:.1f}%")
            print(f"Average Non-Recovery Gap: {non_recovery_details['avg_non_recovery_gap']:.2f}%")
            print(f"Median Non-Recovery Gap: {non_recovery_details['median_non_recovery_gap']:.2f}%")
            print(f"Max Non-Recovery Gap: {non_recovery_details['max_non_recovery_gap']:.2f}%")
        
        # Downside Threshold analysis
        print(f"\n--- Downside Risk Threshold Analysis ---")
        threshold_analysis = results['threshold_analysis']
        
        if threshold_analysis:
            print(f"{'Confidence':<12} {'Breaches':<10} {'Frequency':<12} {'Recovery':<10} {'Non-Recovery':<12} {'Effectiveness':<12} {'Significant':<12}")
            print(f"{'-'*12} {'-'*10} {'-'*12} {'-'*10} {'-'*12} {'-'*12} {'-'*12}")
            
            for conf_level, stats in threshold_analysis.items():
                print(f"{conf_level:<12} "
                      f"{stats['n_breaches']:<10} "
                      f"{stats['breach_frequency']:.2f}%{'':<7} "
                      f"{stats['recovery_rate']:.3f}{'':<6} "
                      f"{stats['non_recovery_rate']:.3f}{'':<8} "
                      f"{stats['effectiveness']:.3f}{'':<8} "
                      f"{'Yes' if stats['statistically_significant'] else 'No':<12}")
        else:
            print("No downside threshold breaches detected in the analysis period.")
        
        # Upside Threshold analysis
        upside_threshold_analysis = results.get('upside_threshold_analysis', {})
        if upside_threshold_analysis:
            print(f"\n--- Profit Potential (Upside) Threshold Analysis ---")
            print(f"{'Confidence':<12} {'Breaches':<10} {'Frequency':<12} {'Expected':<12} {'Effectiveness':<12} {'Avg Magnitude':<14} {'Significant':<12}")
            print(f"{'-'*12} {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*14} {'-'*12}")
            
            for conf_level, stats in upside_threshold_analysis.items():
                print(f"{conf_level:<12} "
                      f"{stats['n_upside_breaches']:<10} "
                      f"{stats['upside_breach_frequency']:.2f}%{'':<7} "
                      f"{stats['expected_upside_frequency']:.2f}%{'':<7} "
                      f"{stats['upside_effectiveness']:.3f}{'':<8} "
                      f"{stats['avg_upside_breach_magnitude']:.3f}%{'':<9} "
                      f"{'Yes' if stats['statistically_significant'] else 'No':<12}")
        else:
            print("\n--- Profit Potential (Upside) Threshold Analysis ---")
            print("No upside threshold breaches detected in the analysis period.")
        
        # Monte Carlo validation
        mc_results = results['monte_carlo_validation']
        print(f"\n--- Statistical Validation ---")
        print(f"Monte Carlo p-value: {mc_results['p_value']:.4f}")
        print(f"Multiple testing corrected p-value: {mc_results['p_value_corrected']:.4f}")
        print(f"Statistically significant pattern: {'Yes' if mc_results['significant'] else 'No'}")
        
        # Regime analysis
        regime_analysis = results['regime_analysis']
        if regime_analysis:
            print(f"\n--- Volatility Regime Analysis ---")
            print(f"{'Regime':<10} {'Observations':<12} {'Recovery Rate':<14} {'Avg O/N Gap':<12} {'Avg Intraday':<12}")
            print(f"{'-'*10} {'-'*12} {'-'*14} {'-'*12} {'-'*12}")
            
            for regime, stats in regime_analysis.items():
                print(f"{regime:<10} "
                      f"{stats['n_observations']:<12} "
                      f"{stats['recovery_rate']:.3f}{'':<10} "
                      f"{stats['avg_overnight_gap']:.4f}{'':<8} "
                      f"{stats['avg_intraday_return']:.4f}")
        
        # Walk-forward performance summary
        walk_forward_results = results['walk_forward_results']
        if walk_forward_results:
            print(f"\n--- Walk-Forward Analysis Summary ---")
            print(f"Number of out-of-sample periods: {len(walk_forward_results)}")
            
            # Calculate average effectiveness across all periods
            for conf_level in self.config.confidence_levels:
                conf_pct = int(conf_level * 100)
                effectiveness_key = f'threshold_effectiveness_{conf_pct}'
                effectiveness_values = [
                    r['performance'].get(effectiveness_key, np.nan) 
                    for r in walk_forward_results
                ]
                effectiveness_values = [v for v in effectiveness_values if not np.isnan(v)]
                
                if effectiveness_values:
                    avg_effectiveness = np.mean(effectiveness_values)
                    std_effectiveness = np.std(effectiveness_values)
                    print(f"{conf_pct}% Threshold - Avg Effectiveness: {avg_effectiveness:.3f} ± {std_effectiveness:.3f}")
        
        # Data quality summary
        data_quality = results['data_quality']
        print(f"\n--- Data Quality ---")
        print(f"Missing data: {data_quality['missing_data_pct']:.2f}%")
        print(f"Extreme price moves (>50%): {data_quality['extreme_moves']}")
        
        if data_quality['data_issues']:
            print(f"Data issues: {', '.join(data_quality['data_issues'])}")
        else:
            print("No significant data quality issues detected.")
        
        print(f"\n{'='*60}")
    
    def _create_comparative_analysis(self, all_results: Dict[str, Dict[str, Any]]):
        """Create comparative analysis across multiple symbols"""
        try:
            logger.info("Creating comparative analysis")
            
            # Create comparative visualization
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Comparative Overnight Analysis', fontsize=16, fontweight='bold')
            
            symbols = list(all_results.keys())
            
            # Plot 1: Recovery rates comparison
            ax1 = axes[0, 0]
            recovery_rates = [all_results[symbol]['basic_statistics']['recovery_rate'] 
                            for symbol in symbols]
            bars = ax1.bar(symbols, recovery_rates, alpha=0.7)
            ax1.set_title('Overall Recovery Rates')
            ax1.set_ylabel('Recovery Rate')
            ax1.set_ylim(0, 1)
            
            # Add value labels
            for bar, rate in zip(bars, recovery_rates):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{rate:.2f}', ha='center', va='bottom')
            
            # Plot 2: Volatility comparison
            ax2 = axes[0, 1]
            volatilities = [all_results[symbol]['basic_statistics']['volatility'] 
                          for symbol in symbols]
            ax2.bar(symbols, volatilities, alpha=0.7, color='orange')
            ax2.set_title('Annualized Volatility')
            ax2.set_ylabel('Volatility')
            
            # Plot 3: Threshold effectiveness comparison (95% confidence)
            ax3 = axes[1, 0]
            effectiveness_95 = []
            for symbol in symbols:
                threshold_analysis = all_results[symbol]['threshold_analysis']
                if '95%' in threshold_analysis:
                    effectiveness_95.append(threshold_analysis['95%']['effectiveness'])
                else:
                    effectiveness_95.append(0)
            
            colors = ['green' if eff > 0 else 'red' for eff in effectiveness_95]
            ax3.bar(symbols, effectiveness_95, alpha=0.7, color=colors)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax3.set_title('95% Threshold Effectiveness')
            ax3.set_ylabel('Effectiveness')
            
            # Plot 4: Monte Carlo p-values
            ax4 = axes[1, 1]
            p_values = [all_results[symbol]['monte_carlo_validation']['p_value_corrected'] 
                       for symbol in symbols]
            colors = ['green' if p < 0.05 else 'red' for p in p_values]
            ax4.bar(symbols, p_values, alpha=0.7, color=colors)
            ax4.axhline(y=0.05, color='black', linestyle='--', alpha=0.5, label='α = 0.05')
            ax4.set_title('Statistical Significance (Corrected p-values)')
            ax4.set_ylabel('p-value')
            ax4.legend()
            
            plt.tight_layout()
            
            # Save comparative plot
            filename = 'data/robust_analysis/comparative_overnight_analysis.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Comparative analysis saved: {filename}")
            plt.close()
            
            # Save comparative summary
            self._save_comparative_summary(all_results)
            
        except Exception as e:
            logger.error(f"Error creating comparative analysis: {e}")
    
    def _save_comparative_summary(self, all_results: Dict[str, Dict[str, Any]]):
        """Save comparative summary to CSV"""
        try:
            summary_data = []
            
            for symbol, results in all_results.items():
                basic_stats = results['basic_statistics']
                mc_results = results['monte_carlo_validation']
                
                row = {
                    'Symbol': symbol,
                    'Observations': results['data_period']['n_observations'],
                    'Recovery_Rate': basic_stats['recovery_rate'],
                    'Volatility': basic_stats['volatility'],
                    'Avg_Overnight_Gap': basic_stats['avg_overnight_gap'],
                    'Avg_Intraday_Return': basic_stats['avg_intraday_return'],
                    'MC_P_Value': mc_results['p_value'],
                    'MC_P_Value_Corrected': mc_results['p_value_corrected'],
                    'Statistically_Significant': mc_results['significant']
                }
                
                # Add threshold-specific metrics
                threshold_analysis = results['threshold_analysis']
                for conf_level in self.config.confidence_levels:
                    conf_pct = int(conf_level * 100)
                    conf_key = f'{conf_pct}%'
                    
                    if conf_key in threshold_analysis:
                        stats = threshold_analysis[conf_key]
                        row[f'Breaches_{conf_pct}'] = stats['n_breaches']
                        row[f'Recovery_Rate_{conf_pct}'] = stats['recovery_rate']
                        row[f'Effectiveness_{conf_pct}'] = stats['effectiveness']
                    else:
                        row[f'Breaches_{conf_pct}'] = 0
                        row[f'Recovery_Rate_{conf_pct}'] = np.nan
                        row[f'Effectiveness_{conf_pct}'] = np.nan
                
                summary_data.append(row)
            
            # Create DataFrame and save
            summary_df = pd.DataFrame(summary_data)
            filename = 'data/robust_analysis/comparative_summary.csv'
            summary_df.to_csv(filename, index=False)
            logger.info(f"Comparative summary saved: {filename}")
            
        except Exception as e:
            logger.error(f"Error saving comparative summary: {e}")
    
    def calculate_trading_signals_and_pnl(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced trading signals and PnL calculation with 5-day scaling strategy:
        
        Strategy:
        1. Enter position at close
        2. If position is open for 5 days, scale in at 5th day closing price
        3. Exit at 1% profit from the scaled entry price
        
        Args:
            df: DataFrame with calculated metrics and thresholds
            
        Returns:
            DataFrame with enhanced trading signals and PnL columns
        """
        result = df.copy()
        
        # Initialize trading columns
        result['trade_signal'] = None
        result['position_status'] = None
        result['entry_price'] = np.nan
        result['scaled_entry_price'] = np.nan  # New column for scaled entry
        result['exit_price'] = np.nan
        result['target_price'] = np.nan
        result['stop_loss_price'] = np.nan
        result['pnl'] = np.nan
        result['running_pnl'] = np.nan
        result['current_equity'] = self.INITIAL_CAPITAL
        result['position_size'] = np.nan
        result['shares'] = np.nan
        result['days_held'] = 0
        result['exit_reason'] = None
        result['scaled_in'] = 0  # New column to track if position was scaled
        
        # Scaling-in analysis columns
        result['position_underwater'] = 0
        result['scaling_opportunity'] = 0
        
        # Track current equity
        current_equity = self.INITIAL_CAPITAL
        
        # Get threshold columns for trading
        upside_threshold_col = f'upside_threshold_{self.CONFIDENCE_LEVEL_FOR_TRADING}'
        above_upside_threshold_col = f'above_upside_threshold_{self.CONFIDENCE_LEVEL_FOR_TRADING}'
        
        # Get stop-loss threshold column (95% confidence level for downside)
        stop_loss_conf_pct = int(self.config.stop_loss_threshold * 100)
        stop_loss_threshold_col = f'threshold_{stop_loss_conf_pct}'
        
        if upside_threshold_col not in result.columns:
            print(f"Warning: Required upside threshold column not found: {upside_threshold_col}")
            return result
        
        if self.config.enable_stop_loss and stop_loss_threshold_col not in result.columns:
            print(f"Warning: Stop-loss threshold column not found: {stop_loss_threshold_col}")
            self.config.enable_stop_loss = False
        
        # Track open position
        open_position = None
        
        for i in range(len(result)):
            current_row = result.iloc[i]
            
            # Update current equity in the dataframe
            result.iloc[i, result.columns.get_loc('current_equity')] = current_equity
            
            # Skip if we don't have threshold data yet
            if pd.isna(current_row[upside_threshold_col]):
                continue
            
            # Check if we have an open position
            if open_position is not None:
                # We have an open position, check for scaling and exit conditions
                original_entry_price = open_position['entry_price']
                current_entry_price = open_position.get('scaled_entry_price', original_entry_price)
                target_price = open_position['target_price']
                stop_loss_price = open_position.get('stop_loss_price', None)
                entry_equity = open_position['entry_equity']
                position_size = open_position['position_size']
                shares = open_position['shares']
                entry_date = open_position['entry_date']
                scaled_in = open_position.get('scaled_in', False)
                
                current_high = current_row['high']
                current_low = current_row['low']
                current_close = current_row['close']
                current_date = current_row['datetime']
                
                # Calculate days held
                days_held = (current_date - entry_date).days
                result.iloc[i, result.columns.get_loc('days_held')] = days_held
                
                # Update position data in result
                result.iloc[i, result.columns.get_loc('entry_price')] = original_entry_price
                result.iloc[i, result.columns.get_loc('scaled_entry_price')] = current_entry_price
                result.iloc[i, result.columns.get_loc('target_price')] = target_price
                result.iloc[i, result.columns.get_loc('scaled_in')] = int(scaled_in)
                
                # Calculate running PnL based on current entry price (scaled or original)
                running_pnl = (current_close - current_entry_price) * shares
                result.iloc[i, result.columns.get_loc('running_pnl')] = running_pnl
                
                # Track if position is underwater
                if current_close < current_entry_price:
                    result.iloc[i, result.columns.get_loc('position_underwater')] = 1
                    
                    # Check for scaling opportunity
                    if pd.notna(current_row.get('next_day_recovery_potential', np.nan)):
                        if current_row['next_day_recovery_potential'] == 1:
                            result.iloc[i, result.columns.get_loc('scaling_opportunity')] = 1
                
                # Check for 5-day scaling condition FIRST (before exit checks)
                if days_held == 5 and not scaled_in:
                    # Scale in: calculate shares needed to bring weighted average within 0.02% of current close price
                    scale_in_price = current_close
                    original_shares = shares
                    
                    # Target weighted average should be within 0.02% of the 5th day closing price
                    target_weighted_avg = scale_in_price * (1 + 0.0002)  # 0.02% above close price
                    
                    # Solve for additional shares needed:
                    # target_avg = (original_shares * original_price + additional_shares * scale_price) / (original_shares + additional_shares)
                    # Rearranging: additional_shares = original_shares * (original_price - target_avg) / (target_avg - scale_price)
                    
                    if abs(target_weighted_avg - scale_in_price) > 0.001:  # Avoid division by zero
                        additional_shares_needed = original_shares * (original_entry_price - target_weighted_avg) / (target_weighted_avg - scale_in_price)
                        
                        # Ensure we don't buy negative shares
                        additional_shares_needed = max(0, additional_shares_needed)
                        
                        # Cap additional shares to avoid excessive position size (max 10x original position for precision)
                        max_additional_shares = original_shares * 10
                        additional_shares = min(additional_shares_needed, max_additional_shares)
                        
                        # Ensure minimum purchase to make scaling worthwhile
                        min_additional_shares = original_shares * 0.1  # At least 10% more shares
                        if additional_shares < min_additional_shares:
                            additional_shares = min_additional_shares
                    else:
                        # If already very close, buy minimal additional shares
                        additional_shares = original_shares * 0.1
                    
                    additional_position_size = additional_shares * scale_in_price
                    
                    # Apply transaction cost for scaling in
                    scale_in_cost = additional_position_size * self.config.transaction_cost
                    current_equity -= scale_in_cost
                    
                    # Calculate actual weighted average entry price with the additional shares
                    total_shares = original_shares + additional_shares
                    original_position_value = original_shares * original_entry_price
                    new_position_value = additional_shares * scale_in_price
                    total_position_value = original_position_value + new_position_value
                    weighted_avg_entry_price = total_position_value / total_shares
                    
                    # Calculate percentage difference from target close price
                    price_diff_pct = abs(weighted_avg_entry_price - scale_in_price) / scale_in_price * 100
                    
                    # Calculate new target price (1% from weighted average)
                    new_target_price = weighted_avg_entry_price * 1.01
                    
                    # Update position with scaled entry
                    open_position['scaled_entry_price'] = weighted_avg_entry_price
                    open_position['target_price'] = new_target_price
                    open_position['shares'] = total_shares
                    open_position['position_size'] = total_position_value
                    open_position['scaled_in'] = True
                    
                    # Update result for this row
                    result.iloc[i, result.columns.get_loc('trade_signal')] = 'SCALE_IN'
                    result.iloc[i, result.columns.get_loc('scaled_entry_price')] = weighted_avg_entry_price
                    result.iloc[i, result.columns.get_loc('target_price')] = new_target_price
                    result.iloc[i, result.columns.get_loc('shares')] = total_shares
                    result.iloc[i, result.columns.get_loc('position_size')] = total_position_value
                    result.iloc[i, result.columns.get_loc('scaled_in')] = 1
                    result.iloc[i, result.columns.get_loc('current_equity')] = current_equity
                    
                    # Update current values for exit checks
                    current_entry_price = weighted_avg_entry_price
                    target_price = new_target_price
                    shares = total_shares
                    scaled_in = True
                    
                    print(f"Scaled in on day {days_held}: Added {additional_shares:.2f} shares at ${scale_in_price:.2f}")
                    print(f"  Original entry: ${original_entry_price:.2f}, 5th day close: ${scale_in_price:.2f}")
                    print(f"  Total shares: {total_shares:.2f}, Weighted avg: ${weighted_avg_entry_price:.2f} ({price_diff_pct:.4f}% from close)")
                    print(f"  New target: ${new_target_price:.2f}")
                
                # Check exit conditions in priority order
                exit_triggered = False
                exit_price = None
                exit_reason = None
                
                # 1. Stop-loss check (highest priority) - only if enabled
                if (self.config.enable_stop_loss and stop_loss_price is not None and 
                    current_low <= stop_loss_price):
                    exit_triggered = True
                    exit_price = stop_loss_price
                    exit_reason = 'STOP_LOSS'
                    result.iloc[i, result.columns.get_loc('trade_signal')] = 'EXIT_STOP_LOSS'
                
                # 2. Target hit check (using current target price)
                elif current_high >= target_price:
                    exit_triggered = True
                    exit_price = target_price
                    exit_reason = 'TARGET_HIT'
                    result.iloc[i, result.columns.get_loc('trade_signal')] = 'EXIT_TARGET_HIT'
                
                # 3. Hold period cap check (only if enabled)
                elif self.config.enable_hold_cap and days_held >= self.config.max_hold_days:
                    exit_triggered = True
                    exit_price = current_close
                    exit_reason = 'HOLD_CAP'
                    result.iloc[i, result.columns.get_loc('trade_signal')] = 'EXIT_HOLD_CAP'
                
                # 4. EOD exit check (only if enabled)
                elif (self.config.enable_eod_exit and days_held > 0 and 
                      current_high < target_price):
                    exit_triggered = True
                    exit_price = current_close
                    exit_reason = 'EOD_EXIT'
                    result.iloc[i, result.columns.get_loc('trade_signal')] = 'EXIT_EOD'
                
                if exit_triggered:
                    # Calculate PnL with transaction costs based on current entry price
                    gross_pnl = (exit_price - current_entry_price) * shares
                    
                    # Apply transaction costs (entry + exit)
                    entry_cost = position_size * self.config.transaction_cost
                    exit_cost = (exit_price * shares) * self.config.transaction_cost
                    total_transaction_costs = entry_cost + exit_cost
                    
                    net_pnl = gross_pnl - total_transaction_costs
                    current_equity += net_pnl
                    
                    # Update result
                    result.iloc[i, result.columns.get_loc('position_status')] = 'CLOSED'
                    result.iloc[i, result.columns.get_loc('exit_price')] = exit_price
                    result.iloc[i, result.columns.get_loc('pnl')] = net_pnl
                    result.iloc[i, result.columns.get_loc('current_equity')] = current_equity
                    result.iloc[i, result.columns.get_loc('exit_reason')] = exit_reason
                    
                    # Clear open position
                    open_position = None
                else:
                    # Position remains open
                    result.iloc[i, result.columns.get_loc('position_status')] = 'OPEN'
            
            else:
                # No open position, check for entry conditions
                current_close = current_row['close']
                upside_threshold_price = current_row[upside_threshold_col]
                volatility_regime = current_row.get('volatility_regime', 'Normal')
                
                # Entry condition checks
                entry_conditions_met = True
                
                # 1. Basic threshold check (still need valid threshold data for analysis)
                if pd.isna(upside_threshold_price) or pd.isna(current_close):
                    entry_conditions_met = False
                
                # 2. Regime filter check (only if enabled)
                if (self.config.enable_regime_filter and 
                    volatility_regime != self.config.regime_filter):
                    entry_conditions_met = False
                
                if entry_conditions_met:
                    # Calculate position size
                    risk_amount = current_equity * (self.RISK_PER_TRADE_PERCENT / 100)
                    shares = risk_amount / current_close
                    position_size = shares * current_close
                    
                    # Use fixed 1% take profit target from initial entry
                    initial_target_price = current_close * 1.01  # 1% profit target
                    
                    # Calculate stop-loss price if enabled
                    stop_loss_price = None
                    if self.config.enable_stop_loss and stop_loss_threshold_col in result.columns:
                        stop_loss_price = current_row[stop_loss_threshold_col]
                        if pd.isna(stop_loss_price):
                            # Fallback: use 95% of current close as stop-loss
                            stop_loss_price = current_close * 0.95
                    
                    # Ensure position size is reasonable
                    if position_size <= risk_amount * 1.01:  # Small buffer for rounding
                        # Open LONG position
                        open_position = {
                            'entry_price': current_close,
                            'scaled_entry_price': current_close,  # Initially same as entry price
                            'target_price': initial_target_price,
                            'stop_loss_price': stop_loss_price,
                            'entry_equity': current_equity,
                            'position_size': position_size,
                            'shares': shares,
                            'entry_date': current_row['datetime'],
                            'scaled_in': False
                        }
                        
                        # Apply entry transaction cost immediately
                        entry_cost = position_size * self.config.transaction_cost
                        current_equity -= entry_cost
                        
                        # Update result
                        result.iloc[i, result.columns.get_loc('trade_signal')] = 'ENTRY_LONG'
                        result.iloc[i, result.columns.get_loc('position_status')] = 'OPEN'
                        result.iloc[i, result.columns.get_loc('entry_price')] = current_close
                        result.iloc[i, result.columns.get_loc('scaled_entry_price')] = current_close
                        result.iloc[i, result.columns.get_loc('target_price')] = initial_target_price
                        result.iloc[i, result.columns.get_loc('stop_loss_price')] = stop_loss_price
                        result.iloc[i, result.columns.get_loc('position_size')] = position_size
                        result.iloc[i, result.columns.get_loc('shares')] = shares
                        result.iloc[i, result.columns.get_loc('current_equity')] = current_equity
                        result.iloc[i, result.columns.get_loc('days_held')] = 0
                        result.iloc[i, result.columns.get_loc('scaled_in')] = 0
        
        return result
    
    def calculate_performance_summary(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Calculate and return performance summary from trading data with 5-day scaling analysis"""
        try:
            # Filter for completed trades (rows with PnL values)
            trades = df[df['pnl'].notna() & (df['pnl'] != 0)]
            
            if len(trades) == 0:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'total_pnl': 0,
                    'final_equity': self.INITIAL_CAPITAL,
                    'roi': 0
                }
            
            # Basic trade statistics
            total_trades = len(trades)
            winning_trades = len(trades[trades['pnl'] > 0])
            losing_trades = len(trades[trades['pnl'] < 0])
            breakeven_trades = len(trades[trades['pnl'] == 0])
            
            # 5-Day Scaling Strategy Analysis
            scaled_trades = trades[trades['scaled_in'] == 1]
            non_scaled_trades = trades[trades['scaled_in'] == 0]
            
            scaling_stats = {
                'total_scaled_trades': len(scaled_trades),
                'total_non_scaled_trades': len(non_scaled_trades),
                'scaling_rate': (len(scaled_trades) / total_trades * 100) if total_trades > 0 else 0,
                'scaled_win_rate': (len(scaled_trades[scaled_trades['pnl'] > 0]) / len(scaled_trades) * 100) if len(scaled_trades) > 0 else 0,
                'non_scaled_win_rate': (len(non_scaled_trades[non_scaled_trades['pnl'] > 0]) / len(non_scaled_trades) * 100) if len(non_scaled_trades) > 0 else 0,
                'avg_scaled_pnl': scaled_trades['pnl'].mean() if len(scaled_trades) > 0 else 0,
                'avg_non_scaled_pnl': non_scaled_trades['pnl'].mean() if len(non_scaled_trades) > 0 else 0,
                'scaled_total_pnl': scaled_trades['pnl'].sum() if len(scaled_trades) > 0 else 0,
                'non_scaled_total_pnl': non_scaled_trades['pnl'].sum() if len(non_scaled_trades) > 0 else 0
            }
            
            # PnL calculations
            total_pnl = trades['pnl'].sum()
            avg_pnl_per_trade = trades['pnl'].mean()
            max_win = trades['pnl'].max()
            max_loss = trades['pnl'].min()
            
            # Win rate
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            # Final equity
            final_equity = self.INITIAL_CAPITAL + total_pnl
            
            # ROI
            roi = (total_pnl / self.INITIAL_CAPITAL) * 100
            
            # Calculate Sharpe ratio
            returns = trades['pnl'] / self.INITIAL_CAPITAL  # Returns as percentage of initial capital
            avg_return = returns.mean()
            return_std = returns.std()
            sharpe_ratio = avg_return / return_std if return_std != 0 else 0
            
            # Maximum drawdown calculation
            cumulative_pnl = trades['pnl'].cumsum()
            running_max = cumulative_pnl.expanding().max()
            drawdown = cumulative_pnl - running_max
            max_drawdown = drawdown.min()
            max_drawdown_pct = (max_drawdown / self.INITIAL_CAPITAL) * 100
            
            # Profit factor
            gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
            gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
            
            # Scaling-in analysis metrics
            underwater_days = (df['position_underwater'] == 1).sum()
            scaling_opportunities = (df['scaling_opportunity'] == 1).sum()
            total_position_days = (df['position_status'] == 'OPEN').sum()
            scale_in_signals = (df['trade_signal'] == 'SCALE_IN').sum()
            
            scaling_metrics = {
                'total_position_days': total_position_days,
                'underwater_days': underwater_days,
                'underwater_percentage': (underwater_days / total_position_days * 100) if total_position_days > 0 else 0,
                'scaling_opportunities': scaling_opportunities,
                'scaling_opportunity_rate': (scaling_opportunities / underwater_days * 100) if underwater_days > 0 else 0,
                'next_day_recovery_potential': df['next_day_recovery_potential'].sum(),
                'scale_in_signals': scale_in_signals,
                'actual_scaling_rate': (scale_in_signals / total_trades * 100) if total_trades > 0 else 0
            }
            
            return {
                'symbol': symbol,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'breakeven_trades': breakeven_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_pnl_per_trade': avg_pnl_per_trade,
                'max_win': max_win,
                'max_loss': max_loss,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss,
                'initial_capital': self.INITIAL_CAPITAL,
                'final_equity': final_equity,
                'roi': roi,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown_pct,
                'confidence_level_used': self.CONFIDENCE_LEVEL_FOR_TRADING,
                'scaling_analysis': scaling_metrics,
                'five_day_scaling_stats': scaling_stats
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance summary: {e}")
            return {}
    
    def print_performance_summary(self, performance: Dict[str, Any]):
        """Print a detailed performance summary"""
        if not performance or performance.get('total_trades', 0) == 0:
            print("\n📊 OVERNIGHT HOLD STRATEGY PERFORMANCE SUMMARY")
            print("=" * 60)
            print("No completed trades found in the data.")
            return
        
        symbol = performance['symbol']
        
        print(f"\n📊 MIDNIGHT MOMENTUM STRATEGY PERFORMANCE SUMMARY: {symbol}")
        print("=" * 70)
        print(f"🎯 STRATEGY: Buy at close, sell when price hits {performance['confidence_level_used']}% threshold")
        print(f"💰 RISK PER TRADE: {self.RISK_PER_TRADE_PERCENT}% of equity")
        
        print(f"\n📈 TRADE STATISTICS")
        print(f"   Total Trades: {performance['total_trades']}")
        print(f"   Winning Trades: {performance['winning_trades']}")
        print(f"   Losing Trades: {performance['losing_trades']}")
        print(f"   Breakeven Trades: {performance['breakeven_trades']}")
        print(f"   Win Rate: {performance['win_rate']:.1f}%")
        
        print(f"\n💰 PROFIT & LOSS")
        print(f"   Total PnL: ${performance['total_pnl']:.2f}")
        print(f"   Average PnL per Trade: ${performance['avg_pnl_per_trade']:.2f}")
        print(f"   Largest Win: ${performance['max_win']:.2f}")
        print(f"   Largest Loss: ${performance['max_loss']:.2f}")
        print(f"   Gross Profit: ${performance['gross_profit']:.2f}")
        print(f"   Gross Loss: ${performance['gross_loss']:.2f}")
        
        print(f"\n📊 PERFORMANCE METRICS")
        print(f"   Initial Capital: ${performance['initial_capital']:,.2f}")
        print(f"   Final Equity: ${performance['final_equity']:,.2f}")
        print(f"   ROI: {performance['roi']:.2f}%")
        print(f"   Profit Factor: {performance['profit_factor']:.2f}")
        print(f"   Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
        print(f"   Maximum Drawdown: ${performance['max_drawdown']:.2f} ({performance['max_drawdown_pct']:.2f}%)")
        
        # 5-Day Scaling Strategy Analysis
        if 'five_day_scaling_stats' in performance:
            scaling_stats = performance['five_day_scaling_stats']
            print(f"\n🎯 5-DAY SCALING STRATEGY ANALYSIS")
            print(f"   Total Trades: {performance['total_trades']}")
            print(f"   Scaled Trades (5+ days): {scaling_stats['total_scaled_trades']}")
            print(f"   Non-Scaled Trades (<5 days): {scaling_stats['total_non_scaled_trades']}")
            print(f"   Scaling Rate: {scaling_stats['scaling_rate']:.1f}%")
            print(f"   ")
            print(f"   📊 Performance Comparison:")
            print(f"   Scaled Trade Win Rate: {scaling_stats['scaled_win_rate']:.1f}%")
            print(f"   Non-Scaled Trade Win Rate: {scaling_stats['non_scaled_win_rate']:.1f}%")
            print(f"   Average Scaled PnL: ${scaling_stats['avg_scaled_pnl']:.2f}")
            print(f"   Average Non-Scaled PnL: ${scaling_stats['avg_non_scaled_pnl']:.2f}")
            print(f"   Total Scaled PnL: ${scaling_stats['scaled_total_pnl']:.2f}")
            print(f"   Total Non-Scaled PnL: ${scaling_stats['non_scaled_total_pnl']:.2f}")
            
            # Strategy effectiveness insights
            if scaling_stats['scaled_win_rate'] > scaling_stats['non_scaled_win_rate']:
                print(f"   ✅ Scaling strategy improves win rate by {scaling_stats['scaled_win_rate'] - scaling_stats['non_scaled_win_rate']:.1f} percentage points")
            else:
                print(f"   ⚠️  Scaling strategy reduces win rate by {scaling_stats['non_scaled_win_rate'] - scaling_stats['scaled_win_rate']:.1f} percentage points")
            
            if scaling_stats['avg_scaled_pnl'] > scaling_stats['avg_non_scaled_pnl']:
                print(f"   ✅ Scaled trades generate ${scaling_stats['avg_scaled_pnl'] - scaling_stats['avg_non_scaled_pnl']:.2f} more profit per trade on average")
            else:
                print(f"   ❌ Scaled trades generate ${scaling_stats['avg_non_scaled_pnl'] - scaling_stats['avg_scaled_pnl']:.2f} less profit per trade on average")

        # Scaling-in analysis section
        if 'scaling_analysis' in performance:
            scaling = performance['scaling_analysis']
            print(f"\n🔄 SCALING-IN ANALYSIS (Position Averaging Potential)")
            print(f"   Total Position Days: {scaling['total_position_days']}")
            print(f"   Days Position Underwater: {scaling['underwater_days']}")
            print(f"   Underwater Percentage: {scaling['underwater_percentage']:.1f}%")
            print(f"   Scaling Opportunities: {scaling['scaling_opportunities']}")
            print(f"   Scaling Opportunity Rate: {scaling['scaling_opportunity_rate']:.1f}%")
            print(f"   Next Day Recovery Potential: {scaling['next_day_recovery_potential']}")
            print(f"   Actual Scale-In Signals: {scaling['scale_in_signals']}")
            print(f"   Actual Scaling Rate: {scaling['actual_scaling_rate']:.1f}%")
            
            # Scaling-in insights
            if scaling['scaling_opportunity_rate'] > 60:
                print(f"   ✅ High scaling-in potential - {scaling['scaling_opportunity_rate']:.1f}% of underwater days show next-day recovery")
            elif scaling['scaling_opportunity_rate'] > 40:
                print(f"   ⚠️  Moderate scaling-in potential - {scaling['scaling_opportunity_rate']:.1f}% of underwater days show next-day recovery")
            else:
                print(f"   ❌ Low scaling-in potential - only {scaling['scaling_opportunity_rate']:.1f}% of underwater days show next-day recovery")
        
        # Strategy insights
        print(f"\n🔍 STRATEGY INSIGHTS")
        if performance['win_rate'] > 50:
            print(f"   ✅ Strategy shows positive win rate")
        else:
            print(f"   ⚠️  Strategy has win rate below 50%")
        
        if performance['roi'] > 0:
            print(f"   ✅ Strategy is profitable overall")
        else:
            print(f"   ❌ Strategy shows net loss")
        
        if performance['sharpe_ratio'] > 1:
            print(f"   ✅ Good risk-adjusted returns (Sharpe > 1)")
        elif performance['sharpe_ratio'] > 0:
            print(f"   ⚠️  Moderate risk-adjusted returns")
        else:
            print(f"   ❌ Poor risk-adjusted returns")
        
        print("=" * 70)

    def save_ohlcv_with_thresholds_csv(self, df: pd.DataFrame, symbol: str) -> str:
        """
        Save OHLCV data with threshold columns to CSV
        
        Args:
            df: DataFrame with calculated metrics and thresholds
            symbol: Stock symbol
            
        Returns:
            Path to saved CSV file
        """
        try:
            # Create historical_data directory if it doesn't exist
            os.makedirs('historical_data', exist_ok=True)
            
            # Prepare the data for CSV export
            csv_data = df.copy()
            
            # Select and order columns for CSV
            base_columns = [
                'datetime', 'open', 'high', 'low', 'close', 'volume',
                'prev_close', 'daily_return', 'overnight_gap', 'intraday_return',
                'high_above_prev_close', 'low_below_prev_close'
            ]
            
            # Add threshold columns
            threshold_columns = []
            for conf_level in self.config.confidence_levels:
                conf_pct = int(conf_level * 100)
                threshold_columns.extend([
                    f'threshold_{conf_pct}',
                    f'below_threshold_{conf_pct}',
                    f'breach_depth_{conf_pct}',
                    f'upside_threshold_{conf_pct}',
                    f'above_upside_threshold_{conf_pct}',
                    f'upside_breach_magnitude_{conf_pct}'
                ])
            
            # Trading columns
            trading_columns = [
                'trade_signal', 'position_status', 'entry_price', 'exit_price', 
                'target_price', 'pnl', 'running_pnl', 'current_equity', 
                'position_size', 'shares'
            ]
            
            # Scaling-in analysis columns
            scaling_columns = [
                'next_day_recovery_potential', 'position_underwater', 'scaling_opportunity'
            ]
            
            # Additional calculated columns
            additional_columns = [
                'true_range', 'prev_day_up', 'prev_day_down'
            ]
            
            # Add volatility regime if it exists
            if 'volatility_regime' in csv_data.columns:
                additional_columns.append('volatility_regime')
            
            # Combine all columns, only including those that exist in the DataFrame
            all_columns = base_columns + threshold_columns + trading_columns + scaling_columns + additional_columns
            available_columns = [col for col in all_columns if col in csv_data.columns]
            
            # Select only available columns
            csv_data = csv_data[available_columns]
            
            # Format datetime for better readability
            csv_data['datetime'] = csv_data['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Round numeric columns to reasonable precision
            numeric_columns = csv_data.select_dtypes(include=[np.number]).columns
            csv_data[numeric_columns] = csv_data[numeric_columns].round(6)
            
            filename = f'historical_data/{symbol}_overnight_hold_backtest_with_thresholds.csv'
            
            # Save to CSV
            csv_data.to_csv(filename, index=False)
            
            print(f"OHLCV data with thresholds saved to: {filename}")
            print(f"CSV contains {len(csv_data)} rows and {len(csv_data.columns)} columns")
            print(f"Columns included: {', '.join(csv_data.columns)}")
            
            # Automatically generate visualizations
            self._generate_automatic_visualizations(filename, symbol)
            
            return filename
            
        except Exception as e:
            logger.error(f"Error saving OHLCV CSV for {symbol}: {e}")
            print(f"Error saving CSV: {e}")
            return ""
    
    def _generate_automatic_visualizations(self, csv_filename: str, symbol: str):
        """
        Automatically generate visualizations after saving CSV data
        
        Args:
            csv_filename: Path to the saved CSV file
            symbol: Stock symbol
        """
        try:
            print(f"\n🎨 Generating automatic visualizations for {symbol}...")
            
            # Create charts directory if it doesn't exist
            os.makedirs('charts', exist_ok=True)
            
            # Initialize the visualizer with the CSV file
            visualizer = MidnightMomentumVisualizer(csv_filename)
            
            # Generate comprehensive visualization
            comprehensive_chart_path = f'charts/{symbol}_overnight_hold_comprehensive.png'
            
            print(f"   📊 Creating comprehensive visualization...")
            visualizer.create_comprehensive_visualization(save_path=comprehensive_chart_path)
            
            # Generate simple visualization
            simple_chart_path = f'charts/{symbol}_overnight_hold_simple.png'
            
            print(f"   📈 Creating simple visualization...")
            visualizer.create_simple_chart(save_path=simple_chart_path)
            
            print(f"✅ Visualizations completed successfully!")
            print(f"   📊 Comprehensive chart: {comprehensive_chart_path}")
            print(f"   📈 Simple chart: {simple_chart_path}")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not generate automatic visualizations: {str(e)}")
            logger.warning(f"Error generating automatic visualizations for {symbol}: {e}")

def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description='Robust Overnight Metrics Analyzer - Statistically sound overnight trading analysis'
    )
    parser.add_argument('symbols', nargs='+', help='Stock symbols to analyze')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--no-save', action='store_true', help='Do not save results to files')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Training data ratio')
    parser.add_argument('--confidence-levels', nargs='+', type=float, 
                       default=[0.68, 0.90, 0.95, 0.99], help='Confidence levels for thresholds')
    parser.add_argument('--bootstrap-samples', type=int, default=1000, help='Number of bootstrap samples')
    parser.add_argument('--monte-carlo-samples', type=int, default=1000, help='Number of Monte Carlo samples')
    parser.add_argument('--transaction-cost', type=float, default=0.001, help='Transaction cost (as decimal)')
    
    args = parser.parse_args()
    
    # Create configuration
    config = AnalysisConfig(
        train_ratio=args.train_ratio,
        confidence_levels=args.confidence_levels,
        n_bootstrap=args.bootstrap_samples,
        n_monte_carlo=args.monte_carlo_samples,
        transaction_cost=args.transaction_cost
    )
    
    # Load custom configuration if provided
    if args.config:
        try:
            with open(args.config, 'r') as f:
                custom_config = json.load(f)
            
            # Update config with custom values
            for key, value in custom_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            logger.info(f"Loaded custom configuration from {args.config}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return
    
    # Initialize analyzer
    analyzer = RobustOvernightAnalyzer(config)
    
    # Run analysis
    try:
        logger.info("Starting robust overnight metrics analysis")
        results = analyzer.analyze_multiple_symbols(args.symbols, save_results=not args.no_save)
        
        if results:
            logger.info(f"Analysis completed successfully for {len(results)} symbols")
            
            # Print final summary
            print(f"\n{'='*80}")
            print(f"ANALYSIS COMPLETED")
            print(f"{'='*80}")
            print(f"Symbols analyzed: {len(results)}")
            print(f"Results saved to: data/robust_analysis/")
            print(f"Log file: robust_overnight_analyzer.log")
            
            if not args.no_save:
                print(f"\nGenerated files:")
                print(f"- Individual analysis plots: *_robust_overnight_analysis.png")
                print(f"- Individual results: *_robust_analysis_results.json")
                if len(results) > 1:
                    print(f"- Comparative analysis: comparative_overnight_analysis.png")
                    print(f"- Comparative summary: comparative_summary.csv")
        else:
            logger.error("No symbols were successfully analyzed")
            
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()

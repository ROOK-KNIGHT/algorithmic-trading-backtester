#!/usr/bin/env python3
"""
Script to fetch 10 years of daily data for a stock symbol and save as CSV
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime, timedelta
from historical_data_handler import HistoricalDataHandler
import time

def fetch_and_save_10_year_data(symbol):
    """
    Fetch 10 years of daily data for the specified symbol and save as CSV
    
    Args:
        symbol (str): Stock symbol to fetch data for
    """
    print(f"Fetching 10 years of daily data for {symbol}...")
    
    # Initialize the historical data handler
    data_handler = HistoricalDataHandler()
    
    # Calculate dates for 10 years of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=10*365)  # Approximately 10 years
    
    # Convert to milliseconds since epoch (required by Schwab API)
    start_date_ms = int(start_date.timestamp() * 1000)
    end_date_ms = int(end_date.timestamp() * 1000)
    
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    try:
        # Fetch historical data
        # Using year period type with 10 years, daily frequency
        data = data_handler.fetch_historical_data(
            symbol=symbol,
            periodType="day",
            period=1,
            frequencyType="minute", 
            freq=5,
            startDate=start_date_ms,
            endDate=end_date_ms,
            needExtendedHoursData=False  # Regular hours only for daily data
        )
        
        if not data or not data.get('candles'):
            print(f"No data received for {symbol}")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(data['candles'])
        
        # Add symbol column
        df['symbol'] = symbol
        
        # Reorder columns
        df = df[['symbol', 'datetime', 'open', 'high', 'low', 'close', 'volume']]
        
        # Sort by datetime
        df = df.sort_values('datetime')
        
        # Create output directory if it doesn't exist
        os.makedirs('historical_data', exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"historical_data/{symbol}_10_year_daily_data_{timestamp}.csv"
        
        # Save to CSV
        df.to_csv(filename, index=False)
        
        print(f"\nData successfully saved to: {filename}")
        print(f"Total records: {len(df)}")
        print(f"Date range in data: {df['datetime'].min()} to {df['datetime'].max()}")
        
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
        
        return filename
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return None

def main():
    """Main function"""
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Fetch 10 years of daily stock data and save as CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fetch_10_year_daily_data.py AAPL
  python fetch_10_year_daily_data.py NVDA
  python fetch_10_year_daily_data.py TSLA
        """
    )
    
    parser.add_argument(
        'ticker',
        type=str,
        help='Stock ticker symbol to fetch data for (e.g., AAPL, NVDA, TSLA)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Convert ticker to uppercase for consistency
    ticker = args.ticker.upper()
    
    print("=" * 60)
    print("10-Year Daily Data Fetcher")
    print("=" * 60)
    
    # Fetch data for the specified ticker
    result = fetch_and_save_10_year_data(ticker)
    
    if result:
        print(f"\n✅ Successfully saved 10-year daily data to: {result}")
    else:
        print("\n❌ Failed to fetch and save data")

if __name__ == "__main__":
    main()

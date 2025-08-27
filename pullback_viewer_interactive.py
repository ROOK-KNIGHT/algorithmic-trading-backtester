#!/usr/bin/env python3
"""
Interactive Pullback Viewer
Based on TradingView's Pullback Viewer by emka

This script creates an interactive chart that identifies valid pullback points in trending markets.
A valid pullback requires a clean candle body close outside the previous candle's range.

Features:
- Real-time data fetching using existing handlers
- Interactive plotly chart with zoom, pan, and hover
- Bullish and bearish pullback identification
- Configurable parameters
- Support for different timeframes
"""

import sys
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import argparse
import numpy as np

# Import required modules for data handling
import requests
import json
import base64
import webbrowser
import urllib.parse
from datetime import timedelta

# Standalone data handler (simplified version of your handlers)
class StandaloneDataHandler:
    def __init__(self):
        self.token_file = "cs_tokens.json"
        self.keys_path = "/Users/isaac/Desktop/Projects/CS_KEYS/KEYS.json"
        
    def load_api_keys(self):
        """Load API keys from the external KEYS.json file"""
        try:
            with open(self.keys_path, 'r') as f:
                keys = json.load(f)
            return keys["APP_KEY"], keys["APP_SECRET"]
        except Exception as e:
            print(f"Error loading API keys from {self.keys_path}: {e}")
            raise
    
    def load_tokens(self):
        if os.path.exists(self.token_file):
            with open(self.token_file, 'r') as f:
                return json.load(f)
        return None
    
    def ensure_valid_tokens(self, refresh=False):
        """Enhanced token validation with refresh capability"""
        tokens = self.load_tokens()
        if tokens:
            expires_at = tokens.get('expires_at')
            
            # Check if 'expires_at' exists and is a valid string
            if expires_at:
                try:
                    expires_at = datetime.fromisoformat(expires_at)
                except ValueError:
                    print("Invalid 'expires_at' format in tokens. Re-authentication required.")
                    tokens = None
            else:
                print("'expires_at' missing from tokens. Re-authentication required.")
                tokens = None

            if tokens:
                refresh_token = tokens.get("refresh_token")
                # Check if access token is expired or about to expire (within 2 minutes) or force refresh
                if refresh or datetime.now() >= expires_at - timedelta(minutes=2):
                    print("Access token expired or refresh requested, attempting to refresh...")
                    new_tokens = self.refresh_tokens(refresh_token)
                    if new_tokens:
                        return new_tokens
                    else:
                        print("Failed to refresh tokens. Please re-authenticate.")
                        return None
                else:
                    return tokens

        print("No valid tokens found. Please run your authentication script first.")
        return None
    
    def save_tokens(self, tokens):
        """Save tokens with expiration time"""
        expires_at = datetime.now() + timedelta(seconds=int(tokens['expires_in']))
        tokens['expires_at'] = expires_at.isoformat()
        
        with open(self.token_file, 'w') as f:
            json.dump(tokens, f)
    
    def refresh_tokens(self, refresh_token):
        """Refresh access token using refresh token"""
        print("Refreshing access token...")
        app_key, app_secret = self.load_api_keys()
        credentials = f"{app_key}:{app_secret}"
        base64_credentials = base64.b64encode(credentials.encode()).decode("utf-8")

        headers = {
            "Authorization": f"Basic {base64_credentials}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token
        }

        token_url = "https://api.schwabapi.com/v1/oauth/token"
        refresh_response = requests.post(token_url, headers=headers, data=payload)
        
        if refresh_response.status_code == 200:
            new_tokens = refresh_response.json()
            self.save_tokens(new_tokens)
            print("Tokens refreshed successfully!")
            return new_tokens
        else:
            print("Failed to refresh tokens")
            print("Status Code:", refresh_response.status_code)
            print("Response:", refresh_response.text)
            return None
    
    def fetch_historical_data(self, symbol, periodType, period, frequencyType, freq, startDate=None, endDate=None, needExtendedHoursData=True):
        """Fetch historical data from Schwab API"""
        tokens = self.ensure_valid_tokens()
        if not tokens:
            return None
            
        access_token = tokens["access_token"]
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json"
        }
        
        url = f"https://api.schwabapi.com/marketdata/v1/pricehistory?symbol={symbol}&periodType={periodType}&period={period}&frequencyType={frequencyType}&frequency={freq}&needExtendedHoursData={str(needExtendedHoursData).lower()}"
        
        if startDate:
            url += f"&startDate={startDate}"
        if endDate:
            url += f"&endDate={endDate}"
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get("empty", True):
                candles = [
                    {
                        "datetime": self.convert_timestamp(bar["datetime"]),
                        "open": bar.get("open"),
                        "high": bar.get("high"),
                        "low": bar.get("low"),
                        "close": bar.get("close"),
                        "volume": bar.get("volume")
                    }
                    for bar in data["candles"]
                ]
                return {
                    "symbol": symbol,
                    "candles": candles,
                    "previousClose": data.get("previousClose"),
                    "previousCloseDate": self.convert_timestamp(data.get("previousCloseDate"))
                }
            else:
                print("No data returned from API.")
                return None
                
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            return None
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return None
    
    def convert_timestamp(self, timestamp):
        """Convert timestamp to datetime string"""
        if timestamp is not None:
            return datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')
        return None

class PullbackViewer:
    def __init__(self):
        self.data_handler = StandaloneDataHandler()
        self.df = None
        self.pullbacks = None
        
    def fetch_data(self, symbol, period_days=30, frequency_minutes=5):
        """
        Fetch historical data for the specified symbol
        
        Args:
            symbol (str): Stock symbol
            period_days (int): Number of days of data to fetch
            frequency_minutes (int): Frequency in minutes (1, 5, 15, 30, 60)
        """
        print(f"Fetching data for {symbol}...")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        
        # Convert to milliseconds
        start_date_ms = int(start_date.timestamp() * 1000)
        end_date_ms = int(end_date.timestamp() * 1000)
        
        try:
            # Fetch data using existing handler
            data = self.data_handler.fetch_historical_data(
                symbol=symbol,
                periodType="day",
                period=1,
                frequencyType="minute",
                freq=frequency_minutes,
                startDate=start_date_ms,
                endDate=end_date_ms,
                needExtendedHoursData=False
            )
            
            if not data or not data.get('candles'):
                print(f"No data received for {symbol}")
                return False
                
            # Convert to DataFrame
            df = pd.DataFrame(data['candles'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime').reset_index(drop=True)
            
            self.df = df
            print(f"Successfully loaded {len(df)} candles for {symbol}")
            return True
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False
    
    def identify_pullbacks(self, structure_only=False):
        """
        Identify valid pullback points in the data
        
        Args:
            structure_only (bool): If True, only show structural pullbacks (last of each type)
        """
        if self.df is None or len(self.df) < 3:
            print("Insufficient data for pullback analysis")
            return
            
        df = self.df.copy()
        pullbacks = []
        
        # Analyze each candle starting from the second one
        for i in range(1, len(df)):
            current = df.iloc[i]
            previous = df.iloc[i-1]
            
            # Check for bearish pullback
            # Current candle body closes above previous candle high
            if current['close'] > previous['high'] and current['open'] < current['close']:
                pullbacks.append({
                    'index': i,
                    'datetime': current['datetime'],
                    'type': 'bearish',
                    'price': current['close'],
                    'previous_high': previous['high'],
                    'valid': True
                })
            
            # Check for bullish pullback  
            # Current candle body closes below previous candle low
            elif current['close'] < previous['low'] and current['open'] > current['close']:
                pullbacks.append({
                    'index': i,
                    'datetime': current['datetime'],
                    'type': 'bullish',
                    'price': current['close'],
                    'previous_low': previous['low'],
                    'valid': True
                })
        
        # If structure_only, keep only the last pullback of each type in recent range
        if structure_only:
            # Group by type and keep only the most recent ones
            bullish_pullbacks = [p for p in pullbacks if p['type'] == 'bullish']
            bearish_pullbacks = [p for p in pullbacks if p['type'] == 'bearish']
            
            filtered_pullbacks = []
            if bullish_pullbacks:
                filtered_pullbacks.append(bullish_pullbacks[-1])  # Most recent bullish
            if bearish_pullbacks:
                filtered_pullbacks.append(bearish_pullbacks[-1])  # Most recent bearish
                
            pullbacks = filtered_pullbacks
        
        self.pullbacks = pd.DataFrame(pullbacks)
        print(f"Identified {len(pullbacks)} pullback points")
        
        return self.pullbacks
    
    def create_interactive_chart(self, symbol, show_volume=True, structure_only=False):
        """
        Create an interactive plotly chart with pullback points
        
        Args:
            symbol (str): Stock symbol for chart title
            show_volume (bool): Whether to show volume subplot
            structure_only (bool): Whether to show only structural pullbacks
        """
        if self.df is None:
            print("No data available for charting")
            return None
            
        # Create subplots
        if show_volume:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=(f'{symbol} Price with Pullback Points', 'Volume'),
                row_width=[0.7, 0.3]
            )
        else:
            fig = go.Figure()
        
        # Add candlestick chart
        candlestick = go.Candlestick(
            x=self.df['datetime'],
            open=self.df['open'],
            high=self.df['high'],
            low=self.df['low'],
            close=self.df['close'],
            name='Price',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444',
            increasing_fillcolor='#00ff88',
            decreasing_fillcolor='#ff4444'
        )
        
        if show_volume:
            fig.add_trace(candlestick, row=1, col=1)
        else:
            fig.add_trace(candlestick)
        
        # Add pullback points if available
        if self.pullbacks is not None and len(self.pullbacks) > 0:
            # Bearish pullbacks (red dots)
            bearish = self.pullbacks[self.pullbacks['type'] == 'bearish']
            if len(bearish) > 0:
                fig.add_trace(go.Scatter(
                    x=bearish['datetime'],
                    y=bearish['price'],
                    mode='markers',
                    marker=dict(
                        symbol='circle',
                        size=12,
                        color='red',
                        line=dict(width=2, color='darkred')
                    ),
                    name='Bearish Pullback',
                    text=[f"Bearish Pullback<br>Price: ${price:.2f}<br>Prev High: ${prev_high:.2f}" 
                          for price, prev_high in zip(bearish['price'], bearish['previous_high'])],
                    hovertemplate='%{text}<extra></extra>'
                ), row=1 if show_volume else None, col=1)
            
            # Bullish pullbacks (green dots)
            bullish = self.pullbacks[self.pullbacks['type'] == 'bullish']
            if len(bullish) > 0:
                fig.add_trace(go.Scatter(
                    x=bullish['datetime'],
                    y=bullish['price'],
                    mode='markers',
                    marker=dict(
                        symbol='circle',
                        size=12,
                        color='green',
                        line=dict(width=2, color='darkgreen')
                    ),
                    name='Bullish Pullback',
                    text=[f"Bullish Pullback<br>Price: ${price:.2f}<br>Prev Low: ${prev_low:.2f}" 
                          for price, prev_low in zip(bullish['price'], bullish['previous_low'])],
                    hovertemplate='%{text}<extra></extra>'
                ), row=1 if show_volume else None, col=1)
        
        # Add volume bars if requested
        if show_volume:
            colors = ['green' if close >= open else 'red' 
                     for close, open in zip(self.df['close'], self.df['open'])]
            
            fig.add_trace(go.Bar(
                x=self.df['datetime'],
                y=self.df['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ), row=2, col=1)
        
        # Update layout
        title_suffix = " (Structure Only)" if structure_only else ""
        fig.update_layout(
            title=f'{symbol} - Interactive Pullback Viewer{title_suffix}',
            xaxis_title='Time',
            yaxis_title='Price ($)',
            template='plotly_dark',
            showlegend=True,
            height=800 if show_volume else 600,
            hovermode='x unified'
        )
        
        # Remove rangeslider for cleaner look
        fig.update_layout(xaxis_rangeslider_visible=False)
        
        if show_volume:
            fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
    
    def analyze_pullbacks(self):
        """
        Provide analysis of identified pullbacks
        """
        if self.pullbacks is None or len(self.pullbacks) == 0:
            print("No pullbacks identified")
            return
            
        print("\n" + "="*60)
        print("PULLBACK ANALYSIS")
        print("="*60)
        
        total_pullbacks = len(self.pullbacks)
        bullish_count = len(self.pullbacks[self.pullbacks['type'] == 'bullish'])
        bearish_count = len(self.pullbacks[self.pullbacks['type'] == 'bearish'])
        
        print(f"Total Pullbacks: {total_pullbacks}")
        print(f"Bullish Pullbacks: {bullish_count}")
        print(f"Bearish Pullbacks: {bearish_count}")
        
        if total_pullbacks > 0:
            print(f"Bullish Ratio: {bullish_count/total_pullbacks:.1%}")
            print(f"Bearish Ratio: {bearish_count/total_pullbacks:.1%}")
        
        print("\nRecent Pullbacks:")
        print("-" * 40)
        
        # Show last 5 pullbacks
        recent = self.pullbacks.tail(5)
        for _, pullback in recent.iterrows():
            pb_type = pullback['type'].upper()
            price = pullback['price']
            time = pullback['datetime'].strftime('%Y-%m-%d %H:%M')
            print(f"{time} | {pb_type:8} | ${price:.2f}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Interactive Pullback Viewer - Identify valid pullback points",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 pullback_viewer_interactive.py AAPL
  python3 pullback_viewer_interactive.py NVDA --days 7 --frequency 1
  python3 pullback_viewer_interactive.py TSLA --structure-only --no-volume
        """
    )
    
    parser.add_argument('symbol', help='Stock symbol (e.g., AAPL, NVDA, TSLA)')
    parser.add_argument('--days', type=int, default=7, help='Number of days of data (default: 7)')
    parser.add_argument('--frequency', type=int, default=5, choices=[1, 5, 15, 30, 60], 
                       help='Frequency in minutes (default: 5)')
    parser.add_argument('--structure-only', action='store_true', 
                       help='Show only structural pullbacks (last of each type)')
    parser.add_argument('--no-volume', action='store_true', help='Hide volume chart')
    
    args = parser.parse_args()
    
    # Initialize viewer
    viewer = PullbackViewer()
    
    print("="*60)
    print("INTERACTIVE PULLBACK VIEWER")
    print("="*60)
    print(f"Symbol: {args.symbol.upper()}")
    print(f"Period: {args.days} days")
    print(f"Frequency: {args.frequency} minutes")
    print(f"Structure Only: {args.structure_only}")
    print("="*60)
    
    # Fetch data
    if not viewer.fetch_data(args.symbol.upper(), args.days, args.frequency):
        print("Failed to fetch data. Exiting.")
        return
    
    # Identify pullbacks
    pullbacks = viewer.identify_pullbacks(structure_only=args.structure_only)
    
    # Analyze pullbacks
    viewer.analyze_pullbacks()
    
    # Create and show interactive chart
    print("\nGenerating interactive chart...")
    fig = viewer.create_interactive_chart(
        args.symbol.upper(), 
        show_volume=not args.no_volume,
        structure_only=args.structure_only
    )
    
    if fig:
        print("Opening interactive chart in browser...")
        fig.show()
        
        # Save as HTML file
        filename = f"{args.symbol.upper()}_pullback_viewer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(filename)
        print(f"Chart saved as: {filename}")
    else:
        print("Failed to create chart")

if __name__ == "__main__":
    main()

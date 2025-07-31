import base64
import requests
import json
import urllib.parse
import os
import time
import pandas as pd
from datetime import datetime, timedelta
import os
import pytz
import argparse
import sys
import subprocess
import logging
from typing import Dict, Any, List, Tuple, Optional
from urllib.parse import quote

# Configure logging
def setup_logging(symbol: str):
    """Setup logging configuration with symbol-specific log file"""
    # First, remove any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        print(f"Creating logs directory: {log_dir}")
        os.makedirs(log_dir)
    
    # Use fixed log filename
    log_file = os.path.join(log_dir, f"{symbol.lower()}_log")
    print(f"Logging to: {log_file}")
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    
    # Setup file handler only
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # Configure root logger with file handler only
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"=== Logging initialized for {symbol} ===")
    return logger



# Schwab API credentials
APP_KEY = "UXvDmuMdEsgAyXAWGMSOblaaLbnR8MhW"
APP_SECRET = "Hl8zGamcb7Valfee"
REDIRECT_URI = "https://127.0.0.1"
TOKEN_FILE = "/Users/isaac/Desktop/CS_TOKENS/cs_tokens.json"

class ExceedenceTrader:
    def __init__(self, symbol, strategy='reversal', auto_mode=False, shares=None, direction='both', use_percentage=False, percentage=None):
        self.symbol = symbol.upper()
        self.strategy = strategy  # 'reversal' or 'momentum'
        
        # Setup logging
        self.logger = setup_logging(self.symbol)
        self.logger.info(f"Initializing {self.symbol} ExceedenceTrader")
        self.logger.info(f"Strategy: {strategy} | Auto Mode: {auto_mode} | Direction: {direction}")
        self.logger.info(f"Position Sizing: {'Percentage: ' + str(percentage) + '%' if use_percentage else 'Fixed Shares: ' + str(shares)}")
        
        self.tokens = self.ensure_valid_tokens()
        self.last_api_minute = None
        self.account_number = self.get_account_number()
        self.auto_mode = auto_mode
        self.shares = shares
        self.direction = direction
        self.use_percentage = use_percentage
        self.percentage = percentage
        self.orders_placed = 0  # Counter to prevent duplicate orders for new positions
        self.scale_in_count = 0  # Separate counter for scaling into existing positions
        self.market_timezone = pytz.timezone('America/New_York')  # NYSE/NASDAQ market timezone
        
        # Position tracking
        self.current_position = "FLAT"  # Can be "FLAT", "LONG", or "SHORT"
        self.position_size = 0
        self.entry_price = 0
        self.entry_time = None
        self.position_id = None
        self.profit_target_pct = 0.00135  # 0.045% profit target percentage
        
    def get_account_number(self):
        """Get the first available account number's hash value"""
        url = "https://api.schwabapi.com/trader/v1/accounts/accountNumbers"
        headers = {
            "Authorization": f"Bearer {self.tokens['access_token']}",
            "Accept": "application/json"
        }
        retries = 5
        for attempt in range(retries):
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()  # Raise error for bad status codes
                accounts = response.json()
                print(f"DEBUG: Account response: {accounts}")
                
                if isinstance(accounts, list) and len(accounts) > 0:
                    account = accounts[0]  # Use first account
                    if isinstance(account, dict) and 'hashValue' in account:
                        return account['hashValue']
                    else:
                        raise ValueError(f"Account missing hashValue: {account}")
                else:
                    raise ValueError(f"No accounts found in response: {accounts}")
                    
            except requests.exceptions.ReadTimeout:
                print(f"Request timed out on attempt {attempt + 1}/{retries}. Retrying...")
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}, attempt {attempt + 1}/{retries}")
            time.sleep(2 ** attempt)  # Exponential backoff
            
        print("Failed to fetch account numbers after all retries")
        return None
   
    def save_orders_to_json(self, orders):
        """Save orders to a JSON file"""
        # Create orders directory if it doesn't exist
        orders_dir = "orders"
        if not os.path.exists(orders_dir):
            os.makedirs(orders_dir)
        
        # Use fixed filename
        filename = os.path.join(orders_dir, "active_orders.json")
        
        # Save orders to file
        with open(filename, 'w') as f:
            json.dump(orders, f, indent=2)
        print(f"\nOrders saved to: {filename}")
    
    def get_active_orders(self, tokens, account_number):
        """Get all active orders for the account"""
        # Get current time in UTC
        now = datetime.now(pytz.UTC)
        from_time = now.strftime("%Y-%m-%dT00:00:00.000Z")
        to_time = now.strftime("%Y-%m-%dT23:59:59.000Z")
        
        # Build URL with query parameters
        base_url = f"https://api.schwabapi.com/trader/v1/accounts/{account_number}/orders"
        params = {
            'maxResults': '3000',  # Maximum allowed
            'fromEnteredTime': from_time,
            'toEnteredTime': to_time,
            'status': 'WORKING'  # Only get active orders
        }
        
        headers = {
            "Authorization": f"Bearer {tokens['access_token']}",
            "Accept": "application/json"
        }
        
        try:
            response = requests.get(base_url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                orders = response.json()
                return orders
            elif response.status_code == 401:
                print("Token expired, refreshing...")
                self.tokens = self.refresh_tokens(self.tokens["refresh_token"])
                return self.get_active_orders(self.tokens, self.account_number)
            else:
                self.logger.error(f"Failed to fetch orders: {response.status_code} - {response.text}")
                return None
        except requests.RequestException as e:
            self.logger.error(f"Error fetching orders: {e}")
            return None
    
    def sync_orders(self):
        """Sync orders from active_orders.json"""
        try:
            # Get active orders
            orders = self.get_active_orders(self.tokens, self.account_number)
            if not orders:
                self.logger.info("No active orders found")
                return [], []
            
            # Save orders to JSON file
            self.save_orders_to_json(orders)
                
            # Filter orders for our symbol
            symbol_orders = [
                order for order in orders
                if any(
                    leg['instrument']['symbol'] == self.symbol 
                    for leg in order.get('orderLegCollection', [])
                )
            ]

            # Check for profit target orders (LIMIT orders)
            profit_targets = [
                order for order in symbol_orders
                if order.get('orderType') == 'LIMIT'
            ]
            
            return symbol_orders, profit_targets
        except Exception as e:
            self.logger.error(f"Error syncing orders: {e}")
            return [], []
            
    def sync_current_positions(self) -> bool:
        """Get current positions and sync internal state"""
        try:
            # Store old position state before updating
            old_position = self.current_position
            old_size = self.position_size
            old_price = self.entry_price
            
            self.logger.info(f"Syncing positions - Current State: {old_position} | Size: {old_size} | Entry: ${old_price:.2f}")
            
            account_data = self.get_account_data()
            if not account_data:
                self.logger.error("Failed to get account data for position sync")
                return False
                
            positions = []
            for account in account_data:
                for position in account.get('positions', []):
                    if position.get('symbol') == self.symbol:
                        positions.append(position)
            
            # If no positions for our symbol, we're flat
            if not positions:
                self.current_position = "FLAT"
                self.position_size = 0
                self.entry_price = 0
                self.entry_time = None
                
                # Check if position was closed (previously had position, now flat)
                if old_position != "FLAT" and self.current_position == "FLAT":
                    self.logger.info(f"Position closed: {old_position} → FLAT")
                    self.logger.info(f"Previous Size: {old_size} | Previous Entry: ${old_price:.2f}")
                    self.logger.info("Resetting orders_placed counter")
                    self.orders_placed = 0
                
                return True
                
            # Check our position type
            position = positions[0]  # Just use the first position if multiple
            quantity = position.get('quantity', 0)
            
            if quantity > 0:
                self.current_position = "LONG"
                self.position_size = quantity
                self.entry_price = position.get('average_price', 0)
                self.logger.info(f"LONG position synced: {self.position_size} shares @ ${self.entry_price:.2f}")
                if old_position != "LONG" or old_size != self.position_size:
                    self.logger.info(f"Position changed: {old_position}({old_size}) → LONG({self.position_size})")
                return True
            elif quantity < 0:
                self.current_position = "SHORT"
                self.position_size = abs(quantity)
                self.entry_price = position.get('average_price', 0)
                self.logger.info(f"SHORT position synced: {self.position_size} shares @ ${self.entry_price:.2f}")
                if old_position != "SHORT" or old_size != self.position_size:
                    self.logger.info(f"Position changed: {old_position}({old_size}) → SHORT({self.position_size})")
                return True
            else:
                self.current_position = "FLAT"
                self.position_size = 0
                self.entry_price = 0
                self.entry_time = None
                return True
                
        except Exception as e:
            print(f"Error syncing positions: {e}")
            return False
            
    def get_current_price(self, symbol: str) -> Dict[str, Any]:
        """Get current price for a symbol with retry logic"""
        if not self.tokens:
            raise ValueError("Failed to get valid tokens")
            
        url = f"https://api.schwabapi.com/marketdata/v1/{symbol}/quotes?fields=quote,reference"
        headers = {
            "Authorization": f"Bearer {self.tokens['access_token']}",
            "Accept": "application/json"
        }
        
        retries = 5
        for attempt in range(retries):
            try:
                # Add timeout parameter and log attempt
                print(f"Fetching quote for {symbol}, attempt {attempt + 1}/{retries}")
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    quote_data = data.get(symbol, {}).get('quote', {})
                    return {
                        'lastPrice': quote_data.get('lastPrice'),
                        'askPrice': quote_data.get('askPrice'),
                        'bidPrice': quote_data.get('bidPrice'),
                        'totalVolume': quote_data.get('totalVolume'),
                        'netChange': quote_data.get('netChange'),
                        'netPercentChange': quote_data.get('netPercentChange')
                    }
                elif response.status_code == 401:
                    print("Token expired, refreshing...")
                    self.tokens = self.refresh_tokens(self.tokens["refresh_token"])
                    # Update headers with new token
                    headers["Authorization"] = f"Bearer {self.tokens['access_token']}"
                else:
                    print(f"Request failed with status code: {response.status_code}")
                    
            except requests.exceptions.ReadTimeout:
                print(f"Request timed out on attempt {attempt + 1}/{retries}")
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}, attempt {attempt + 1}/{retries}")
            
            # Don't sleep on the last attempt
            if attempt < retries - 1:
                sleep_time = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8, 16 seconds
                print(f"Waiting {sleep_time} seconds before next attempt...")
                time.sleep(sleep_time)
        
        raise ValueError(f"Failed to get quote after {retries} attempts")

    def place_market_order(self, order_type: str, quantity: int) -> Dict[str, Any]:
        """Place a market order to enter or exit a position"""
        self.logger.info(f"Attempting to place {order_type} market order for {quantity} shares of {self.symbol}")
        
        if not self.account_number:
            self.logger.error("No account number available for order placement")
            return {"error": "No account number available"}
            
        # Determine instruction based on order type
        if order_type == "BUY":
            instruction = "BUY"
        elif order_type == "SELL":
            instruction = "SELL"
        elif order_type == "SELL_SHORT":
            instruction = "SELL_SHORT"
        elif order_type == "BUY_TO_COVER":
            instruction = "BUY_TO_COVER"
        else:
            return {"error": f"Invalid order type: {order_type}"}
            
        # Create order payload
        order_payload = {
            "orderStrategyType": "SINGLE",
            "orderType": "MARKET",
            "session": "NORMAL",
            "duration": "DAY",
            "orderLegCollection": [
                {
                    "instruction": instruction,
                    "quantity": quantity,
                    "instrument": {
                        "symbol": self.symbol,
                        "assetType": "EQUITY"
                    }
                }
            ]
        }
        
        url = f"https://api.schwabapi.com/trader/v1/accounts/{self.account_number}/orders"
        headers = {
            "Authorization": f"Bearer {self.tokens['access_token']}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        try:
            response = requests.post(url, json=order_payload, headers=headers)
            
            if response.status_code in [200, 201]:
                order_id = response.headers.get('Location', '').split('/')[-1]
                self.logger.info(f"Successfully placed {order_type} market order: ID {order_id}")
                self.logger.info(f"Order details: {quantity} shares @ market price")
                return {"success": True, "orderId": order_id, "type": order_type}
            else:
                error_msg = f"Failed to place order: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                return {"error": error_msg}
        except Exception as e:
            error_msg = f"Error placing order: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg}
    
    def place_limit_order(self, order_type: str, quantity: int, price: float) -> Dict[str, Any]:
        """Place a limit order to enter or exit a position at a specific price"""
        self.logger.info(f"Attempting to place {order_type} limit order for {quantity} shares of {self.symbol} @ ${price:.2f}")
        
        if not self.account_number:
            self.logger.error("No account number available for order placement")
            return {"error": "No account number available"}
            
        # Determine instruction based on order type
        if order_type == "BUY":
            instruction = "BUY"
        elif order_type == "SELL":
            instruction = "SELL"
        elif order_type == "SELL_SHORT":
            instruction = "SELL_SHORT"
        elif order_type == "BUY_TO_COVER":
            instruction = "BUY_TO_COVER"
        else:
            return {"error": f"Invalid order type: {order_type}"}
            
        # Create order payload
        order_payload = {
            "orderStrategyType": "SINGLE",
            "orderType": "LIMIT",
            "session": "SEAMLESS",   # SEAMLESS for after hours
            "duration": "GOOD_TILL_CANCEL",  # GOOD_TILL_CANCEL keeps the order active until filled or cancelled
            "price": str(price),  # Price must be a string in API
            "orderLegCollection": [
                {
                    "instruction": instruction,
                    "quantity": quantity,
                    "instrument": {
                        "symbol": self.symbol,
                        "assetType": "EQUITY"
                    }
                }
            ]
        }
        
        url = f"https://api.schwabapi.com/trader/v1/accounts/{self.account_number}/orders"
        headers = {
            "Authorization": f"Bearer {self.tokens['access_token']}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        try:
            response = requests.post(url, json=order_payload, headers=headers)
            
            if response.status_code in [200, 201]:
                order_id = response.headers.get('Location', '').split('/')[-1]
                self.logger.info(f"Successfully placed {order_type} limit order: ID {order_id}")
                self.logger.info(f"Order details: {quantity} shares @ ${price:.2f}")
                return {"success": True, "orderId": order_id, "type": order_type}
            else:
                error_msg = f"Failed to place limit order: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                return {"error": error_msg}
        except Exception as e:
            error_msg = f"Error placing limit order: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg}

    def scale_into_position(self, signal_direction: str, suggested_quantity: int = None) -> Tuple[bool, str, Optional[Dict]]:
        """Scale into position only with appropriate strategy direction and exceedence signal"""
        # Check for pending orders first
        if self.orders_placed > 0 or self.scale_in_count > 0:
            self.logger.info("Checking for pending orders before proceeding...")
            # Sync positions to see if previous order was filled
            if not self.sync_current_positions():
                self.logger.error("Failed to sync positions to check pending orders")
                return False, "Failed to validate pending orders", None
                
            # If we still have counters > 0, we might have pending orders
            if self.orders_placed > 0 or self.scale_in_count > 0:
                self.logger.warning("Pending orders detected - waiting for fills before placing new orders")
                return False, "Pending orders must be filled before placing new orders", None
        
        # Log entry into scale_into_position
        self.logger.info(f"Evaluating scale into position: Signal={signal_direction} | Current Position={self.current_position}")
        
        # Check if already scaled in (cost > percentage)
        if self.has_already_scaled_in():
            self.logger.info("Scale rejected: Position already scaled in (cost exceeds percentage limit)")
            return False, "Position already scaled in (cost exceeds percentage limit)", None
        
        # Check if signal direction is compatible with scaling based on strategy
        if not self.should_scale_based_on_signal(signal_direction):
            strategy_reason = "same direction" if self.strategy == 'reversal' else "opposite direction"
            msg = f"{self.strategy.title()} strategy: Signal ({signal_direction}) is {strategy_reason} as position ({self.current_position}) - no scaling"
            self.logger.info(f"Scale rejected: {msg}")
            return False, msg, None
        
        # Sync positions first to make sure we're working with current data
        if not self.sync_current_positions():
            return False, "Failed to sync positions", None
            
        result_message = ""
        position_result = None
        
        # Get current price data
        price_data = self.get_current_price(self.symbol)
        if not price_data or 'lastPrice' not in price_data:
            self.logger.error("Failed to get current price data for scaling decision")
            return False, "Unable to get current price for scaling", None
            
        self.logger.info(f"Current price data: ${price_data['lastPrice']:.2f}")
            
        current_price = price_data['lastPrice']
        
        # Determine if we need to add to an existing position or open a new one
        if self.current_position == "FLAT":
            # Open a new position - use suggested quantity or calculate from percentage/shares
            quantity = suggested_quantity if suggested_quantity else self.calculate_shares_from_percentage(current_price)
            
            # When opening a new position, use the signal direction
            direction = signal_direction
            
            self.logger.info(f"Opening new {direction} position")
            self.logger.info(f"Position parameters: Quantity={quantity} | Strategy={self.strategy}")
            self.logger.info(f"Account sizing: {'Percentage: ' + str(self.percentage) + '%' if self.use_percentage else 'Fixed Shares: ' + str(self.shares)}")
            
            if direction == "LONG":
                open_result = self.place_market_order("BUY", quantity)
            else:  # SHORT
                open_result = self.place_market_order("SELL_SHORT", quantity)
                
            if not open_result.get("success", False):
                error_msg = open_result.get("error", "Unknown error")
                return False, f"Failed to open {direction} position: {error_msg}", None
                
            # Increment orders_placed immediately after successful order placement
            self.orders_placed += 1
            print(f"New position order placed: counter = {self.orders_placed}")
                
            # Wait for the market order to be filled before placing limit order
            print("Market order placed for new position. Waiting for order completion...")
            order_filled = self.wait_for_order_fill(timeout_seconds=60)
            if not order_filled:
                print("Warning: Timeout waiting for new position order fill. Proceeding with caution...")
            
            # Sync positions to get actual broker data after market order fills
            sync_success = self.sync_current_positions()
            if sync_success:
                print(f"New position confirmed. Size: {self.position_size}, Entry: ${self.entry_price:.2f}")
            else:
                # Fallback to estimated values if sync fails
                self.position_id = open_result.get("orderId")
                self.current_position = direction
                self.position_size = quantity
                self.entry_price = current_price
                self.entry_time = datetime.now(self.market_timezone)
                print(f"Using estimated position data. Size: {self.position_size}, Entry: ${self.entry_price:.2f}")
            
            # Now place profit target limit order (with profit for new positions)
            # Don't wait for limit order to fill
            self.set_profit_target_order(use_breakeven=False)
            
            result_message = f"Opened new {direction} position of {quantity} shares at ${current_price:.2f}."
            position_result = open_result
            
        elif self.current_position in ["LONG", "SHORT"]:
            # First, cancel any existing profit target orders before scaling
            _, profit_targets = self.sync_orders()
            if profit_targets:
                print("Canceling existing profit target orders before scaling position...")
                for pt in profit_targets:
                    order_id = pt.get('orderId')
                    self.cancel_order(order_id)
            
            # Always use current position direction when scaling, regardless of signal
            direction = self.current_position
            # Calculate optimal number of shares to add to bring average price within 0.02% of current price
            price_diff = abs(current_price - self.entry_price)
            target_diff = current_price * 0.0002  # Target to get within 0.02% of current price
            
            self.logger.info(f"Scaling calculation parameters:")
            self.logger.info(f"Current Price: ${current_price:.2f} | Entry Price: ${self.entry_price:.2f}")
            self.logger.info(f"Price Difference: ${price_diff:.2f} | Target Difference: ${target_diff:.2f}")
            
            if price_diff <= target_diff:  # If prices are already within target range
                print(f"Current average price (${self.entry_price:.2f}) is already within ${target_diff:.2f} of current price (${current_price:.2f})")
                additional_quantity = self.calculate_shares_from_percentage(current_price)  # Use percentage or fixed shares
            else:
                # Calculate shares needed to get average price within target_diff of current price
                if direction == "LONG":
                    if current_price > self.entry_price:
                        # Can't average up to current price (would need to sell shares)
                        print(f"Current price (${current_price:.2f}) is above entry price (${self.entry_price:.2f}). Using default quantity.")
                        additional_quantity = self.calculate_shares_from_percentage(current_price)
                    else:
                        # Calculate shares needed to average down to within target_diff of current price
                        try:
                            target_avg = current_price + target_diff  # Aim for slightly above current price
                            
                            # Prevent division by zero or very small denominators
                            denominator = target_avg - current_price
                            if abs(denominator) < 0.001:  # Avoid division by very small numbers
                                raise ValueError("Target average too close to current price, would require too many shares")
                                
                            # Formula: add_qty = position_size * (entry_price - target_avg) / (target_avg - current_price)
                            calc_quantity = self.position_size * (self.entry_price - target_avg) / denominator
                            
                            # Validate the calculated quantity
                            if calc_quantity <= 0:
                                print(f"Calculated quantity ({calc_quantity:.2f}) is invalid. Using default quantity.")
                                additional_quantity = self.calculate_shares_from_percentage(current_price)
                            else:
                                additional_quantity = int(round(calc_quantity))
                                print(f"Calculated {additional_quantity} shares to get average within ${target_diff:.2f} of current price")
                        except Exception as e:
                            print(f"Error calculating scaling quantity: {e}. Using default quantity.")
                            additional_quantity = self.calculate_shares_from_percentage(current_price)
                else:  # SHORT
                    if current_price < self.entry_price:
                        # Can't average down to current price for shorts (would need to cover shares)
                        print(f"Current price (${current_price:.2f}) is below entry price (${self.entry_price:.2f}) for SHORT. Using default quantity.")
                        additional_quantity = self.calculate_shares_from_percentage(current_price)
                    else:
                        # Calculate shares needed to average up to within target_diff of current price for shorts
                        try:
                            target_avg = current_price - target_diff  # Aim for slightly below current price for shorts
                            
                            # Prevent division by zero or very small denominators
                            denominator = current_price - target_avg
                            if abs(denominator) < 0.001:  # Avoid division by very small numbers
                                raise ValueError("Target average too close to current price, would require too many shares")
                            
                            # Formula for shorts
                            calc_quantity = self.position_size * (target_avg - self.entry_price) / denominator
                            
                            # Validate the calculated quantity
                            if calc_quantity <= 0:
                                print(f"Calculated quantity ({calc_quantity:.2f}) is invalid for SHORT. Using default quantity.")
                                additional_quantity = self.calculate_shares_from_percentage(current_price)
                            else:
                                additional_quantity = int(round(calc_quantity))
                                print(f"Calculated {additional_quantity} shares to get SHORT average within ${target_diff:.2f} of current price")
                        except Exception as e:
                            print(f"Error calculating scaling quantity for SHORT: {e}. Using default quantity.")
                            additional_quantity = self.calculate_shares_from_percentage(current_price)
            
            # Ensure we're adding at least one share
            additional_quantity = max(1, additional_quantity)
            
            # Add to existing position
            if direction == "LONG":
                add_result = self.place_market_order("BUY", additional_quantity)
            else:  # SHORT
                add_result = self.place_market_order("SELL_SHORT", additional_quantity)
                
            if not add_result.get("success", False):
                error_msg = add_result.get("error", "Unknown error")
                return False, f"Failed to add to {direction} position: {error_msg}", None
                
            # Increment scale_in_count immediately after order placement
            self.scale_in_count += 1
            print(f"Scale-in order placed: counter = {self.scale_in_count}")
            
            # After placing the order, wait for order completion by checking positions
            print("Market order placed. Waiting for order completion...")
            
            # Wait for order to be filled by checking if position exists for our symbol
            order_filled = self.wait_for_order_fill(timeout_seconds=60)
            if not order_filled:
                print("Warning: Timeout waiting for order fill. Proceeding with position sync...")
            
            # Sync positions to get the actual broker-reported average price
            sync_success = self.sync_current_positions()
            if not sync_success:
                print("Warning: Failed to sync positions after scaling. Using estimated values.")
                # Calculate estimated average price as fallback
                total_cost = (self.entry_price * self.position_size) + (current_price * additional_quantity)
                estimated_new_size = self.position_size + additional_quantity
                self.entry_price = total_cost / estimated_new_size
                self.position_size = estimated_new_size
                print(f"Using estimated average price: ${self.entry_price:.2f}")
            else:
                print(f"Position synced. Actual broker-reported average price: ${self.entry_price:.2f}")
            
            # Set new profit target at break-even (exact entry price) when scaling in
            # (We already cancelled the previous profit target order at the beginning of this method)
            # Don't wait for limit order to fill
            self.set_profit_target_order(use_breakeven=True)
            
            # Show how close we got to the target average price
            avg_diff = abs(self.entry_price - current_price)
            result_message = (f"Added {additional_quantity} shares to {direction} position. "
                             f"New size: {self.position_size}, New avg price: ${self.entry_price:.2f} "
                             f"(${avg_diff:.2f} from current price)")
            position_result = add_result
            
            # # Spawn new process after successful scaling
            # spawn_success, spawn_msg = self.spawn_new_process()
            # if spawn_success:
            #     result_message += f" | {spawn_msg}"
            # else:
            #     print(f"Warning: {spawn_msg}")
            
        else:
            # Currently have a position in the opposite direction
            return False, f"Cannot scale into {signal_direction} with existing {self.current_position} position - no position reversal", None
            
        return True, result_message, position_result
    
    def set_profit_target_order(self, use_breakeven=True) -> bool:
        """Set a profit target limit order based on current position
           If use_breakeven is True, sets target at entry price (breakeven) when scaling
           If use_breakeven is False, sets target at profit_target (.045%) from entry price"""
        self.logger.info("\n" + "="*80)
        self.logger.info("PROFIT TARGET ORDER SETUP")
        self.logger.info("="*80)
        
        # Log initial state
        self.logger.info("\nCurrent Position State:")
        self.logger.info("-"*40)
        self.logger.info(f"Position: {self.current_position}")
        self.logger.info(f"Size: {self.position_size} shares")
        self.logger.info(f"Entry Price: ${self.entry_price:.2f}")
        self.logger.info(f"Order Type: {'BREAKEVEN' if use_breakeven else 'PROFIT TARGET'}")
        
        if self.current_position == "FLAT" or self.position_size == 0:
            self.logger.info("\n❌ No position to set profit target for")
            return False
            
        # Get current price data to ensure we have the correct entry price
        self.logger.info("\nPrice Verification:")
        self.logger.info("-"*40)
        
        price_data = self.get_current_price(self.symbol)
        if not price_data or 'lastPrice' not in price_data:
            self.logger.error("❌ Unable to get current price for profit target")
            return False
        
        self.logger.info(f"Current Market Price: ${price_data['lastPrice']:.2f}")
        self.logger.info(f"Bid: ${price_data.get('bidPrice', 0):.2f}")
        self.logger.info(f"Ask: ${price_data.get('askPrice', 0):.2f}")
            
        # Use actual entry price if available, otherwise use last price
        if self.entry_price and self.entry_price > 0:
            base_price = self.entry_price
            self.logger.info(f"Using actual entry price: ${base_price:.2f}")
        else:
            base_price = price_data['lastPrice']
            self.entry_price = base_price
            self.logger.info(f"Using current market price as base: ${base_price:.2f}")
        
        # Calculate target price based on percentage of current price
        self.logger.info("\nTarget Price Calculation:")
        self.logger.info("-"*40)
        
        if use_breakeven:
            # Use exact entry price for breakeven
            target_price = base_price
            target_amount = 0
            self.logger.info("Mode: BREAKEVEN")
            self.logger.info(f"Target Price = Entry Price: ${base_price:.2f}")
        else:
            # Calculate profit target amount as percentage of base price
            target_amount = base_price * self.profit_target_pct
            self.logger.info("Mode: PROFIT TARGET")
            self.logger.info(f"Profit Target: {self.profit_target_pct*100:.3f}% = ${target_amount:.2f}")
            
            if self.current_position == "LONG":
                # For LONG positions: profit = entry + (price * percentage)
                target_price = base_price + target_amount
                self.logger.info("LONG Position Calculation:")
                self.logger.info(f"  Entry Price:   ${base_price:.2f}")
                self.logger.info(f"  Target Amount: +${target_amount:.2f}")
                self.logger.info(f"  Target Price:  ${target_price:.2f}")
            else:  # SHORT
                # For SHORT positions: profit = entry - (price * percentage)
                target_price = base_price - target_amount
                self.logger.info("SHORT Position Calculation:")
                self.logger.info(f"  Entry Price:   ${base_price:.2f}")
                self.logger.info(f"  Target Amount: -${target_amount:.2f}")
                self.logger.info(f"  Target Price:  ${target_price:.2f}")
    
        # Prepare order details
        self.logger.info("\nOrder Details:")
        self.logger.info("-"*40)
        
        order_type = "SELL" if self.current_position == "LONG" else "BUY_TO_COVER"
        target_type = "BREAKEVEN" if use_breakeven else f"PROFIT (${target_amount:.2f} = 0.045% of ${base_price:.2f})"
        
        self.logger.info(f"Order Type: {order_type}")
        self.logger.info(f"Target Type: {target_type}")
        self.logger.info(f"Quantity: {self.position_size} shares")
        
        # Round to 2 decimal places
        target_price = round(target_price, 2)
        self.logger.info(f"Final Target Price: ${target_price:.2f}")
        
        # Check for existing profit target orders in active_orders.json
        self.logger.info("\nChecking Existing Orders:")
        self.logger.info("-"*40)
        _, profit_targets = self.sync_orders()
        
        if profit_targets:
            self.logger.info(f"Found {len(profit_targets)} existing profit target order(s):")
            for pt in profit_targets:
                order_id = pt.get('orderId')
                price = pt.get('price', 'N/A')
                quantity = sum(leg.get('quantity', 0) for leg in pt.get('orderLegCollection', []))
                self.logger.info(f"  Order ID: {order_id}")
                self.logger.info(f"  Price: ${price}")
                self.logger.info(f"  Quantity: {quantity} shares")
            
            self.logger.info("\n✓ Using existing profit target order")
            return True
            
        # Cancel any existing profit target orders
        if profit_targets:
            self.logger.info("\nCanceling Existing Orders:")
            self.logger.info("-"*40)
            for pt in profit_targets:
                order_id = pt.get('orderId')
                self.logger.info(f"Canceling order ID: {order_id}")
                cancel_success = self.cancel_order(order_id)
                if cancel_success:
                    self.logger.info("✓ Successfully canceled existing order")
                else:
                    self.logger.info("⚠️ Failed to cancel existing order")
            
        # Place limit order at profit target
        self.logger.info("\nSubmitting Order:")
        self.logger.info("-"*40)
        
        result = self.place_limit_order(order_type, self.position_size, target_price)
        
        if result.get("success", False):
            order_id = result.get("orderId")
            self.logger.info("\n✅ LIMIT ORDER PLACED SUCCESSFULLY")
            self.logger.info(f"Order ID: {order_id}")
            self.logger.info(f"Type: {order_type} LIMIT")
            self.logger.info(f"Size: {self.position_size} shares")
            self.logger.info(f"Price: ${target_price:.2f}")
            self.logger.info(f"Target: {target_type}")
            # Don't wait for limit order to fill, just return success
            return True
        else:
            error_msg = result.get("error", "Unknown error")
            self.logger.error("\n❌ LIMIT ORDER PLACEMENT FAILED")
            self.logger.error(f"Error: {error_msg}")
            self.logger.error("Order Details:")
            self.logger.error(f"  Type: {order_type} LIMIT")
            self.logger.error(f"  Size: {self.position_size} shares")
            self.logger.error(f"  Price: ${target_price:.2f}")
            return False
            
    def wait_for_order_fill(self, timeout_seconds: int = 60) -> bool:
        """Wait for market order to be filled by checking if position exists for our symbol"""
        self.logger.info("\n" + "="*80)
        self.logger.info("ORDER FILL MONITORING")
        self.logger.info("="*80)
        
        self.logger.info("\nInitial State:")
        self.logger.info("-"*40)
        self.logger.info(f"Symbol: {self.symbol}")
        self.logger.info(f"Position Size: {self.position_size} shares")
        self.logger.info(f"Timeout: {timeout_seconds} seconds")
        
        start_time = time.time()
        initial_position_size = self.position_size
        attempt_count = 0
        
        while time.time() - start_time < timeout_seconds:
            attempt_count += 1
            elapsed = time.time() - start_time
            
            self.logger.info(f"\nCheck Attempt {attempt_count}:")
            self.logger.info("-"*40)
            self.logger.info(f"Elapsed Time: {elapsed:.1f}s")
            self.logger.info("Fetching account data...")
            
            # Get current account data to check positions
            account_data = self.get_account_data()
            if not account_data:
                self.logger.warning("❌ Failed to get account data")
                self.logger.info(f"Sleeping for 1 second before retry...")
                time.sleep(1)
                continue
            
            self.logger.info("✓ Account data received")
            
            # Check if we have a position for our symbol
            position_found = False
            current_size = 0
            
            for account in account_data:
                for position in account.get('positions', []):
                    if position.get('symbol') == self.symbol:
                        position_found = True
                        current_size = abs(position.get('quantity', 0))
                        break
                if position_found:
                    break
            
            # Log position check results
            self.logger.info("\nPosition Check:")
            self.logger.info("-"*40)
            self.logger.info(f"Position Found: {'Yes' if position_found else 'No'}")
            self.logger.info(f"Current Size: {current_size}")
            self.logger.info(f"Initial Size: {initial_position_size}")
            
            # If we found a position and it's different from initial size, order was filled
            if position_found and current_size != initial_position_size:
                self.logger.info("\n✅ ORDER FILLED")
                self.logger.info(f"Position size changed: {initial_position_size} → {current_size}")
                self.logger.info(f"Total time to fill: {elapsed:.1f} seconds")
                return True
            
            # If we were flat and now have a position, order was filled
            if initial_position_size == 0 and position_found and current_size != 0:
                self.logger.info("\n✅ ORDER FILLED")
                self.logger.info(f"New position created: {current_size} shares")
                self.logger.info(f"Total time to fill: {elapsed:.1f} seconds")
                return True
            
            remaining = timeout_seconds - elapsed
            self.logger.info(f"\nOrder not filled yet. Sleeping for 1 second...")
            self.logger.info(f"Time remaining: {remaining:.1f}s")
            time.sleep(1)  # Check every second
        
        self.logger.warning("\n⚠️ ORDER FILL TIMEOUT")
        self.logger.warning(f"No fill detected after {timeout_seconds} seconds")
        self.logger.warning(f"Final state: Size={current_size} | Initial={initial_position_size}")
        return False

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order by ID"""
        if not self.account_number or not order_id:
            self.logger.error("Missing account number or order ID for cancellation")
            return False
        
        # Ensure order_id is properly formatted
        try:
            # Convert to integer if it's a string containing digits
            if isinstance(order_id, str) and order_id.isdigit():
                order_id = int(order_id)
                self.logger.info(f"Converting order ID to integer: {order_id}")
        except ValueError as e:
            self.logger.warning(f"Order ID format conversion failed: {e}")
            # If conversion fails, keep original value
            pass
            
        url = f"https://api.schwabapi.com/trader/v1/accounts/{self.account_number}/orders/{order_id}"
        headers = {
            "Authorization": f"Bearer {self.tokens['access_token']}",
            "Accept": "*/*"  # Per API documentation
        }
        
        try:
            self.logger.info(f"Attempting to cancel order ID: {order_id}")
            response = requests.delete(url, headers=headers)
            
            if response.status_code in [200, 201, 204]:
                self.logger.info(f"Successfully cancelled order {order_id}")
                return True
            else:
                error_msg = f"Failed to cancel order: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                return False
        except Exception as e:
            error_msg = f"Error cancelling order: {str(e)}"
            self.logger.error(error_msg)
            return False
        
    def save_tokens(self, tokens):
        # Add expiration timestamps if not present
        if 'expires_in' in tokens and 'access_token_expires_at' not in tokens:
            current_time = int(time.time())
            tokens['access_token_expires_at'] = current_time + tokens['expires_in']
            tokens['refresh_token_expires_at'] = current_time + (7 * 24 * 60 * 60)  # 7 days
        
        with open(TOKEN_FILE, 'w') as f:
            json.dump(tokens, f)
        print("Tokens saved successfully")

    def load_tokens(self):
        if os.path.exists(TOKEN_FILE):
            with open(TOKEN_FILE, 'r') as f:
                return json.load(f)
        return None

    def get_authorization_code(self):
        auth_url = f"https://api.schwabapi.com/v1/oauth/authorize?response_type=code&client_id={APP_KEY}&redirect_uri={REDIRECT_URI}&scope=readonly"
        print("Opening browser for authentication...")
        print(f"If browser doesn't open, visit: {auth_url}")
        import webbrowser
        webbrowser.open(auth_url)
        returned_url = input("Paste the full returned URL here: ")
        
        parsed_url = urllib.parse.urlparse(returned_url)
        code = urllib.parse.parse_qs(parsed_url.query).get('code', [None])[0]
        if not code:
            raise ValueError("Failed to extract authorization code")
        return code

    def get_initial_tokens(self, code):
        credentials = f"{APP_KEY}:{APP_SECRET}"
        base64_credentials = base64.b64encode(credentials.encode()).decode("utf-8")
        
        headers = {
            "Authorization": f"Basic {base64_credentials}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        payload = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI
        }
        
        response = requests.post("https://api.schwabapi.com/v1/oauth/token", headers=headers, data=payload)
        if response.status_code != 200:
            raise Exception(f"Failed to get tokens: {response.text}")
            
        tokens = response.json()
        # Add expiration timestamps
        current_time = int(time.time())
        tokens['access_token_expires_at'] = current_time + tokens['expires_in']
        tokens['refresh_token_expires_at'] = current_time + (7 * 24 * 60 * 60)  # 7 days
        self.save_tokens(tokens)
        return tokens

    def refresh_tokens(self, refresh_token):
        try:
            credentials = f"{APP_KEY}:{APP_SECRET}"
            base64_credentials = base64.b64encode(credentials.encode()).decode("utf-8")
            
            headers = {
                "Authorization": f"Basic {base64_credentials}",
                "Content-Type": "application/x-www-form-urlencoded"
            }
            payload = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token
            }
            
            response = requests.post("https://api.schwabapi.com/v1/oauth/token", headers=headers, data=payload)
            
            if response.status_code == 200:
                tokens = response.json()
                # Add expiration timestamps
                current_time = int(time.time())
                tokens['access_token_expires_at'] = current_time + tokens['expires_in']
                tokens['refresh_token_expires_at'] = current_time + (7 * 24 * 60 * 60)  # 7 days
                self.save_tokens(tokens)
                return tokens
                
            # If refresh token is expired/invalid, we need to restart OAuth flow
            if response.status_code == 400 and "invalid_request" in response.text:
                print("Refresh token expired or invalid. Initiating new OAuth flow...")
                code = self.get_authorization_code()
                return self.get_initial_tokens(code)
                
            # Other errors
            raise Exception(f"Failed to refresh tokens: {response.text}")
            
        except Exception as e:
            print(f"Error during token refresh: {e}")
            print("Initiating new OAuth flow...")
            code = self.get_authorization_code()
            return self.get_initial_tokens(code)

    def get_account_equity(self):
        """Get current account equity"""
        account_data = self.get_account_data()
        if account_data and len(account_data) > 0:
            return account_data[0]['balances'].get('equity', 0)
        return 0
    
    def has_already_scaled_in(self) -> bool:
        """Check if we already have the maximum number of scaled positions (2 total) across all positions"""
        if self.current_position == "FLAT":
            return False
        
        # Get all account positions from brokerage
        account_data = self.get_account_data()
        if not account_data:
            return False
        
        scaled_positions_count = 0
        
        # Check all positions across all accounts
        for account in account_data:
            for position in account.get('positions', []):
                if abs(position.get('quantity', 0)) > 0:  # Has a position
                    symbol = position.get('symbol', '')
                    position_cost = abs(position.get('quantity', 0)) * position.get('average_price', 0)
                    
                    # Determine if this position is scaled by checking if cost exceeds normal percentage
                    if self.use_percentage and self.percentage:
                        equity = account['balances'].get('equity', 0)
                        if equity > 0:
                            normal_position_cost = equity * (self.percentage / 100.0)
                            # If position cost is greater than normal percentage, it has been scaled
                            if position_cost > normal_position_cost:
                                scaled_positions_count += 1
                                print(f"Detected scaled position: {symbol} (${position_cost:.2f} > ${normal_position_cost:.2f})")
        
        # Allow maximum of 2 scaled positions across all tickers
        max_scaled_positions = 2
        if scaled_positions_count >= max_scaled_positions:
            print(f"Maximum scaled positions reached: {scaled_positions_count}/{max_scaled_positions}")
            return True
        
        return False

    def calculate_band_stability(self, band_values: list) -> bool:
        """Calculate if a volatility band has been stable over the period"""
        self.logger.info("\n" + "="*80)
        self.logger.info("BAND STABILITY ANALYSIS")
        self.logger.info("="*80)
        
        # Log initial band values with timestamps
        self.logger.info("\nHistorical Band Values:")
        self.logger.info("-"*40)
        current_time = datetime.now(pytz.timezone('America/Los_Angeles'))
        for i, value in enumerate(band_values):
            # Calculate approximate time for each bar (5 min intervals)
            bar_time = current_time - timedelta(minutes=5*(len(band_values)-i-1))
            self.logger.info(f"Bar {i} ({bar_time.strftime('%H:%M:%S')}): ${value:.2f}")
        
        # Calculate and log statistical measures
        mean_value = sum(band_values) / len(band_values)
        std_dev = (sum((x - mean_value) ** 2 for x in band_values) / len(band_values)) ** 0.5
        max_value = max(band_values)
        min_value = min(band_values)
        total_range = max_value - min_value
        
        self.logger.info("\nStatistical Analysis:")
        self.logger.info("-"*40)
        self.logger.info(f"Mean Value: ${mean_value:.2f}")
        self.logger.info(f"Standard Deviation: ${std_dev:.2f}")
        self.logger.info(f"Range: ${total_range:.2f} (${min_value:.2f} - ${max_value:.2f})")
        
        # Calculate percentage changes for the band relative to itself
        self.logger.info("\nBar-to-Bar Changes:")
        self.logger.info("-"*40)
        
        band_changes = []
        for i in range(1, len(band_values)):
            prev_value = band_values[i-1]
            curr_value = band_values[i]
            pct_change = abs((curr_value - prev_value) / prev_value)
            band_changes.append(pct_change)
            
            # Calculate time for these bars
            prev_time = current_time - timedelta(minutes=5*(len(band_values)-i))
            curr_time = current_time - timedelta(minutes=5*(len(band_values)-i-1))
            
            # Log each change with details and direction
            change_direction = "▲" if curr_value > prev_value else "▼" if curr_value < prev_value else "="
            self.logger.info(f"\n{prev_time.strftime('%H:%M:%S')} → {curr_time.strftime('%H:%M:%S')}:")
            self.logger.info(f"  Previous: ${prev_value:.2f}")
            self.logger.info(f"  Current:  ${curr_value:.2f} {change_direction}")
            self.logger.info(f"  Change:   {pct_change*100:.3f}% ({change_direction})")
        
        # Define stability threshold (0.1% change per bar)
        stability_threshold = 0.0019
        self.logger.info(f"\nStability Analysis:")
        self.logger.info("-"*40)
        self.logger.info(f"Threshold: {stability_threshold*100:.2f}% maximum change allowed per bar")
        
        # Calculate average change
        avg_change = sum(band_changes) / len(band_changes) if band_changes else 0
        self.logger.info(f"Average Change: {avg_change*100:.3f}%")
        
        # Check each change against threshold
        unstable_changes = [
            (i+1, change) 
            for i, change in enumerate(band_changes) 
            if change >= stability_threshold
        ]
        
        if unstable_changes:
            self.logger.info("\n⚠️ UNSTABLE CHANGES DETECTED:")
            for bar, change in unstable_changes:
                bar_time = current_time - timedelta(minutes=5*(len(band_values)-bar))
                self.logger.info(f"  Bar {bar} ({bar_time.strftime('%H:%M:%S')}): {change*100:.3f}% change")
                self.logger.info(f"  Exceeds {stability_threshold*100:.2f}% threshold by {(change-stability_threshold)*100:.3f}%")
            self.logger.info("\n❌ RESULT: Band is NOT stable")
            return False
        else:
            self.logger.info("\nAll changes within stability threshold:")
            for i, change in enumerate(band_changes):
                bar_time = current_time - timedelta(minutes=5*(len(band_values)-i-1))
                self.logger.info(f"  Bar {i+1} ({bar_time.strftime('%H:%M:%S')}): {change*100:.3f}% ✓")
            self.logger.info("\n✅ RESULT: Band is stable")
            return True

    def should_scale_based_on_signal(self, signal_direction: str) -> bool:
        """
        Determine if we should scale into a position based on:
        1. For new positions (FLAT): Always allow
        2. For scaling existing positions:
           - Check for opposite exceedences in last 30 minutes
           - Verify band stability over last hour
           - Apply strategy direction logic
        """
        # If we're flat, always allow new positions
        if self.current_position == "FLAT":
            self.logger.info(f"Position is FLAT - Allowing new {signal_direction} position")
            return True
            
        # For scaling, get price data for the last 1/2 hour
        df = self.get_price_data()
        if df is None or len(df) < 12:  # Need at least 12 bars
            return False
        
        # Get volatility metrics for all bars we'll analyze
        metrics_list = [self.calculate_volatility_metrics(df.iloc[:i+1]) 
                       for i in range(len(df)-12, len(df))]
        
        # Last 6 bars (30 minutes) for exceedence check
        last_6_metrics = metrics_list[-6:]
        last_6_bars = df.iloc[-6:]
        
        # Check for opposite exceedences using close prices
        self.logger.info("\nChecking for Opposite Exceedences:")
        self.logger.info("-" * 40)
        
        has_opposite_exceedence = False
        for i, metrics in enumerate(last_6_metrics):
            close_price = last_6_bars.iloc[i]['close']
            bar_time = last_6_bars.index[i].strftime("%H:%M:%S")
            
            self.logger.info(f"\nBar {i} ({bar_time}):")
            self.logger.info(f"  Close: ${close_price:.2f}")
            self.logger.info(f"  High Band: ${metrics['high_band']:.2f}")
            self.logger.info(f"  Low Band: ${metrics['low_band']:.2f}")
            
            if signal_direction == "LONG" and close_price < metrics['low_band']:
                self.logger.info(f"  ⚠️ Found opposite exceedence: Close ${close_price:.2f} < Low Band ${metrics['low_band']:.2f}")
                has_opposite_exceedence = True
                break
            if signal_direction == "SHORT" and close_price > metrics['high_band']:
                self.logger.info(f"  ⚠️ Found opposite exceedence: Close ${close_price:.2f} > High Band ${metrics['high_band']:.2f}")
                has_opposite_exceedence = True
                break
            self.logger.info("  ✓ No opposite exceedence found")
        
        if has_opposite_exceedence:
            self.logger.info("\nScaling rejected: Found opposite exceedence")
            return False
        
        self.logger.info("\nNo opposite exceedences found in last 6 bars")
        
        # Check band stability over last hour (12 bars)
        self.logger.info("\nAnalyzing Band Stability Over Last Hour:")
        self.logger.info("-" * 40)
        
        high_band_values = [m['high_band'] for m in metrics_list]
        low_band_values = [m['low_band'] for m in metrics_list]
        
        # Check stability based on signal direction
        if signal_direction == "LONG":
            self.logger.info("\nLONG Signal - Checking Low Band Stability:")
            if not self.calculate_band_stability(low_band_values):
                self.logger.info("Scaling rejected: Low band not stable enough for LONG scaling")
                return False
            self.logger.info("✓ Low band stability confirmed for LONG scaling")
        else:  # SHORT
            self.logger.info("\nSHORT Signal - Checking High Band Stability:")
            if not self.calculate_band_stability(high_band_values):
                self.logger.info("Scaling rejected: High band not stable enough for SHORT scaling")
                return False
            self.logger.info("✓ High band stability confirmed for SHORT scaling")
        
        # If we get here, bands are stable and no opposite exceedences
        # Apply strategy direction logic for scaling
        if self.strategy == 'momentum':
            should_scale = self.current_position == signal_direction and self.current_position != "FLAT"
            self.logger.info(f"Momentum strategy check: Current={self.current_position}, Signal={signal_direction}, Allow Scale={should_scale}")
            return should_scale
        else:  # reversal
            should_scale = self.current_position != signal_direction and self.current_position != "FLAT"
            self.logger.info(f"Reversal strategy check: Current={self.current_position}, Signal={signal_direction}, Allow Scale={should_scale}")
            return should_scale

    # def count_active_processes(self) -> int:
    #     """Count currently running exceedence_strategy processes"""
    #     count = 0
    #     for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
    #         try:
    #             if proc.info['name'] == 'python' and proc.info['cmdline']:
    #                 if 'exceedence_strategy.py' in ' '.join(proc.info['cmdline']):
    #                     count += 1
    #         except (psutil.NoSuchProcess, psutil.AccessDenied):
    #             continue
    #     return count

    # def spawn_new_process(self) -> Tuple[bool, str]:
    #     """Spawn new exceedence_strategy process if under limit"""
    #     if self.count_active_processes() >= 50:
    #         return False, "Maximum process limit (50) reached"
        
    #     cmd = [
    #         'python', 'exceedence_strategy.py', self.symbol,
    #         '--strategy', self.strategy,
    #         '--auto', 'yes',
    #         '--direction', self.direction
    #     ]
        
    #     if self.use_percentage and self.percentage:
    #         cmd.extend(['--percentage', str(self.percentage)])
    #     elif self.shares:
    #         cmd.extend(['--shares', str(self.shares)])
        
    #     try:
    #         proc = subprocess.Popen(cmd)
    #         return True, f"Spawned new process (PID: {proc.pid}) for {self.symbol}"
    #     except Exception as e:
    #         return False, f"Failed to spawn process: {e}"
    
    def calculate_shares_from_percentage(self, current_price):
        """Calculate number of shares based on percentage of account equity"""
        if not self.use_percentage or not self.percentage:
            return self.shares if self.shares else 1
            
        equity = self.get_account_equity()
        if equity <= 0 or current_price <= 0:
            print(f"Unable to calculate shares: equity=${equity:.2f}, price=${current_price:.2f}")
            return 1  # Default to 1 share
            
        # Calculate position value based on percentage
        position_value = equity * (self.percentage / 100.0)
        shares = int(position_value / current_price)
        
        # Ensure at least 1 share
        shares = max(1, shares)
        
        print(f"Calculated {shares} shares: {self.percentage}% of ${equity:.2f} = ${position_value:.2f} / ${current_price:.2f}")
        return shares

    def get_account_data(self):
        """Get account data including positions and balances"""
        url = "https://api.schwabapi.com/trader/v1/accounts?fields=positions"
        headers = {"Authorization": f"Bearer {self.tokens['access_token']}"}
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                accounts = response.json()
                account_data = []
                
                for account in accounts:
                    if 'securitiesAccount' not in account:
                        continue
                        
                    sec_account = account['securitiesAccount']
                    current_balances = sec_account.get('currentBalances', {})
                    
                    # Extract relevant data
                    account_info = {
                        'account_id': sec_account.get('accountNumber'),
                        'type': sec_account.get('type'),
                        'is_day_trader': sec_account.get('isDayTrader', False),
                        'round_trips': sec_account.get('roundTrips', 0),
                        'balances': {
                            'cash': current_balances.get('cashBalance', 0),
                            'buying_power': current_balances.get('buyingPower', 0),
                            'day_trading_buying_power': current_balances.get('dayTradingBuyingPower', 0),
                            'liquidation_value': current_balances.get('liquidationValue', 0),
                            'long_market_value': current_balances.get('longMarketValue', 0),
                            'short_market_value': current_balances.get('shortMarketValue', 0),
                            'equity': current_balances.get('equity', 0),
                            'maintenance_requirement': current_balances.get('maintenanceRequirement', 0),
                            'maintenance_call': current_balances.get('maintenanceCall', 0),
                            'reg_t_call': current_balances.get('regTCall', 0)
                        },
                        'positions': []
                    }
                    
                    # Extract positions if available
                    positions = sec_account.get('positions', [])
                    for pos in positions:
                        instrument = pos.get('instrument', {})
                        position_info = {
                            'symbol': instrument.get('symbol', ''),
                            'quantity': pos.get('longQuantity', 0) - pos.get('shortQuantity', 0),
                            'market_value': pos.get('marketValue', 0),
                            'average_price': pos.get('averagePrice', 0),
                            'current_day_profit_loss': pos.get('currentDayProfitLoss', 0),
                            'current_day_profit_loss_pct': pos.get('currentDayProfitLossPercentage', 0)
                        }
                        account_info['positions'].append(position_info)
                    
                    account_data.append(account_info)
                
                return account_data
            elif response.status_code == 401:
                print("Token expired, refreshing...")
                self.tokens = self.refresh_tokens(self.tokens["refresh_token"])
                return self.get_account_data()
            else:
                print(f"Failed to fetch account data: {response.status_code} - {response.text}")
                return None
        except requests.RequestException as e:
            print(f"Error fetching account data: {e}")
            return None

    def ensure_valid_tokens(self):
        tokens = self.load_tokens()
        if tokens:
            # Check if refresh token is expired
            current_time = int(time.time())
            if 'refresh_token_expires_at' in tokens and current_time >= tokens['refresh_token_expires_at']:
                print("Refresh token expired. Initiating new OAuth flow...")
                code = self.get_authorization_code()
                return self.get_initial_tokens(code)
            
            # Check if access token is expired
            if 'access_token_expires_at' in tokens and current_time >= tokens['access_token_expires_at']:
                print("Access token expired, attempting refresh...")
                return self.refresh_tokens(tokens["refresh_token"])
            
            # Validate access token
            test_url = "https://api.schwabapi.com/trader/v1/accounts"
            headers = {"Authorization": f"Bearer {tokens['access_token']}"}
            try:
                response = requests.get(test_url, headers=headers, timeout=10)
                if response.status_code == 200:
                    print("Access token validated successfully")
                    return tokens
                elif response.status_code == 401:
                    print("Access token expired, refreshing...")
                    return self.refresh_tokens(tokens["refresh_token"])
            except requests.RequestException as e:
                print(f"Token validation failed: {e}")
        
        print("No valid tokens found. Initiating new OAuth flow...")
        code = self.get_authorization_code()
        return self.get_initial_tokens(code)

    def get_price_data(self):
        self.logger.info("\n" + "="*80)
        self.logger.info("PRICE DATA FETCH")
        self.logger.info("="*80)
        
        # Setup request parameters
        url = "https://api.schwabapi.com/marketdata/v1/pricehistory"
        end_date = int(datetime.now().timestamp() * 1000)
        
        self.logger.info("\nRequest Parameters:")
        self.logger.info("-"*40)
        
        params = {
            "symbol": self.symbol,
            "periodType": "day",
            "period": 5,
            "frequencyType": "minute",
            "frequency": 5,
            "needExtendedHoursData": "true",
            "endDate": end_date
        }
        
        # Log request details
        self.logger.info(f"URL: {url}")
        for key, value in params.items():
            self.logger.info(f"{key}: {value}")
            
        headers = {
            "Authorization": f"Bearer {self.tokens['access_token']}",
            "Accept": "application/json"
        }
        
        try:
            self.logger.info("\nSending Request...")
            self.logger.info("-"*40)
            
            start_time = time.time()
            response = requests.get(url, headers=headers, params=params, timeout=15)
            request_time = time.time() - start_time
            
            self.logger.info(f"Request completed in {request_time:.2f} seconds")
            self.logger.info(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                self.logger.info("\nProcessing Response...")
                self.logger.info("-"*40)
                
                data = response.json()
                if not data.get("empty", True) and "candles" in data:
                    candle_count = len(data["candles"])
                    self.logger.info(f"Received {candle_count} candles")
                    
                    # Create DataFrame
                    df = pd.DataFrame(data["candles"])
                    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms', utc=True)
                    df['datetime'] = df['datetime'].dt.tz_convert('America/Los_Angeles')
                    df.set_index('datetime', inplace=True)
                    
                    # Log data summary
                    start_time = df.index[0].strftime('%Y-%m-%d %H:%M:%S')
                    end_time = df.index[-1].strftime('%Y-%m-%d %H:%M:%S')
                    self.logger.info(f"Data Range: {start_time} to {end_time}")
                    self.logger.info(f"Total Bars: {len(df)}")
                    
                    self.logger.info("\n✅ Price data processed successfully")
                    return df
                else:
                    self.logger.error("\n❌ No price data in response")
                    return None
            elif response.status_code == 401:
                self.logger.info("\n⚠️ Token expired, refreshing...")
                self.tokens = self.refresh_tokens(self.tokens["refresh_token"])
                return self.get_price_data()
            else:
                self.logger.error(f"\n❌ Failed to fetch price data: {response.status_code}")
                self.logger.error(f"Response: {response.text}")
                return None
        except requests.RequestException as e:
            self.logger.error(f"\n❌ Request error: {str(e)}")
            return None

    def calculate_volatility_metrics(self, df):
        """Calculate volatility metrics using the previous bar's data"""
        if len(df) < 20:  # Need minimum data for meaningful analysis
            return None
            
        # Use previous bar's close for calculations
        prev_bar = df.iloc[-2]  # Previous bar
        lookback = min(2000, len(df)-2)  # Use last 20 bars excluding current bar
        
        # Get current system time for accurate second tracking
        current_time = datetime.now(pytz.timezone('America/Los_Angeles'))
        api_minute = df.index[-1].minute
        
        # Only update last_api_minute, counters are now reset after order fill confirmation
        if api_minute != self.last_api_minute:
            self.last_api_minute = api_minute
            
            # Log counter state for debugging
            if self.orders_placed > 0 or self.scale_in_count > 0:
                self.logger.info(f"Current counter state - orders_placed: {self.orders_placed}, scale_in_count: {self.scale_in_count}")
        
        # Calculate rolling means and standard deviations up to previous bar
        highside_vol = df['high'] - df['close']
        lowside_vol = df['low'] - df['close']
        mean_highside = highside_vol.iloc[:-1].rolling(window=lookback).mean().iloc[-1]
        mean_lowside = lowside_vol.iloc[:-1].rolling(window=lookback).mean().iloc[-1]
        std_highside = highside_vol.iloc[:-1].rolling(window=lookback).std().iloc[-1]
        std_lowside = lowside_vol.iloc[:-1].rolling(window=lookback).std().iloc[-1]
        
        # Calculate volatility bands based on previous bar's close
        high_side_limit = prev_bar['close'] + (std_highside + mean_highside)
        low_side_limit = prev_bar['close'] - (std_lowside - mean_lowside)
        
        # Calculate exceedances for current bar
        current_bar = df.iloc[-1]
        high_exceedance = current_bar['high'] - high_side_limit if current_bar['high'] > high_side_limit else 0
        low_exceedance = low_side_limit - current_bar['low'] if current_bar['low'] < low_side_limit else 0
        
        # Calculate relative distances and levels
        current_price = current_bar['close']
        band_range = high_side_limit - low_side_limit
        band_midpoint = low_side_limit + (band_range / 2)
        
        # Calculate distance from each band as percentage
        distance_to_high = ((high_side_limit - current_price) / band_range) * 100
        distance_to_low = ((current_price - low_side_limit) / band_range) * 100
        
        # Calculate position within band range as percentage (0% = at lower band, 100% = at upper band)
        position_in_range = ((current_price - low_side_limit) / band_range) * 100
        
        # Check for trading signals using system time
        seconds = current_time.second
        near_minute_end = seconds >= 55 and seconds <= 59
        
        trading_signal = None
        signal_direction = None
        
        if near_minute_end:
            self.logger.info(f"Evaluating trading signals at {current_time.strftime('%H:%M:%S')}")
            self.logger.info(f"Current Price: ${current_price:.2f} | Position in Range: {position_in_range:.1f}%")
            self.logger.info(f"Bands: High=${high_side_limit:.2f} | Low=${low_side_limit:.2f}")
            
            # Check if we're past 12:30 PM PST cutoff
            current_hour = current_time.hour
            current_minute = current_time.minute
            after_cutoff = (current_hour > 12) or (current_hour == 12 and current_minute >= 30)
            
            if after_cutoff:
                self.logger.info("No signal generated - After 12:30 PM PST cutoff")
            
            # Only generate trading signals before cutoff time
            if not after_cutoff:
                # Strategy determines the signal direction based on price position in the band
                if self.strategy == 'reversal':
                    self.logger.info("Evaluating reversal strategy conditions")
                    # Reversal strategy: SHORT at top of range, LONG at bottom of range
                    if position_in_range >= 99 and self.direction in ['short', 'both']:
                        trading_signal = f"SHORT Signal @ {current_time.strftime('%H:%M:%S')}"
                        signal_direction = "SHORT"
                        self.logger.info(f"Reversal SHORT signal triggered: Price at {position_in_range:.1f}% of range")
                    elif position_in_range <= 1 and self.direction in ['long', 'both']:
                        trading_signal = f"LONG Signal @ {current_time.strftime('%H:%M:%S')}"
                        signal_direction = "LONG"
                        self.logger.info(f"Reversal LONG signal triggered: Price at {position_in_range:.1f}% of range")
                    else:
                        self.logger.info(f"No reversal signal: Price at {position_in_range:.1f}% of range")
                else:  # momentum strategy
                    self.logger.info("Evaluating momentum strategy conditions")
                    # Momentum strategy: LONG at top of range, SHORT at bottom of range
                    if position_in_range >= 99 and self.direction in ['long', 'both']:
                        trading_signal = f"LONG Signal @ {current_time.strftime('%H:%M:%S')}"
                        signal_direction = "LONG"
                        self.logger.info(f"Momentum LONG signal triggered: Price at {position_in_range:.1f}% of range")
                    elif position_in_range <= 1 and self.direction in ['short', 'both']:
                        trading_signal = f"SHORT Signal @ {current_time.strftime('%H:%M:%S')}"
                        signal_direction = "SHORT"
                        self.logger.info(f"Momentum SHORT signal triggered: Price at {position_in_range:.1f}% of range")
                    else:
                        self.logger.info(f"No momentum signal: Price at {position_in_range:.1f}% of range")
        
        return {
            'high_band': high_side_limit,
            'low_band': low_side_limit,
            'high_exceedance': high_exceedance,
            'low_exceedance': low_exceedance,
            'trading_signal': trading_signal,
            'signal_direction': signal_direction,
            'distance_to_high': distance_to_high,
            'distance_to_low': distance_to_low,
            'position_in_range': position_in_range,
            'current_price': current_price
        }

    def close_position(self) -> Tuple[bool, str]:
        """Close current position without opening a new one"""
        self.logger.info(f"Attempting to close position: {self.current_position} | Size: {self.position_size} | Entry: ${self.entry_price:.2f}")
        
        # Sync positions first to make sure we're working with current data
        if not self.sync_current_positions():
            self.logger.error("Failed to sync positions before closing")
            return False, "Failed to sync positions"
            
        # Cancel any existing profit target orders
        _, profit_targets = self.sync_orders()
        if profit_targets:
            self.logger.info("Canceling profit target orders before closing position")
            for pt in profit_targets:
                order_id = pt.get('orderId')
                self.logger.info(f"Canceling order ID: {order_id}")
                self.cancel_order(order_id)
            
        # Close existing position if any
        if self.current_position != "FLAT" and self.position_size > 0:
            close_type = "SELL" if self.current_position == "LONG" else "BUY_TO_COVER"
            
            # Get current price for P&L calculation
            price_data = self.get_current_price(self.symbol)
            current_price = price_data.get('lastPrice', 0) if price_data else 0
            
            self.logger.info(f"Closing {self.current_position} position:")
            self.logger.info(f"Size: {self.position_size} shares | Entry: ${self.entry_price:.2f} | Current: ${current_price:.2f}")
            
            # Calculate P&L if we have valid prices
            if current_price > 0 and self.entry_price > 0:
                if self.current_position == "LONG":
                    pnl = (current_price - self.entry_price) * self.position_size
                else:  # SHORT
                    pnl = (self.entry_price - current_price) * self.position_size
                pnl_pct = (pnl / (self.entry_price * self.position_size)) * 100
                self.logger.info(f"P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
            
            close_result = self.place_market_order(close_type, self.position_size)
            
            if not close_result.get("success", False):
                error_msg = close_result.get("error", "Unknown error")
                self.logger.error(f"Failed to close position: {error_msg}")
                return False, f"Failed to close {self.current_position} position: {error_msg}"
                
            result_message = f"Closed {self.current_position} position of {self.position_size} shares"
            
            # Reset position tracking
            old_position = self.current_position
            old_size = self.position_size
            
            self.current_position = "FLAT"
            self.position_size = 0
            self.entry_price = 0
            self.entry_time = None
            
            self.logger.info(f"Position tracking reset: {old_position}({old_size}) → FLAT(0)")
            
            # Reset both counters to allow new positions immediately
            if self.orders_placed > 0 or self.scale_in_count > 0:
                self.logger.info("Resetting position counters:")
                if self.orders_placed > 0:
                    self.logger.info(f"  orders_placed: {self.orders_placed} → 0")
                    self.orders_placed = 0
                if self.scale_in_count > 0:
                    self.logger.info(f"  scale_in_count: {self.scale_in_count} → 0")
                    self.scale_in_count = 0
            
            return True, result_message
        
        self.logger.info("No position to close")
        return False, "No position to close"
    
    def update_data(self):
        # Get current time and check if we're in the last 10 seconds of the minute
        current_time = datetime.now(pytz.timezone('America/Los_Angeles'))
        if current_time.second < 50:  # Only update in last 10 seconds of minute
            return
            
        # Get price data
        df = self.get_price_data()
        if df is None or df.empty:
            print("No price data available")
            return
        
        current_price = df['close'].iloc[-1]
        price_change = df['close'].iloc[-1] - df['close'].iloc[0]
        price_change_pct = (price_change / df['close'].iloc[0]) * 100
        change_symbol = '▲' if price_change >= 0 else '▼'
        volume = df['volume'].iloc[-1]
        
        # Calculate volatility metrics
        vol_metrics = self.calculate_volatility_metrics(df)
        
        # Get account data and sync positions
        self.sync_current_positions()
        account_data = self.get_account_data()
        
        # Check if we need to place a profit target order
        if self.current_position != "FLAT" and self.position_size > 0:
            _, profit_targets = self.sync_orders()
            if not profit_targets:
                self.set_profit_target_order()
        
        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Print price and volatility update
        # Get current system time
        current_time = datetime.now(pytz.timezone('America/Los_Angeles'))
        print(f"{self.symbol}: ${current_price:.2f} {change_symbol} ${abs(price_change):.2f} ({abs(price_change_pct):.2f}%) Vol: {volume:,.0f} | {current_time.strftime('%H:%M:%S')} PST")
        print(f"Strategy: {self.strategy.upper()} | Current Position: {self.current_position} | Size: {self.position_size}")
        print("-" * 80)
        
        if vol_metrics:
            print("\nVolatility Bands (from previous bar):")
            print(f"Upper: ${vol_metrics['high_band']:.2f} | Distance: {vol_metrics['distance_to_high']:.1f}%")
            print(f"Lower: ${vol_metrics['low_band']:.2f} | Distance: {vol_metrics['distance_to_low']:.1f}%")
            print(f"Position in Range: {vol_metrics['position_in_range']:.1f}%")
            
            # Check for trading signals
            if vol_metrics['trading_signal']:
                signal_direction = vol_metrics['signal_direction']
                print("\n🚨 TRADING SIGNAL 🚨")
                print(f"{vol_metrics['trading_signal']} - Price: ${current_price:.2f}")
                
                # Check if we need to take action (scale into existing position or open new)
                take_action = False
                quantity = self.shares if self.shares else 1  # Default to 1 if no shares specified
                
                # If we're flat and get a signal, open new position
                if self.current_position == "FLAT":
                    take_action = True
                    action_msg = f"NEW SIGNAL: Open {signal_direction} position"
                
                # If we have an existing position (any direction), scale in
                elif self.current_position in ["LONG", "SHORT"]:
                    # Always scale in regardless of signal direction
                    take_action = True
                    
                    # If signal matches position, it's a normal scale
                    if self.current_position == signal_direction:
                        action_msg = f"SCALE SIGNAL: Add to existing {self.current_position} position (averaging)"
                    # If signal is opposite, we still scale in
                    else:
                        action_msg = f"OPPOSITE SIGNAL: Still scaling into {self.current_position} position (averaging)"
                
                # Handle order execution based on signal
                if take_action:
                    print(f"\n{action_msg}")
                    
                    # Check if we've already placed an order for this signal
                    current_minute = current_time.minute
                    
                    # Use different counters based on whether we're opening a new position or scaling in
                    if self.current_position == "FLAT" and self.orders_placed > 0:
                        print(f"Order already placed for new position. Skipping to prevent duplicate orders.")
                    elif self.current_position != "FLAT" and self.scale_in_count > 0:
                        print(f"Scale-in order already placed. Skipping to prevent duplicate orders.")
                    else:
                        if self.auto_mode:
                            if self.shares or self.use_percentage:
                                print("Auto-trading enabled - Executing trade...")
                                success, message, result = self.scale_into_position(signal_direction)
                                if success:
                                    print(f"SUCCESS: {message}")
                                    print("SUCCESS: Market order placed")
                                    self.logger.info("Order placed successfully")
                                else:
                                    print(f"ERROR: {message}")
                            else:
                                print("Auto-trading enabled but no shares quantity or percentage specified")
                        else:
                            # Manual mode
                            print("Would you like to execute this trade? (y/n)")
                            user_input = input().lower()
                            if user_input == 'y':
                                success, message, result = self.scale_into_position(signal_direction)
                                if success:
                                    print(f"SUCCESS: {message}")
                                    print("SUCCESS: Market order placed")
                                    self.logger.info("Order placed successfully")
                                else:
                                    print(f"ERROR: {message}")
            
            if vol_metrics['high_exceedance'] > 0 or vol_metrics['low_exceedance'] > 0:
                print("\nExceedances:")
                if vol_metrics['high_exceedance'] > 0:
                    print(f"Above Upper: +${vol_metrics['high_exceedance']:.2f}")
                if vol_metrics['low_exceedance'] > 0:
                    print(f"Below Lower: -${vol_metrics['low_exceedance']:.2f}")
            print("-" * 80)
        
            
        # Print account data
        if account_data:
            print("\nAccount Information:")
            print("-" * 80)
            for account in account_data:
                print(f"\nAccount: {account['account_id']} ({account['type']})")
                print(f"Day Trader: {'Yes' if account['is_day_trader'] else 'No'}")
                print(f"Round Trips: {account['round_trips']}")
                
                # Print balances
                balances = account['balances']
                print("\nBalances:")
                print(f"Cash: ${balances['cash']:,.2f}")
                print(f"Equity: ${balances['equity']:,.2f}")
                print(f"Buying Power: ${balances['buying_power']:,.2f}")
                print(f"Day Trading Buying Power: ${balances['day_trading_buying_power']:,.2f}")
                print(f"Liquidation Value: ${balances['liquidation_value']:,.2f}")
                
                # Print positions with P&L
                if account['positions']:
                    print("\nPositions:")
                    for pos in account['positions']:
                        position_type = "LONG" if pos['quantity'] > 0 else "SHORT"
                        print(f"\n{position_type} {pos['symbol']}: {abs(pos['quantity']):,.0f} shares")
                        print(f"Market Value: ${abs(pos['market_value']):,.2f}")
                        print(f"Average Price: ${pos['average_price']:,.2f}")
                        pnl = pos['current_day_profit_loss']
                        pnl_pct = pos['current_day_profit_loss_pct']
                        print(f"P&L: {'▲' if pnl >= 0 else '▼'} ${abs(pnl):,.2f} ({abs(pnl_pct):.2f}%)")
                else:
                    print("\nNo open positions")
                
                # Print calls if any
                if balances['maintenance_call'] > 0 or balances['reg_t_call'] > 0:
                    print("\nCalls:")
                    if balances['maintenance_call'] > 0:
                        print(f"Maintenance Call: ${balances['maintenance_call']:,.2f}")
                    if balances['reg_t_call'] > 0:
                        print(f"Reg T Call: ${balances['reg_t_call']:,.2f}")
        
        print("\n" + "-" * 80)


    def is_market_open(self):
        """Check if the market is currently open"""
        # Get current time in market timezone (Eastern Time)
        now = datetime.now(self.market_timezone)
        
        # Check if it's a weekday (Monday to Friday)
        if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            return False
        
        # Regular market hours: 9:30 AM - 4:00 PM ET
        market_open = now.replace(hour=9, minute=35, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # Check if current time is within market hours
        return market_open <= now <= market_close
    
    def time_until_market_open(self):
        """Calculate time until market opens"""
        now = datetime.now(self.market_timezone)
        
        # If it's weekend, calculate time until Monday
        days_to_add = 0
        if now.weekday() >= 5:  # Weekend
            days_to_add = 7 - now.weekday() if now.weekday() == 6 else 1
        
        # Get next market open time
        next_open = now.replace(hour=9, minute=35, second=0, microsecond=0)
        
        # If we're past market open today, move to next business day
        if now.time() >= next_open.time() and days_to_add == 0:
            if now.weekday() == 4:  # Friday
                days_to_add = 3  # Next Monday
            else:
                days_to_add = 1  # Next day
        
        # Add required days
        if days_to_add > 0:
            next_open = (next_open + timedelta(days=days_to_add))
        
        # Calculate time difference
        time_diff = next_open - now
        
        return time_diff
    
    def time_until_market_close(self):
        """Calculate time until market closes"""
        now = datetime.now(self.market_timezone)
        today_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # Calculate time difference if market is still open
        if now < today_close:
            time_diff = today_close - now
            return time_diff
        
        return timedelta(0)  # Market is already closed
    
    def wait_for_market_open(self):
        """Wait until the market opens"""
        if self.is_market_open():
            print(f"Market is already open. Starting {self.symbol} tracker...")
            return
        
        wait_time = self.time_until_market_open()
        hours, remainder = divmod(wait_time.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Get market open time in both Eastern and Pacific time
        market_open_et = datetime.now(self.market_timezone) + wait_time
        pacific_tz = pytz.timezone('America/Los_Angeles')
        market_open_pt = market_open_et.astimezone(pacific_tz)
        
        # Display times in both timezones
        print(f"Market is closed. Waiting for market to open in {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"Expected market open: {market_open_et.strftime('%Y-%m-%d %H:%M:%S')} ET / {market_open_pt.strftime('%Y-%m-%d %H:%M:%S')} PT")
        
        # Wait with updates every minute
        remaining_seconds = wait_time.total_seconds()
        update_interval = 60  # Update wait message every minute
        
        while remaining_seconds > 0:
            sleep_time = min(update_interval, remaining_seconds)
            time.sleep(sleep_time)
            remaining_seconds -= sleep_time
            
            if remaining_seconds > 0:
                hours, remainder = divmod(remaining_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                print(f"Market opens in {int(hours)}h {int(minutes)}m {int(seconds)}s (Pacific Time)")
        
        print("Market is now open! Starting tracker...")

    def run(self):
        print(f"\nInitializing {self.symbol} exceedence trader...")
        print("Will run during market hours (9:30 AM - 4:00 PM ET / 6:30 AM - 1:00 PM PT)")
        print(f"Strategy: {self.strategy.upper()} | Direction: {self.direction.upper()}")
        if self.strategy == 'reversal':
            print("  Reversal - Scale into SHORT at upper band, Scale into LONG at lower band")
        else:
            print("  Momentum - Scale into LONG at upper band, Scale into SHORT at lower band")
        print("  Scaling Strategy - Add shares when price reaches the band again (average up/down)")
        print(f"Profit Target: 0.045% of entry price (using limit orders)")
        print("Press Ctrl+C to stop at any time")
        
        try:
            # Sync positions at startup
            self.sync_current_positions()
            print(f"Current position: {self.current_position} | Size: {self.position_size}")
            
            while True:
                # Check if market is open
                if not self.is_market_open():
                    # If we were running and market closed
                    print("\nMarket is now closed. Waiting for next market open...")
                    self.wait_for_market_open()
                
                # Get time until market closes
                close_time = self.time_until_market_close()
                hours, remainder = divmod(close_time.total_seconds(), 3600)
                minutes, seconds = divmod(remainder, 60)
                
                # Update data and display time until close
                self.update_data()
                print(f"Time until market close: {int(hours)}h {int(minutes)}m {int(seconds)}s (Pacific Time)")
                
                # Sleep for a short time before next update
                time.sleep(2)  # Update every 1 seconds
        except KeyboardInterrupt:
            print("\nStopping tracker...")


def validate_symbol(symbol):
    """Basic symbol validation"""
    if not symbol or not isinstance(symbol, str):
        return False
    # Remove any whitespace and check if we have a non-empty string
    symbol = symbol.strip()
    if not symbol:
        return False
    # Check for invalid characters
    invalid_chars = set('<>{}[]~`')
    if any(char in invalid_chars for char in symbol):
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Exceedence Trading Strategy with Average Scaling')
    parser.add_argument('symbol', type=str, help='Stock symbol to track (e.g. AAPL, MSFT, NVDA)')
    parser.add_argument('--strategy', type=str, choices=['reversal', 'momentum'], default='reversal',
                       help='Trading strategy: reversal (short at top, long at bottom) or momentum (long at top, short at bottom)')
    parser.add_argument('--auto', type=str, choices=['yes', 'no'], default='no',
                      help='Enable auto-trading mode (yes/no)')
    parser.add_argument('--shares', type=int, help='Fixed number of shares to add per scaling signal')
    parser.add_argument('--percentage', type=float, help='Percentage of account equity to use per trade')
    parser.add_argument('--direction', type=str, choices=['long', 'short', 'both'], default='both',
                      help='Trading direction (long/short/both). Default is both')
    
    args = parser.parse_args()
    symbol = args.symbol.upper()
    
    if not validate_symbol(symbol):
        print(f"Error: Invalid symbol '{symbol}'")
        sys.exit(1)
    
    # Validate auto mode parameters
    auto_mode = args.auto == 'yes'
    use_percentage = args.percentage is not None
    
    if auto_mode and args.shares is None and args.percentage is None:
        print("Error: Either --shares or --percentage must be specified when auto mode is enabled")
        sys.exit(1)
        
    if args.shares is not None and args.percentage is not None:
        print("Error: Cannot specify both --shares and --percentage. Choose one.")
        sys.exit(1)
        
    if args.percentage is not None and (args.percentage <= 0 or args.percentage > 100):
        print("Error: Percentage must be between 0 and 100")
        sys.exit(1)
    
    # Print colorful header with strategy parameters
    # ANSI color codes
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    
    # Create a rainbow border
    rainbow_border = f"{RED}█{YELLOW}█{GREEN}█{CYAN}█{BLUE}█{MAGENTA}█{RED}█{YELLOW}█{GREEN}█{CYAN}█{BLUE}█{MAGENTA}█{RESET}"
    
    print("\n" + rainbow_border * 7)
    print(f"{BG_BLUE}{BOLD}{WHITE} EXCEEDENCE STRATEGY TRADING SYSTEM {RESET}")
    print(f"{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║{RESET} {YELLOW}SYMBOL:{RESET} {GREEN}{symbol.ljust(65)}{BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}╠══════════════════════════════════════════════════════════════════════════╣{RESET}")
    print(f"{BOLD}{CYAN}║{RESET} {MAGENTA}PARAMETERS:{RESET}{' ' * 65}{BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}║{RESET}   {BLUE}Strategy:{RESET} {GREEN}{args.strategy.upper().ljust(58)}{BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}║{RESET}   {BLUE}Direction:{RESET} {GREEN}{args.direction.upper().ljust(56)}{BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}║{RESET}   {BLUE}Auto Trading:{RESET} {GREEN}{'ENABLED' if auto_mode else 'DISABLED'}{' ' * (53 - len('ENABLED' if auto_mode else 'DISABLED'))}{BOLD}{CYAN}║{RESET}")
    if auto_mode:
        if use_percentage:
            print(f"{BOLD}{CYAN}║{RESET}   {BLUE}Position Size:{RESET} {GREEN}{args.percentage}% of account equity{' ' * (51 - len(f'{args.percentage}% of account equity'))}{BOLD}{CYAN}║{RESET}")
        else:
            print(f"{BOLD}{CYAN}║{RESET}   {BLUE}Position Size:{RESET} {GREEN}{args.shares} shares (fixed){' ' * (52 - len(f'{args.shares} shares (fixed)'))}{BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}║{RESET}   {BLUE}Profit Target:{RESET} {GREEN}0.045% of entry price{' ' * 45}{BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}╚══════════════════════════════════════════════════════════════════════════╝{RESET}")
    print(rainbow_border * 7 + "\n")
    
    trader = ExceedenceTrader(
        symbol=symbol, 
        strategy=args.strategy,
        auto_mode=auto_mode, 
        shares=args.shares, 
        direction=args.direction,
        use_percentage=use_percentage,
        percentage=args.percentage
    )
    trader.run()

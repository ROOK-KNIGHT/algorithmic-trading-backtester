#!/usr/bin/env python3
"""
Charles Schwab Order Handler for Trading Operations
Supports market orders, limit orders, short and long positions with proper action types.
Makes actual API requests to Charles Schwab using connection_manager.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import logging
import requests
import json
import sys
import os
sys.path.append(os.path.dirname(__file__))
import connection_manager

class OrderHandler:
    """
    Charles Schwab order handler for managing different types of trading orders.
    Supports proper action types: BUY, SELL, SELL_SHORT, BUY_TO_COVER.
    Makes actual API requests to Charles Schwab.
    """
    
    def __init__(self):
        """
        Initialize the order handler with Schwab API integration.
        """
        self.order_history = []
        
        # Get valid tokens and account info using connection manager
        self.tokens = connection_manager.ensure_valid_tokens()
        if not self.tokens:
            raise ValueError("Failed to get valid Schwab API tokens")
        
        self.account_numbers = connection_manager.get_account_numbers(self.tokens['access_token'])
        if not self.account_numbers or len(self.account_numbers) == 0:
            raise ValueError("No account numbers found")
        
        # Use the first account
        self.account_number = self.account_numbers[0]['hashValue']
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"OrderHandler initialized with account: {self.account_number}")

    def _get_auth_headers(self):
        """Get authorization headers for API requests."""
        return {
            "Authorization": f"Bearer {self.tokens['access_token']}",
            "Accept": "application/json"
        }

    def get_account(self) -> Dict[str, Any]:
        """
        Get account information and balances
        
        Returns:
            Dict containing account information
        """
        tokens = connection_manager.ensure_valid_tokens()
        access_token = tokens["access_token"]
        
        if not self.account_number:
            # Get accounts linked to the user
            accounts_url = "https://api.schwabapi.com/trader/v1/accounts"
            headers = self._get_auth_headers()
            
            response = requests.get(accounts_url, headers=headers)
            
            if response.status_code == 200:
                accounts = response.json()
                if accounts and 'accounts' in accounts and len(accounts['accounts']) > 0:
                    # Use the first account
                    self.account_number = accounts['accounts'][0]['accountId']
                else:
                    print("No accounts found")
                    return {}
            else:
                print(f"Failed to retrieve accounts: {response.status_code}, {response.text}")
                return {}
        
        # Get account details
        account_url = f"https://api.schwabapi.com/trader/v1/accounts/{self.account_number}"
        headers = self._get_auth_headers()
        
        response = requests.get(account_url, headers=headers, 
                              params={"fields": "positions"})
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to retrieve account: {response.status_code}, {response.text}")
            return {}
    

    
    def place_market_order(self, action_type: str, symbol: str, shares: int, 
                          current_price: float = None, timestamp: datetime = None) -> Dict:
        """
        Place a market order with proper action types using Schwab API.
        
        Args:
            action_type: Order action ("BUY", "SELL", "SELL_SHORT", "BUY_TO_COVER")
            symbol: Stock symbol
            shares: Number of shares
            current_price: Current market price (optional for market orders)
            timestamp: Order timestamp
            
        Returns:
            Order execution result
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # Validate action type
        valid_actions = ["BUY", "SELL", "SELL_SHORT", "BUY_TO_COVER"]
        if action_type not in valid_actions:
            return {
                'status': 'rejected',
                'reason': f'Invalid action type: {action_type}. Must be one of {valid_actions}',
                'timestamp': timestamp
            }
        
        self.logger.info(f"Attempting to place {action_type} market order for {shares} shares of {symbol}")
        
        try:
            if shares <= 0:
                return {
                    'status': 'rejected',
                    'reason': 'Invalid share quantity',
                    'timestamp': timestamp
                }
            
            # Create order payload for Schwab API
            order_payload = {
                "orderStrategyType": "SINGLE",
                "orderType": "MARKET",
                "session": "NORMAL",
                "duration": "DAY",
                "orderLegCollection": [
                    {
                        "instruction": action_type,
                        "quantity": shares,
                        "instrument": {
                            "symbol": symbol,
                            "assetType": "EQUITY"
                        }
                    }
                ]
            }
            
            # Make API request to Schwab
            url = f"https://api.schwabapi.com/trader/v1/accounts/{self.account_number}/orders"
            headers = {
                "Authorization": f"Bearer {self.tokens['access_token']}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            response = requests.post(url, json=order_payload, headers=headers)
            
            if response.status_code in [200, 201]:
                order_id = response.headers.get('Location', '').split('/')[-1]
                
                # Calculate dollar amount if price provided
                if current_price is not None:
                    dollar_amount = shares * current_price
                    price_info = f" at ${current_price:.2f}"
                else:
                    dollar_amount = None
                    price_info = " at market price"
                
                # Record successful order
                order_record = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'action_type': action_type,
                    'order_type': 'market',
                    'shares': shares,
                    'price': current_price,
                    'dollar_amount': dollar_amount,
                    'order_id': order_id,
                    'status': 'submitted'
                }
                
                self.order_history.append(order_record)
                
                self.logger.info(f"{action_type} market order submitted: {shares} shares of {symbol}{price_info}, Order ID: {order_id}")
                
                return {
                    'status': 'submitted',
                    'symbol': symbol,
                    'action_type': action_type,
                    'shares': shares,
                    'fill_price': current_price,
                    'dollar_amount': dollar_amount,
                    'order_id': order_id,
                    'timestamp': timestamp
                }
            else:
                error_msg = f"Failed to place order: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                return {
                    'status': 'rejected',
                    'reason': error_msg,
                    'timestamp': timestamp
                }
            
        except Exception as e:
            self.logger.error(f"Error executing {action_type} market order: {str(e)}")
            return {
                'status': 'error',
                'reason': str(e),
                'timestamp': timestamp
            }
    
    def place_limit_order(self, action_type: str, symbol: str, shares: int,
                         limit_price: float, timestamp: datetime = None) -> Dict:
        """
        Place a limit order with proper action types using Schwab API.
        
        Args:
            action_type: Order action ("BUY", "SELL", "SELL_SHORT", "BUY_TO_COVER")
            symbol: Stock symbol
            shares: Number of shares
            limit_price: Limit price for the order
            timestamp: Order timestamp
            
        Returns:
            Order placement result
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # Validate action type
        valid_actions = ["BUY", "SELL", "SELL_SHORT", "BUY_TO_COVER"]
        if action_type not in valid_actions:
            return {
                'status': 'rejected',
                'reason': f'Invalid action type: {action_type}. Must be one of {valid_actions}',
                'timestamp': timestamp
            }
        
        self.logger.info(f"Attempting to place {action_type} limit order for {shares} shares of {symbol} @ ${limit_price:.2f}")
        
        try:
            if shares <= 0:
                return {
                    'status': 'rejected',
                    'reason': 'Invalid share quantity',
                    'timestamp': timestamp
                }
            
            # Create order payload for Schwab API
            order_payload = {
                "orderStrategyType": "SINGLE",
                "orderType": "LIMIT",
                "session": "SEAMLESS",  # SEAMLESS for after hours
                "duration": "GOOD_TILL_CANCEL",  # GTC keeps the order active until filled or cancelled
                "price": str(limit_price),  # Price must be a string in API
                "orderLegCollection": [
                    {
                        "instruction": action_type,
                        "quantity": shares,
                        "instrument": {
                            "symbol": symbol,
                            "assetType": "EQUITY"
                        }
                    }
                ]
            }
            
            # Make API request to Schwab
            url = f"https://api.schwabapi.com/trader/v1/accounts/{self.account_number}/orders"
            headers = {
                "Authorization": f"Bearer {self.tokens['access_token']}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            response = requests.post(url, json=order_payload, headers=headers)
            
            if response.status_code in [200, 201]:
                order_id = response.headers.get('Location', '').split('/')[-1]
                
                # Calculate dollar amount
                dollar_amount = shares * limit_price
                
                # Record successful order
                order_record = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'action_type': action_type,
                    'order_type': 'limit',
                    'shares': shares,
                    'limit_price': limit_price,
                    'dollar_amount': dollar_amount,
                    'order_id': order_id,
                    'status': 'submitted'
                }
                
                self.order_history.append(order_record)
                
                self.logger.info(f"{action_type} limit order submitted: {shares} shares of {symbol} at ${limit_price:.2f}, Order ID: {order_id}")
                
                return {
                    'status': 'submitted',
                    'symbol': symbol,
                    'action_type': action_type,
                    'order_type': 'limit',
                    'shares': shares,
                    'limit_price': limit_price,
                    'dollar_amount': dollar_amount,
                    'order_id': order_id,
                    'timestamp': timestamp
                }
            else:
                error_msg = f"Failed to place limit order: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                return {
                    'status': 'rejected',
                    'reason': error_msg,
                    'timestamp': timestamp
                }
            
        except Exception as e:
            self.logger.error(f"Error placing {action_type} limit order: {str(e)}")
            return {
                'status': 'error',
                'reason': str(e),
                'timestamp': timestamp
            }
    
    def buy_market(self, symbol: str, shares: int, current_price: float = None,
                   timestamp: datetime = None) -> Dict:
        """Convenience method for BUY market orders."""
        return self.place_market_order("BUY", symbol, shares, current_price, timestamp)
    
    def sell_market(self, symbol: str, shares: int, current_price: float = None,
                    timestamp: datetime = None) -> Dict:
        """Convenience method for SELL market orders."""
        return self.place_market_order("SELL", symbol, shares, current_price, timestamp)
    
    def sell_short_market(self, symbol: str, shares: int, current_price: float = None,
                         timestamp: datetime = None) -> Dict:
        """Convenience method for SELL_SHORT market orders."""
        return self.place_market_order("SELL_SHORT", symbol, shares, current_price, timestamp)
    
    def buy_to_cover_market(self, symbol: str, shares: int, current_price: float = None,
                           timestamp: datetime = None) -> Dict:
        """Convenience method for BUY_TO_COVER market orders."""
        return self.place_market_order("BUY_TO_COVER", symbol, shares, current_price, timestamp)
    
    def buy_limit(self, symbol: str, shares: int, limit_price: float,
                  timestamp: datetime = None) -> Dict:
        """Convenience method for BUY limit orders."""
        return self.place_limit_order("BUY", symbol, shares, limit_price, timestamp)
    
    def sell_limit(self, symbol: str, shares: int, limit_price: float,
                   timestamp: datetime = None) -> Dict:
        """Convenience method for SELL limit orders."""
        return self.place_limit_order("SELL", symbol, shares, limit_price, timestamp)
    
    def sell_short_limit(self, symbol: str, shares: int, limit_price: float,
                        timestamp: datetime = None) -> Dict:
        """Convenience method for SELL_SHORT limit orders."""
        return self.place_limit_order("SELL_SHORT", symbol, shares, limit_price, timestamp)
    
    def buy_to_cover_limit(self, symbol: str, shares: int, limit_price: float,
                          timestamp: datetime = None) -> Dict:
        """Convenience method for BUY_TO_COVER limit orders."""
        return self.place_limit_order("BUY_TO_COVER", symbol, shares, limit_price, timestamp)
    
    def get_order_history_df(self) -> pd.DataFrame:
        """
        Get order history as a pandas DataFrame.
        
        Returns:
            DataFrame with order history
        """
        if not self.order_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.order_history)
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an existing order
        
        Parameters:
            order_id: The ID of the order to cancel
            
        Returns:
            Dictionary containing result of cancellation
        """
        if not self.account_number:
            print("No account number available")
            return {"error": "No account number"}
        
        url = f"https://api.schwabapi.com/trader/v1/accounts/{self.account_number}/orders/{order_id}"
        headers = self._get_auth_headers()
        
        try:
            response = requests.delete(url, headers=headers)
            
            if response.status_code == 200:
                return {"status": "SUCCESS", "message": "Order cancelled successfully"}
            else:
                error_message = f"Failed to cancel order: {response.status_code}, {response.text}"
                print(error_message)
                return {"error": error_message}
        except Exception as e:
            error_message = f"Error cancelling order: {str(e)}"
            print(error_message)
            return {"error": error_message}


def main():
    """Command-line interface for OrderHandler."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Execute trading orders using OrderHandler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # BUY market order for 100 shares of AAPL at current price $150
  python3 handlers/order_handler.py AAPL BUY market --shares 100 
  
  # SELL_SHORT market order for 50 shares of TSLA at current price $200
  python3 handlers/order_handler.py TSLA SELL_SHORT market --shares 50 
  
  # uSELL limit order for 25 shares at limit price $155
  python3 handlers/order_handler.py AAPL SELL limit --shares 25 --price 155.0
  
  # BUY_TO_COVER market order for 75 shares at current price $195
  python3 handlers/order_handler.py TSLA BUY_TO_COVER market --shares 75 
        """
    )
    
    parser.add_argument('symbol', help='Stock symbol (e.g., AAPL, TSLA)')
    parser.add_argument('action', choices=['BUY', 'SELL', 'SELL_SHORT', 'BUY_TO_COVER'], 
                       help='Order action type')
    parser.add_argument('order_type', choices=['market', 'limit'], help='Order type')
    parser.add_argument('--shares', type=int, required=True, help='Number of shares')
    parser.add_argument('--price', type=float, help='Market price or limit price (required for limit orders)')
    args = parser.parse_args()
    
    # Validate price requirement for limit orders
    if args.order_type == 'limit' and args.price is None:
        print("Error: --price is required for limit orders")
        return
    
    # Create order handler
    handler = OrderHandler()
    
    print("OrderHandler initialized")
    
    # Execute order based on parameters
    if args.order_type == 'market':
        result = handler.place_market_order(
            args.action, args.symbol, args.shares, args.price
        )
    else:  # limit
        result = handler.place_limit_order(
            args.action, args.symbol, args.shares, args.price
        )
    print(f"Order Result:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

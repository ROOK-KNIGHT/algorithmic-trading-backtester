#!/usr/bin/env python3
"""
Enhanced script to fetch historical data with flexible parameters and save as CSV
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime, timedelta
from historical_data_handler import HistoricalDataHandler
import time

class EnhancedDataFetcher:
    def __init__(self):
        self.data_handler = HistoricalDataHandler()
        self.output_dir = 'historical_data'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def fetch_historical_data(self, symbol, period_type="year", period=1, frequency_type="daily", 

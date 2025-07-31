"""Test script for the IMF Scraper."""
import sys
import os
from pathlib import Path
from bs4 import BeautifulSoup
import requests
import pandas as pd
from datetime import datetime
import logging
import time
from typing import List, Dict, Any, Optional

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from scraper import IMFScraper

def main():
    """Test the IMF Scraper."""
    print("Testing IMF Scraper...")
    
    # Initialize the scraper
    scraper = IMFScraper()
    
    # Fetch the data
    print("Fetching lending arrangements data...")
    df = scraper.fetch_loans_data()
    
    if df is not None and not df.empty:
        print(f"\nSuccessfully fetched {len(df)} lending arrangements:")
        print("\nFirst 5 rows:")
        print(df.head())
        
        # Save to CSV
        csv_path = scraper.save_to_csv(df, 'test_imf_loans.csv')
        if csv_path:
            print(f"\nData saved to: {csv_path}")
        else:
            print("\nFailed to save data to CSV")
    else:
        print("\nFailed to fetch lending arrangements data")
        print("Check the logs for more information")

if __name__ == "__main__":
    main()

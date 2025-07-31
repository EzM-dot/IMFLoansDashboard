#!/usr/bin/env python3
"""
Test script to explore the structure of data returned by imfdatapy.
"""
import os
import pandas as pd
from imfdatapy.imf import IFS, BOP
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_ifs_data():
    """Test fetching data from IMF's International Financial Statistics (IFS)."""
    logger.info("Testing IFS data access...")
    
    # Set date range (last 5 years)
    end_year = datetime.now().year
    start_year = end_year - 5
    
    # Initialize IFS client
    ifs = IFS(
        search_terms=["gross domestic product, real"],
        countries=["US", "GB", "DE", "FR", "JP", "CN"],
        period='A',  # Annual data
        start_date=str(start_year),
        end_date=str(end_year)
    )
    
    # Fetch the data
    logger.info(f"Fetching IFS data from {start_year} to {end_year}...")
    df = ifs.download_data()
    
    if df is not None and not df.empty:
        logger.info(f"\nSuccessfully retrieved {len(df)} records")
        logger.info("\nDataFrame info:")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"\nData types:\n{df.dtypes}")
        logger.info(f"\nSample data (first 5 rows):\n{df.head().to_string()}")
        
        # Save sample data to CSV for inspection
        output_file = "imf_ifs_sample.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"\nSaved sample data to {output_file}")
    else:
        logger.warning("No data returned from the IFS API")
    
    return df

def test_bop_data():
    """Test fetching data from IMF's Balance of Payments (BOP)."""
    logger.info("\nTesting BOP data access...")
    
    # Set date range (last 5 years)
    end_year = datetime.now().year
    start_year = end_year - 5
    
    # Initialize BOP client
    bop = BOP(
        search_terms=["current account, total, credit"],
        countries=["US", "GB", "DE", "FR", "JP", "CN"],
        period='A',  # Annual data
        start_date=str(start_year),
        end_date=str(end_year)
    )
    
    # Fetch the data
    logger.info(f"Fetching BOP data from {start_year} to {end_year}...")
    df = bop.download_data()
    
    if df is not None and not df.empty:
        logger.info(f"\nSuccessfully retrieved {len(df)} records")
        logger.info("\nDataFrame info:")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"\nData types:\n{df.dtypes}")
        logger.info(f"\nSample data (first 5 rows):\n{df.head().to_string()}")
        
        # Save sample data to CSV for inspection
        output_file = "imf_bop_sample.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"\nSaved sample data to {output_file}")
    else:
        logger.warning("No data returned from the BOP API")
    
    return df

if __name__ == "__main__":
    # Test IFS data
    ifs_df = test_ifs_data()
    
    # Test BOP data
    bop_df = test_bop_data()

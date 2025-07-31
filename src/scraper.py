import os
import pandas as pd
import requests
from imfdatapy.imf import IFS, BOP
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any, Optional, Union
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('imf_scraper.log')
    ]
)
logger = logging.getLogger(__name__)

class IMFScraper:
    """Client for accessing IMF lending arrangements data."""
    
    def __init__(self, data_dir: str = 'data'):
        """Initialize the IMF Data Client.
        
        Args:
            data_dir: Directory to save data files
        """
        self.data_dir = os.path.abspath(data_dir)
        os.makedirs(self.data_dir, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'IMF Data Tool/1.0',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.5',
        })
    
    def _inspect_dataframe_structure(self, df: pd.DataFrame) -> None:
        """Log detailed information about the DataFrame structure.
        
        Args:
            df: The DataFrame to inspect
        """
        if df is None:
            logger.warning("DataFrame is None")
            return
            
        logger.info("="*80)
        logger.info("DATAFRAME INSPECTION")
        logger.info("="*80)
        
        # Basic info
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Index type: {type(df.index).__name__}")
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        # Detailed column info
        logger.info("\nCOLUMN DETAILS:")
        logger.info("-"*40)
        for col in df.columns:
            logger.info(f"\nColumn: {col}")
            logger.info(f"  Type: {df[col].dtype}")
            logger.info(f"  Non-null count: {df[col].count()} (of {len(df)} total)")
            logger.info(f"  Null count: {df[col].isna().sum()}")
            logger.info(f"  Unique values: {df[col].nunique()}")
            
            # Show sample values
            sample_values = df[col].dropna().head(5).tolist()
            if sample_values:
                logger.info(f"  Sample values: {sample_values}")
            
            # For string columns, show length statistics
            if df[col].dtype == 'object':
                lengths = df[col].astype(str).str.len()
                logger.info(f"  String length - Min: {lengths.min()}, "
                          f"Max: {lengths.max()}, "
                          f"Avg: {lengths.mean():.1f}")
        
        # Show first few rows
        logger.info("\nFIRST 5 ROWS:")
        logger.info("-"*40)
        for i, (_, row) in enumerate(df.head(5).iterrows(), 1):
            logger.info(f"\nRow {i}:")
            for col, val in row.items():
                logger.info(f"  {col} ({type(val).__name__}): {val}")
        
        # Show summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if not numeric_cols.empty:
            logger.info("\nNUMERIC COLUMNS SUMMARY:")
            logger.info("-"*40)
            logger.info(df[numeric_cols].describe().to_string())
        
        logger.info("="*80)
    
    def _clean_amount(self, amount_input) -> float:
        """Convert amount string or Series to numeric value(s).
        
        Args:
            amount_input: String, float, or Series containing amount values
            
        Returns:
            float or Series: Cleaned numeric value(s)
        """
        if pd.api.types.is_float(amount_input) or pd.api.types.is_integer(amount_input):
            return float(amount_input)
            
        if pd.isna(amount_input):
            return None
            
        if isinstance(amount_input, str):
            # Remove non-numeric characters except decimal point and negative sign
            amount_str = re.sub(r'[^\d.-]', '', amount_input)
            
            # Handle cases where comma is used as thousand separator
            if ',' in amount_str and '.' in amount_str:
                amount_str = amount_str.replace(',', '')
            elif ',' in amount_str:
                amount_str = amount_str.replace('.', '').replace(',', '.')
                
            try:
                return float(amount_str) if amount_str else None
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not convert amount '{amount_input}': {str(e)}")
                return None
            
        if isinstance(amount_input, pd.Series):
            return amount_input.astype(str).apply(self._clean_amount)
            
        return None
        
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the DataFrame after transformation.
        
        Args:
            df: DataFrame with raw data after wide-to-long transformation
            
        Returns:
            pd.DataFrame: Cleaned and standardized data
        """
        if df is None or df.empty:
            return df
            
        logger.info("Cleaning and standardizing data...")
        
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Ensure required columns exist
        required_columns = ['country', 'date', 'amount_agreed']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            logger.error(f"Missing required columns for cleaning: {missing}")
            return df
            
        # Clean country names
        df['country'] = df['country'].astype(str).str.strip()
        
        # Ensure date is datetime type
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
        # Clean amount_agreed - ensure it's numeric
        if not pd.api.types.is_numeric_dtype(df['amount_agreed']):
            df['amount_agreed'] = df['amount_agreed'].apply(self._clean_amount)
            
        # Remove any rows with missing required values
        initial_count = len(df)
        df = df.dropna(subset=required_columns)
        
        if len(df) < initial_count:
            logger.info(f"Removed {initial_count - len(df)} rows with missing values")
            
        # Sort by country and date
        df = df.sort_values(['country', 'date'])
        
        # Reset index for cleaner output
        df = df.reset_index(drop=True)
        
        logger.info(f"Cleaned data shape: {df.shape}")
        return df
            
    def _infer_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Try to infer column types based on data patterns.
        
        Args:
            df: Input DataFrame with raw data
            
        Returns:
            DataFrame with inferred column types and possibly renamed columns
        """
        if df.empty:
            return df
            
        logger.info("Attempting to infer column types...")
        
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Try to identify columns by their content
        for col in df.columns:
            col_lower = str(col).lower()
            
            # Check for date columns
            if any(keyword in col_lower for keyword in ['date', 'approval', 'expir']):
                try:
                    df[col] = pd.to_datetime(df[col], errors='ignore')
                except Exception as e:
                    logger.warning(f"Could not convert column '{col}' to datetime: {str(e)}")
            
            # Check for amount columns
            elif any(keyword in col_lower for keyword in ['amount', 'sdr', 'value']):
                df[col] = df[col].apply(self._clean_amount)
                
            # Check for country/borrower columns
            elif any(keyword in col_lower for keyword in ['country', 'member', 'borrower']):
                # Standardize country names
                df[col] = df[col].astype(str).str.strip()
                
        return df
        
    def _extract_missing_columns(self, df: pd.DataFrame, missing_columns: list) -> pd.DataFrame:
        """Attempt to extract missing columns from existing data.
        
        Args:
            df: Input DataFrame
            missing_columns: List of column names that need to be extracted
            
        Returns:
            DataFrame with extracted columns if possible
        """
        if df.empty or not missing_columns:
            return df
            
        logger.info(f"Attempting to extract missing columns: {missing_columns}")
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # First, try to understand the structure of the data
        if len(df.columns) == 2 and all(isinstance(col, (int, str)) for col in df.columns):
            logger.info("Detected a two-column structure, attempting to parse...")
            
            # Check if the first column contains country names
            col0_sample = df[df.columns[0]].head(5).astype(str).str.cat(sep=' ').lower()
            col1_sample = df[df.columns[1]].head(5).astype(str).str.cat(sep=' ').lower()
            
            # Common patterns for different data types
            country_terms = ['united states', 'france', 'germany', 'japan', 'china', 
                           'brazil', 'india', 'russia', 'uk', 'canada', 'mexico']
            
            date_patterns = [r'\d{1,2}/\d{1,2}/\d{2,4}',  # MM/DD/YYYY or DD/MM/YYYY
                           r'\d{4}-\d{1,2}-\d{1,2}',    # YYYY-MM-DD
                           r'[A-Za-z]{3} \d{1,2},? \d{4}']  # Jan 1 2020 or Jan 1, 2020
            
            amount_patterns = [r'\$[\d,]+',  # $1,000,000
                             r'\d+(\.\d+)?\s*(?:million|billion|trillion)?',
                             r'SDR\s*[\d,.]+']  # SDR 1,000,000
            
            # Check which column is more likely to be countries
            col0_country_score = sum(1 for term in country_terms if term in col0_sample)
            col1_country_score = sum(1 for term in country_terms if term in col1_sample)
            
            # Assign country column
            if col0_country_score > col1_country_score and 'country' in missing_columns:
                result_df['country'] = result_df[result_df.columns[0]].astype(str).str.strip()
                logger.info(f"Identified column 0 as country")
            elif 'country' in missing_columns:
                result_df['country'] = result_df[result_df.columns[1]].astype(str).str.strip()
                logger.info(f"Identified column 1 as country")
            
            # Check for dates in the other column
            date_col = None
            if 'date' in missing_columns:
                for i, col in enumerate([result_df.columns[0], result_df.columns[1]]):
                    try:
                        # Try to convert to datetime
                        dates = pd.to_datetime(result_df[col], errors='coerce')
                        if not dates.isna().all():
                            result_df['date'] = dates
                            logger.info(f"Identified column {i} as date")
                            date_col = col
                            break
                    except Exception as e:
                        logger.debug(f"Column {i} is not a date column: {str(e)}")
            
            # If we didn't find dates, try to extract from the other column
            if 'date' in missing_columns and 'date' not in result_df.columns:
                other_col = [c for c in [result_df.columns[0], result_df.columns[1]] 
                           if c != date_col][0] if date_col else None
                
                if other_col is not None:
                    # Try to extract dates from text using multiple patterns
                    try:
                        # Try extracting dates in various formats
                        date_patterns = [
                            # Month DD, YYYY (e.g., January 1, 2023)
                            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]+\d{1,2},?[\s,]+\d{4})',
                            # MM/DD/YYYY or DD/MM/YYYY
                            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                            # YYYY-MM-DD
                            r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
                            # Month YYYY (e.g., January 2023)
                            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]+\d{4})',
                            # Quarter YYYY (e.g., Q1 2023)
                            r'(Q[1-4]\s*\d{4})',
                            # Year only (as last resort)
                            r'(\b\d{4}\b)'
                        ]
                        
                        # Try each pattern until we find dates
                        for pattern in date_patterns:
                            try:
                                date_strs = result_df[other_col].astype(str).str.extract(
                                    pattern,
                                    flags=re.IGNORECASE
                                )
                                if not date_strs.empty and not date_strs[0].isna().all():
                                    result_df['date'] = pd.to_datetime(date_strs[0], errors='coerce')
                                    if not result_df['date'].isna().all():
                                        logger.info(f"Extracted dates using pattern: {pattern}")
                                        break
                            except Exception as e:
                                logger.debug(f"Pattern {pattern} failed: {str(e)}")
                        
                        # If we still don't have dates, try to infer from the column name
                        if 'date' not in result_df.columns or result_df['date'].isna().all():
                            col_name = str(other_col).lower()
                            if any(term in col_name for term in ['date', 'time', 'year', 'month']):
                                result_df['date'] = pd.to_datetime(result_df[other_col], errors='coerce')
                                logger.info(f"Inferred date from column name: {other_col}")
                        
                        # If we still don't have dates, add a default date (today)
                        if 'date' not in result_df.columns or result_df['date'].isna().all():
                            result_df['date'] = pd.Timestamp.now().normalize()
                            logger.info("Added current date as default")
                            
                    except Exception as e:
                        logger.warning(f"Could not extract dates from text: {str(e)}")
                        # Add a default date if all else fails
                        result_df['date'] = pd.Timestamp.now().normalize()
                        logger.info("Added current date as fallback")
            
            # Check for amounts in the remaining column
            if 'amount_agreed' in missing_columns and len(result_df.columns) >= 2:
                # Try the column that's not the date column
                amount_col = [c for c in [result_df.columns[0], result_df.columns[1]] 
                            if c != date_col or date_col is None][0]
                
                if amount_col is not None:
                    try:
                        # Clean and convert amounts
                        cleaned_amounts = self._clean_amount(result_df[amount_col])
                        if not cleaned_amounts.isna().all():
                            result_df['amount_agreed'] = cleaned_amounts
                            logger.info(f"Identified column {amount_col} as amount_agreed")
                    except Exception as e:
                        logger.warning(f"Error processing column {amount_col} for amount: {str(e)}")
            
            return result_df
        
        # Fallback to the original logic for other cases
        for col in missing_columns:
            if col == 'country':
                # Look for a column that might contain country names
                for existing_col in result_df.columns:
                    try:
                        # Check if this is a single column (not a multi-index)
                        if isinstance(existing_col, str):
                            # Get a sample of the column values
                            sample = result_df[existing_col].dropna().astype(str).str.lower().str.cat(sep=' ')
                            
                            # Check for common country names or codes in the sample
                            country_terms = [
                                'united states', 'france', 'germany', 'japan', 'china', 
                                'brazil', 'india', 'russia', 'uk', 'canada', 'mexico',
                                'aus', 'chn', 'deu', 'fra', 'gbr', 'ita', 'jpn', 'rus', 'usa'
                            ]
                            
                            if any(term in sample for term in country_terms):
                                # Clean and assign the country column
                                result_df['country'] = result_df[existing_col].astype(str).str.strip()
                                logger.info(f"Identified '{existing_col}' as the country column")
                                break
                    except Exception as e:
                        logger.warning(f"Error processing column '{existing_col}' for country: {str(e)}")
                        continue
                        
            elif col == 'date':
                # Look for date-like columns
                for existing_col in result_df.columns:
                    if isinstance(existing_col, str):
                        try:
                            # Try to convert to datetime
                            dates = pd.to_datetime(result_df[existing_col], errors='coerce')
                            if not dates.isna().all():  # If we have at least one valid date
                                result_df['date'] = dates
                                logger.info(f"Identified '{existing_col}' as the date column")
                                break
                        except Exception as e:
                            logger.debug(f"Column '{existing_col}' is not a date column: {str(e)}")
                            continue
                        
            elif col == 'amount_agreed':
                # Look for amount columns
                for existing_col in result_df.columns:
                    if isinstance(existing_col, str) and any(term in existing_col.lower() for term in 
                         ['amount', 'sdr', 'total', 'agreed', 'value']):
                        try:
                            # Clean the amount column
                            cleaned_amounts = self._clean_amount(result_df[existing_col])
                            if not cleaned_amounts.isna().all():  # If we have at least one valid amount
                                result_df['amount_agreed'] = cleaned_amounts
                                logger.info(f"Identified '{existing_col}' as the amount_agreed column")
                                break
                        except Exception as e:
                            logger.warning(f"Error processing column '{existing_col}' for amount: {str(e)}")
                            continue
        
        return result_df
    
    def _make_api_request(self, url: str, params: dict = None, max_retries: int = 3) -> dict:
        """Helper method to make API requests with retries and error handling."""
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    url,
                    params=params,
                    timeout=30,
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} attempts failed for {url}")
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff

    def _parse_imf_data(self, data: dict) -> pd.DataFrame:
        """Parse IMF API response into a DataFrame."""
        try:
            records = []
            for series in data.get('Structure', {}).get('Series', []):
                country = series.get('REF_AREA', 'Unknown')
                for obs in series.get('Obs', []):
                    records.append({
                        'country': country,
                        'date': obs.get('TIME_PERIOD', ''),
                        'amount_agreed': obs.get('OBS_VALUE'),
                        'indicator': series.get('INDICATOR', '')
                    })
            return pd.DataFrame(records)
        except Exception as e:
            logger.error(f"Error parsing IMF data: {str(e)}")
            return pd.DataFrame()

    def _get_imf_datasets(self) -> list:
        """Get available IMF datasets that might contain lending data."""
        return [
            {
                'name': 'IFS',
                'indicators': ['FCL', 'PLL', 'SBA', 'EFF', 'RCF', 'RFI']
            },
            {
                'name': 'GFS',
                'indicators': ['FCL', 'PLL', 'SBA', 'EFF']
            },
            {
                'name': 'BOP',
                'indicators': ['FCL', 'PLL']
            }
        ]

    def _fetch_imf_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data from IMF API."""
        base_url = "https://sdmxdata.imf.org/ws/sdmx/rest/data"
        all_data = []
        
        for dataset in self._get_imf_datasets():
            for indicator in dataset['indicators']:
                try:
                    url = f"{base_url}/IMF/{dataset['name']}"
                    params = {
                        'startPeriod': start_date[:4],
                        'endPeriod': end_date[:4],
                        'format': 'jsondata',
                        'key': f"{indicator}...A"
                    }
                    
                    logger.info(f"Fetching {dataset['name']} - {indicator}")
                    data = self._make_api_request(url, params)
                    
                    if data:
                        df = self._parse_imf_data(data)
                        if not df.empty:
                            df['source'] = f"{dataset['name']}_{indicator}"
                            all_data.append(df)
                            
                except Exception as e:
                    logger.warning(f"Error processing {dataset['name']}/{indicator}: {str(e)}")
        
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    def _fetch_from_alternative_source(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data from alternative data source if primary fails."""
        try:
            # Try World Bank as an alternative
            url = "https://api.worldbank.org/v2/country/all/indicator/DT.ODA.ODAT.GI.ZS"
            params = {
                'format': 'json',
                'per_page': 10000,
                'date': f"{start_date[:4]}:{end_date[:4]}"
            }
            
            data = self._make_api_request(url, params)
            if data and len(data) > 1 and 'value' in data[1][0]:
                records = []
                for item in data[1]:
                    records.append({
                        'country': item['country']['value'],
                        'date': f"{item['date']}-01-01",
                        'amount_agreed': item['value'],
                        'source': 'worldbank'
                    })
                return pd.DataFrame(records)
                
        except Exception as e:
            logger.warning(f"Alternative data source failed: {str(e)}")
            
        return pd.DataFrame()

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the data."""
        if df.empty:
            return df
            
        # Ensure required columns exist
        for col in ['country', 'date', 'amount_agreed']:
            if col not in df.columns:
                df[col] = None
        
        # Clean country names
        if 'country' in df.columns:
            df['country'] = df['country'].str.strip().str.title()
        
        # Convert date to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Convert amount to numeric
        if 'amount_agreed' in df.columns:
            df['amount_agreed'] = pd.to_numeric(df['amount_agreed'], errors='coerce')
        
        return df

    def fetch_loans_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Fetch lending arrangement data from IMF's data services.
        
        This method attempts to fetch data from multiple sources and returns
        a cleaned and standardized DataFrame with lending arrangement data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format (optional)
            end_date: End date in YYYY-MM-DD format (optional)
            
        Returns:
            DataFrame containing lending arrangement data with columns:
            - country: Name of the country
            - date: Date of the arrangement
            - amount_agreed: Amount agreed in SDR (Special Drawing Rights)
            - source: Data source identifier
        """
        try:
            # Set default date range if not provided (last 5 years)
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if not start_date:
                start_date = (datetime.now() - timedelta(days=7*365)).strftime('%Y-%m-%d') # last 5 years
                
            logger.info(f"Fetching lending arrangements from {start_date} to {end_date}")
            
            # Try primary IMF data source
            logger.info("Trying primary IMF data source...")
            df = self._fetch_imf_data(start_date, end_date)
            
            # If primary source fails, try alternative source
            if df.empty:
                logger.warning("Primary source returned no data, trying alternative source...")
                df = self._fetch_from_alternative_source(start_date, end_date)
            
            # If still no data, try web scraping as last resort
            if df.empty:
                logger.warning("Alternative source returned no data, trying web scraping...")
                df = self._scrape_imf_website(start_date, end_date)
            
            # If we still have no data, log an error and return empty DataFrame
            if df.empty:
                logger.error("All data sources returned no data")
                return pd.DataFrame()
            
            # Clean and process the data
            df = self._clean_data(df)
            
            # Ensure we have the required columns
            required_columns = ['country', 'date', 'amount_agreed']
            for col in required_columns:
                if col not in df.columns:
                    logger.error(f"Required column '{col}' not found in data")
                    return pd.DataFrame()
            
            # Filter by date range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
            
            # Drop rows with missing values in required columns
            initial_count = len(df)
            df = df.dropna(subset=required_columns)
            
            if len(df) < initial_count:
                logger.info(f"Dropped {initial_count - len(df)} rows with missing values")
            
            # Log summary of the data
            logger.info(f"Retrieved {len(df)} valid records")
            if not df.empty:
                logger.info(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
                logger.info(f"Countries: {df['country'].nunique()}")
                logger.info(f"Total amount: {df['amount_agreed'].sum():,.2f} SDR")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in fetch_loans_data: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error in fetch_loans_data: {str(e)}")
            return pd.DataFrame()
    
    def _fetch_loans_data_imfdatapy(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fallback method to fetch data using imfdatapy if SDMX API fails."""
        try:
            logger.info("Falling back to imfdatapy implementation")
            from imfdatapy.imf import IFS
            
            # Initialize IFS client
            ifs = IFS(
                countries=[],  # All countries
                period='A',    # Annual data
                start_date=start_date[:4],
                end_date=end_date[:4]
            )
            
            # Try to download data
            df = ifs.download_data()
            
            if df is not None and not df.empty:
                # Process the data to match expected format
                result_df = pd.DataFrame()
                
                # Map columns case-insensitively
                col_mapping = {}
                for req in ['country', 'period', 'value']:
                    matches = [col for col in df.columns if req.lower() == col.lower()]
                    if matches:
                        col_mapping[req] = matches[0]
                
                if len(col_mapping) >= 3:  # Need at least country, period, and value
                    df = df.rename(columns={v: k for k, v in col_mapping.items()})
                    
                    # Create result DataFrame
                    result_df['country'] = df['country']
                    result_df['date'] = pd.to_datetime(df['period'], errors='coerce')
                    result_df['amount_agreed'] = pd.to_numeric(df['value'], errors='coerce')
                    
                    # Clean and return
                    return self._clean_data(result_df)
                
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error in fallback imfdatapy method: {str(e)}")
            return pd.DataFrame()
            
        # This point should not be reached
        logger.error("Unexpected code path in fetch_loans_data")
        return pd.DataFrame()
    
    def _parse_table_manually(self, table) -> pd.DataFrame:
        """Manually parse HTML table when pandas fails.
        
        Args:
            table: BeautifulSoup table element
            
        Returns:
            pd.DataFrame: Parsed table data or None if parsing fails
        """
        try:
            logger.info("Attempting to parse table manually...")
            
            # Get all rows from the table
            rows = table.find_all('tr')
            if not rows:
                logger.warning("No rows found in table")
                return pd.DataFrame()
                
            # Get headers from the first row
            headers = []
            header_cells = rows[0].find_all(['th', 'td'])
            for cell in header_cells:
                headers.append(cell.get_text(strip=True))
                
            if not headers:
                logger.warning("No headers found in table")
                return pd.DataFrame()
                
            # Get data rows
            data = []
            for row in rows[1:]:  # Skip header row
                cells = row.find_all(['td', 'th'])
                if len(cells) != len(headers):
                    continue  # Skip rows with incorrect number of cells
                    
                row_data = [cell.get_text(strip=True) for cell in cells]
                data.append(row_data)
                
            if not data:
                logger.warning("No data rows found in table")
                return pd.DataFrame()
                
            # Create DataFrame
            df = pd.DataFrame(data, columns=headers)
            logger.info(f"Manually parsed table with {len(df)} rows and {len(df.columns)} columns")
            return df
            
            logger.error(f"Request failed: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}", exc_info=True)
            return pd.DataFrame()
    
    def save_to_csv(self, df: pd.DataFrame, filename: str = 'imf_loans.csv') -> Optional[str]:
        """Save DataFrame to CSV file with proper error handling and logging.
        
        Args:
            df: DataFrame to save
            filename: Output filename (will be saved in the data_dir)
            
        Returns:
            str: Path to the saved file if successful, None otherwise
        """
        if df is None or df.empty:
            logger.warning("No data to save to CSV")
            return None
            
        try:
            # Ensure the data directory exists
            os.makedirs(self.data_dir, exist_ok=True)
            filepath = os.path.join(self.data_dir, filename)
            
            # Save to CSV with appropriate parameters
            df.to_csv(filepath, index=False, encoding='utf-8')
            
            # Save the file
            df.to_csv(filepath, index=False)
            logger.info(f"Data saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving to CSV: {str(e)}")
            return None

if __name__ == "__main__":
    scraper = IMFScraper()
    loans_df = scraper.fetch_loans_data()
    if not loans_df.empty:
        scraper.save_to_csv(loans_df)

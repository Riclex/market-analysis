import os
import requests
import pandas as pd
from dotenv import load_dotenv
import yfinance as yf
import time 
import logging
from typing import Optional
import numpy as np
from datetime import datetime

# Logging configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

load_dotenv()
API_KEY = os.getenv("ALPHA_VANTAGE_KEY")

class APIError(Exception):
    '''Base class for API errors exceptions'''
    pass

class InvalidResponseError(APIError):
    '''Exception raised for invalid responses from the API'''
    pass

def fetch_stock_data(symbol: str, max_retries: int = 3, backoff_factor: float = 0.5) -> Optional[pd.DataFrame]:
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}&outputsize=full"

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()

            # Check for API errors
            if "Error Message" in data:
                raise APIError(f"API Error: {data['Error Message']}")
            if "Note" in data: # API rate limit
                raise APIError(f"API Error: {data['Note']}")
            
            # Validate response
            time_series = data.get("Time Series (Daily)")
            if not time_series:
                raise InvalidResponseError("Missing 'Time Series (Daily)' in response")
            
            # Parse data
            df = pd.DataFrame(time_series).T
            df.reset_index(inplace=True)
            df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            df["symbol"] = symbol

            # Convert data types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            
            return df
        
        except requests.exceptions.RequestException as e: # Catch all requests exceptions
            logging.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                sleep_time = backoff_factor * 2 ** attempt
                logging.info(f"Retrying in {sleep_time:.2f} seconds...") # Log retry attempt
                time.sleep(sleep_time)
                continue
            else:
                logging.error(f"Failed to fetch data after {max_retries} attempts")
                return None
        
        except (KeyError, ValueError) as e: # Catch all other exceptions
            logging.error(f"Data processing error: {str(e)}")
            return None

def validate_stock_data(df: pd.DataFrame, symbol: str) -> bool:
    """Validate fetched stock data"""
    try:
        # Check for required columns
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'symbol']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logging.error(f"Missing columns for {symbol}: {missing_cols}")
            return False
            
        # Check for null values
        null_counts = df.isnull().sum()
        if null_counts.any():
            logging.warning(f"Null values found in {symbol} data:")
            for col, count in null_counts[null_counts > 0].items():
                logging.warning(f"{col}: {count} nulls")
        
        # Check data types
        expected_types = {
            'date': 'datetime64[ns]',
            'open': 'float64',
            'high': 'float64',
            'low': 'float64',
            'close': 'float64',
            'volume': 'int64'
        }
        
        for col, expected_type in expected_types.items():
            if df[col].dtype != expected_type:
                logging.warning(f"Column {col} has type {df[col].dtype}, expected {expected_type}")
                
        return True
        
    except Exception as e:
        logging.error(f"Validation failed for {symbol}: {str(e)}")
        return False

def fetch_fundamentals(symbol: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(symbol)
            try:
                fundamentals = {
                    "symbol": symbol,
                    "marketCap": ticker.fast_info.get("marketCap"),
                    "peRatio": ticker.fast_info.get("lastPrice", 0) / ticker.fast_info.get("trailingEps", 1),
                    "dividendYield": ticker.fast_info.get("lastDividendValue", 0) / ticker.fast_info.get("lastPrice", 1) * 100
                }
            except Exception as e:
                logging.warning(f"Fast info failed, falling back to regular info: {e}")
                info = ticker.info
                fundamentals = {
                    "symbol": symbol,
                    "marketCap": info.get("marketCap"),
                    "peRatio": info.get("trailingPE"),
                    "dividendYield": info.get("dividendYield")
                }

            # Validate data
            df = pd.DataFrame([fundamentals])
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Log missing fields
            null_fields = df.columns[df.isnull().any()].tolist()
            if null_fields:
                logging.warning(f"Missing/invalid fields for {symbol}: {null_fields}")
            
            return df

        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    symbols = ["AAPL", "AMZN", "MSFT"]
    
    for symbol in symbols:
        # Fetch and validate stock data
        logging.info(f"Processing {symbol}...")
        stock_data = fetch_stock_data(symbol)
        
        if stock_data is not None and validate_stock_data(stock_data, symbol):
            try:
                filename = f"data/raw/{symbol.lower()}_daily.csv"
                stock_data.to_csv(filename, index=False)
                logging.info(f"Stock data saved: {filename}")
            except Exception as e:
                logging.error(f"Failed to save stock data for {symbol}: {e}")
        
        time.sleep(2)  # Rate limiting
        
        # Fetch fundamentals
        fundamentals_data = fetch_fundamentals(symbol)
        if fundamentals_data is not None:
            try:
                filename = f"data/raw/{symbol.lower()}_fundamentals.csv"
                fundamentals_data.to_csv(filename, index=False)
                logging.info(f"Fundamentals saved: {filename}")
            except Exception as e:
                logging.error(f"Failed to save fundamentals for {symbol}: {e}")
        
        time.sleep(2)  # Rate limiting
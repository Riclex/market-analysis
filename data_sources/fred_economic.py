import os
import pandas as pd
import requests
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import logging

load_dotenv()

class FREDclient:
    def __init__(self, api_key):
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
        self.api_key = os.getenv("FRED_API_KEY")

        # logging configuration
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def save_data(self, df: pd.DataFrame, series_id: str):
        '''Save data to db and parquet file'''
        try:
            # Create and save to directory
            save_dir = "data/processed/fred"
            os.makedirs(save_dir, exists_ok=True)
            
            # Save to parquet
            parquet_path = f"{save_dir}/{series_id}.parquet"
            df.to_parquet(parquet_path)
            
            # Save data to PostgreSql
            engine = create_engine(os.getenv("DB_URI"))
            with engine.begin() as conn:
                df.to_sql(
                    name='economic_indicators',
                    schema='fred',
                    con=conn,
                    if_exists='append',
                    index=False,
                    method='multi'
                )

            self.logger.info(f"Saved {len(df)} records for {series_id}")
        
        except Exception as e:
            self.logger.error(f"Failed to save data: {str(e)}")
            return False
        
        return True

    def get_economic_indicators(self, series_ids:list):
        '''Fetch and store economic data'''
        if not self.api_key:
            raise ValueError("API key not found. Please set FRED_API_KEY environment variable")
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')

        # dfs = []
        for series_id in series_ids:
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "observation_start": start_date,
                "observation_end": end_date
            }

            try:
                self.logger.info(f"Fetching data for {series_id}")
                response = requests.get(self.base_url, params=params, timeout=15)
                response.raise_for_status()

                data = response.json().get("observations",[])

                if not data:
                    self.logger.warning(f"No data found for {series_id}")
                    continue

                df = pd.DataFrame(data)
                df["series_id"] = series_id
                df["date"] = pd.to_datetime(df["date"])

                # Convert to numeric
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                df = df.dropna(subset=["value"])

                # Persist data
                self.save_data(df, series_id)

            except Exception as e:
                print(f"Error fetching {series_id}: {str(e)}")
        
if __name__ == "__main__":
    client = FREDclient(api_key=os.getenv("FRED_API_KEY"))
    client.get_economic_indicators(["GDP", "UNRATE", "CPIAUCSL"])
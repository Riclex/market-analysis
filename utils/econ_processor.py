# Import required libraries
import os
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Union

class EconomicDataProcessor:
    def __init__(self, 
                 econ_series: Optional[List[str]] = None,
                 input_dir: str = "data/processed/fred/", 
                 output_path: str = "data/processed/fred/combined.parquet"):
        """Initialize the economic data processor        
        Args:
            econ_series: List of FRED series IDs to process. Defaults to common
                         macroeconomic indicators if None is provided.
            input_dir: Directory containing the input parquet files.
            output_path: Path where the combined data will be saved.
        """
        # Default to common economic indicators if none provided
        self.econ_series = econ_series or ['GDP', 'UNRATE', 'CPIAUCSL']
        self.input_dir = input_dir
        self.output_path = output_path
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
    def _load_series(self, series_id: str) -> pd.DataFrame:
        """Load a single economic time series from parquet file
        Args:
            series_id: The FRED series identifier    
        Returns:
            DataFrame containing the date and series values
        Raises:
            FileNotFoundError: If the parquet file doesn't exist
        """
        file_path = os.path.join(self.input_dir, f"{series_id}.parquet")
        
        try:
            df = pd.read_parquet(file_path)
            
            # Validate required columns exist
            required_cols = {'date', 'value'}
            if not required_cols.issubset(df.columns):
                missing = required_cols - set(df.columns)
                self.logger.error(f"Missing required columns {missing} in {series_id}")
                raise ValueError(f"Series {series_id} is missing columns: {missing}")
                
            # Extract only needed columns and rename for clarity
            return df[['date', 'value']].rename(columns={'value': series_id})
            
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            raise
    
    def process_economic_data(self, 
                             normalization_method: str = 'zscore',
                             fill_method: Optional[str] = 'ffill') -> pd.DataFrame:
        """Combine and normalize economic indicators
        Args:
            normalization_method:'zscore' - (x - mean) / std
                                 'minmax' - (x - min) / (max - min)
                                 'none' - no normalization
            fill_method:'ffill', 'bfill', None (leave NaN).
        Returns:
            DataFrame containing the processed economic data.
        """
        self.logger.info(f"Processing {len(self.econ_series)} economic series")
        
        # Load all data series
        dfs = []
        for series_id in self.econ_series:
            try:
                df = self._load_series(series_id)
                dfs.append(df)
                self.logger.info(f"Loaded {series_id} with {len(df)} records")
            except Exception as e:
                self.logger.warning(f"Failed to load {series_id}: {str(e)}")
        
        if not dfs:
            self.logger.error("No data series could be loaded")
            raise ValueError("Failed to load any economic data series")
            
        # Merge all economic data series
        self.logger.info("Merging data series")
        econ_df = dfs[0]
        for i, df in enumerate(dfs[1:], 1):
            econ_df = econ_df.merge(df, on='date', how='outer')
            self.logger.debug(f"Merged series {i} of {len(dfs)-1}")
        
        # Sort by date for time series consistency
        econ_df = econ_df.sort_values('date')
        
        # Fill missing values 
        if fill_method:
            self.logger.info(f"Filling missing values using '{fill_method}'")
            econ_df = econ_df.fillna(method=fill_method)
        
        # Apply normalization
        if normalization_method != 'none' and len(self.econ_series) > 0:
            self.logger.info(f"Normalizing data using '{normalization_method}'")
            
            # Normalize the economic series columns
            available_series = [s for s in self.econ_series if s in econ_df.columns]
            
            if normalization_method == 'zscore':
                econ_df[available_series] = econ_df[available_series].apply(
                    lambda x: (x - x.mean()) / x.std()
                )
            elif normalization_method == 'minmax':
                econ_df[available_series] = econ_df[available_series].apply(
                    lambda x: (x - x.min()) / (x.max() - x.min())
                )
            else:
                self.logger.warning(f"Unknown normalization method: {normalization_method}")
        
        # Calculate basic statistics for reporting
        stats = {
            'num_rows': len(econ_df),
            'date_range': (econ_df['date'].min(), econ_df['date'].max()),
            'missing_values': econ_df[econ_df.columns[1:]].isna().sum().to_dict()
        }
        self.logger.info(f"Processed data stats: {stats}")
        
        # Save the results
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        econ_df.to_parquet(self.output_path)
        self.logger.info(f"Saved combined data to {self.output_path}")
        
        return econ_df
    
    def get_correlation_matrix(self, econ_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Calculate correlation matrix between economic indicators
        Args:
            econ_df: Optional DataFrame to use. If None, loads the saved combined data    
        Returns:
            DataFrame containing the correlation matrix
        """
        if econ_df is None:
            econ_df = pd.read_parquet(self.output_path)
            
        # Select only the economic series columns for correlation
        available_series = [s for s in self.econ_series if s in econ_df.columns]
        return econ_df[available_series].corr()
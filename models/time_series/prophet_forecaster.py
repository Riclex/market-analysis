# Import required libraries
import os
import logging
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio
from dotenv import load_dotenv
import json
from prophet.diagnostics import cross_validation, performance_metrics
from typing import Dict, Any

# Loggging configuration
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
pio.renderers.default = 'browser'

class ProphetForecaster:
    def __init__(self, symbol: str = "MSFT", forecast_periods: int = 30):
        ''' Initialize Prophet model with stock symbol
            Args:
                symbol: stock ticker symbol
                forecast_periods: nr of days to forecast
                '''
        self.symbol = symbol
        self.forecast_periods = forecast_periods
        self.model = None
        self.forecast = None
        self.metrics = {}

    def get_data_path(self) -> str:
        """Get the path to the data file."""
        return f"data/processed/{self.symbol}_features.parquet"

    def load_data(self) -> pd.DataFrame:
        """Load processed stock data."""
        try:
            df = pd.read_parquet(self.get_data_path())
            return self.process_data(df)
        except FileNotFoundError:
            logging.error("Data file not found.")
            raise
        except Exception as e:
            logging.error(f"Data loading failed: {str(e)}")
            raise

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the dataframe for Prophet."""
        df = df[['date', 'close']].rename(columns={'date': 'ds', 'close': 'y'})
        df['ds'] = pd.to_datetime(df['ds'])
        return df.dropna()


    def add_economic_regressor(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add FRED economic data as regressors."""
        try:
            econ_df = self.load_economic_data()
            merged = df.merge(econ_df, on='ds', how='left')
            return self.handle_missing_values(merged)
        except Exception as e:
            logging.warning(f"Failed to add economic regressor: {str(e)}")
            return df

    def load_economic_data(self) -> pd.DataFrame:
        """Load and preprocess economic data."""
        econ_df = pd.read_parquet("data/processed/fred/combined.parquet")
        econ_df['ds'] = pd.to_datetime(econ_df['date'])
        for col in ['GDP', 'UNRATE', 'CPIAUCSL']:
            if col in econ_df.columns:
                econ_df[col] = econ_df[col].ffill().bfill()
        return econ_df[['ds', 'GDP', 'UNRATE', 'CPIAUCSL']]
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataframe."""
        if df.isna().any().any():
            logging.warning("NaN values found in data after processing")
        return df.ffill().bfill()
    
    def train(self, include_economics: bool = False) -> 'ProphetForecaster':
        """Train model with optimized parameters."""
        try:
            self.model = self.initialize_model()
            df = self.load_data()

            if include_economics:
                df = self.add_economic_regressor(df)
                logging.info("Added economic regressors to the model")

            self.validate_data(df)
            self.model.fit(df)
            logging.info("Successfully trained Prophet model")
            return self
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            raise

    def initialize_model(self) -> Prophet:
        """Initialize the Prophet model with specified parameters."""
        return Prophet(
            growth='linear',
            yearly_seasonality=True,
            daily_seasonality=True,
            changepoint_prior_scale=0.01,
            changepoint_range=0.9,
            seasonality_prior_scale=10.0,
            holidays_prior_scale=10.0,
            seasonality_mode='multiplicative'
        )
    
    def validate_data(self, df: pd.DataFrame) -> None:
        """Validate the dataframe for NaN values."""
        if df.isna().any().any():
            raise ValueError("Dataset contains NaN values after processing")
        
    def predict(self) -> 'ProphetForecaster':
        """Generate predictions."""
        try:
            future = self.create_future_dataframe()
            if self.model.extra_regressors:
                future = self.add_future_regressors(future)
            self.validate_data(future)
            self.forecast = self.model.predict(future)
            self.calculate_metrics()
            return self
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            raise

    def create_future_dataframe(self) -> pd.DataFrame:
        """Create a future dataframe for predictions."""
        return self.model.make_future_dataframe(
            periods=self.forecast_periods,
            freq='D',
            include_history=True
        )
    
    def add_future_regressors(self, future: pd.DataFrame) -> pd.DataFrame:
        """Add future economic regressors to the dataframe"""
        econ_df = self.load_economic_data()
        return future.merge(econ_df, on='ds', how='left').ffill().bfill()
    

    def calculate_metrics(self) -> None:
        """Calculate performance metrics."""
        try:
            merged = self.model.history.merge(
                self.forecast[['ds', 'yhat']], on='ds'
            )
            self.metrics = {
                'MAE': mean_absolute_error(merged['y'], merged['yhat']),
                'RMSE': np.sqrt(mean_squared_error(merged['y'], merged['yhat'])),
                'MAPE': np.mean(np.abs((merged['y'] - merged['yhat']) / merged['y'])) * 100,
                'R2': r2_score(merged['y'], merged['yhat'])
            }
            logging.info(f"Model metrics: {self.metrics}")
        except Exception as e:
            logging.warning(f"Metric calculation failed: {str(e)}")


    def visualize(self) -> 'ProphetForecaster':
        """Generate interactive visualizations."""
        try:
            fig = plot_plotly(self.model, self.forecast)
            fig.update_layout(
                title=f"{self.symbol} Price Forecast",
                xaxis_title="Date",
                yaxis_title="Price (USD)"
            )
            fig.show()

            comp_fig = plot_components_plotly(self.model, self.forecast)
            comp_fig.show()
            return self
        except Exception as e:
            logging.error(f"Visualization failed: {str(e)}")

    def save_model(self) -> 'ProphetForecaster':
        """Save model to /models/time_series."""
        try:
            model_data = self.extract_model_data()
            self.save_to_json(model_data)
            logging.info(f"Model saved to models/time_series/{self.symbol}_prophet.json")
            return self
        except Exception as e:
            logging.error(f"Failed to save model: {str(e)}")
            raise
    
    def extract_model_data(self) -> Dict[str, Any]:
        """Extract model data for saving."""
        model_data = {
            'growth': self.model.growth,
            'changepoints': self.model.changepoints.astype(str).tolist() if self.model.changepoints is not None else None,
            'n_changepoints': int(self.model.n_changepoints),
            'changepoint_range': float(self.model.changepoint_range),
            'yearly_seasonality': self.model.yearly_seasonality,
            'weekly_seasonality': self.model.weekly_seasonality,
            'daily_seasonality': self.model.daily_seasonality,
            'seasonality_mode': self.model.seasonality_mode,
            'seasonality_prior_scale': float(self.model.seasonality_prior_scale),
            'changepoint_prior_scale': float(self.model.changepoint_prior_scale),
            'holidays_prior_scale': float(self.model.holidays_prior_scale),
            'mcmc_samples': self.model.mcmc_samples,
            'interval_width': float(self.model.interval_width),
            'uncertainty_samples': self.model.uncertainty_samples,
            'stan_backend': str(self.model.stan_backend)
        }
        if self.model.extra_regressors:
            model_data['extra_regressors'] = {
                name: {
                    'prior_scale': float(reg['prior_scale']),
                    'standardize': reg['standardize'],
                    'mode': reg['mode']
                } for name, reg in self.model.extra_regressors.items()
            }
        return model_data

    def save_to_json(self, model_data: Dict[str, Any]) -> None:
        """Save model data to a JSON file."""
        with open(f"models/time_series/{self.symbol}_prophet.json", "w") as f:
            json.dump(model_data, f, indent=4)

    def load_model(self, symbol: str) -> 'ProphetForecaster':
        """Load model from /models/time_series."""
        try:
            with open(f"models/time_series/{symbol}_prophet.json", "r") as f:
                model_data = json.load(f)
            self.model = self.initialize_loaded_model(model_data)
            logging.info(f"Loaded model for {symbol}")
            return self
        except Exception as e:
            logging.error(f"Model loading failed: {str(e)}")
            raise

    def initialize_loaded_model(self, model_data: Dict[str, Any]) -> Prophet:
        """Initialize the Prophet model with loaded parameters."""
        model = Prophet(
            growth=model_data['growth'],
            changepoint_prior_scale=model_data['changepoint_prior_scale'],
            seasonality_prior_scale=model_data['seasonality_prior_scale'],
            holidays_prior_scale=model_data['holidays_prior_scale'],
            yearly_seasonality=model_data['yearly_seasonality'],
            weekly_seasonality=model_data['weekly_seasonality'],
            daily_seasonality=model_data['daily_seasonality']
        )
        if model_data.get('extra_regressors'):
            for name, regressor in model_data['extra_regressors'].items():
                model.add_regressor(name)
        return model


    def cross_validate_model(self) -> Dict[str, Any]:
        """Perform time series cross-validation."""
        try:
            cv_results = cross_validation(
                self.model,
                initial='730 days',
                period='180 days',
                horizon='30 days',
                parallel="processes"
            )
            metrics = performance_metrics(cv_results)
            logging.info(f"Cross-validation metrics: {metrics}")
            return metrics
        except Exception as e:
            logging.error(f"Cross-validation failed: {str(e)}")
            raise



    def optimize_hyperparameters(self) -> Dict[str, Any]:
        """Find optimal hyperparameters."""
        try:
            param_grid = self.get_param_grid()
            best_params = self.grid_search(param_grid)
            logging.info(f"Best parameters found: {best_params}")
            return best_params
        except Exception as e:
            logging.error(f"Hyperparameter optimization failed: {str(e)}")
            raise

    def get_param_grid(self) -> Dict[str, Any]:
        """Define the parameter grid for hyperparameter tuning."""
        return {
            'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
            'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
            'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
            'seasonality_mode': ['additive', 'multiplicative']
        }

    def grid_search(self, param_grid: Dict[str, Any]) -> Dict[str, Any]:
        """Perform grid search for hyperparameter tuning."""
        best_rmse = float('inf')
        best_params = None
        results = []

        for cp in param_grid['changepoint_prior_scale']:
            for sp in param_grid['seasonality_prior_scale']:
                for hp in param_grid['holidays_prior_scale']:
                    for sm in param_grid['seasonality_mode']:
                        self.model = Prophet(
                            changepoint_prior_scale=cp,
                            seasonality_prior_scale=sp,
                            holidays_prior_scale=hp,
                            seasonality_mode=sm
                        )
                        self.train()
                        self.predict()

                        if self.metrics['RMSE'] < best_rmse:
                            best_rmse = self.metrics['RMSE']
                            best_params = {
                                'changepoint_prior_scale': cp,
                                'seasonality_prior_scale': sp,
                                'holidays_prior_scale': hp,
                                'seasonality_mode': sm
                            }

                        results.append({
                            'params': best_params,
                            'metrics': self.metrics
                        })

        return best_params


if __name__ == "__main__":
    try:
        forecaster = ProphetForecaster(symbol="AAPL", forecast_periods=30)

        # Check if best parameters are already known
        best_params_path = f"models/time_series/{forecaster.symbol}_best_params.json"
        if os.path.exists(best_params_path):
            with open(best_params_path, "r") as f:
                best_params = json.load(f)
            logging.info(f"Loaded best parameters from {best_params_path}")
        else:
            # Optimize hyperparameters
            best_params = forecaster.optimize_hyperparameters()
            with open(best_params_path, "w") as f:
                json.dump(best_params, f, indent=4)
            logging.info(f"Saved best parameters to {best_params_path}")

        # Train with best parameters
        forecaster.model = Prophet(
            changepoint_prior_scale=best_params['changepoint_prior_scale'],
            seasonality_prior_scale=best_params['seasonality_prior_scale'],
            holidays_prior_scale=best_params['holidays_prior_scale'],
            seasonality_mode=best_params['seasonality_mode']
        )
        (forecaster
            .train(include_economics=True)
            .predict()
            .save_model()
            .visualize())
        logging.info(f"Model Metrics: {forecaster.metrics}")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
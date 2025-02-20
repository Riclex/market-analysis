# Import libraries
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import logging 
from dotenv import load_dotenv
from sqlalchemy import create_engine
import ta


# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load env variables
load_dotenv()


def load_economic_data(series_ids: list):
    """Load economic data from Parquet files."""
    dfs = [pd.read_parquet(f"data/processed/fred/{sid}.parquet") for sid in series_ids]
    return pd.concat(dfs)

class LSTMForecaster:
    def __init__(self, symbol: str = 'MSFT', sequence_length: int = 60):
        self.symbol = symbol
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0 , 1))
        self.model = None
        self.data = None


    def load_data(self):
        """Load and process data from multiple sources"""
        try:
            # Load processed price data
            price_df = pd.read_parquet(f"data/processed/{self.symbol}_features.parquet")
            logging.info(f"Loaded price data: {price_df.shape}")

            # Load fundamental data
            fundamental_df = pd.read_csv(f"data/raw/{self.symbol}_fundamentals.csv")
            logging.info(f"Loaded fundamental data: {fundamental_df.shape}")

            # Load economic data
            econ_df = pd.read_parquet("data/processed/fred/combined.parquet")
            logging.info(f"Loaded economic data: {econ_df.shape}")

            # Convert dates to datetime
            for df in [price_df, econ_df]:
                df['date'] = pd.to_datetime(df['date'])

            # Merge datasets
            merged_data = price_df.merge(fundamental_df, on="symbol", how="left")
            merged_data = merged_data.merge(econ_df, on="date", how="left")
            logging.info(f"Merged data shape: {merged_data.shape}")

            # Drop non-numeric columns except date
            non_numeric_cols = merged_data.select_dtypes(exclude=['float64', 'int64']).columns
            cols_to_drop = [col for col in non_numeric_cols if col != 'date' and col != 'volume']
            merged_data = merged_data.drop(cols_to_drop, axis=1)

            # Forward fill missing data
            merged_data = merged_data.fillna(method='ffill').fillna(method='bfill')

            # Verify data quality
            logging.info(f"Columns: {merged_data.columns.tolist()}")
            logging.info(f"Data shape: {merged_data.shape}")
            logging.info(f"Close price range: {merged_data['close'].min():.2f} to {merged_data['close'].max():.2f}")
            logging.info(f"Sample of close prices: {merged_data['close'].head()}")

            # Check for constant values
            for col in merged_data.columns:
                if merged_data[col].std() == 0:
                    logging.warning(f"Column {col} has zero variance")

            merged_data = merged_data.set_index('date')
            self.data = merged_data

            logging.info(f"Successfully loaded data for {self.symbol}")
            return True

        except Exception as e:
            logging.error(f"Data loading failed: {str(e)}")
            return False
        

    def preprocess_data(self):
        """Feature engineering and data normalization"""
        try:
            # Log available columns
            logging.info(f"Available columns: {self.data.columns.tolist()}")

            # Feature selection
            base_feature_columns = ['close', 'ma_7', 'volume', 'peRatio', 'dividendYield',
                                    'GDP', 'UNRATE', 'CPIAUCSL']
            self.features = [col for col in base_feature_columns if col in self.data.columns]
            features = self.data[self.features].copy()

            # Log transformation
            for col in ['volume', 'close']:
                if col in features.columns:
                    features[col] = np.log1p(features[col])
                    logging.info(f"Applied log transform to {col}")

            # Temporal features
            features['day_of_week'] = self.data.index.dayofweek
            features['month'] = self.data.index.month

            # Add technical indicators
            features['rsi'] = ta.momentum.RSIIndicator(close=self.data['close']).rsi()
            features['macd'] = ta.trend.MACD(close=self.data['close']).macd()
            features['bb_high'] = ta.volatility.BollingerBands(close=self.data['close']).bollinger_hband()

            # Add price changes
            features['price_change'] = self.data['close'].pct_change()
            features['volatility'] = self.data['close'].rolling(window=20).std()

            # Handle NaN values
            features = features.ffill().bfill()

            # Update features list
            self.all_features = features.columns.tolist()
            self.n_features = len(self.all_features)

            # Normalize data
            scaled_data = self.scaler.fit_transform(features)

            # Create sequences
            X, y = [], []
            for i in range(len(scaled_data) - self.sequence_length - 1):
                X.append(scaled_data[i:i + self.sequence_length])
                y.append(scaled_data[i + self.sequence_length, 0])  # Close price prediction

            X = np.array(X)
            y = np.array(y)

            # Train-test split, 80-20
            split = int(0.8 * len(X))
            self.X_train, self.X_test = X[:split], X[split:]
            self.y_train, self.y_test = y[:split], y[split:]

            logging.info(f"Features used: {self.n_features}")
            logging.info(f"Data shape: X_train={self.X_train.shape}, X_test={self.X_test.shape}")
            return True

        except Exception as e:
            logging.error(f"Preprocessing data failed: {str(e)}")
            return False    

    def build_model(self):
        """Create LSTM architecture"""
        self.model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.sequence_length, self.n_features),
                 kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            Dropout(0.3),
            LSTM(64, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            Dropout(0.3),
            LSTM(32, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])

        self.model.compile(
            optimizer='adam',
            loss='huber_loss',
            metrics=['mae']
        )

        self.model.summary()


    def train(self, epochs: int = 100, batch_size: int = 32):
        """Train with early stopping to prevent overfitting"""
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            mode='min'
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001
        )

        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stop, reduce_lr],
            shuffle=True
        )

        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss during Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"plots/{self.symbol}_training_history.png")
        plt.close()

        return history    

    def evaluate(self):
        try:
            predictions = self.model.predict(self.X_test)

            n_total_features = len(self.all_features)
            actual_padded = np.zeros((len(self.y_test), n_total_features))
            actual_padded[:, 0] = self.y_test

            pred_padded = np.zeros((len(predictions), n_total_features))
            pred_padded[:, 0] = predictions.flatten()

            # Handle NaN values before inverse transform
            actual_padded = np.nan_to_num(actual_padded)
            pred_padded = np.nan_to_num(pred_padded)

            # Inverse transform
            actual = self.scaler.inverse_transform(actual_padded)[:, 0]
            predictions = self.scaler.inverse_transform(pred_padded)[:, 0]

            # Remove any NaN values
            mask = ~np.isnan(actual) & ~np.isnan(predictions)
            actual = actual[mask]
            predictions = predictions[mask]

            if len(actual) == 0 or len(predictions) == 0:
                logging.error("No valid predictions after removing NaN values")
                return

            # Calculate metrics
            metrics = {
                'mae': mean_absolute_error(actual, predictions),
                'mape': np.mean(np.abs((actual - predictions) / actual)) * 100,
                'rmse': np.sqrt(np.mean((actual - predictions) ** 2)),
                'r2': r2_score(actual, predictions),
                'direction_accuracy': np.mean(np.sign(np.diff(actual)) ==
                                              np.sign(np.diff(predictions))) * 100
            }

             # Log metrics
            for metric, value in metrics.items():
                logging.info(f"Test {metric.upper()}: {value:.2f}")

            # Visualize results
            plt.figure(figsize=(12, 6))
            plt.plot(actual[-100:], label='Actual Price')
            plt.plot(predictions[-100:], label='Predicted Price')
            plt.title(f"{self.symbol} Stock Price Prediction")
            plt.xlabel("Trading Days")
            plt.ylabel("Price ($)")
            plt.legend()
            plt.savefig(f"plots/{self.symbol}_lstm_prediction.png", dpi=300, bbox_inches='tight')
            plt.close()

            return metrics

        except Exception as e:
            logging.error(f"Evaluation failed: {str(e)}")


    def save_artifacts(self):
        """Save model, scaler, and performance metrics."""
        try:
            # Create directories if they don't exist
            os.makedirs("models", exist_ok=True)
            os.makedirs("plots", exist_ok=True)

            # Save model and scaler
            self.model.save(f"models/{self.symbol}_lstm.keras")
            pd.to_pickle(self.scaler, f"models/{self.symbol}_scaler.pkl")

            # Save model architecture and parameters
            with open(f"models/{self.symbol}_lstm_summary.txt", 'w') as f:
                self.model.summary(print_fn=lambda x: f.write(x + '\n'))

            logging.info("Model artifacts saved successfully")

        except Exception as e:
            logging.error(f"Failed to save artifacts: {str(e)}")

    def plot_predictions(self, actual, predictions, intervals=None):
        plt.figure(figsize=(15, 8))

        # Plot actual and predicted values
        plt.plot(actual[-100:], 'b-', label='Actual Price', alpha=0.7)
        plt.plot(predictions[-100:], 'r--', label='Predicted Price', alpha=0.7)

        if intervals:
            plt.fill_between(range(len(predictions[-100:])),
                             intervals['lower'][-100:],
                             intervals['upper'][-100:],
                             color='r', alpha=0.1,
                             label='95% Confidence Interval')

        plt.title(f"{self.symbol} Stock Price Prediction")
        plt.xlabel("Trading Days")
        plt.ylabel("Price ($)")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Add metrics annotation
        mae = mean_absolute_error(actual, predictions)
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100
        plt.annotate(f'MAE: ${mae:.2f}\nMAPE: {mape:.2f}%',
                     xy=(0.05, 0.95), xycoords='axes fraction')

        plt.tight_layout()
        plt.savefig(f"plots/{self.symbol}_lstm_prediction.png", dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    #forecaster = LSTMForecaster(symbol="MSFT")
    forecaster = LSTMForecaster(symbol="AAPL")
    #forecaster = LSTMForecaster(symbol="AMZN")

    if forecaster.load_data() and forecaster.preprocess_data():
        forecaster.build_model()
        forecaster.train(epochs=100)
        forecaster.evaluate()
        forecaster.save_artifacts()
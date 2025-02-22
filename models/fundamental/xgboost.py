import os
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import numpy as np
import json
import logging
from dotenv import load_dotenv

# Supress warnings
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings('ignore', category=DataConversionWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*DataConversionWarning.*')

# Logging configuration
load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def train_xgboost_model(symbol: str, test_size: float = 0.2, cross_validate: bool = True):
    """ Train with comprehensive data integration and hyperparameter tuning
        Args:
            symbol: Stock ticker 
            test_size: Proportion of data for testing
            cross_validate: Enable time-series cross-validation
        Returns:
            Trained XGBoost model and evaluation metrics
    """ 
          
    try:
        # Load and prepare data
        df = load_and_preprocess_data(symbol)
        
        # Ensure date is dropped
        X = df.drop(columns=['target', 'date'] if 'date' in df.columns else ['target'])
        y = df['target']
        
        # Time-series aware split
        X_train, X_test, y_train, y_test = time_series_split(df, test_size)

        logging.info(f"Training data shape: {X_train.shape}")
        logging.info(f"Test data shape: {X_test.shape}")

        # Preprocessing pipeline
        preprocessor = build_preprocessor(X.columns)

        # Model pipeline
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=500,
                learning_rate=0.05,
                random_state=42
            ))
        ])

        # Hyperparameter grid
        param_grid = {
            'regressor__max_depth': [3, 5, 7],
            'regressor__subsample': [0.8, 1.0],
            'regressor__colsample_bytree': [0.8, 1.0]
        }
            
        # Train with cross validation
        if cross_validate:
            ts_cv = TimeSeriesSplit(n_splits=5).split(X_train)
            grid_search = GridSearchCV(
                model, 
                param_grid,
                cv=ts_cv,
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
        else:
            best_model = model.fit(X_train, y_train)

        # Evaluate model
        metrics = evaluate_model(best_model, X_test, y_test, symbol)

        # Save artifacts
        save_model_artifacts(best_model, symbol, df.columns)

        return best_model, metrics
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")

def load_and_preprocess_data(symbol: str):
    """ Load and merge data sources"""
    try:
        # Load
        price_df = pd.read_parquet(f"data/processed/{symbol}_features.parquet")
        fundamental_df = pd.read_csv(f"data/raw/{symbol}_fundamentals.csv")
        econ_df = pd.read_parquet("data/processed/fred/combined.parquet")

        # Convert date 
        price_df['date'] = pd.to_datetime(price_df['date'])
        econ_df['date'] = pd.to_datetime(econ_df['date'])
        
        # Add symbol if not present
        if 'symbol' not in price_df.columns:
            price_df['symbol'] = symbol

        # Merge datasets
        df = (price_df.merge(fundamental_df, on='symbol', how='left')
            .merge(econ_df, on='date', how='left')
            )

        # Feature engineering
        df['target'] = df['close'].shift(-1)    # Predict next day's close
        df['price_to_ma'] = df['close'] / df['ma_7']
        df['volume_change'] = df['volume'].pct_change()
        df['market_cap_log'] = np.log(df['marketCap'])

        # Data validation
        df = validate_data(df)

        # Drop null rows
        df = df.dropna(subset=['target'])

        return df
    except Exception as e:
        
        logging.error(f"Data loading failed: {str(e)}")
        raise

def engineer_features(df: pd.DataFrame, config=None):
    """Apply configurable feature engineering transformations"""
    if config is None:
        config = {
            'moving_averages': [7, 20, 50],
            'bollinger_bands': {'window': 20, 'num_std': 2},
            'momentum_indicators': True,
            'volume_indicators': True,
            'target_horizon': 1  # Predict 1 day ahead by default
        }
    
    # Original feature count
    original_feature_count = len(df.columns)
    
    # Create target variable - configurable horizon
    df['target'] = df['close'].shift(-config['target_horizon'])
    
    # Moving averages
    for window in config['moving_averages']:
        df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
        df[f'price_to_ma_{window}'] = df['close'] / df[f'ma_{window}']
    
    # Bollinger Bands
    if config['bollinger_bands']:
        window = config['bollinger_bands']['window']
        num_std = config['bollinger_bands']['num_std']
        rolling_std = df['close'].rolling(window=window).std()
        df[f'bollinger_upper'] = df[f'ma_{window}'] + (rolling_std * num_std)
        df[f'bollinger_lower'] = df[f'ma_{window}'] - (rolling_std * num_std)
        df['bollinger_pct'] = (df['close'] - df[f'bollinger_lower']) / (df[f'bollinger_upper'] - df[f'bollinger_lower'])
    
    # Momentum indicators
    if config['momentum_indicators']:
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Volume indicators
    if config['volume_indicators']:
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']
    
    # Financial ratios
    if 'marketCap' in df.columns and 'revenue' in df.columns:
        df['market_cap_log'] = np.log(df['marketCap'])
        df['price_to_sales'] = df['marketCap'] / df['revenue']
    
    # Calendar features
    if 'date' in df.columns:
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        
    # Lagged features
    for lag in [1, 3, 5]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'return_lag_{lag}'] = df['close'].pct_change(lag)
    
    # Log new feature count
    new_features = len(df.columns) - original_feature_count
    logging.info(f"Created {new_features} new features through feature engineering")
    
    return df

def time_series_split(df: pd.DataFrame, test_size: float):
    """ Time-series aware data splitting"""
    split_idx = int(len(df) * (1 - test_size))

    # Remove date colum if present
    features_to_drop = ['target']
    if 'date' in df.columns:
        features_to_drop.append('date')

    return (
        df.iloc[:split_idx].drop(columns=features_to_drop),
        df.iloc[split_idx:].drop(columns=features_to_drop),
        df.iloc[:split_idx]['target'],
        df.iloc[split_idx:]['target']
        )

def build_preprocessor(features):
    """Build feature preprocessing pipeline"""
    categorical_features = ['symbol']

    # Remove date column
    numeric_features = [f for f in features if f not in categorical_features + ['date', 'target']]

    logging.info(f"Numeric features: {numeric_features}")
    logging.info(f"Categorial features: {categorical_features}")

    # Transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
        ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
        ('onehot', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'))
        ])

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
        )

    return preprocessor

def predict_next_day(symbol: str, model_version=None):
    """Make prediction for the next trading day"""
    try:
        # Load latest data
        df = load_and_preprocess_data(symbol)
        
        # Load the model (latest or specific version)
        if model_version:
            model_path = f"models/{symbol}/version_{model_version}/xgboost_model.pkl"
        else:
            # Find latest model version
            model_dirs = sorted(os.listdir(f"models/{symbol}/"))
            if not model_dirs:
                raise FileNotFoundError(f"No models found for {symbol}")
            latest_dir = model_dirs[-1]
            model_path = f"models/{symbol}/{latest_dir}/xgboost_model.pkl"
        
        model = joblib.load(model_path)
        
        # Get latest row (without target)
        latest_data = df.iloc[-1:].drop(columns=['target'])
        
        # Make prediction
        prediction = model.predict(latest_data)[0]
        
        # Calculate confidence interval based on recent model performance
        recent_errors = []
        if len(df) > 30:  # If we have enough historical data
            recent_predictions = model.predict(df.iloc[-30:].drop(columns=['target']))
            recent_actuals = df.iloc[-30:]['target'].values
            recent_errors = np.abs(recent_predictions - recent_actuals)
            error_std = np.std(recent_errors)
            confidence_interval = (prediction - 1.96 * error_std, prediction + 1.96 * error_std)
        else:
            confidence_interval = None
        
        # Get current price for comparison
        current_price = df.iloc[-1]['close']
        predicted_change_pct = ((prediction - current_price) / current_price) * 100
        
        result = {
            'symbol': symbol,
            'prediction_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
            'current_price': current_price,
            'predicted_price': prediction,
            'predicted_change_pct': predicted_change_pct,
            'confidence_interval': confidence_interval,
            'model_version': model_version or latest_dir
        }
        return result
        
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise

def monitor_model_performance(symbol: str, lookback_days=30):
    """Monitor ongoing model performance and detect drift"""
    try:
        # Load latest data
        df = load_and_preprocess_data(symbol)
        
        # Load latest model
        model_dirs = sorted(os.listdir(f"models/{symbol}/"))
        if not model_dirs:
            raise FileNotFoundError(f"No models found for {symbol}")
        latest_dir = model_dirs[-1]
        model_path = f"models/{symbol}/{latest_dir}/xgboost_model.pkl"
        model = joblib.load(model_path)
        
        # Get performance metrics from training time
        with open(f"models/{symbol}/{latest_dir}/metrics.json", 'r') as f:
            training_metrics = json.load(f)
        
        # Calculate recent performance
        recent_data = df.iloc[-lookback_days:]
        X_recent = recent_data.drop(columns=['target'])
        y_recent = recent_data['target']
        
        predictions = model.predict(X_recent)
        recent_metrics = {
            'mae': mean_absolute_error(y_recent, predictions),
            'rmse': np.sqrt(mean_squared_error(y_recent, predictions)),
            'r2': r2_score(y_recent, predictions),
            'mape': np.mean(np.abs((y_recent - predictions) / y_recent)) * 100
        }
        
        # Compare current vs. training performance
        performance_change = {
            metric: (recent_metrics[metric] - training_metrics[metric]) / training_metrics[metric] * 100
            for metric in training_metrics
        }
        
        # Flag significant drift
        drift_detected = any(abs(change) > 20 for change in performance_change.values())
        
        # Create monitoring report
        report = {
            'symbol': symbol,
            'model_version': latest_dir,
            'evaluation_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
            'training_metrics': training_metrics,
            'recent_metrics': recent_metrics,
            'performance_change_pct': performance_change,
            'drift_detected': drift_detected,
            'lookback_days': lookback_days
        }
        
        # Save monitoring report
        os.makedirs(f"models/{symbol}/{latest_dir}/monitoring", exist_ok=True)
        with open(f"models/{symbol}/{latest_dir}/monitoring/report_{pd.Timestamp.now().strftime('%Y%m%d')}.json", 'w') as f:
            json.dump(report, f, indent=4)
            
        if drift_detected:
            logging.warning(f"Model drift detected for {symbol}. Consider retraining the model.")
        
        return report
        
    except Exception as e:
        logging.error(f"Model monitoring failed: {str(e)}")
        raise


def validate_data(df: pd.DataFrame):
    """Validate data before preprocessing with more comprehensive checks"""
    # Basic shape and type checks
    logging.info(f"DataFrame shape: {df.shape}")
    
    # Check for expected columns
    expected_columns = ['date', 'close', 'volume', 'ma_7', 'marketCap']
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"Missing expected columns: {missing_columns}")
        raise ValueError(f"Data missing critical columns: {missing_columns}")
    
    # Check for nulls
    null_pct = (df.isnull().sum() / len(df)) * 100
    high_null_cols = null_pct[null_pct > 5].index.tolist()
    if high_null_cols:
        logging.warning(f"Columns with >5% null values: {high_null_cols}")
    
    # Check for data range validity
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].min() < 0 and col not in ['GDP', 'UNRATE', 'CPIAUCSL']:  # Allow negative values for some econ indicators
            logging.warning(f"Column {col} contains negative values: min={df[col].min()}")
    
    # Check for outliers
    for col in numeric_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        outlier_count = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
        if outlier_count > 0:
            outlier_pct = (outlier_count / len(df)) * 100
            if outlier_pct > 1:  # More than 1% outliers
                logging.warning(f"Column {col} has {outlier_pct:.2f}% potential outliers")
    
    # Check for data freshness
    if 'date' in df.columns:
        latest_date = df['date'].max()
        days_old = (pd.Timestamp.now() - latest_date).days
        if days_old > 7:  # Data is more than a week old
            logging.warning(f"Data may be stale - latest date is {latest_date} ({days_old} days old)")
    
    return df

def evaluate_model(model, X_test, y_test, symbol: str):
    """Calculate evaluation metrics and visualizations"""
    predictions = model.predict(X_test)

    metrics = {
          'mae': mean_absolute_error(y_test, predictions),
          'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
          'r2': r2_score(y_test, predictions),
          'mape': np.mean(np.abs((y_test - predictions) / y_test)) * 100    
    }

    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    xgb.plot_importance(model.named_steps['regressor'], ax=ax)
    plt.savefig(f"plots/xgboost_{symbol}_feature_importance.png")
    plt.close()

    # Predictions vs actuals
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test.values, label='Actual')
    ax.plot(predictions, label='Predicted')
    ax.set_title(f"{symbol} Price Predictions")
    ax.legend()
    plt.savefig(f"plots/xgboost_{symbol}_predictions.png")
    plt.close()

    return metrics

def backtest_model(symbol: str, start_date=None, end_date=None, window_size=90, step_size=30):
    """Backtest model performance through walk-forward validation"""
    try:
        # Load full dataset
        df = load_and_preprocess_data(symbol)
        df = df.sort_values('date')
        
        # Filter by date range if specified
        if start_date:
            df = df[df['date'] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df['date'] <= pd.Timestamp(end_date)]
            
        # Setup for walk-forward validation
        results = []
        dates = df['date'].unique()
        
        for i in range(0, len(dates) - window_size, step_size):
            # Define train/test periods
            train_end_idx = i + int(window_size * 0.8)  # 80% for training
            train_dates = dates[i:train_end_idx]
            test_dates = dates[train_end_idx:i+window_size]
            
            if len(test_dates) < 5:  # Skip if test period too small
                continue
                
            # Split data
            train_df = df[df['date'].isin(train_dates)]
            test_df = df[df['date'].isin(test_dates)]
            
            # Prepare features and target
            X_train = train_df.drop(columns=['target', 'date'])
            y_train = train_df['target']
            X_test = test_df.drop(columns=['target', 'date'])
            y_test = test_df['target']
            
            # Build and train model (simplified for backtesting)
            preprocessor = build_preprocessor(X_train.columns)
            model = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200))
            ])
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate on test set
            predictions = model.predict(X_test)
            period_metrics = {
                'start_date': test_dates[0],
                'end_date': test_dates[-1],
                'train_size': len(train_df),
                'test_size': len(test_df),
                'mae': mean_absolute_error(y_test, predictions),
                'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                'r2': r2_score(y_test, predictions),
                'mape': np.mean(np.abs((y_test - predictions) / y_test)) * 100
            }
            
            results.append(period_metrics)
            
        # Compile results
        results_df = pd.DataFrame(results)
        
        # Calculate overall and volatility metrics
        summary = {
            'mean_rmse': results_df['rmse'].mean(),
            'median_rmse': results_df['rmse'].median(),
            'std_rmse': results_df['rmse'].std(),
            'worst_period': results_df.loc[results_df['rmse'].idxmax(), 'start_date'].strftime('%Y-%m-%d'),
            'best_period': results_df.loc[results_df['rmse'].idxmin(), 'start_date'].strftime('%Y-%m-%d'),
            'performance_stability': results_df['rmse'].std() / results_df['rmse'].mean()
        }
        
        # Visualize backtest results
        plt.figure(figsize=(12, 6))
        plt.plot(results_df['start_date'], results_df['rmse'], marker='o')
        plt.title(f"{symbol} Model Performance Across Time Periods")
        plt.ylabel("RMSE")
        plt.xlabel("Test Period Start")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        #os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/{symbol}_backtest_performance.png")
        plt.close()
        
        return results_df, summary
        
    except Exception as e:
        logging.error(f"Backtesting failed: {str(e)}")
        raise


def save_model_artifacts(model, symbol, features, metrics=None):
    """Save model, metadata, and training metrics"""
    version = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"models/{symbol}/version_{version}"
    os.makedirs(model_dir, exist_ok=True)

    # Save Model
    joblib.dump(model, f"{model_dir}/xgboost_model.pkl")

    # Feature list
    pd.Series(features).to_csv(f"{model_dir}/features.csv")

    # Save model hyperparameters
    params = model.named_steps['regressor'].get_params()
    with open(f"{model_dir}/hyperparameters.json", 'w') as f:
        json.dump(params, f, indent=4)

    # Save evaluation metrics if provided
    if metrics:
        with open(f"{model_dir}/metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)

    # Save a model card with metadata
    with open(f"{model_dir}/model_card.md", 'w') as f:
        f.write(f"# XGBoost Model for {symbol}\n")
        f.write(f"Created: {pd.Timestamp.now()}\n\n")
        f.write("## Performance Metrics\n")
        if metrics:
            for metric, value in metrics.items():
                f.write(f"- {metric}: {value:.4f}\n")
    
    logging.info(f"Saved XGBoost model artifacts for {symbol} in {model_dir}")
    return model_dir
        

def main():
    """Command line interface for model training and prediction"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Stock Price Prediction with XGBoost')
    subparsers = parser.add_subparsers(dest='command')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train XGBoost model')
    train_parser.add_argument('--symbol', type=str, required=True, help='Stock ticker symbol')
    train_parser.add_argument('--test-size', type=float, default=0.2, help='Test split proportion')
    train_parser.add_argument('--cross-validate', action='store_true', help='Use time-series cross validation')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions with trained model')
    predict_parser.add_argument('--symbol', type=str, required=True, help='Stock ticker symbol')
    predict_parser.add_argument('--model-version', type=str, help='Specific model version to use')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor model performance')
    monitor_parser.add_argument('--symbol', type=str, required=True, help='Stock ticker symbol')
    monitor_parser.add_argument('--lookback-days', type=int, default=30, help='Number of days to evaluate')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        print(f"Training XGBoost model for {args.symbol}...")
        model, metrics = train_xgboost_model(args.symbol, args.test_size, args.cross_validate)
        print(f"Training complete. Model metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
            
    elif args.command == 'predict':
        print(f"Predicting next day price for {args.symbol}...")
        result = predict_next_day(args.symbol, args.model_version)
        print(f"Current price: ${result['current_price']:.2f}")
        print(f"Predicted price: ${result['predicted_price']:.2f}")
        print(f"Predicted change: {result['predicted_change_pct']:.2f}%")
        if result['confidence_interval']:
            print(f"95% Confidence interval: ${result['confidence_interval'][0]:.2f} to ${result['confidence_interval'][1]:.2f}")
            
    elif args.command == 'monitor':
        print(f"Monitoring model performance for {args.symbol}...")
        report = monitor_model_performance(args.symbol, args.lookback_days)
        print("Model performance summary:")
        print(f"  Training RMSE: {report['training_metrics']['rmse']:.4f}")
        print(f"  Current RMSE: {report['recent_metrics']['rmse']:.4f}")
        print(f"  Performance change: {report['performance_change_pct']['rmse']:.2f}%")
        if report['drift_detected']:
            print("⚠️ WARNING: Significant model drift detected. Consider retraining.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
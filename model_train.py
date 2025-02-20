# Import required libraries
import logging
import joblib
import os
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, YearLocator
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression as LinearReg
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

def configure_logging():
    """
    Configures logging to output messages to the console with a specific format
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """
    Loads data from a specified parquet file
    Parameters:
    - file_path (str): The path to the parquet file
    Returns:
    - pd.DataFrame: The loaded data as a pandas DataFrame
    """
    try:
        return pd.read_parquet(file_path)
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        return None

def impute_missing_values(df, columns, imputer_path):
    """
    Imputes missing values in specified columns using a SimpleImputer model
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data
    - columns (list): The list of columns to impute
    - imputer_path (str): The path to save/load the imputer model
    Returns:
    - pd.DataFrame: The DataFrame with imputed values
    """
    try:
        imputer = joblib.load(imputer_path)
        logging.info("Loaded existing imputer model.")
    except FileNotFoundError:
        imputer = SimpleImputer(strategy='mean')
        df[columns] = imputer.fit_transform(df[columns])
        joblib.dump(imputer, imputer_path)
        logging.info("Fitted and saved new imputer model.")
    else:
        df[columns] = imputer.transform(df[columns])
    return df

def train_model(X, y, model_path):
    """
    Trains a linear regression model using time-series split for cross-validation
    Parameters:
    - X (pd.DataFrame): The feature set
    - y (pd.Series): The target variable
    - model_path (str): The path to save the trained model
    Returns:
    - model: The trained linear regression model
    - X_test (pd.DataFrame): The test feature set from the last fold
    - y_test (pd.Series): The test target variable from the last fold
    """
    model = LinearReg()
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        logging.info("Training Completed for current fold")

    joblib.dump(model, model_path)
    logging.info("Model saved.")
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the test set and logs the performance metrics
    Parameters:
    - model: The trained linear regression model
    - X_test (pd.DataFrame): The test feature set
    - y_test (pd.Series): The test target variable
    Returns:
    - y_pred (np.array): The predicted values
    - y_test (pd.Series): The actual values
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    logging.info(f"Mean Squared Error on test set: {mse}")
    logging.info(f"R^2 Score on test set: {r2}")
    logging.info(f"Mean Absolute Error: {mae:.2f}")
    logging.info(f"Mean Absolute Percentage Error: {mape:.2%}")

    return y_pred, y_test

def plot_results(y_test, y_pred, plot_path):
    """
    Plots the actual vs predicted values and saves the plot to a file
    Parameters:
    - y_test (pd.Series): The actual values
    - y_pred (np.array): The predicted values
    - plot_path (str): The path to save the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label='Actual', alpha=0.7)
    plt.plot(y_test.index, y_pred, label='Predicted', alpha=0.7)
    plt.title('Stock Price: Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(YearLocator())
    plt.gcf().autofmt_xdate()
    plt.savefig(plot_path, dpi=150)
    plt.close()

def main():
    """
    Main function to execute the data loading, preprocessing, model training, evaluation, and plotting.
    """
    configure_logging()

    # Load data
    #msft = load_data("data/processed/msft_features.parquet")
    aapl = load_data("data/processed/aapl_features.parquet")
    # amzn = load_data("data/processed/amzn_features.parquet")

    # Use the appropriate dataset
    df = aapl  # Change as needed

    # Impute missing values
    df = impute_missing_values(df, ["ma_7", "volume"], "models/imputer.pkl")

    # Define features and target
    X = df[["ma_7", "volume"]]
    y = df["close"]

    # Train the model
    model, X_test, y_test = train_model(X, y, "models/linear_regression.pkl")

    # Evaluate the model
    y_pred, y_test = evaluate_model(model, X_test, y_test)

    # Plot the results
    plot_results(y_test, y_pred, 'plots/aapl_linear_regression_predictions.png') # change as needed

if __name__ == "__main__":
    main()
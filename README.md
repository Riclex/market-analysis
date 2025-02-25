# Market Analysis & Predictive Modeling Platform

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A comprehensive system for analysing financial markets, predictive modelling, and visualisation. This platform integrates market data, utilises various AI and machine learning models for forecasting, and provides real-time interactive dashboards.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Docker Deployment](#docker-deployment)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)


## Overview

This platform is designed to assist users in analysing financial markets and generating predictive insights through a combination of time series models, regression techniques, and natural language processing for sentiment analysis. It integrates data from various sources, including stock prices, economic indicators, and news sentiment.

## Features

- **Data Integration:**
  - Stock prices from Alpha Vantage/Yahoo Finance
  - Fundamental data from Yahoo Finance
  - Economic indicators via FRED
  - News sentiment through NewsAPI

- **AI/ML Models**
    - LSTM Neural Networks
    - Facebook Prophet
    - XGBoost Regression
    - ARIMA Time Series (in progress)
    - NLP Sentiment Analysis

- **Dashboards & Visualizations:**
  - Interactive Plotly Dash dashboards
  - Model performance metrics and market trend visualizations

- **Architecture & Deployment:**
  - PostgreSQL database integration
  - Dockerized deployment (in progress)
  - Automated data pipelines
  - CI/CD with GitHub Actions

## Installation

### Prerequisites
    - Python 3.10+
    - PostgreSQL
    - VS code (recommended)
    - Docker (optional)

### Setup
1. Clone repository:
    ```bash
        git clone https://github.com/riclex/market-analysis.git
        cd market-analysis
    ```    
2. Create a virtual environment:
    ```bash
        python -m venv venv
        venv\Scripts/activate
    ```
3. Install dependencies:
    ```bash
        pip install -r requirements.txt
    ```
4. Configure environment variables:
    ```bash
        # .env
        ALPHA_VANTAGE_KEY="your_api_key"
        NEWS_API_KEY="your_api_key"
        FRED_API_KEY="your_api_key"
        DB_URI="postgresql:://finance_user:securepass@localhost:5432/finance"
    ```
5. Initialize databse:
    ```bash
        psql -U postgres -f db/schema.sql
    ```
## Usage
- Data ingestion

    ```bash
        python data_ingestion.py
    ```
- Feature Engineering 
 
    ```bash
        python data_processing.py
    ```
- Model Training
    ```bash
        # Train Prophet model
        python models/time_series/prophet_forecaster.py

        # Train XGBoost model
        python models/fundamental/xgboost.py
    ```
- Dashboard
    ```bash
        python dashboard.py
    ```

 Access at -> http://localhost:8050

 - Docker deployment
    ```bash
        docker build -t finance-dashboard .
        docker run -p 8050:8050 finance-dashboard
    ```

## Project Structure
        market-analysis/
            ├── data/                   # Data storage
            │   ├── raw/                # Raw datasets
            │   ├── processed/          # Processed data
            │   └── news/               # News articles
            ├── models/                 # Trained models
            ├── utils/                  # Helper functions
            ├── tests/                  # Unit tests
            ├── docs/                   # Documentation
            ├── docker-compose.yml      # Container orchestration
            └── requirements.txt        # Dependencies

## Ackowledgements
- Market data providers: Alpha Vantage, Yahoo Finance
- Economic data: Federal Reserve Economic Data (FRED)
- Machine Learning: TensorFlow, Scikit-Learn, PySpark
- NLP Model: Hugging Face Transformers

## Contributing
Please read [CONTRIBUTING.md](docs\CONTRIBUTING.md) for development setup and contributing process.

## Possible future enhancements
- Real-time streaming data integration
- Portfolio optimization module
- Risk analysis models
- Cloud deployment (AWS/GCP/Azure)
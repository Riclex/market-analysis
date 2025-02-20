CREATE TABLE IF NOT EXISTS stock_prices (
    date DATE PRIMARY KEY,
    symbol VARCHAR(10),
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    volume BIGINT
);

CREATE TABLE IF NOT EXISTS fundamentals (
    symbol VARCHAR(10) PRIMARY KEY,
    market_cap BIGINT,
    pe_ratio FLOAT,
    dividendYield FLOAT
);
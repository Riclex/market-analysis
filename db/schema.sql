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
    marketcap BIGINT,
    peratio FLOAT,
    dividendyield FLOAT
);

CREATE TABLE IF NOT EXISTS news_articles (
    published_at TIMESTAMP,
    source VARCHAR(255),
    title TEXT,
    content TEXT,
    url TEXT,
    symbols VARCHAR(255),
    url_hash VARCHAR(32) PRIMARY KEY,
    sentiment VARCHAR(10),
    sentiment_score FLOAT,
);

CREATE TABLE IF NOT EXISTS classified_news (
    published_at TIMESTAMP,
    source VARCHAR(255),
    title TEXT,
    content TEXT,
    url TEXT,
    symbols VARCHAR(255),
    url_hash VARCHAR(32) PRIMARY KEY,
    sentiment VARCHAR(10),
    sentiment_score FLOAT,
    negative FLOAT,
    neutral FLOAT,
    positive FLOAT
);
# Import required libraries
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dotenv import load_dotenv
import logging
from transformers import pipeline
from sqlalchemy import create_engine
import hashlib

# Logging configuration
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NewsClient:
    def __init__(self):
        self.api_key = os.getenv("NEWS_API_KEY")
        self.base_url = "https://newsapi.org/v2/everything"
        self.engine = create_engine(os.getenv("DB_URI"))


    def generate_query(self, symbols: List[str]) -> str:
        """Generate search query for multiple stock symbols"""
        return f"({' OR '.join(symbols)}) AND (stock OR market OR earnings)"

    def fetch_news(self, symbols: List[str], days_back: int = 7, max_pages: int = 5) -> pd.DataFrame:
        """Fetch news articles related to stock symbols"""
        articles = []
        from_date = (datetime.now() - timedelta(days=days_back)).date().isoformat()
        params = self.build_request_params(symbols, from_date)

        for page in range(1, max_pages + 1):
            params["page"] = page
            response = self.make_request(params)
            if response:
                articles.extend(response.get("articles", []))
                if self.should_stop_pagination(response, page):
                    break

        return self.process_articles(articles, symbols)

    def build_request_params(self, symbols: List[str], from_date: str) -> Dict:
        """Build request parameters for the News API"""
        return {
            "q": self.generate_query(symbols),
            "from": from_date,
            "sortBy": "relevancy",
            "language": "en",
            "pageSize": 50,
            "apiKey": self.api_key
        }

    def make_request(self, params: Dict) -> Optional[Dict]:
        """Make a request to the News API and handle errors"""
        try:
            response = requests.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logging.error(f"News fetch failed: {str(e)}")
            return None

    def should_stop_pagination(self, response: Dict, page: int) -> bool:
        """Determine if pagination should stop"""
        return response.get("totalResults", 0) <= page * 50
    
    def process_articles(self, articles: List[Dict], symbols: List[str]) -> pd.DataFrame:
        """Process raw articles into structured DataFrame"""
        df = pd.DataFrame([{
            "published_at": article["publishedAt"],
            "source": article["source"]["name"],
            "title": article["title"],
            "content": article["content"] or article["description"],
            "url": article["url"],
            "symbols": ", ".join(symbols),
            "url_hash": hashlib.md5(article["url"].encode()).hexdigest()
        }
        for article in articles if article["content"]])

        if not df.empty:
            df = df.drop_duplicates("url_hash")
            df["published_at"] = pd.to_datetime(df["published_at"])

        return df
    
    def save_to_db(self, df: pd.DataFrame):
        """Save news data to PostgreSQL db"""
        try:
            if not df.empty:
                df.to_sql(
                    name="news_articles",
                    con=self.engine,
                    if_exists="append",
                    index=False,
                    method="multi"
                )
                logging.info(f"Saved {len(df)} news articles to database")
        except Exception as e:
            logging.error(f"Database save failed: {str(e)}")
    
    def save_to_parquet(self, df: pd.DataFrame, symbol: str):
        """Save news data to Parquet file"""
        try:
            os.makedirs("data/news", exist_ok=True)
            df.to_parquet(f"data/news/{symbol}_{datetime.now().date()}.parquet")
            logging.info("Saved news data to Parquet file")
        except Exception as e:
            logging.error(f"Parquet save failed: {str(e)}")



if __name__ == "__main__":
    # Initialize client
    news_client = NewsClient()

    # Fetch news for Apple and Microsoft
    news_df = news_client.fetch_news(["AAPL"])

    if not news_df.empty:
        # Save to databse
        news_client.save_to_db(news_df)

        # Save to Parquet
        news_client.save_to_parquet(news_df, "AAPL")

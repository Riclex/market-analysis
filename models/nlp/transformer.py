# Import required libraries
import os
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline
)
from sklearn.metrics import classification_report
import torch
import logging
from dotenv import load_dotenv
from typing import List, Dict, Optional
from tqdm import tqdm
from sqlalchemy import create_engine
import glob

# Logging configuration
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FinancialTextAnalyzer:
    def __init__(self, model_name: str = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"):
        """Initialize financial text analyzer
        Args:
            model_name: Pretrained model from Hugging Face hub
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.pipeline = TextClassificationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            top_k=None 
        )
        logging.info(f"Loaded model: {model_name} on {self.device.upper()}")
        self.engine = create_engine(os.getenv("DB_URI"))

    def analyze_sentiment(self, texts: List[str], batch_size: int = 32) -> pd.DataFrame:
        """Perform sentiment analysis on financial texts
        Args:
            texts: List of text documents
            batch_size: Number of texts to process at once
        Returns:
            DataFrame with sentiment analysis results
        """
        try:
            results = []

            # Process in batches
            for i in tqdm(range(0, len(texts), batch_size)):
                batch = texts[i:i + batch_size]
                batch_results = self.pipeline(batch)
                results.extend(batch_results)

            # Process results
            sentiment_data = []
            for result in results:
                scores = {item["label"]: item["score"] for item in result}
                sentiment_data.append({
                    "sentiment": max(scores, key=scores.get),
                    "sentiment_score": max(scores.values()),
                    **scores  # Include individual class probabilities
                })

            return pd.DataFrame(sentiment_data)
        except Exception as e:
            logging.error(f"Sentiment analysis failed: {str(e)}")
            return pd.DataFrame()

    def evaluate(self, test_data: pd.DataFrame) -> Dict:
        """Evaluate model performance on labeled test data
        Args:
            test_data: DataFrame with 'text' and 'label' columns
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Predict sentiments
            predictions = self.analyze_sentiment(test_data["text"].tolist())
            predicted_labels = predictions["sentiment"]

            # Generate classification report
            report = classification_report(
                test_data["label"],
                predicted_labels,
                output_dict=True
            )

            return {
                "accuracy": report["accuracy"],
                "precision": report["weighted avg"]["precision"],
                "recall": report["weighted avg"]["recall"],
                "f1": report["weighted avg"]["f1-score"]
            }

        except Exception as e:
            logging.error(f"Evaluation failed: {str(e)}")
            return {}

    def save_model(self, output_dir: str = "models/nlp"):
        """Save the model to a directory"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            logging.info(f"Model saved to {output_dir}")
        except Exception as e:
            logging.error(f"Model saving failed: {str(e)}")
            raise

    @classmethod
    def load_model(cls, model_dir: str = "models/nlp"):
        """Load a saved model from a directory"""
        try:
            return cls(model_name=model_dir)
        except Exception as e:
            logging.error(f"Model loading failed: {str(e)}")
            raise

    def load_news_data(self, source: str = "db", symbol: Optional[str] = None) -> pd.DataFrame:
        """Load news data from a database or Parquet files
        Args:
            source: 'db' for database or 'file' for Parquet
            symbol: Optional stock symbol to filter
        Returns:
            DataFrame with news articles
        """
        try:
            if source == "db":
                engine = self.engine
                query = "SELECT * FROM news_articles" 
                if symbol:
                    query += " WHERE symbols LIKE %s"
                    with engine.connect() as conn:
                        return pd.read_sql(query, conn, params=('%' + symbol + '%',))
                with engine.connect() as conn:
                    return pd.read_sql(query, conn)
            elif source == "file":
                files = glob.glob("data/news/*.parquet")
                if symbol:
                    files = [f for f in files if symbol in f]
                return pd.concat([pd.read_parquet(f) for f in files])
            else:
                raise ValueError("Invalid source. Use 'db' or 'file'")

        except Exception as e:
            logging.error(f"Failed to load news data: {str(e)}")
            return pd.DataFrame()

    def classify_news(self, symbol: Optional[str] = None, save_results: bool = True) -> pd.DataFrame:
        """Classify news articles and save results
        Args:
            symbol: Optional stock symbol to filter
            save_results: Save classified data to database
        Returns:
            DataFrame with classified news articles
        """
        try:
            # Load news data
            news_df = self.load_news_data(source="db", symbol=symbol)
            #news_df = self.load_news_data(source="file", symbol=symbol)

            if news_df.empty:
                logging.warning("No news data found")
                return pd.DataFrame()

            # Analyze sentiments
            logging.info(f"Classifying {len(news_df)} news articles...")
            sentiment_df = self.analyze_sentiment(news_df["content"].tolist())

            # Merge results
            classified_df = pd.concat([news_df.reset_index(drop=True), sentiment_df], axis=1)

            # Save to db
            if save_results:
                self.save_classified_data(classified_df)

            return classified_df

        except Exception as e:
            logging.error(f"News classification failed: {str(e)}")
            return pd.DataFrame()
        
    def save_classified_data(self, df: pd.DataFrame):
        """Save classified news data to a database"""
        try:
            columns_to_drop = ['sentiment']
            df = df.drop(columns=columns_to_drop, errors='ignore')

            df.to_sql(
                name="classified_news",
                con=self.engine,
                if_exists="append",
                index=False,
                method="multi"
            )
            logging.info(f"Saved {len(df)} classified articles to database")
        except Exception as e:
            logging.error(f"Failed to save classified data: {str(e)}")


if __name__ == "__main__":
    # Initialize analyzer
    analyzer = FinancialTextAnalyzer()

    # Classify news
    classified_news = analyzer.classify_news(symbol="AAPL")

    if not classified_news.empty:
        print(classified_news[["published_at", "source", "sentiment", "title"]].head())
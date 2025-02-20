import psycopg2
from psycopg2 import sql
import pandas as pd
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def db_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST")
        )

def load_csv_to_table(conn: psycopg2.extensions.connection, csv_path: str, table_name: str):
    df = pd.read_csv(csv_path)

    with conn.cursor() as cur:
        try:
            records = [tuple(
                x.to_pydatetime() if isinstance(x, pd.Timestamp)
                else float(x) if isinstance(x, pd.Float64Dtype)
                else x
                for x in row)
            for row in df.to_records(index=False)]
            
            columns = sql.SQL(', ').join(map(sql.Identifier, df.columns))

            query = sql.SQL("""
                            INSERT INTO {} ({})
                            VALUES ({})
                            ON CONFLICT (date, symbol) DO NOTHING
                            """).format(
                                sql.Identifier(table_name),
                                columns,
                                sql.SQL(', ').join(sql.Placeholder() * len(df.columns))
                            )
            # Execute batch insert
            cur.executemany(query, records)
            conn.commit()
            logger.info(f"Sucessfully loaded data from {csv_path} to {table_name}")

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to load data from {csv_path} to {table_name}")
            raise



if __name__ == "__main__":
    stock_files = [
        "data/raw/msft_daily.csv",
        "data/raw/aapl_daily.csv",
        "data/raw/amzn_daily.csv"
    ]


    try:
        with db_connection() as conn:
            logger.info("Connected to database")

            for file in stock_files:
                try:
                    load_csv_to_table(conn, file, "stock_prices")
                except Exception as e:
                    logger.error(f"Failed to process{file}: {e}")
                    continue
        conn.close()
        logger.info("Database connection closed")
 
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")
        raise
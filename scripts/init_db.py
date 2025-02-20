# Import required libraries
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create db for FRED
def init_database():
    try:
        load_dotenv()
        engine = create_engine(os.getenv("DB_URI"))
        
        with engine.connect() as conn:
            # Create schema
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS fred;"))
            
            # Create table with proper types
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS fred.economic_indicators (
                    id SERIAL PRIMARY KEY,
                    series_id VARCHAR(50),
                    date TIMESTAMP,
                    value DOUBLE PRECISION,
                    realtime_start DATE,
                    realtime_end DATE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))
            conn.commit()
            
        logger.info("Database initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        return False

if __name__ == "__main__":
    init_database()
# Data Pipeline Flow

1. **Ingestion**
    - Daily cron job fetches market data
    - News API triggers on market open

2. **Processing**
    ```python
    # Sample PySpark job
    df = spark.read.csv("raw_data")
    df.withColumn("ma_7", avg("close").over(window))
    ```
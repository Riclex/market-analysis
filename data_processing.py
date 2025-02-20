import os
import sys
from pathlib import Path
import tempfile
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg
from pyspark.sql.window import Window
import logging

# Create Spark session
spark = (SparkSession.builder
        .appName("StockFeatureEngieering")
        .getOrCreate())

df = spark.read.csv("data/raw/aapl_daily.csv", header=True, inferSchema=True)

# Calculate 7 day moving avg
window = Window.orderBy("date").rowsBetween(-6, 0)
df = df.withColumn("ma_7", avg(col("close")).over(window))

# Save data
df.write.mode("overwrite").parquet("data/processed/aapl_features.parquet")
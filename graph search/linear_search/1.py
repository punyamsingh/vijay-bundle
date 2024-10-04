from pyspark.sql import SparkSession

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("TestSpark") \
    .getOrCreate()

# Get SparkContext
sc = spark.sparkContext

# Print basic info about SparkContext
print(f"Spark version: {sc.version}")
print(f"Python version: {sc.pythonVer}")
print(f"Master: {sc.master}")
print(f"Spark Web UI: {sc.uiWebUrl}")

# Stop the Spark session
spark.stop()

from pyspark.sql import SparkSession
import numpy as np

# Create Spark session with cluster configuration
spark = SparkSession.builder \
    .appName("MatrixGeneration") \
    .master("spark://10.58.0.158:7077") \
    .getOrCreate()

# Matrix dimensions
total_rows = 3000
columns_A = 3000
columns_B = 1

# Output paths for matrices
output_A_path = "hdfs://hadoop-namenode:9820/project/matrix_A3.csv"
output_B_path = "hdfs://hadoop-namenode:9820/project/matrix_B3.csv"

# Generate random matrices using NumPy
matrix_A = np.random.rand(total_rows, columns_A)  # Matrix A (5000 x 5000)
matrix_B = np.random.rand(total_rows, columns_B)  # Matrix B (5000 x 1)

# Convert matrices to list of rows for Spark DataFrame
data_A = [(int(i), [int(x * 100) for x in row]) for i, row in enumerate(matrix_A)]
data_B = [(int(i), [int(x * 100) for x in row]) for i, row in enumerate(matrix_B)]

# Create DataFrames with index and vector (row)
df_A = spark.createDataFrame(data_A, ["index", "vector"])
df_B = spark.createDataFrame(data_B, ["index", "vector"])

# Convert list of floats to string for CSV storage
df_A = df_A.withColumn("vector", df_A["vector"].cast("string"))
df_B = df_B.withColumn("vector", df_B["vector"].cast("string"))

# Write as CSV
try:
    df_A.write.mode("overwrite").option("header", "true").csv(output_A_path)
    df_B.write.mode("overwrite").option("header", "true").csv(output_B_path)

    print(f"Matrix A saved to {output_A_path}")
    print(f"Matrix B saved to {output_B_path}")
except Exception as e:
    print(f"Error while saving DataFrames: {e}")

# Stop the Spark session
spark.stop()
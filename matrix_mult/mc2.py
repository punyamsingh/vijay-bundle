from pyspark.sql import SparkSession
import numpy as np

# Create Spark session
# spark = SparkSession.builder.appName("MatrixGeneration").master("spark://10.58.0.158:7077").getOrCreate()
spark = SparkSession.builder.appName("MatrixGeneration").getOrCreate()

# Matrix dimensions
matrix_A_size = (5000, 5000)
matrix_B_size = (5000, 1)

# Generate random matrices using NumPy
matrix_A = np.random.rand(*matrix_A_size)
matrix_B = np.random.rand(*matrix_B_size)

# Convert matrices to list of rows for Spark DataFrame
data_A = [(int(i), [int(x*100) for x in row]) for i, row in enumerate(matrix_A)]
data_B = [(int(i), [int(x*100) for x in row]) for i, row in enumerate(matrix_B)]

# Create DataFrames with index and vector (row)
df_A = spark.createDataFrame(data_A, ["index", "vector"])
df_B = spark.createDataFrame(data_B, ["index", "vector"])

# # Save the DataFrames to HDFS in Parquet format
# output_A_path = "hdfs://hadoop-namenode:9820/project/matrix_A.parquet"
# output_B_path = "hdfs://hadoop-namenode:9820/project/matrix_B.parquet"
# df_A.write.mode("overwrite").parquet(output_A_path)
# df_B.write.mode("overwrite").parquet(output_B_path)

# print(f"Matrix A (1000x1000) saved to HDFS at {output_A_path}")
# print(f"Matrix B (1000x4) saved to HDFS at {output_B_path}")

# Save DataFrames as CSV (for better visualization)
output_A_path = "hdfs://hadoop-namenode:9820/project/matrix_A1.csv"
output_B_path = "hdfs://hadoop-namenode:9820/project/matrix_B1.csv"

# Convert list of floats to string for CSV storage
df_A = df_A.withColumn("vector", df_A["vector"].cast("string"))
df_B = df_B.withColumn("vector", df_B["vector"].cast("string"))

# Write as CSV
df_A.write.mode("overwrite").option("header", "true").csv(output_A_path)
df_B.write.mode("overwrite").option("header", "true").csv(output_B_path)

print(f"Matrix A (1000x1000) saved to HDFS as CSV at {output_A_path}")
print(f"Matrix B (1000x4) saved to HDFS as CSV at {output_B_path}")

# Collect and print the entire matrix (be cautious if the matrix is large)
matrix = df_B.collect()

for row in matrix:
    print(row)

# Stop the Spark session
spark.stop()

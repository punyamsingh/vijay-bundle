from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType
import numpy as np

# Create Spark session
# spark = SparkSession.builder \
#     .appName("MatrixToHDFS") \
#     .master("spark://10.58.0.158:7077") \
#     .getOrCreate()
spark = SparkSession.builder \
    .appName("MatrixMultiplication") \
    .getOrCreate()

# Path where the matrix is stored in HDFS
matrix_path = "hdfs://hadoop-namenode:9820/project/large_matrix.parquet"

# Load the matrix from HDFS
df = spark.read.parquet(matrix_path)

# Collect matrix rows into an array (as list) and broadcast it
matrix_rows = np.array(df.select("vector").rdd.map(lambda x: x[0]).collect())
broadcast_matrix = spark.sparkContext.broadcast(matrix_rows)

# Define UDF to multiply a row by the matrix (for squaring)
def multiply_row(row):
    matrix = broadcast_matrix.value
    return np.dot(np.array(row), matrix).tolist()

multiply_udf = udf(multiply_row, ArrayType(FloatType()))

# Apply row-wise multiplication using UDF
squared_df = df.withColumn("squared_vector", multiply_udf("vector"))

# Save the squared matrix back to HDFS
squared_output_path = "hdfs://hadoop-namenode:9820/project/squared_matrix.parquet"
squared_df.write.mode("overwrite").parquet(squared_output_path)

print(f"Squared matrix saved to HDFS at {squared_output_path}")

# Collect and print the entire matrix (be cautious if the matrix is large)
matrix = df.collect()

for row in matrix:
    print(row)


# Stop the Spark session
spark.stop()
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType
import numpy as np

# Create Spark session
spark = SparkSession.builder.appName("MatrixMultiplication").master("spark://10.58.0.158:7077").getOrCreate()

# Paths where the matrices are stored in HDFS
matrix_A_path = "hdfs://hadoop-namenode:9820/project/matrix_A.csv"
matrix_B_path = "hdfs://hadoop-namenode:9820/project/matrix_B.csv"

# Load the matrices from CSV in HDFS
df_A = spark.read.option("header", "true").csv(matrix_A_path)
df_B = spark.read.option("header", "true").csv(matrix_B_path)

# Convert vectors (stored as strings) back to arrays
def string_to_array(vector_str):
    return list(map(float, vector_str.strip('[]').split(',')))

convert_udf = udf(string_to_array, ArrayType(FloatType()))
df_A = df_A.withColumn("vector", convert_udf("vector"))
df_B = df_B.withColumn("vector", convert_udf("vector"))

# Collect matrix rows into arrays and broadcast them
matrix_A_rows = np.array(df_A.select("vector").rdd.map(lambda x: x[0]).collect())
matrix_B_rows = np.array(df_B.select("vector").rdd.map(lambda x: x[0]).collect())
broadcast_B = spark.sparkContext.broadcast(matrix_B_rows)

# Define UDF to multiply a row of matrix A with matrix B
def multiply_row(row):
    matrix_B = broadcast_B.value
    return np.dot(np.array(row), matrix_B).tolist()

multiply_udf = udf(multiply_row, ArrayType(FloatType()))

# Apply row-wise multiplication using UDF
multiplied_df = df_A.withColumn("multiplied_vector", multiply_udf("vector"))

# Convert the vectors to string for CSV storage
multiplied_df = multiplied_df.withColumn("vector", multiplied_df["vector"].cast("string"))
multiplied_df = multiplied_df.withColumn("multiplied_vector", multiplied_df["multiplied_vector"].cast("string"))

# Save the multiplied result back to HDFS as CSV
multiplied_output_path = "hdfs://hadoop-namenode:9820/project/multiplied_matrix.csv"
multiplied_df.write.mode("overwrite").option("header", "true").csv(multiplied_output_path)

# Print the result (first 10 rows)
multiplied_df["multiplied_vector"].show(10, truncate=False)

print(f"Multiplied matrix saved to HDFS at {multiplied_output_path}")

# Stop the Spark session
spark.stop()

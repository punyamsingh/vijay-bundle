from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType
import numpy as np

# Create Spark session with HDFS configuration
spark = SparkSession.builder \
    .appName("MatrixMultiplication") \
    .master("spark://10.58.0.158:7077") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://hadoop-namenode:9820") \
    .getOrCreate()

# Paths where the matrices are stored in HDFS
matrix_A_path = "hdfs://hadoop-namenode:9820/project/matrix_A.csv"
matrix_B_path = "hdfs://hadoop-namenode:9820/project/matrix_B.csv"

# Load the matrices from CSV in HDFS
try:
    df_A = spark.read.option("header", "true").csv(matrix_A_path)
    df_B = spark.read.option("header", "true").csv(matrix_B_path)
except Exception as e:
    print(f"Error loading matrices: {e}")
    spark.stop()
    exit(1)

# Convert vectors (stored as strings) back to arrays
def string_to_array(vector_str):
    return list(map(float, vector_str.strip('[]').split(',')))

convert_udf = udf(string_to_array, ArrayType(FloatType()))
df_A = df_A.withColumn("vector", convert_udf("vector"))
df_B = df_B.withColumn("vector", convert_udf("vector"))

# Collect matrix B rows for multiplication
try:
    matrix_B = np.array(df_B.select("vector").rdd.map(lambda x: x[0]).collect())
except Exception as e:
    print(f"Error collecting matrix B: {e}")
    spark.stop()
    exit(1)

# Perform matrix multiplication
result_rows = []
for row in df_A.select("vector").rdd.collect():
    a_vector = np.array(row[0])
    result_row = np.dot(a_vector, matrix_B)
    result_rows.append(result_row.tolist())

# Create DataFrame from the result
result_df = spark.createDataFrame(result_rows, schema=["multiplied_vector"])

# Convert the multiplied vector to a string for CSV storage
result_df = result_df.withColumn("multiplied_vector", result_df["multiplied_vector"].cast("string"))

# Check the DataFrame before writing to HDFS
result_df.printSchema()
result_df.show(truncate=False)

# Save the multiplied result back to HDFS as CSV
multiplied_output_path = "hdfs://hadoop-namenode:9820/project/multiplied_output/multiplied_matrix.csv"
try:
    result_df.write.mode("overwrite").option("header", "true").csv(multiplied_output_path)
    print(f"Multiplied matrix saved to HDFS at {multiplied_output_path}")
except Exception as e:
    print(f"An error occurred while saving the DataFrame: {e}")

# Stop the Spark session
spark.stop()

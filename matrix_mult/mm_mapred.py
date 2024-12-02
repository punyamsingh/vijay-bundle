from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType
import numpy as np

# Create Spark session with HDFS configuration
spark = SparkSession.builder \
    .appName("MatrixMultiplication_12core") \
    .master("spark://10.58.0.158:7077") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://hadoop-namenode:9820") \
    .getOrCreate()

# Paths where the matrices are stored in HDFS
matrix_A_path = "hdfs://hadoop-namenode:9820/project/matrix_A5.csv"
matrix_B_path = "hdfs://hadoop-namenode:9820/project/matrix_B5.csv"

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

# Perform matrix multiplication using parallelized dot product
def multiply_row_with_matrix(a_vector, matrix_B):
    return np.dot(a_vector, matrix_B).tolist()

multiply_udf = udf(lambda a: multiply_row_with_matrix(a, matrix_B), ArrayType(FloatType()))

# Apply UDF to perform the matrix multiplication
df_result = df_A.withColumn("multiplied_vector", multiply_udf("vector"))

# Convert the multiplied vector to a string for CSV storage
df_result = df_result.withColumn("multiplied_vector", df_result["multiplied_vector"].cast("string"))

# Check the DataFrame before writing to HDFS
df_result.printSchema()
df_result.show(truncate=False)

# Save the multiplied result back to HDFS as CSV
multiplied_output_path = "hdfs://hadoop-namenode:9820/project/multiplied_matrix_output_mapred_12core.csv"
try:
    df_result.write.mode("overwrite").option("header", "true").csv(multiplied_output_path)
    print(f"Multiplied matrix saved to HDFS at {multiplied_output_path}")
except Exception as e:
    print(f"An error occurred while saving the DataFrame: {e}")

# Function to get dimensions of a DataFrame
def get_dimensions(file_path):
    try:
        df = spark.read.option("header", "true").csv(file_path)
        num_rows = df.count()
        num_cols = len(df.columns)
        return num_rows, num_cols
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0, 0

# Get and print the dimensions of the saved matrices
rows_A, cols_A = get_dimensions(multiplied_output_path)

print(f"Final dimensions of Matrix A: {rows_A} x {cols_A}")

# Stop the Spark session
spark.stop()

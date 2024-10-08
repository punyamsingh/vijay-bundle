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
matrix_A_path = "hdfs://hadoop-namenode:9820/project/matrix_As.csv"
matrix_B_path = "hdfs://hadoop-namenode:9820/project/matrix_Bs.csv"
multiplied_output_path = "hdfs://hadoop-namenode:9820/project/multiplied_outputs/multiplied_matrixs.csv"

# Load the matrices from CSV in HDFS
try:
    df_A = spark.read.option("header", "true").csv(matrix_A_path)
    df_B = spark.read.option("header", "true").csv(matrix_B_path)
except Exception as e:
    print(f"Error loading matrices: {e}")
    spark.stop()
    exit(1)

# Check the schema of the loaded DataFrames
print("Schema of Matrix A:")
df_A.printSchema()
print("Schema of Matrix B:")
df_B.printSchema()

# Rename the columns if necessary
if 'vector' not in df_A.columns:
    df_A = df_A.withColumnRenamed(df_A.columns[1], "vector")  # Assuming the second column contains the vector

if 'vector' not in df_B.columns:
    df_B = df_B.withColumnRenamed(df_B.columns[1], "vector")  # Assuming the second column contains the vector

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

# Determine the dimensions
num_rows_A = df_A.count()
num_cols_A = len(df_A.select("vector").first()[0])  # Assuming all rows have the same length
num_rows_B = len(matrix_B)
num_cols_B = 1  # Since it's a column vector

print(f"Matrix A dimensions: {num_rows_A} x {num_cols_A}")
print(f"Matrix B dimensions: {num_rows_B} x {num_cols_B}")

# Determine the minimum size for multiplication
min_rows = min(num_rows_A, num_rows_B)
min_cols = min(num_cols_A, num_cols_B)

# Calculate the chunk size (size of matrix A / 50)
chunk_size = max(1, min_rows // 50)  # Ensure at least chunk size of 1

# Initialize an empty list to store results
result_rows = []

for start in range(0, min_rows, chunk_size):
    end = min(start + chunk_size, min_rows)
    chunk = df_A.select("vector").rdd.zipWithIndex().filter(lambda x: start <= x[1] < end).map(lambda x: x[0]).collect()
    
    for row in chunk:
        a_vector = np.array(row[0][:min_cols])  # Use only the elements up to min_cols
        b_vector = matrix_B[:min_rows]  # Use only the elements up to min_rows
        result_row = np.dot(a_vector, b_vector)
        result_rows.append(result_row.tolist())

    print(f"Processed rows {start} to {end}")

# Create DataFrame from the result
result_df = spark.createDataFrame(result_rows, schema=["multiplied_vector"])

# Convert the multiplied vector to a string for CSV storage
result_df = result_df.withColumn("multiplied_vector", result_df["multiplied_vector"].cast("string"))

# Check the DataFrame before writing to HDFS
result_df.printSchema()
result_df.show(truncate=False)

# Save the multiplied result back to HDFS as CSV
try:
    result_df.write.mode("append").option("header", True).csv(multiplied_output_path)
    print(f"Multiplied matrix saved to HDFS at {multiplied_output_path}")
except Exception as e:
    print(f"An error occurred while saving the DataFrame: {e}")

# Stop the Spark session
spark.stop()

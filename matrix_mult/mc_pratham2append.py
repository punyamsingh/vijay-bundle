from pyspark.sql import SparkSession
import numpy as np

# Create Spark session with cluster configuration
spark = SparkSession.builder \
    .appName("MatrixGeneration") \
    .master("spark://10.58.0.158:7077") \
    .getOrCreate()

# Matrix dimensions
total_rows = 1000
columns_A = 1000
columns_B = 1

# Output paths for matrices
output_A_path = "hdfs://hadoop-namenode:9820/project/matrix_Ay.csv"
output_B_path = "hdfs://hadoop-namenode:9820/project/matrix_By.csv"

# Function to get dimensions of an existing DataFrame
def get_dimensions(file_path):
    try:
        df = spark.read.option("header", "true").csv(file_path)
        num_rows = df.count()
        num_cols = len(df.columns)
        return num_rows, num_cols
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0, 0

# Print existing dimensions before generating new data
existing_rows_A, existing_cols_A = get_dimensions(output_A_path)
existing_rows_B, existing_cols_B = get_dimensions(output_B_path)

print(f"Existing dimensions of Matrix A: {existing_rows_A} x {existing_cols_A}")
print(f"Existing dimensions of Matrix B: {existing_rows_B} x {existing_cols_B}")

# Generate random matrices using NumPy
matrix_A = np.random.rand(total_rows, columns_A)  # Matrix A (1000 x 1000)
matrix_B = np.random.rand(total_rows, columns_B)  # Matrix B (1000 x 1)

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
    # Append mode requires no header for subsequent writes
    df_A.write.mode("append").option("header", "false").csv(output_A_path)
    df_B.write.mode("append").option("header", "false").csv(output_B_path)

    print(f"Matrix A appended to {output_A_path}")
    print(f"Matrix B appended to {output_B_path}")
except Exception as e:
    print(f"Error while saving DataFrames: {e}")

# Print updated dimensions after appending
updated_rows_A, updated_cols_A = get_dimensions(output_A_path)
updated_rows_B, updated_cols_B = get_dimensions(output_B_path)

print(f"Updated dimensions of Matrix A: {updated_rows_A} x {updated_cols_A}")
print(f"Updated dimensions of Matrix B: {updated_rows_B} x {updated_cols_B}")

# Stop the Spark session
spark.stop()

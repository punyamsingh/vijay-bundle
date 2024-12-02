from pyspark.sql import SparkSession
import numpy as np

# Create Spark session with cluster configuration
spark = SparkSession.builder \
    .appName("AppendMatrixRowsWithRandomValues") \
    .master("spark://10.58.0.158:7077") \
    .getOrCreate()

# Paths to the existing matrices
matrix_A_path = "hdfs://hadoop-namenode:9820/project/matrix_A5.csv"
matrix_B_path = "hdfs://hadoop-namenode:9820/project/matrix_B5.csv"

# Matrix dimensions
columns_A = 4000  # Same as the original number of columns for matrix A
columns_B = 1     # Same as the original number of columns for matrix B
extra_rows = 1000  # Number of new rows to append

# Load the existing matrices from HDFS
df_A = spark.read.option("header", "true").csv(matrix_A_path)
df_B = spark.read.option("header", "true").csv(matrix_B_path)

# Get the current number of rows in the matrices
existing_rows_A = df_A.count()
existing_rows_B = df_B.count()

# Generate new rows with random values
new_rows_A = np.random.rand(extra_rows, columns_A)  # Random values for new rows for matrix A
new_rows_B = np.random.rand(extra_rows, columns_B)  # Random values for new rows for matrix B

# Convert new rows to list of rows for Spark DataFrame
new_data_A = [(existing_rows_A + i, [int(x * 100) for x in row]) for i, row in enumerate(new_rows_A)]
new_data_B = [(existing_rows_B + i, [int(x * 100) for x in row]) for i, row in enumerate(new_rows_B)]

# Create DataFrames for the new rows
new_df_A = spark.createDataFrame(new_data_A, ["index", "vector"])
new_df_B = spark.createDataFrame(new_data_B, ["index", "vector"])

# Convert list of floats to string for CSV storage (same format as the existing matrices)
new_df_A = new_df_A.withColumn("vector", new_df_A["vector"].cast("string"))
new_df_B = new_df_B.withColumn("vector", new_df_B["vector"].cast("string"))

# Combine the original DataFrames with the new rows
combined_df_A = df_A.union(new_df_A)
combined_df_B = df_B.union(new_df_B)

# Append the new rows to the matrices without overwriting any existing data
try:
    combined_df_A.write.mode("overwrite").option("header", "true").csv(matrix_A_path)
    combined_df_B.write.mode("overwrite").option("header", "true").csv(matrix_B_path)

    print(f"Matrix A (with new rows) saved to {matrix_A_path}")
    print(f"Matrix B (with new rows) saved to {matrix_B_path}")
except Exception as e:
    print(f"Error while saving updated DataFrames: {e}")

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

# Get and print the updated dimensions of the matrices
updated_rows_A, updated_cols_A = get_dimensions(matrix_A_path)
updated_rows_B, updated_cols_B = get_dimensions(matrix_B_path)

print(f"Updated dimensions of Matrix A: {updated_rows_A} x {updated_cols_A}")
print(f"Updated dimensions of Matrix B: {updated_rows_B} x {updated_cols_B}")

# Stop the Spark session
spark.stop()

from pyspark.sql import SparkSession
import numpy as np

# Create Spark session with cluster configuration
spark = SparkSession.builder \
    .appName("MatrixGeneration") \
    .master("spark://10.58.0.158:7077") \
    .getOrCreate()

# Matrix dimensions and chunk size
total_rows = 5000
chunk_size = 100  # Number of rows to generate in each iteration

# Output paths for matrices
output_A_path = "hdfs://hadoop-namenode:9820/project/matrix_As.csv"
output_B_path = "hdfs://hadoop-namenode:9820/project/matrix_Bs.csv"

# Check if the output files already exist to handle header
file_exists_A = False
file_exists_B = False

# Calculate the total number of chunks
total_chunks = total_rows // chunk_size

for chunk_index in range(total_chunks):
    # Generate random matrices in chunks using NumPy
    matrix_A_chunk = np.random.rand(chunk_size, 5000)  # Adjust the column size if necessary
    matrix_B_chunk = np.random.rand(chunk_size, 1)      # Adjust the column size if necessary

    # Convert chunks to list of rows for Spark DataFrame
    data_A = [(int(chunk_index * chunk_size + i), [int(x * 100) for x in row]) for i, row in enumerate(matrix_A_chunk)]
    data_B = [(int(chunk_index * chunk_size + i), [int(x * 100) for x in row]) for i, row in enumerate(matrix_B_chunk)]

    # Create DataFrames with index and vector (row)
    df_A = spark.createDataFrame(data_A, ["index", "vector"])
    df_B = spark.createDataFrame(data_B, ["index", "vector"])

    # Convert list of floats to string for CSV storage
    df_A = df_A.withColumn("vector", df_A["vector"].cast("string"))
    df_B = df_B.withColumn("vector", df_B["vector"].cast("string"))

    # Write as CSV
    try:
        df_A.write.mode("append").option("header", not file_exists_A).csv(output_A_path)
        df_B.write.mode("append").option("header", not file_exists_B).csv(output_B_path)
        
        # Update file existence flags
        file_exists_A = True
        file_exists_B = True

        print(f"Chunk {chunk_index + 1}/{total_chunks}: Matrix A appended to {output_A_path}")
        print(f"Chunk {chunk_index + 1}/{total_chunks}: Matrix B appended to {output_B_path}")
    except Exception as e:
        print(f"Error while saving DataFrames for chunk {chunk_index}: {e}")

# Stop the Spark session
spark.stop()

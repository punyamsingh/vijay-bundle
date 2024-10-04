from pyspark.sql import SparkSession
import numpy as np

# Create Spark session
spark = SparkSession.builder \
    .appName("MatrixToHDFS") \
    .getOrCreate()

# spark = SparkSession.builder \
#     .appName("lsearch test core 12 :)") \
#     .master("spark://10.58.0.158:7077") \
#     .config("spark.jars.packages", "graphframes:graphframes:0.8.4-spark3.5-s_2.12") \
#     .config("spark.executor.memory", "4g") \
#     .config("spark.executor.cores", "6") \
#     .config("spark.default.parallelism", "200") \
#     .config("spark.sql.shuffle.partitions", "200") \
#     .config("spark.memory.fraction", "0.8") \
#     .getOrCreate()

# Matrix dimensions
matrix_size = 1000  # Change this for a larger/smaller matrix

# Generate a random large matrix using NumPy
large_matrix = np.random.rand(matrix_size, matrix_size)

# Convert NumPy array to list of lists of Python floats
data = [(int(i), [float(x) for x in row]) for i, row in enumerate(large_matrix)]

# Create DataFrame with index and vector (row)
df = spark.createDataFrame(data, ["index", "vector"])

# Save the DataFrame to HDFS in Parquet format (or CSV, depending on your need)
# output_path = "hdfs://<namenode>/user/<your-username>/large_matrix.parquet"
path = "hdfs://hadoop-namenode:9820/project/large_matrix.parquet"
df.write.mode("overwrite").parquet(path)

print(f"Matrix saved to HDFS at {path}")

# Stop the Spark session
spark.stop()

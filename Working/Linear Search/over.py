import time
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql import Row
from graphframes import GraphFrame
import random

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Synthetic Large Dataset Parallelism") \
    .master("spark://10.58.0.158:7077") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.4-spark3.5-s_2.12") \
    .getOrCreate()

# Set number of partitions to control parallelism
num_partitions = 1000  # Adjust this for dataset size

# Start time for overall execution
start_time = time.time()

# Schema definition for edges DataFrame
schema = StructType([
    StructField("src", StringType(), True),
    StructField("dst", StringType(), True)
])

# Generate a large synthetic graph dataset (e.g., 1 billion edges)
def generate_synthetic_data(num_edges):
    for _ in range(num_edges):
        src = str(random.randint(1, 1000000))  # Random user IDs
        dst = str(random.randint(1, 1000000))  # Random user IDs
        yield (src, dst)

# Create RDD from synthetic dataset
num_edges = 1000000  # Adjust for dataset size
edges_rdd = spark.sparkContext.parallelize(generate_synthetic_data(num_edges), num_partitions)

# Convert RDD to DataFrame using schema
edges_df = spark.createDataFrame(edges_rdd.map(lambda x: Row(src=x[0], dst=x[1])), schema=schema)

# Create vertices DataFrame by flattening the edges into distinct vertices
vertices_rdd = edges_rdd.flatMap(lambda row: [Row(id=row[0]), Row(id=row[1])]).distinct()
vertices_df = spark.createDataFrame(vertices_rdd, schema=StructType([StructField("id", StringType(), True)]))

# Create GraphFrame from vertices and edges
graph = GraphFrame(vertices_df, edges_df)

# Basic graph operations - counting vertices and edges
vertex_count = graph.vertices.count()
edge_count = graph.edges.count()

# Print results (parallelized)
print(f"Number of vertices: {vertex_count}")
print(f"Number of edges: {edge_count}")

# Perform a sample operation - filter for specific edges
target_user_id = "500000"  # Example target user for filtering
user_edges = graph.edges.filter(f"src = '{target_user_id}' or dst = '{target_user_id}'")

# Count filtered edges
filtered_edge_count = user_edges.count()

# Save filtered results to HDFS
output_path = "hdfs://hadoop-namenode:9870/project/output_filtered_edges.txt"  # HDFS output path
user_edges.write.mode("overwrite").csv(output_path)

# Print filtered result count
print(f"Filtered edge count for user {target_user_id}: {filtered_edge_count}")

# Total execution time
end_time = time.time()
print(f"Total execution time: {end_time - start_time} seconds")

# Stop Spark session
spark.stop()

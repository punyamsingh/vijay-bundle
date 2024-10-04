import time
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, StringType
from graphframes import GraphFrame

# Initialize Spark session
start_time = time.time()  # Start time for overall execution
spark = SparkSession.builder \
    .appName("Pratham using sometime core 24 :)") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.4-spark3.5-s_2.12") \
    .getOrCreate()

# Start time for data loading
data_load_start_time = time.time()

# Load the Facebook dataset (edges) from HDFS
path = "hdfs://hadoop-namenode:9820/project/as-skitter.txt"  # Update with the actual HDFS path

def parse_line(line):
    parts = line.split("\t")
    if len(parts) == 2:
        return (parts[0], parts[1])
    else:
        return None

# Create RDD and filter out invalid lines
edges_rdd = spark.sparkContext.textFile(path) \
    .map(parse_line) \
    .filter(lambda x: x is not None)

# Data loading time
data_load_end_time = time.time()
print(f"Data loading time: {data_load_end_time - data_load_start_time} seconds")

# Define schema for edges DataFrame
schema = StructType([
    StructField("src", StringType(), True),
    StructField("dst", StringType(), True)
])

# Time for DataFrame creation and graph generation
graph_creation_start_time = time.time()

# Convert RDD to DataFrame using schema
edges_df = spark.createDataFrame(edges_rdd, schema=schema)

# Create vertices DataFrame
vertices_rdd = edges_rdd.flatMap(lambda row: [Row(id=row[0]), Row(id=row[1])]).distinct()
vertices_df = spark.createDataFrame(vertices_rdd, schema=StructType([StructField("id", StringType(), True)]))

# Create GraphFrame
graph = GraphFrame(vertices_df, edges_df)

# Time taken for creating DataFrame and GraphFrame
graph_creation_end_time = time.time()
print(f"Graph creation time: {graph_creation_end_time - graph_creation_start_time} seconds")

# Basic graph operations - fully parallelized
vertex_count = graph.vertices.count()
edge_count = graph.edges.count()
print(f"Number of vertices: {vertex_count}")
print(f"Number of edges: {edge_count}")

# Perform parallel search for a specific user ID
user_search_start_time = time.time()  # Time for searching user
target_user_id = "1691593"  # Update with the user ID you want to search for
user_found = vertices_df.filter(vertices_df.id == target_user_id)

# Check if the user is found - fully parallelized
result = ""
if user_found.count() > 0:
    result = f"User ID {target_user_id} found in the dataset."
else:
    result = f"User ID {target_user_id} not found in the dataset."

# User search time
user_search_end_time = time.time()
print(f"User search time: {user_search_end_time - user_search_start_time} seconds")

# Time for writing result to HDFS
output_write_start_time = time.time()
output_path = "hdfs://hadoop-namenode:9820/project/output12tt89.txt"  # Specify the HDFS output path
rdd = spark.sparkContext.parallelize([result])
rdd.saveAsTextFile(output_path)

# Write operation time
output_write_end_time = time.time()
print(f"Output write time: {output_write_end_time - output_write_start_time} seconds")

# Time for finding neighbors
neighbor_search_start_time = time.time()

# Find the neighbors of the specified user - parallelized
neighbors = graph.edges.filter(f"src = '{target_user_id}' or dst = '{target_user_id}'")
print(f"Neighbors of user {target_user_id}:")
neighbors.show()

# Neighbors search time
neighbor_search_end_time = time.time()
print(f"Neighbors search time: {neighbor_search_end_time - neighbor_search_start_time} seconds")

# Save neighbors list to HDFS - parallelized
neighbors_output_path = "hdfs://hadoop-namenode:9820/project/neighbors_outputt2.txt"
neighbors.write.mode("overwrite").csv(neighbors_output_path)

# Time for sampling for visualization
sampling_start_time = time.time()

# Sampling 0.001% of the vertices and their edges for visualization
sampled_vertices = vertices_df.sample(withReplacement=False, fraction=0.00001)
sampled_edges = edges_df.join(sampled_vertices, edges_df.src == sampled_vertices.id, "inner") \
    .select(edges_df.src, edges_df.dst)

# Save sampled data for visualization
sampled_vertices_output_path = "hdfs://hadoop-namenode:9820/project/sample_verticest2.csv"
sampled_edges_output_path = "hdfs://hadoop-namenode:9820/project/sample_edgest2.csv"

sampled_vertices.write.mode("overwrite").csv(sampled_vertices_output_path)
sampled_edges.write.mode("overwrite").csv(sampled_edges_output_path)

# Sampling time
sampling_end_time = time.time()
print(f"Sampling time: {sampling_end_time - sampling_start_time} seconds")

# Total execution time
end_time = time.time()
print(f"Total execution time: {end_time - start_time} seconds")

# Stop Spark session
spark.stop()

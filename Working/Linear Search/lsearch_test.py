from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, StringType
from graphframes import GraphFrame

# Initialize Spark session with optimized configurations
spark = SparkSession.builder \
    .appName("lsearch test core 12 :)") \
    .master("spark://10.58.0.158:7077") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.4-spark3.5-s_2.12") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "6") \
    .config("spark.default.parallelism", "200") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.memory.fraction", "0.8") \
    .getOrCreate()

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

# Define schema for edges
schema = StructType([
    StructField("src", StringType(), True),
    StructField("dst", StringType(), True)
])

# Convert RDD to DataFrame using schema
edges_df = spark.createDataFrame(edges_rdd, schema=schema)

# Create vertices DataFrame directly from edges_df
vertices_df = edges_df.select("src").union(edges_df.select("dst")).distinct().toDF("id")

# Create GraphFrame
graph = GraphFrame(vertices_df, edges_df)

# Basic graph operations
num_vertices = graph.vertices.count()
num_edges = graph.edges.count()
print(f"Number of vertices: {num_vertices}")
print(f"Number of edges: {num_edges}")

# Perform linear search for a specific user ID
target_user_id = "1691593"  # Update with the user ID you want to search for
user_found = vertices_df.filter(vertices_df.id == target_user_id)

# Collect the search result without using collect()
result = f"User ID {target_user_id} {'found' if user_found.count() > 0 else 'not found'} in the dataset."

# Write the result to an HDFS file
output_path = "hdfs://hadoop-namenode:9820/project/outputtest50.txt"  # Specify the HDFS output path
spark.sparkContext.parallelize([result]).saveAsTextFile(output_path)

# Find the neighbors of the specified user without collecting data
neighbors = graph.edges.filter(f"src = '{target_user_id}' or dst = '{target_user_id}'")
print(f"Neighbors of user {target_user_id}:")
neighbors.show()

# Stop Spark session
spark.stop()

from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, StringType
from graphframes import GraphFrame

# Initialize Spark session
spark = SparkSession.builder \
    .appName("FacebookGraphFrames") \
    .master("local[*]") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.4-spark3.5-s_2.12") \
    .getOrCreate()

# Load the Facebook dataset (edges) from HDFS
path = "hdfs://hadoop-namenode:9820/project/facebook_combined.txt"  # Update with the actual HDFS path
def parse_line(line):
    parts = line.split(" ")
    if len(parts) == 2:
        return (parts[0], parts[1])
    else:
        return None

# Create RDD and filter out invalid lines
edges_rdd = spark.sparkContext.textFile(path) \
    .map(parse_line) \
    .filter(lambda x: x is not None)

# Define schema
schema = StructType([
    StructField("src", StringType(), True),
    StructField("dst", StringType(), True)
])

# Convert RDD to DataFrame using schema
edges_df = spark.createDataFrame(edges_rdd, schema=schema)

# Create vertices DataFrame
vertices_rdd = edges_rdd.flatMap(lambda row: [Row(id=row[0]), Row(id=row[1])]).distinct()
vertices_df = spark.createDataFrame(vertices_rdd, schema=StructType([StructField("id", StringType(), True)]))

# Create GraphFrame
graph = GraphFrame(vertices_df, edges_df)

# Basic graph operations
print(f"Number of vertices: {graph.vertices.count()}")
print(f"Number of edges: {graph.edges.count()}")

# Find the neighbors of a specific user (e.g., user 0)
user_id = "0"
neighbors = graph.edges.filter(f"src = '{user_id}' or dst = '{user_id}'")
neighbors.show()

# Run a simple PageRank algorithm
pagerank_results = graph.pageRank(resetProbability=0.15, maxIter=10)
pagerank_results.vertices.select("id", "pagerank").orderBy("pagerank", ascending=False).show()

# Stop Spark session
spark.stop()

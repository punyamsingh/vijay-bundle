from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, StringType
from graphframes import GraphFrame
import matplotlib.pyplot as plt
import networkx as nx
import os

# Initialize Spark session
spark = SparkSession.builder \
    .appName("FacebookGraphLinearSearchAndVisualization") \
    .master("local[*]") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.4-spark3.5-s_2.12") \
    .getOrCreate()

# Initialize Spark session
# spark = SparkSession.builder \
#     .appName("FacebookGraphLinearSearchAndVisualization") \
#     .master("spark://10.58.0.158:7077") \
#     .config("spark.jars.packages", "graphframes:graphframes:0.8.4-spark3.5-s_2.12") \
#     .getOrCreate()


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

# Perform linear search for a specific user ID
target_user_id = "1911"  # Update with the user ID you want to search for
user_found = vertices_df.filter(vertices_df.id == target_user_id)

# Collect the search result
result = ""
if user_found.count() > 0:
    result = f"User ID {target_user_id} found in the dataset."
else:
    result = f"User ID {target_user_id} not found in the dataset."

# Write the result to an HDFS file
output_path = "hdfs://hadoop-namenode:9820/project/output2.txt"  # Specify the HDFS output path
rdd = spark.sparkContext.parallelize([result])
rdd.saveAsTextFile(output_path)

# Find the neighbors of the specified user
neighbors = graph.edges.filter(f"src = '{target_user_id}' or dst = '{target_user_id}'")
print(f"Neighbors of user {target_user_id}:")
neighbors.show()

# Convert to NetworkX graph for visualization
# nx_graph = nx.Graph()

# # Add edges to NetworkX graph
# for row in edges_df.collect():
#     nx_graph.add_edge(row['src'], row['dst'])

# # Plotting the graph
# plt.figure(figsize=(12, 12))
# pos = nx.spring_layout(nx_graph)  # positions for all nodes
# nx.draw(nx_graph, pos, with_labels=True, node_size=50, font_size=8, font_color='black', node_color='cyan', edge_color='gray')
# plt.title("Facebook Graph Visualization")
# plt.show()

# Stop Spark session
spark.stop()

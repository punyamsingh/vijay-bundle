from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql import Row
from pyspark.graphx import Graph

# Initialize Spark session and context
spark = SparkSession.builder \
    .appName("Facebook Social Graph Analysis") \
    .getOrCreate()

sc = spark.sparkContext

# Load Facebook edges file
# Path to your unzipped 'facebook_combined.txt' file
edges_file = "project\\linear search\\facebook_combined.txt"

# Load edges as an RDD
edges_rdd = sc.textFile(edges_file).map(lambda line: tuple(map(int, line.split())))

# Transform the edges into a form GraphX can understand
# GraphX needs the vertices and edges as RDDs
# Vertices are individual users (which we extract from the edge tuples)
vertices_rdd = edges_rdd.flatMap(lambda edge: edge).distinct().map(lambda x: (x, x))

# Create the graph
graph = Graph(vertices_rdd, edges_rdd)

# Perform some basic operations
print(f"Number of vertices (users): {graph.vertices.count()}")
print(f"Number of edges (friendships): {graph.edges.count()}")

# Find all friends of a specific user, e.g., user 0
user_id = 0
friends_of_user = graph.edges.filter(lambda edge: edge.srcId == user_id or edge.dstId == user_id)\
                             .map(lambda edge: edge.dstId if edge.srcId == user_id else edge.srcId).collect()

print(f"Friends of user {user_id}: {friends_of_user}")

# Stop the Spark session
spark.stop()

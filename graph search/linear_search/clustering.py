from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, StringType
from graphframes import GraphFrame
import matplotlib.pyplot as plt

# Initialize Spark session
spark = SparkSession.builder \
    .appName("AdvancedFacebookGraphAnalysis") \
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

# PageRank calculation
pagerank_results = graph.pageRank(resetProbability=0.15, maxIter=10)

# Show top 10 important nodes
top_n = 10
important_nodes = pagerank_results.vertices.select("id", "pagerank").orderBy("pagerank", ascending=False).limit(top_n)
print("Top 10 important nodes based on PageRank:")
important_nodes.show()

# Clustering Coefficient Calculation
clustering_df = graph.triangleCount()
print("Clustering Coefficient Results:")
clustering_df.show()

# Community Detection using Label Propagation
communities = graph.labelPropagation(maxIter=5)
print("Community Detection Results:")
communities.select("id", "label").show()

# Degree Distribution
degree_df = graph.degrees
degree_distribution = degree_df.toPandas().groupby("degree").size().reset_index(name='count')

# Plotting Degree Distribution
plt.figure(figsize=(10, 6))
plt.bar(degree_distribution['degree'], degree_distribution['count'])
plt.title('Degree Distribution')
plt.xlabel('Degree')
plt.ylabel('Number of Nodes')
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.show()

# Stop Spark session
spark.stop()

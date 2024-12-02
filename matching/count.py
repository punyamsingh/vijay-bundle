from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

# Initialize Spark
conf = SparkConf().setAppName("CountEdges").setMaster("spark://10.58.0.158:7077")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

# Path to the output file
output_path = "hdfs://hadoop-namenode:9820/project/protein_matching_result2"

# Read the output file from HDFS
edges = sc.textFile(output_path)

# Count the number of edges
edge_count = edges.count()

print(f"Total number of edges: {edge_count}")

# Stop Spark session
sc.stop()

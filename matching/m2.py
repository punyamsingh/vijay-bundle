from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import random

# Initialize Spark
conf = SparkConf().setAppName("FacebookMaximalMatching")
# .setMaster("spark://10.58.0.158:7077")
conf.set("spark.driver.memory", "4g")
conf.set("spark.executor.memory", "4g")
conf.set("spark.rpc.message.maxSize", "512")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

# Load Dataset from HDFS
# Assuming each line represents an edge in the format: "vertex1 vertex2"
file_path = "hdfs://hadoop-namenode:9820/project/facebook_combined.txt"

# Load Dataset
edges = sc.textFile(file_path)
edges = edges.map(lambda line: tuple(map(int, line.split())))

# Parameters
sample_probability = 0.5  # Probability for edge sampling
num_iterations = 3  # Number of matching rounds

# Function to perform edge sampling
def sample_edges(edge):
    return random.random() < sample_probability

# Function to perform greedy matching within each partition
def greedy_matching(partition):
    matched = set()
    matching = set()
    for u, v in partition:
        if u not in matched and v not in matched:
            matching.add((u, v))
            matched.add(u)
            matched.add(v)
    return list(matching)

# Checkpoint intermediate RDDs to HDFS
sc.setCheckpointDir("/tmp/spark-checkpoints")

# Empty RDD to collect final matching
final_matching = sc.emptyRDD()

# Perform maximal matching iteratively
for i in range(num_iterations):
    # Sample edges
    sampled_edges = edges.filter(sample_edges)
    
    # Local greedy matching in each partition
    local_matchings = sampled_edges.mapPartitions(greedy_matching)
    
    # Union results with the final matching for each iteration, checkpoint if large
    if i > 0 and i % 2 == 0:
        final_matching = final_matching.union(local_matchings).distinct().checkpoint()
    else:
        final_matching = final_matching.union(local_matchings).distinct()

    # Cache the final matching to avoid recomputation in subsequent iterations
    final_matching.cache()

    # Remove matched vertices from edges for the next iteration
    matched_vertices = final_matching.flatMap(lambda edge: [edge[0], edge[1]]).distinct()
    edges = edges.filter(lambda edge: edge[0] not in matched_vertices and edge[1] not in matched_vertices)

# Save final results to HDFS
final_matching.saveAsTextFile("hdfs://hadoop-namenode:9820/project/facebook_matching_result")

# Stop Spark session
sc.stop()
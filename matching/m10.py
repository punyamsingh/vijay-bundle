from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from math import pow
import math
import random

# Initialize Spark
conf = SparkConf().setAppName("FacebookMaximalMatching").setMaster("spark://10.58.0.158:7077")
conf.set("spark.driver.memory", "4g")
conf.set("spark.executor.memory", "4g")
conf.set("spark.rpc.message.maxSize", "2047")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

spark.conf.set("spark.hadoop.mapreduce.output.fileoutputformat.compress", "true")
spark.conf.set("spark.hadoop.mapreduce.output.fileoutputformat.compress.codec", "org.apache.hadoop.io.compress.GzipCodec")

# Load Dataset from HDFS
file_path = "hdfs://hadoop-namenode:9820/project/facebook_combined.txt"
edges = sc.textFile(file_path)
edges = edges.map(lambda line: tuple(map(int, line.split())))

# Compute the degree of each vertex to determine delta (maximum degree)
vertex_degrees = edges.flatMap(lambda edge: [(edge[0], 1), (edge[1], 1)]).reduceByKey(lambda a, b: a + b)
delta = vertex_degrees.map(lambda x: x[1]).max()  # Maximum degree
print(f"Maximum degree (delta): {delta}")

# Parameters
sample_probability = pow(delta, -0.77)
num_partitions = int(pow(delta, 0.12))  # Number of partitions for vertex partitioning
print(f"Sample probability: {sample_probability}")
print(f"Number of partitions (k): {num_partitions}")

# Greedy Matching Function
def greedy_matching(partition):
    edges = sorted(partition, key=lambda edge: random.random())  # Assign random priority
    matched = set()
    matching = set()
    for edge in edges:
        try:
            u, v = edge  # Unpack edge into u and v
            if u not in matched and v not in matched:
                matching.add((u, v))
                matched.add(u)
                matched.add(v)
        except TypeError as e:
            print("Error unpacking edge:", edge)  # Log edges that caused the error
            raise e
    return list(matching)

# Checkpoint intermediate RDDs to HDFS
sc.setCheckpointDir("hdfs://hadoop-namenode:9820/project/matching-checkpoints21")
final_matching = sc.emptyRDD()

# Estimate max iterations
estimated_max_iterations = math.ceil(math.log(delta) / math.log(1.1))  # Approx. log(log(delta))
print(f"Estimated maximum iterations: {estimated_max_iterations}")

num_iterations = estimated_max_iterations

# Perform maximal matching iteratively
for i in range(num_iterations):
    # Sample edges
    sampled_edges = edges.sample(False, sample_probability)
    
    # Partition sampled edges based on vertices
    sampled_edges = sampled_edges.map(lambda edge: (edge[0], edge)).partitionBy(num_partitions, lambda key: key % num_partitions)

    # Perform greedy matching locally within partitions
    local_matchings = sampled_edges.mapPartitions(greedy_matching)

    # Combine local matchings with the global final matching
    final_matching = final_matching.union(local_matchings).distinct()

    # Periodic checkpointing
    if i % 5 == 0:
        final_matching.checkpoint()

    final_matching.cache()

    # Extract matched vertices
    matched_vertices = final_matching.flatMap(lambda edge: [edge[0], edge[1]]).distinct()

    # Convert matched vertices to key-value pairs for efficient join
    matched_vertices = matched_vertices.map(lambda v: (v, None))

    # Perform left outer join to filter edges
    edges_with_match = edges.leftOuterJoin(matched_vertices)

    # Keep only edges that are not matched
    edges = edges_with_match.filter(lambda x: x[1][1] is None).map(lambda x: x[1][0])

    # Only pass valid edges to the greedy_matching function
    edges = edges.filter(lambda edge: isinstance(edge, tuple) and len(edge) == 2)

    # Exit if no edges remain
    if edges.isEmpty():
        break

# Simplified validate_matching function
def validate_matching(matching):
    vertices_in_matching = matching.flatMap(lambda edge: [edge[0], edge[1]])
    vertex_counts = vertices_in_matching.countByValue()

    for vertex, count in vertex_counts.items():
        if count > 1:
            print(f"Validation Failed: Vertex {vertex} is part of multiple matched edges.")
            return False

    print("Validation Successful: Maximal matching is correct.")
    return True

# Final Processing and Saving
final_matching = final_matching.filter(lambda edge: edge[0] is not None and edge[1] is not None)

# Flatten any nested tuples for final output
def flatten_edge(edge):
    u, v = edge
    if isinstance(v, tuple):
        v = v[1]
    return f"{u} {v}"

final_matching_clean = final_matching.map(flatten_edge)

# Validate the matching
validate_matching(final_matching)

# Save the result to HDFS
final_matching_clean.saveAsTextFile("hdfs://hadoop-namenode:9820/project/facebook_matching_result101")

# Stop Spark session
sc.stop()
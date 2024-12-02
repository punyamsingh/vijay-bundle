from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from math import pow
import random
import math

# Initialize Spark
conf = SparkConf().setAppName("PPI Maximal Matching").setMaster("spark://10.58.0.158:7077")
conf.set("spark.driver.memory", "4g")
conf.set("spark.executor.memory", "4g")
conf.set("spark.rpc.message.maxSize", "2047")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

spark.conf.set("spark.hadoop.mapreduce.output.fileoutputformat.compress", "true")
spark.conf.set("spark.hadoop.mapreduce.output.fileoutputformat.compress.codec", "org.apache.hadoop.io.compress.GzipCodec")

# Load PPI dataset from HDFS
file_path = "hdfs://hadoop-namenode:9820/project/9606.protein.links.v12.0.txt"
edges = sc.textFile(file_path)

# Skip the header row
header = edges.first()  # Get the first row (header)
edges = edges.filter(lambda line: line != header)  # Filter out the header

# Parse the edges
edges = edges.map(lambda line: line.split())  # Split each line into columns
edges = edges.map(lambda cols: (cols[0], cols[1], int(cols[2])))  # Convert the combined_score to integer

# Parse the edges
# edges = edges.map(lambda line: line.split()).map(lambda cols: (cols[0], cols[1], int(cols[2])))

# Step 1: Compute average combined score for each node
def node_scores(edge):
    u, v, score = edge
    return [(u, (score, 1)), (v, (score, 1))]  # Map both endpoints to their scores

# Sum up scores and counts for each node
node_totals = edges.flatMap(node_scores).reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))

# Compute average combined score for each node
node_averages = node_totals.mapValues(lambda x: x[0] / x[1])  # Total score / count

# Step 2: Filter edges based on the average scores of their endpoints
def filter_edges(edge, avg_scores):
    u, v, score = edge
    avg_u = avg_scores.get(u, 0)  # Average score for node u
    avg_v = avg_scores.get(v, 0)  # Average score for node v
    return score > max(avg_u, avg_v)  # Match only if score > max(average scores of endpoints)

# Broadcast node averages for efficient filtering
node_averages_broadcast = sc.broadcast(dict(node_averages.collect()))

# Apply filtering
filtered_edges = edges.filter(lambda edge: filter_edges(edge, node_averages_broadcast.value))

######## MPC
# Compute the degree of each vertex to determine delta (maximum degree)
vertex_degrees = filtered_edges.flatMap(lambda edge: [(edge[0], 1), (edge[1], 1)]).reduceByKey(lambda a, b: a + b)
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
    for edge in filtered_edges:
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
sc.setCheckpointDir("hdfs://hadoop-namenode:9820/project/matching-checkpoints_PPI_maximal1")
final_matching = sc.emptyRDD()

# Estimate max iterations
estimated_max_iterations = math.ceil(math.log(delta) / math.log(1.1))  # Approx. log(log(delta))
print(f"Estimated maximum iterations: {estimated_max_iterations}")

num_iterations = estimated_max_iterations

# Perform maximal matching iteratively
for i in range(num_iterations):
    # Sample edges
    sampled_edges = filtered_edges.sample(False, sample_probability)
    
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
    filtered_edges = edges_with_match.filter(lambda x: x[1][1] is None).map(lambda x: x[1][0])

    # Only pass valid edges to the greedy_matching function
    filtered_edges = filtered_edges.filter(lambda edge: isinstance(edge, tuple) and len(edge) == 2)

    # Exit if no edges remain
    if filtered_edges.isEmpty():
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
final_matching_clean.saveAsTextFile("hdfs://hadoop-namenode:9820/project/facebook_matching_result_PPI_maximal1")

# Stop Spark session
sc.stop()
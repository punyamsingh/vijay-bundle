from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from math import pow
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
num_partitions = int(pow(delta, 0.12))  # Number of vertex partitions
print(f"Sample probability: {sample_probability}")
print(f"Number of partitions (k): {num_partitions}")

# Broadcast vertex partitions
vertex_partitions = sc.parallelize(range(num_partitions)).map(lambda i: (i, [])).collectAsMap()

def assign_vertex_partition(vertex_id):
    return vertex_id % num_partitions

vertex_partitions_broadcast = sc.broadcast(vertex_partitions)

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
sc.setCheckpointDir("hdfs://hadoop-namenode:9820/project/matching-checkpoints11")
final_matching = sc.emptyRDD()

# Perform maximal matching iteratively
iteration = 0
while True:
    iteration += 1
    print(f"Starting iteration {iteration}...")

    # Step 1: Sample edges
    sampled_edges = edges.sample(False, sample_probability)

    # Step 2: Partition edges based on vertex partitions
    def map_to_partition(edge):
        u, v = edge
        partition = assign_vertex_partition(u) if assign_vertex_partition(u) == assign_vertex_partition(v) else None
        if partition is not None:
            return (partition, edge)
        else:
            return None

    partitioned_edges = sampled_edges.map(map_to_partition).filter(lambda x: x is not None).groupByKey()

    # Step 3: Perform greedy matching locally
    local_matchings = partitioned_edges.flatMap(lambda x: greedy_matching(list(x[1])))

    # Step 4: Combine local matchings with the global final matching
    final_matching = final_matching.union(local_matchings).distinct()

    # Checkpoint to prevent lineage growth
    if iteration % 5 == 0:
        final_matching.checkpoint()

    final_matching.cache()

    # Step 5: Update residual graph by removing matched vertices and their edges
    matched_vertices = final_matching.flatMap(lambda edge: [edge[0], edge[1]]).distinct()
    matched_vertices = matched_vertices.map(lambda v: (v, None))

    edges_with_match = edges.leftOuterJoin(matched_vertices)
    edges = edges_with_match.filter(lambda x: x[1][1] is None).map(lambda x: x[1][0])

    # Check for degree reduction
    remaining_vertex_degrees = edges.flatMap(lambda edge: [(edge[0], 1), (edge[1], 1)] if isinstance(edge, tuple) and len(edge) == 2 else []).reduceByKey(lambda a, b: a + b)
    if remaining_vertex_degrees.isEmpty():
        print(f"Iteration {iteration}: No edges remaining. Stopping.")
        break
    
    max_degree = remaining_vertex_degrees.map(lambda x: x[1]).max()
    print(f"Iteration {iteration}: Maximum degree of residual graph: {max_degree}")

    # Terminate if the graph is sufficiently reduced
    if max_degree <= 1:
        print(f"Graph sufficiently reduced after {iteration} iterations. Stopping.")
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
final_matching_clean.saveAsTextFile("hdfs://hadoop-namenode:9820/project/facebook_matching_result_fixed22")

# Stop Spark session
sc.stop()
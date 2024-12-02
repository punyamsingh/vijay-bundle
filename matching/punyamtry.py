from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from math import pow
import math
import random

# Initialize Spark
conf = SparkConf().setAppName("PPIMaximalMatching").setMaster("spark://10.58.0.158:7077")
conf.set("spark.driver.memory", "4g")
conf.set("spark.executor.memory", "4g")
conf.set("spark.rpc.message.maxSize", "2047")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

spark.conf.set("spark.hadoop.mapreduce.output.fileoutputformat.compress", "true")
spark.conf.set("spark.hadoop.mapreduce.output.fileoutputformat.compress.codec", "org.apache.hadoop.io.compress.GzipCodec")

# Load Dataset from HDFS
file_path = "hdfs://hadoop-namenode:9820/project/9606.protein.links.v12.0.txt"
edges = sc.textFile(file_path)

# Skip the header row
header = edges.first()  # Get the first row (header)
edges = edges.filter(lambda line: line != header)  # Filter out the header

# Safely parse edges, ensuring there are exactly two proteins
def parse_edge(line):
    parts = line.split()
    if len(parts) == 2:
        return tuple(parts)
    else:
        return None  # Skip malformed lines

edges = edges.map(parse_edge).filter(lambda edge: edge is not None)  # Filter out invalid lines

# Parse dataset into edges and optionally filter by combined_score
# Assuming the dataset has columns: protein1, protein2, combined_score
edges = edges.map(lambda line: line.split()).map(lambda cols: ((cols[0], cols[1]), int(cols[2])))

# Filter edges based on combined_score (optional)
min_score_threshold = 200  # Adjust this as needed
edges = edges.filter(lambda edge: edge[1] >= min_score_threshold).map(lambda edge: edge[0])

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
        u, v = edge  # Unpack edge into u and v
        if u not in matched and v not in matched:
            matching.add((u, v))
            matched.add(u)
            matched.add(v)
    return list(matching)

# Checkpoint intermediate RDDs to HDFS
sc.setCheckpointDir("hdfs://hadoop-namenode:9820/project/matching-checkpoints123")
final_matching = sc.emptyRDD()

# Estimate max iterations
estimated_max_iterations = math.ceil(math.log(delta) / math.log(1.1))  # Approx. log(log(delta))
print(f"Estimated maximum iterations: {estimated_max_iterations}")

num_iterations = estimated_max_iterations

# Perform maximal matching iteratively
for i in range(num_iterations):
    sampled_edges = edges.sample(False, sample_probability)
    sampled_edges = sampled_edges.map(lambda edge: (edge[0], edge)).partitionBy(num_partitions, lambda key: hash(key) % num_partitions)
    local_matchings = sampled_edges.mapPartitions(greedy_matching)

    final_matching = final_matching.union(local_matchings).distinct()

    if i % 5 == 0:
        final_matching.checkpoint()

    final_matching.cache()

    matched_vertices = final_matching.flatMap(lambda edge: [edge[0], edge[1]]).distinct()
    matched_vertices = matched_vertices.map(lambda v: (v, None))

    edges_with_match = edges.leftOuterJoin(matched_vertices)
    edges = edges_with_match.filter(lambda x: x[1][1] is None).map(lambda x: x[1][0])

    # if edges.isEmpty():
    #     break

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

def flatten_edge(edge):
    u, v = edge
    return f"{str(u)} {str(v)}"

final_matching_clean = final_matching.map(flatten_edge).filter(lambda line: line is not None)

# Debug the output before saving
print(final_matching_clean.take(10))

# Save the result to HDFS
output_path = "hdfs://hadoop-namenode:9820/project/protein_matching_result123"
final_matching_clean.repartition(10).saveAsTextFile(output_path)
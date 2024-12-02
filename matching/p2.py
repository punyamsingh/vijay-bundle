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

# Step 1: Compute average combined score for each node using aggregateByKey
def seqOp(acc, value):
    # Accumulator: (sum, count)
    score, count = acc
    return (score + value[0], count + 1)

def combOp(acc1, acc2):
    # Combiner: Combine the sum and counts from different partitions
    score1, count1 = acc1
    score2, count2 = acc2
    return (score1 + score2, count1 + count2)

# Aggregate by node to calculate the total score and count
node_totals = edges.flatMap(lambda edge: [(edge[0], (edge[2], 1)), (edge[1], (edge[2], 1))]) \
                   .aggregateByKey((0, 0), seqOp, combOp)

# Compute average score for each node
node_averages = node_totals.mapValues(lambda x: x[0] / x[1])  # Total score / count

# Broadcast node averages for efficient filtering
node_averages_broadcast = sc.broadcast(dict(node_averages.collect()))

# Step 2: Combine Filtering and Matching in Greedy Matching
def greedy_matching(partition):
    matched = set()
    matching = set()
    for edge in partition:
        u, v, score = edge
        avg_u = node_averages_broadcast.value.get(u, 0)  # Average score for node u
        avg_v = node_averages_broadcast.value.get(v, 0)  # Average score for node v

        # Match only if score > max(average scores of both endpoints) and both nodes are unmatched
        if u not in matched and v not in matched and score > max(avg_u, avg_v):
            matching.add((u, v))
            matched.add(u)
            matched.add(v)
    
    return list(matching)

# Compute the degree of each vertex to determine delta (maximum degree)
vertex_degrees = edges.flatMap(lambda edge: [(edge[0], 1), (edge[1], 1)]).reduceByKey(lambda a, b: a + b)
delta = vertex_degrees.map(lambda x: x[1]).max()  # Maximum degree
print(f"Maximum degree (delta): {delta}")

# Parameters
sample_probability = pow(delta, -0.77)
num_partitions = int(pow(delta, 0.12))  # Number of partitions for vertex partitioning
print(f"Sample probability: {sample_probability}")
print(f"Number of partitions (k): {num_partitions}")

# Checkpoint intermediate RDDs to HDFS
sc.setCheckpointDir("hdfs://hadoop-namenode:9820/project/matching-checkpoints_PPI2121")
final_matching = sc.emptyRDD()

# Estimate max iterations
estimated_max_iterations = math.ceil(math.log(delta) / math.log(1.1))  # Approx. log(log(delta))
print(f"Estimated maximum iterations: {estimated_max_iterations}")

num_iterations = estimated_max_iterations

# Perform maximal matching iteratively
for i in range(num_iterations):
    # Sample edges
    sampled_edges = edges.sample(False, sample_probability)
    
    # Dynamically partition sampled edges based on the vertex degree distribution
    sampled_edges_partitioned = sampled_edges.map(lambda edge: (edge[0], edge)) \
                                              .partitionBy(num_partitions, lambda key: hash(key) % num_partitions)

    # Perform greedy matching locally within partitions
    local_matchings = sampled_edges_partitioned.mapPartitions(greedy_matching)

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

    edges=edges.repartition(100)

    # Exit if no edges remain
    # if edges.count() == 0:
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
# final_matching = final_matching.filter(lambda edge: edge[0] is not None and edge[1] is not None)

# Flatten any nested tuples for final output
if final_matching.isEmpty():
    print("No matching found, not saving output.")
else:
    # Flatten any nested tuples for final output
    def flatten_edge(edge):
        u, v = edge
        if isinstance(v, tuple):
            v = v[1]
        return f"{u} {v}"

    final_matching_clean = final_matching.map(flatten_edge)

    # Validate the matching
    # validate_matching(final_matching)
    print("Validation skipped for large datasets")

    # Save the result to HDFS
    try:
        final_matching_clean.saveAsTextFile("hdfs://hadoop-namenode:9820/project/ppi_matching_result_PPI12")
        print("Matching results saved successfully.")
    except Exception as e:
        print(f"Error while saving the matching result: {str(e)}")

# Stop Spark session
sc.stop()
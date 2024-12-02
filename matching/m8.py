## Accurate Working code according to research paper.
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from math import pow
import math

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

# Partition edges based on vertices
def vertex_partition(key):
    return key % num_partitions  # Partition by vertex ID mod k

edges = edges.map(lambda edge: (edge[0], edge)).partitionBy(num_partitions, vertex_partition)

# Greedy Matching Function
def greedy_matching(partition):
    matched = set()
    matching = set()
    for edge in partition:
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
sc.setCheckpointDir("hdfs://hadoop-namenode:9820/project/matching-checkpoints")
final_matching = sc.emptyRDD()

# Estimate max iterations
estimated_max_iterations = math.ceil(math.log(edges.count()) / math.log(1 / sample_probability))
print(f"Estimated maximum iterations: {estimated_max_iterations}")

num_iterations = estimated_max_iterations

# Perform maximal matching iteratively
for i in range(num_iterations):  # Fixed number of iterations
    # Sample edges
    sampled_edges = edges.sample(False, sample_probability)
    
    # Perform greedy matching locally within partitions
    local_matchings = sampled_edges.mapPartitions(greedy_matching)

    # Combine local matchings with the global final matching
    if i > 0 and i % 2 == 0:
        final_matching = final_matching.union(local_matchings).distinct()
        final_matching.checkpoint()
    else:
        final_matching = final_matching.union(local_matchings).distinct()

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


# Simplified validate_matching function
def validate_matching(matching):
    # Flatten all edges to extract the vertices
    vertices_in_matching = matching.flatMap(lambda edge: [edge[0], edge[1]])

    # Count occurrences of each vertex
    vertex_counts = vertices_in_matching.countByValue()

    # # Print the counts of each vertex
    # print("Vertex counts:")
    # for vertex, count in vertex_counts.items():
    #     print(f"Vertex {vertex}: {count}")

    # Check if any vertex appears more than once
    for vertex, count in vertex_counts.items():
        # print(f"Vertex {vertex}: {count}")
        if count > 1:
            print(f"Vertex {vertex}: {count}")
            print(f"Validation Failed: Vertex {vertex} is part of multiple matched edges.")
            # return False

    # print("Validation Successful: Maximal matching is correct.")
    return True


# Final Processing and Saving
final_matching = final_matching.filter(lambda edge: edge[0] is not None and edge[1] is not None)

# Flatten any nested tuples, e.g., (1576, (1576, 1628)) -> 1576 1628
def flatten_edge(edge):
    u, v = edge
    if isinstance(v, tuple):  # If v is a tuple like (1576, 1628)
        v = v[1]  # Get the second element of the tuple (e.g., 1628)
    return f"{u} {v}"

# Apply the flattening function to each edge
final_matching_clean = final_matching.map(flatten_edge)

# Validate the matching
validate_matching(final_matching_clean)

# Save the result to HDFS in the correct format
final_matching_clean.saveAsTextFile("hdfs://hadoop-namenode:9820/project/facebook_matching_result8")

# Stop Spark session
sc.stop()

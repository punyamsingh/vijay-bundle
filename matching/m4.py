from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import math

# Initialize Spark
conf = SparkConf().setAppName("FacebookMPC").setMaster("spark://10.58.0.158:7077")
conf.set("spark.driver.memory", "4g")
conf.set("spark.executor.memory", "4g")
conf.set("spark.rpc.message.maxSize", "2047")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

spark.conf.set("spark.hadoop.mapreduce.output.fileoutputformat.compress", "true")
spark.conf.set("spark.hadoop.mapreduce.output.fileoutputformat.compress.codec", "org.apache.hadoop.io.compress.GzipCodec")

# Load Dataset from HDFS
file_path = "hdfs://hadoop-namenode:9820/project/facebook_combined.txt"
edges = sc.textFile(file_path).map(lambda line: tuple(map(int, line.split())))

# Calculate maximum degree (delta) and graph statistics
degrees = edges.flatMap(lambda edge: [(edge[0], 1), (edge[1], 1)]) \
               .reduceByKey(lambda a, b: a + b)
max_degree = degrees.map(lambda x: x[1]).max()  # Find the max degree
num_vertices = degrees.count()
num_edges = edges.count()
print(f"Graph Statistics: Vertices = {num_vertices}, Edges = {num_edges}, Max Degree (delta) = {max_degree}")

# Sampling probability and partitions based on delta
sampling_probability = max_degree ** -0.77
num_partitions = int(math.ceil(max_degree ** 0.12))
print(f"Sampling Probability: {sampling_probability}, Number of Partitions: {num_partitions}")

# Estimate number of iterations (heuristic)
estimated_iterations = int(math.ceil(math.log(num_edges)))
print(f"Estimated Iterations: {estimated_iterations}")

# Ensure edges are tuples with two elements
edges = edges.filter(lambda edge: isinstance(edge, tuple) and len(edge) == 2)

# Partition edges
edges = edges.partitionBy(num_partitions, lambda key: key % num_partitions)
# edges = edges.partitionBy(num_partitions, lambda edge: edge[0] % num_partitions)

# Greedy Matching Function
def greedy_matching(partition):
    matched = set()
    matching = set()
    for edge in partition:
        try:
            u, v = edge
            if u not in matched and v not in matched:
                matching.add((u, v))
                matched.add(u)
                matched.add(v)
        except TypeError as e:
            print("Error unpacking edge:", edge)
            raise e
    return list(matching)

# Checkpoint intermediate RDDs to HDFS
sc.setCheckpointDir("hdfs://hadoop-namenode:9820/project/matching-checkpoints")
final_matching = sc.emptyRDD()

# Perform maximal matching iteratively
previous_edge_count = num_edges
for iteration in range(estimated_iterations):
    print(f"Starting iteration {iteration + 1}...")

    # Sample edges
    sampled_edges = edges.sample(False, sampling_probability)
    
    # Perform greedy matching locally within partitions
    local_matchings = sampled_edges.mapPartitions(greedy_matching)

    # Combine local matchings with the global final matching
    final_matching = final_matching.union(local_matchings).distinct()
    final_matching.cache()

    # Extract matched vertices
    matched_vertices = final_matching.flatMap(lambda edge: [edge[0], edge[1]]).distinct()
    matched_vertices = matched_vertices.map(lambda v: (v, None))

    # Perform left outer join to filter edges
    edges_with_match = edges.leftOuterJoin(matched_vertices)
    edges = edges_with_match.filter(lambda x: x[1][1] is None).map(lambda x: x[1][0])

    # Count remaining edges to check for convergence
    current_edge_count = edges.count()
    print(f"Remaining edges: {current_edge_count}")
    
    if current_edge_count == 0 or abs(previous_edge_count - current_edge_count) < 1:
        print("Converged: No significant change in edge count.")
        break
    previous_edge_count = current_edge_count

# Validate maximal matching
def validate_matching(matching, edges):
    # No two matched edges share a vertex
    vertices_in_matching = matching.flatMap(lambda edge: [edge[0], edge[1]]).countByValue()
    if any(count > 1 for count in vertices_in_matching.values()):
        print("Validation Failed: Overlapping matched edges detected.")
        return False

    # No unmatched edge exists where both vertices are unmatched
    unmatched_edges = edges.subtract(matching)
    unmatched_check = unmatched_edges.filter(lambda edge: all(vertex not in vertices_in_matching for vertex in edge))
    if unmatched_check.count() > 0:
        print("Validation Failed: Unmatched edges with both vertices free.")
        return False

    print("Validation Successful: Maximal matching is correct.")
    return True

# Validate the final matching
final_matching = final_matching.filter(lambda edge: edge[0] is not None and edge[1] is not None)
validate_matching(final_matching, edges)

# Save results
final_matching_txt = final_matching.map(lambda edge: f"{edge[0]},{edge[1]}")
final_matching_txt.saveAsTextFile("hdfs://hadoop-namenode:9820/project/facebook_matching_result_mpc")

# Stop Spark session
sc.stop()

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from math import pow
import math
import random

# Initialize Spark
conf = SparkConf().setAppName("ProteinMaximalMatching").setMaster("spark://10.58.0.158:7077")
conf.set("spark.driver.memory", "4g")
conf.set("spark.executor.memory", "4g")
conf.set("spark.rpc.message.maxSize", "2047")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

spark.conf.set("spark.hadoop.mapreduce.output.fileoutputformat.compress", "true")
spark.conf.set("spark.hadoop.mapreduce.output.fileoutputformat.compress.codec", "org.apache.hadoop.io.compress.GzipCodec")

# Load Dataset from HDFS (adjust path as needed)
file_path = "hdfs://hadoop-namenode:9820/project/9606.protein.links.v12.0.txt"
edges = sc.textFile(file_path)

header = edges.first()  # Get the first row (header)
edges = edges.filter(lambda line: line != header)  # Filter out the header

# Parse lines into tuples (protein1, protein2, combined_score)
edges = edges.map(lambda line: (line.split()[0], line.split()[1], float(line.split()[2])))

# Step 1: Calculate the average combined score for each protein
# Flatten edges to associate each protein with its scores
protein_scores = edges.flatMap(lambda edge: [(edge[0], edge[2]), (edge[1], edge[2])])
protein_avg_scores = (
    protein_scores
    .aggregateByKey((0, 0), 
                    lambda acc, value: (acc[0] + value, acc[1] + 1),  # Sum and count
                    lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1]))  # Combine sums and counts
    .mapValues(lambda acc: acc[0] / acc[1])  # Compute averages
)

# Broadcast the average scores for efficient access
protein_avg_scores_broadcast = sc.broadcast(dict(protein_avg_scores.collect()))

# Step 2: Greedy Matching Function
def greedy_matching_with_score_check(partition):
    avg_scores = protein_avg_scores_broadcast.value
    edges = sorted(partition, key=lambda edge: random.random())  # Assign random priority
    matched = set()
    matching = set()
    for edge in edges:
        u, v, score = edge  # Unpack edge
        # Check if the score is greater than the average of both vertices
        if score > avg_scores.get(u, 0) and score > avg_scores.get(v, 0):
            if u not in matched and v not in matched:
                matching.add((u, v))
                matched.add(u)
                matched.add(v)
    return list(matching)

# Step 3: Perform maximal matching iteratively
sc.setCheckpointDir("hdfs://hadoop-namenode:9820/project/matching-checkpoints_protein3")
final_matching = sc.emptyRDD()

# Compute the degree of each protein (vertex) to determine delta (maximum degree)
vertex_degrees = edges.flatMap(lambda edge: [(edge[0], 1), (edge[1], 1)]).reduceByKey(lambda a, b: a + b)
delta = vertex_degrees.map(lambda x: x[1]).max()  # Maximum degree
sample_probability = pow(delta, -0.77)
num_partitions = int(pow(delta, 0.12))  # Number of partitions for vertex partitioning
estimated_max_iterations = math.ceil(math.log(delta) / math.log(1.1))

for i in range(estimated_max_iterations):
    # Sample edges
    sampled_edges = edges.sample(False, sample_probability)
    
    # Partition sampled edges
    sampled_edges = sampled_edges.map(lambda edge: (edge[0], edge)).partitionBy(num_partitions, lambda key: hash(key) % num_partitions)

    # Perform greedy matching with score check locally within partitions
    local_matchings = sampled_edges.mapPartitions(greedy_matching_with_score_check)

    # Combine local matchings with the global final matching
    final_matching = final_matching.union(local_matchings).distinct()

    # Periodic checkpointing
    if i % 5 == 0:
        final_matching.checkpoint()

    final_matching.cache()

    # Extract matched proteins
    matched_proteins = final_matching.flatMap(lambda edge: [edge[0], edge[1]]).distinct()

    # Convert matched proteins to key-value pairs for efficient join
    matched_proteins = matched_proteins.map(lambda v: (v, None))

    # Perform left outer join to filter edges
    edges_with_match = edges.map(lambda edge: ((edge[0], edge[1]), edge)).leftOuterJoin(matched_proteins)

    # Keep only edges that are not matched
    edges = edges_with_match.filter(lambda x: x[1][1] is None).map(lambda x: x[1][0])

    # Exit if no edges remain
    if edges.isEmpty():
        break

# Step 4: Validate and Save Results
final_matching = final_matching.filter(lambda edge: edge[0] is not None and edge[1] is not None)
def flatten_edge(edge):
    u, v = edge
    return f"{u} {v}"

final_matching_clean = final_matching.map(flatten_edge)
output_path = "hdfs://hadoop-namenode:9820/project/protein_matching_result3"
final_matching_clean.repartition(10).saveAsTextFile(output_path)

sc.stop()
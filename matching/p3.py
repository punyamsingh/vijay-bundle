from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from math import pow
import math
from subprocess import call

# Initialize Spark
conf = SparkConf().setAppName("PPI Maximal Matching").setMaster("spark://10.58.0.158:7077")
conf.set("spark.driver.memory", "4g")
conf.set("spark.executor.memory", "8g")  # Increased executor memory
conf.set("spark.rpc.message.maxSize", "2047")
conf.set("spark.network.timeout", "600s")  # Increase network timeout
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

spark.conf.set("spark.hadoop.mapreduce.output.fileoutputformat.compress", "true")
spark.conf.set("spark.hadoop.mapreduce.output.fileoutputformat.compress.codec", "org.apache.hadoop.io.compress.GzipCodec")

# Load PPI dataset from HDFS
file_path = "hdfs://hadoop-namenode:9820/project/9606.protein.links.v12.0.txt"
edges = sc.textFile(file_path)

# Skip the header row
header = edges.first()
edges = edges.filter(lambda line: line != header)

# Parse the edges
edges = edges.map(lambda line: line.strip().split())
edges = edges.map(lambda cols: (cols[0], cols[1], int(cols[2])))

# Validate that each edge is a tuple with correct format and types
def validate_edge(edge):
    return (
        isinstance(edge, tuple) and
        len(edge) == 3 and
        isinstance(edge[0], str) and
        isinstance(edge[1], str) and
        isinstance(edge[2], int)
    )

edges = edges.filter(validate_edge)

# Step 1: Compute average combined score for each node using aggregateByKey
def seqOp(acc, value):
    return (acc[0] + value[0], acc[1] + 1)

def combOp(acc1, acc2):
    return (acc1[0] + acc2[0], acc1[1] + acc2[1])

node_totals = edges.flatMap(lambda edge: [(edge[0], (edge[2], 1)), (edge[1], (edge[2], 1))]) \
                   .aggregateByKey((0, 0), seqOp, combOp)

node_averages = node_totals.mapValues(lambda x: x[0] / x[1])
node_averages_broadcast = sc.broadcast(dict(node_averages.collect()))

# Step 2: Combine Filtering and Matching in Greedy Matching
def greedy_matching(partition):
    matched = set()
    matching = set()
    for edge in partition:
        u, v, score = edge
        avg_u = node_averages_broadcast.value.get(u, 0)
        avg_v = node_averages_broadcast.value.get(v, 0)
        if u not in matched and v not in matched and score > max(avg_u, avg_v):
            matching.add((u, v))
            matched.add(u)
            matched.add(v)
    return list(matching)

vertex_degrees = edges.flatMap(lambda edge: [(edge[0], 1), (edge[1], 1)]).reduceByKey(lambda a, b: a + b)
delta = vertex_degrees.map(lambda x: x[1]).max()
sample_probability = pow(delta, -0.77)
num_partitions = int(pow(delta, 0.12))

sc.setCheckpointDir("hdfs://hadoop-namenode:9820/project/matching-checkpoints_PPI_maximal")
final_matching = sc.emptyRDD()

estimated_max_iterations = math.ceil(math.log(delta) / math.log(1.1))
num_iterations = estimated_max_iterations

for i in range(num_iterations):
    sampled_edges = edges.sample(False, sample_probability)
    sampled_edges_partitioned = sampled_edges.map(lambda edge: (edge[0], edge)) \
                                              .partitionBy(num_partitions, lambda key: hash(key) % num_partitions)
    local_matchings = sampled_edges_partitioned.mapPartitions(greedy_matching)
    final_matching = final_matching.union(local_matchings).distinct()

    if i % 5 == 0:
        final_matching.checkpoint()

    final_matching.cache()
    matched_vertices = final_matching.flatMap(lambda edge: [edge[0], edge[1]]).distinct()
    matched_vertices = matched_vertices.map(lambda v: (v, None))

    edges_with_match = edges.leftOuterJoin(matched_vertices)
    edges = edges_with_match.filter(lambda x: x[1][1] is None).map(lambda x: x[1][0])
    edges = edges.filter(validate_edge)
    edges = edges.repartition(100)

# Validate data before saving
def validate_matching_data(edge):
    return isinstance(edge, tuple) and len(edge) == 2 and edge[0] is not None and edge[1] is not None

final_matching = final_matching.filter(validate_matching_data)

# Flatten edges for final output
def flatten_edge(edge):
    u, v = edge
    return f"{u} {v}"

final_matching_clean = final_matching.map(flatten_edge)

# Remove existing directory before saving
output_path = "hdfs://hadoop-namenode:9820/project/ppi_matching_result_PPIMaximal"
call(["hdfs", "dfs", "-rm", "-r", output_path])

# Save the result to HDFS
final_matching_clean.saveAsTextFile(output_path)

# Stop Spark session
sc.stop()

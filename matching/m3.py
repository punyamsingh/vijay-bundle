from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

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

# Partition edges
num_partitions = 100  # Adjust based on your cluster size
edges = edges.map(lambda edge: (edge[0], edge))  # Convert to key-value pairs

# Parameters
sample_probability = 0.5
num_iterations = 73

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
sc.setCheckpointDir("hdfs://hadoop-namenode:9820/project/matching-checkpoints3")
final_matching = sc.emptyRDD()

# Perform maximal matching iteratively
for i in range(num_iterations):
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


# Checkpoint final matching
final_matching.checkpoint()


final_matching = final_matching.filter(lambda edge: edge[0] is not None and edge[1] is not None)
final_matching_txt = final_matching.map(lambda edge: f"{edge[0]},{edge[1]}")
# final_matching_txt.saveAsTextFile("C:\\Users\\ps232\\vijay-bundle\\project\\matching_result")
final_matching_txt.saveAsTextFile("hdfs://hadoop-namenode:9820/project/facebook_matching_result42")



# Convert final matching to DataFrame
# final_matching_df = final_matching.map(lambda edge: (edge[0], edge[1])).toDF(["vertex1", "vertex2"])

# Convert final matching to RDD of strings, where each string is "Vertex1,Vertex2"
# final_matching_txt = final_matching.map(lambda edge: f"{edge[0]},{edge[1]}")

# Save as text file in the specified directory (local path or HDFS path)
# final_matching_txt.saveAsTextFile("C:\\Users\\ps232\\vijay-bundle\\project\\matching_result.txt")

# If saving to HDFS:
# final_matching_txt.saveAsTextFile("hdfs://hadoop-namenode:9820/project/facebook_matching_result")


# Write as a distributed Parquet file
# final_matching_df.write.mode("overwrite").parquet("C:\\Users\\ps232\\vijay-bundle\\project\\matching_result")
# final_matching_df.write.mode("overwrite").parquet("hdfs://10.58.0.129:9820/project/facebook_matching_result")

# Save after checkpointing (optional step for saving as text)
final_matching = final_matching.repartition(200)  # Adjust partitions

# Save the RDD
# final_matching.saveAsTextFile("C:\\Users\\ps232\\vijay-bundle\\project\\matching_result")
# final_matching.saveAsTextFile("hdfs://hadoop-namenode:9820/project/facebook_matching_result")

# Stop Spark session
sc.stop()

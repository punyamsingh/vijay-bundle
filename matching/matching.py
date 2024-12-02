from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

# Initialize Spark
# Initialize Spark session
# spark = SparkSession.builder \
#     .appName("Pratham using sometime core 24 :)") \
#     .config("spark.jars.packages", "graphframes:graphframes:0.8.4-spark3.5-s_2.12") \
#     .getOrCreate()


conf = SparkConf().setAppName("FacebookGraph").setMaster("spark://10.58.0.158:7077")
conf.set("spark.rpc.message.maxSize", "512")
conf.set("spark.rpc.askTimeout", "300s")
conf.set("spark.sql.shuffle.partitions", "200")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)
sc.setLogLevel("DEBUG")


# Load the dataset from HDFS
file_path = "hdfs://10.58.0.129:9870/project/facebook-combined.txt"
edges_rdd = sc.wholeTextFiles(file_path)


edges_rdd = edges_rdd.repartition(200)

# Parse edges
# Assuming each line has two integers separated by space representing an edge
edges_rdd = edges_rdd.mapPartitions(lambda iter: (tuple(map(int, line.split())) for line in iter))

# # Display the first few edges to verify
# print("Sample edges:")
# for edge in edges_rdd.take(10):
#     print(edge)

# Optional: Convert to DataFrame if needed for DataFrame-based operations
edges_df = spark.createDataFrame(edges_rdd, ["Vertex1", "Vertex2"])
edges_df.show(10)

# Stop Spark session after operations
sc.stop()

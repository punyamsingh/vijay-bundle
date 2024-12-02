from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from math import pow
import math
import random

# Initialize Spark Session
conf = SparkConf().setAppName("PPI Analysis").setMaster("spark://10.58.0.158:7077")
conf.set("spark.driver.memory", "4g")
conf.set("spark.executor.memory", "4g")
conf.set("spark.rpc.message.maxSize", "2047")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

# Load the dataset into a DataFrame
data_path = "hdfs://hadoop-namenode:9820/project/9606.protein.links.v12.0.txt"
ppi_df = spark.read.csv(data_path, sep=" ", header=True)

# Count total number of interactions (rows)
num_interactions = ppi_df.count()

# Find the distinct proteins
unique_proteins = (
    ppi_df.select("protein1")
    .union(ppi_df.select("protein2"))
    .distinct()
)

num_proteins = unique_proteins.count()

# Display results
print(f"Total number of interactions: {num_interactions}")
print(f"Total number of unique proteins: {num_proteins}")

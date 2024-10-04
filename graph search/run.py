# import findspark
# findspark.init()

from pyspark import SparkConf, SparkContext

# Initialize Spark Context
conf = SparkConf().setAppName("WordCount").setMaster("local")
sc = SparkContext(conf=conf)

# Read input file (update the path accordingly)
input_file = "file:///C:/Users/ps232/vijay-bundle/project/text.txt"
text_file = sc.textFile(input_file)

# Perform the word count
counts = text_file.flatMap(lambda line: line.split(" ")) \
                  .map(lambda word: (word, 1)) \
                  .reduceByKey(lambda a, b: a + b)

# Collect and print the results
print("Hello"*1000)
# for (word, count) in counts.collect():
#     print(f"{word}: {count}")

# Stop the Spark Context
sc.stop()

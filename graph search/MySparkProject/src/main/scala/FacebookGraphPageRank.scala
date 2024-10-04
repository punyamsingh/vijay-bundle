import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object FacebookGraphPageRank {
  def main(args: Array[String]): Unit = {
    // Create a Spark session
    val spark = SparkSession.builder
      .appName("Facebook Graph PageRank")
      .getOrCreate()

    val sc = spark.sparkContext

    // Load the edges as an RDD from HDFS
    val edges: RDD[Edge[Int]] = sc.textFile("hdfs:///project/facebook_combined.txt")
      .map(line => {
        val parts = line.split("\\s+")
        Edge(parts(0).toLong, parts(1).toLong, 1)
      })

    // Create the graph
    val graph: Graph[Int, Int] = Graph.fromEdges(edges, defaultValue = 1)

    // Run the PageRank algorithm
    val ranks = graph.pageRank(0.0001).vertices

    // Collect and print the result
    ranks.collect.foreach { case (vertexId, rank) =>
      println(s"Vertex $vertexId has rank $rank")
    }

    // Stop the Spark session
    spark.stop()
  }
}

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object MySparkApp {
  def main(args: Array[String]): Unit = {
    // Create Spark Session
    val spark = SparkSession.builder()
      .appName("Join Example")
      .master("local[*]") // Run locally with all available cores
      .getOrCreate()

    import spark.implicits._

    // Create two DataFrames (tables)
    val table1 = Seq(
      (1, "Alice", 25),
      (2, "Bob", 30),
      (3, "Charlie", 35)
    ).toDF("id", "name", "age")

    val table2 = Seq(
      (1, "HR"),
      (2, "Engineering"),
      (3, "Marketing")
    ).toDF("id", "department")

    // Perform an inner join on the 'id' column
    val joinedTable = table1.join(table2, "id")

    // Show the result of the join
    joinedTable.show()

    // Stop the Spark session
    spark.stop()
  }
}

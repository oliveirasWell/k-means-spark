import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator

object KMeansSpark {

  def sparkExampleMain(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName(s"${this.getClass.getSimpleName}")
      .getOrCreate()

    // Loads data.
    val dataset = spark.read.format("libsvm").load("data/sample_kmeans_data.txt")

    // Trains a k-means model.
    val kmeans = new KMeans().setK(2).setSeed(1L)
    val model = kmeans.fit(dataset)

    // Make predictions
    val predictions = model.transform(dataset)

    // Evaluate clustering by computing Silhouette score
    val evaluator = new ClusteringEvaluator()

    val silhouette = evaluator.evaluate(predictions)
    println(s"Silhouette with squared euclidean distance = $silhouette")

    // Shows the result.
    println("Cluster Centers: ")
    println(model)
    println(model.clusterCenters)
    model.clusterCenters.foreach(println)
    // $example off$
    spark.stop()
  }

  def main(args: Array[String]) {
    print("\n\n>>>>> START OF PROGRAM <<<<<\n\n");

    println("Hello World.")

    print("\n\n>>>>> END OF PROGRAM <<<<<\n\n");
  }

}
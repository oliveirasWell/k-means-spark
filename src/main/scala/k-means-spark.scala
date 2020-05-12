import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.util.MLUtils

object KMeansSpark {

  val DataFile = "data/sample_kmeans_data.txt"
  val K = 2

  def sparkExampleMain(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName(s"${this.getClass.getSimpleName}")
      .getOrCreate()

    // Loads data.
    val dataset = spark.read.format("libsvm").load(DataFile)

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

  /** ***************************************************************************************/
  // MAIN

  def distance(x: (Double, Double, Double), y: (Double, Double, Double)): Double = {
    return x._1 * y._1 + x._2 * y._2 + x._3 * y._3;
  }

  def minDistPoint(centroidInput: Array[(Double, Double, Double)], pointInput: (Double, Double, Double)): Int = {
    val value = centroidInput.map(x => (x, distance(x, pointInput))).sortBy(p => p._2).min
    return centroidInput.indexOf(value);
  }

  def main(args: Array[String]) {
    println("\n\n>>>>> START <<<<<\n\n");

    val r = scala.util.Random;
    val conf = new SparkConf().setAppName(s"${this.getClass.getSimpleName}");
    val sc = new SparkContext(conf)

    sc.setLogLevel("ERROR")
    var converged = false;
    val rdd = MLUtils.loadLibSVMFile(sc, DataFile)
    // parallelize
    // mapWithPartition
    val rddMapped = rdd.map(
      (x) => (x.features(0), x.features(1), x.features(2))
    )

    val maxCoord = rddMapped.reduce(
      (x, y) =>
        (x._1 max y._1, x._2 max y._2, x._3 max y._3)
    )


    var centroids = (0 to K).toVector.map((x) =>
      (
        (r.nextInt(maxCoord._1.toInt) + r.nextFloat()).toDouble,
        (r.nextInt(maxCoord._2.toInt) + r.nextFloat()).toDouble,
        (r.nextInt(maxCoord._3.toInt) + r.nextFloat()).toDouble
      )
    ).toArray

    println("centroids")
    centroids.foreach(println)
    println("centroids")


    do {
      val newCentroids = rddMapped.map(
          x => (minDistPoint(centroids, x), (x._1, x._2, x._3, 1))
        )
        .reduceByKey(
          (accum, n) => (accum._1 + n._1, accum._2 + n._2, accum._3 + n._3, accum._4 + n._4)
        ).map(
            x => (x._2._1 / x._2._4, x._2._2 / x._2._4, x._2._3 / x._2._4)
        ).collect()

      println("newCentroids")
      newCentroids.foreach(println)

      if (newCentroids.sameElements(centroids)) {
        converged = true;
      }

      centroids = newCentroids;
    }
    while (!converged)

    println("new centroids")
    centroids
    centroids.foreach(println)

    println("\n\n>>>>> END <<<<<\n\n");
  }

}
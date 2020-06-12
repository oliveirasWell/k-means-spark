import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.util.MLUtils

object KMeansSpark {

  val DataFile = "data/sample_kmeans_data.txt"
  val K = 2

  /****************************************************************************************
    Here is the spark example provided at spark documentation
   ****************************************************************************************/
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

  /****************************************************************************************
    Here begin my implementation
  ****************************************************************************************/

  def mapDistanceBetweenPoints(x: (Double, Double, Double), y: (Double, Double, Double)): ((Double, Double, Double), Double) = {
    (x, distance(x, y))
  }

  def distance(p1: (Double, Double, Double), p2: (Double, Double, Double)): Double = {
    val d1 = p1._1 - p2._1
    val d2 = p1._2 - p2._2
    val d3 = p1._3 - p2._3
    d1 * d1 + d2 * d2 + d3 * d3;
  }

  def minDistPoint(centroidInput: Array[(Double, Double, Double)], pointInput: (Double, Double, Double)): Int = {
    val value = {
      centroidInput.map(x => mapDistanceBetweenPoints(x, pointInput)).reduceLeft((p1, p2) => if (p1._2 < p2._2) (p1) else (p2))
    }
    centroidInput.indexOf(value._1);
  }

  def main(args: Array[String]) {
    println("\n\n>>>>> START <<<<<\n\n");

    val RANDOM = scala.util.Random;
    val conf = new SparkConf().setAppName(s"${this.getClass.getSimpleName}");
    val sc = new SparkContext(conf)

    sc.setLogLevel("ERROR")

    var converged = false;
    val rdd = MLUtils.loadLibSVMFile(sc, DataFile)
    val rddMapped = rdd.map(
      (x) => (x.features(0), x.features(1), x.features(2))
    )

    val maxCoord = rddMapped.reduce(
      (x, y) =>
        (x._1 max y._1, x._2 max y._2, x._3 max y._3)
    )

    println("------------------------ Max Coordenates ------------------------")
    println(maxCoord)
    println("-----------------------------------------------------------------")

    var centroids = (0 to K).toVector.map((x) =>
      (
        (RANDOM.nextFloat() * maxCoord._1.toInt).toDouble,
        (RANDOM.nextFloat() * maxCoord._2.toInt).toDouble,
        (RANDOM.nextFloat() * maxCoord._3.toInt).toDouble
      )
    ).toArray

    println("----------- Centroids Created Randonly -----------")
    centroids.foreach(println)
    println("--------------------------------------------------")

    var interation = 0

    while(!converged) {
      var newCentroids = centroids.map(x => (x._1,x._2,x._3))
      val mapped = rddMapped.map(
        x => ( minDistPoint(centroids, x), (x._1, x._2, x._3, 1))
      )

      val reduced = mapped.reduceByKey(
                (accum, n) => (accum._1 + n._1, accum._2 + n._2, accum._3 + n._3, accum._4 + n._4)
      ).collect()

      for( w <- 0 to K)
        {
          try {
            val item = reduced(w)
            val tuple = (item._2._1 / item._2._4, item._2._2 / item._2._4, item._2._3 / item._2._4)
            newCentroids = newCentroids.updated(item._1,tuple)
//            println("newCentroids")
//            println(item._1)
//            println(tuple)
//            println("1newCentroids")
//            newCentroids.foreach(println)
//            println("newCentroids1")
          } catch {
            case unknown: Exception => {
              None
            }
          }
        }

      println(s"----------- Iteration $interation -----------")
      println(s"------------------ Mapped -------------------")
      mapped.foreach(println)
      println(s"------------------ Reduced -------------------")
      reduced.foreach(println)
      println(s"------------------ Old Centroids -------------------")
      centroids.foreach(println)
      println(s"------------------ New Centroids -------------------")
      newCentroids.foreach(println)
      println(s"----------- Iteration $interation End -----------")

      interation = interation +1
      converged = newCentroids sameElements centroids
      centroids = newCentroids.map(x => (x._1,x._2,x._3))
    }
  }
}
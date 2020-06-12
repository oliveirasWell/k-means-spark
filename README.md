# k-means-spark

A k-means implementation using scala and spark of three dimensions points

## Pre Requirements

- Scala version 2.12.11
- Spark version 2.4.0
- Only tested on Linux environment (Ubuntu 19.10)
 
## Variables

- `k` variable to number of centroids, setted at `src/main/scala/br.com.oliveiraswell/k-means-spark.scala:11`
- The input file `data/sample_kmeans_data.txt` using 3d points on the format SVMFile - `0 1:0.0 2:0.0 3:0.0`

## Build

Use the command `sbt package`

## To run:

I recommend you to build and run at the same time using the following command:

```
sbt package && spark-submit ./target/scala-2.12/hello_2.12-1.0.jar
```

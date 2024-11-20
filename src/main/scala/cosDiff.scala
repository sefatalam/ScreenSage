import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import scala.math._
import scala.collection.mutable
// hash functions
import com.google.common.hash.Hashing
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import scala.util.Random

object cosDiff {
  type Signaturematrix = Array[(String, Array[Int])]
  val bandSize = 5
  val numVectors = 25

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("cosDiff").setMaster("local[*]")
    val sc = new SparkContext(conf)

    val spark = SparkSession
      .builder()
      .appName("preparation")
      .getOrCreate()

    import spark.implicits._

    val training_df = spark.read.option("sep", "\t").option("header", value = true)
      .csv(args(0))
      .na.fill(0.0,Seq("avgRuntimeMinutes"))

    val test_df = spark.read.option("sep", "\t").option("header", value = true)
      .csv(args(1))
      .na.fill(0.0, Seq("avgRuntimeMinutes"))
      .limit(2)

    // Cosine Similarity without LSH
    // val cosineRatings: Array[(String, Double)] = test_df.collect().map{ row: Row => predictedRating(training_df,row)}





    /** LSH implementation */
    //Generate comparison vectors
    val vectors: Array[Array[Double]] = (1 to numVectors).map(_ => generateRandomArray()).distinct.toArray

    //Generate signature matrix of movies
    val signatureMatrix: Signaturematrix  = training_df.collect().map{ row:Row => signature(row, vectors)}

    // Call hash function
    val hashed: Map[Int, Seq[String]] = hashing(signatureMatrix, bandSize, 13)

    // Convert hashed data to a suitable Spark data structure
    val hashedRDD = sc.parallelize(hashed.toSeq) // Convert the Map to an RDD
    hashedRDD.cache() // Persist the RDD in memory

    // rating all movies in test_df
    training_df.cache()
    val predictedRatings: Array[(String, Double, Double)] = test_df.collect().map {
      row: Row => predictedRatingLSH(training_df, hashedRDD.collect().toMap, row, vectors)
    }
    training_df.unpersist()
    hashedRDD.unpersist()

    // output
//    cosineRatings.foreach { case (tconst, rating) =>
//      println(s"$tconst: ${"%.1f".format(rating)}")
//    }


//    println(predictedRatings.mkString("Array(", ", ", ")"))

    //Calculating RMSE of cosine similarity
//    val rmseValue = rmse(predictedRatings.map { case (tconst, predicted, time) => (test_df.filter($"tconst" === tconst)
//      .select("averageRating").head().getDouble(0), predicted)
//    })
//
//    println(s"\u001B[32mRMSE: ${rmseValue}\u001B[0m")

  }

  /**
   * Hashes movie signatures using Locality-Sensitive Hashing (LSH) and organizes them into buckets.
   *
   * @param signatureMatrix A map containing all movie signatures.
   * @param bandSize        The size of each band for hashing.
   * @param prime           The prime number used for hashing calculations.
   * @return A map that organizes movie identifiers into buckets based on their hashed signatures.
   */
  def hashing(signatureMatrix: Signaturematrix, bandSize: Int, prime: Int): Map[Int, Seq[String]] = {
    val buckets = mutable.Map.empty[Int, Seq[String]]

    // Split the signatures into bands.
    val bands: Array[(String, Array[Array[Int]])] = signatureMatrix.map {
      case (tconst, signature) => (tconst, splitArray(signature, bandSize))
    }

    // Iterate over all movies in your signature matrix.
    for ((tconst, bandSignatures) <- bands) {
      for ((signature, bandIndex) <- bandSignatures.zipWithIndex) {

        // Hash the band for this movie.
        val bucketIndex = murmurHash(signature)

        // Generate a unique key for the bucket (bandIndex * prime + bucketIndex).
        val uniqueBucketIndex = bandIndex * prime + bucketIndex

        // Add the movie to the corresponding bucket.
        val updatedBucket = buckets.get(uniqueBucketIndex) match {
          case Some(movies) => movies :+ tconst
          case None => Seq(tconst)
        }

        buckets(uniqueBucketIndex) = updatedBucket
      }
    }

    buckets.toMap
  }

  // simple division hash with prime
  def simpleHash(band: Array[Int], prime: Int): Int = {
    band.sum % prime
  }

  // random hash
  def randomHash(band: Array[Int]): Int = {
    val random = new Random()
    random.nextInt()
  }

  // crypto hash
  def sha256Hash(band: Array[Int]): Int = {
    val md = MessageDigest.getInstance("SHA-256")
    val hashBytes = md.digest(band.mkString(",").getBytes)
    val hash = java.nio.ByteBuffer.wrap(hashBytes).getInt
    Math.abs(hash)
  }

  // non crypto hash
  def murmurHash(band: Array[Int]): Int = {
    val hash = Hashing.murmur3_32().hashString(band.mkString(","), StandardCharsets.UTF_8)
    hash.asInt()
  }

  /**
   * Predicts the average rating for a given movie using Locality-Sensitive Hashing (LSH) and cosine similarity.
   *
   * @param allMovies   The dataset containing information about all movies we can compare with.
   * @param hashMap     The map that organizes movie identifiers into buckets based on their hashed signatures.
   * @param movieToRate The input movie for which the average rating is to be predicted.
   * @param vector      The array of arrays representing movie signatures for LSH.
   * @return A tuple containing the "tconst" identifier of the input movie, the predicted average rating,
   *         and the time taken for the prediction in seconds.
   */
  def predictedRatingLSH(allMovies: Dataset[Row], hashMap: Map[Int, Seq[String]], movieToRate: Row, vector: Array[Array[Double]]): (String, Double, Double) = {
    val startTime = System.currentTimeMillis()

    val movieToRateSignatureMatrix: (String, Array[Int]) = signature(movieToRate, vector)
    val hashedMovieToRate: Map[Int, Seq[String]] = hashing(Array(movieToRateSignatureMatrix), bandSize, 13)

    val possiblePairs: Map[String, Seq[String]] = hashedMovieToRate.flatMap { case (bucketIndex, bucket) =>
      hashMap.get(bucketIndex).map(bucketIndex.toString -> _)
    }.toMap

    // Filter the buckets that meet the condition (maximum half the size of allMovies).
    val filteredBuckets = possiblePairs.filter { case (_, bucket) =>
      bucket.size <= allMovies.count() / 2
    }

    // If none of the buckets meet the condition, choose the smallest bucket.
    val selectedBucket = if (filteredBuckets.isEmpty) {
      possiblePairs.minBy { case (_, bucket) => bucket.size }._2
    } else {
      filteredBuckets.values.flatten.toSeq
    }

    // Create combined sequence using selected buckets
    val combinedSequences: Seq[String] = selectedBucket ++ selectedBucket.flatMap(bucketIndex =>
      possiblePairs.getOrElse(bucketIndex, Seq.empty)
    ).toArray.distinct.toSeq

    // Filter all relevant Movies
    val relevantMovies = allMovies.filter(allMovies("tconst").isin(combinedSequences: _*))

    // Fetch cosine similarity for chosen movies
    val cosineSim: Array[(String, Double)] = relevantMovies.drop("averageRating").collect().map { row: Row =>
      val tconst = row.getAs[String]("tconst")
      val similarity = cosineSimilarity(row, movieToRate)
      (tconst, similarity)
    }

    val topSimilarMovies = cosineSim.sorted.reverse.take(10).map { case (tcon, sim) =>
      val averageRatingStr = relevantMovies
        .filter(relevantMovies("tconst") === tcon)
        .select("averageRating")
        .head()
        .getString(0)

      averageRatingStr.toDouble
    }

    val result = topSimilarMovies.sum / 10

    val roundedResult = BigDecimal(result).setScale(1, BigDecimal.RoundingMode.HALF_UP).toDouble

    val endTime = System.currentTimeMillis()
    val predictionTimeMillis = endTime - startTime
    val predictionTimeSeconds = predictionTimeMillis.toDouble / 1000.0

    (movieToRate.getAs[String]("tconst"), roundedResult, predictionTimeSeconds)
  }

  /**
   * Predicts the average rating for a given movie using cosine similarity with other movies in the dataset.
   *
   * @param allMovies   The dataset containing information about all movies.
   * @param movieToRate The input movie for which the average rating is to be predicted.
   * @return A tuple containing the "tconst" identifier of the input movie and the predicted average rating.
   */
  def predictedRating(allMovies: Dataset[Row], movieToRate: Row): (String, Double) = {
    val tconstA = movieToRate.getAs[String]("tconst")

    val similarityArray = allMovies.collect().map { row: Row =>
      val tconstB = row.getAs[String]("tconst")
      val similarityValue = cosineSimilarity(movieToRate, row)
      (tconstB, similarityValue)
    }

    val predictRatings: Array[(String, Double)] = similarityArray.sortBy(-_._2).take(10)

    val ratings = predictRatings.map { case (tconstB, sim) =>
      val averageRating = allMovies
        .filter(allMovies("tconst") === tconstB)
        .select("averageRating")
        .head()
        .getString(0)
        .toDouble
      (averageRating, sim)
    }

    val sum = ratings.map(_._1).sum
    val predictedRating = sum / ratings.length
    (tconstA, predictedRating)
  }

  /**
   * Splits an input array into smaller arrays, each containing a specified number of elements or less.
   *
   * @param inputArr The input array to be split.
   * @param size     The maximum number of elements in each smaller array.
   * @return An array of arrays, where each inner array contains up to 'size' elements from the input array.
   */
  def splitArray(inputArr: Array[Int], size: Int):Array[Array[Int]]= {
    val chunkSize = (inputArr.length + size - 1) / size
    inputArr.grouped(chunkSize).toArray
  }

  /**
   * Calculates the cosine similarity between two rows.
   *
   * @param rowA The first input row containing numeric values for the similarity calculation.
   * @param rowB The second input row containing numeric values for the similarity calculation.
   * @return The cosine similarity between the two input rows, a value between -1.0 and 1.0.
   *         A higher value indicates a higher similarity, with 1.0 indicating identical vectors,
   *         -1.0 indicating completely opposite vectors, and 0 indicating no similarity.
   */
  def cosineSimilarity(rowA: Row, rowB: Row):  Double = {
    val arrRowA = rowA.toSeq.toArray.dropWhile(!_.isInstanceOf[Double]).collect { case value: Double => value }
    val arrRowB = rowB.toSeq.toArray.dropWhile(!_.isInstanceOf[Double]).collect { case value: Double => value }

    dotProduct(arrRowA, arrRowB) / (magnitude(arrRowA) * magnitude(arrRowB))
  }

  /**
   * Calculates the dot product of two arrays of double values.
   *
   * @param x The first input array.
   * @param y The second input array.
   * @return The dot product of the two input arrays.
   */
  def dotProduct(x: Array[Double], y: Array[Double]): Double = {
    (for ((a, b) <- x zip y) yield a * b) sum
  }

  /**
   * Calculates the magnitude (Euclidean norm) of an array of double values.
   *
   * @param x The input array of double values.
   * @return The magnitude of the input array, representing the Euclidean norm.
   */
  def magnitude(x: Array[Double]): Double = {
    math.sqrt(x map (i => i * i) sum)
  }

  /**
   * Calculates the Root Mean Square Error (RMSE) for a set of pairs of actual and predicted values.
   *
   * @param inputArr An array of pairs, where the first Double represents the original rating,
   *                 and the second Double represents the predicted rating.
   * @return The Root Mean Square Error (RMSE) representing the average magnitude of the differences between
   *         the actual and predicted values.
   */
  def rmse(inputArr: Array[(Double,Double)]):Double = {
     // d1 = original Rating, d2 = predicted
      val diff = inputArr.map { case (d1, d2) => pow((d2-d1), 2)}.sum
    sqrt(diff/ inputArr.length)
  }

  /**
   * Generates an array of random double values containing -1.0 and 1.0.
   *
   * @return An array of double values where each element is randomly set to either -1.0 or 1.0.
   */
  def generateRandomArray(): Array[Double] = {
    Array.fill(534)(scala.util.Random.nextInt(2) * 2 - 1)
  }

  /**
   * Generates a signature for a given input row and a set of vectors.
   *
   * @param row     The input data row, which contains all fields.
   * @param vectors An array of arrays representing a set of vectors to compare against the input row data.
   * @return A tuple containing the "tconst" identifier and a sketch array of integers representing the similarity
   *         of the input row to each vector. Each element in the sketch is either -1, 1, or a randomly chosen value.
   *         -1 indicates negative similarity, 1 indicates positive similarity, and random values are used for zero similarity.
   */
  def signature(row: Row, vectors: Array[Array[Double]]): (String, Array[Int]) = {
    val tconst = row.getAs[String]("tconst")
    val vectorValues = row.toSeq.tail.collect { case x: Double => x }.toArray

    val similarityVector = vectors.map(vector => dotProduct(vectorValues, vector))
    val sketch = similarityVector.map { similarity =>
      if (similarity < 0) {
        -1
      } else if (similarity > 0) {
        1
      } else {
        val randomValue = scala.util.Random.nextInt(2) * 2 - 1
        randomValue
      }
    }
    (tconst, sketch)
  }
}

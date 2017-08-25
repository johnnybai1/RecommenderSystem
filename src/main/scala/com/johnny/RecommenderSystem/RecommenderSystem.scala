package com.johnny.RecommenderSystem

import org.apache.spark.ml.linalg.{SparseVector, Vector, Vectors}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by Johnny on 8/21/17.
  */
object RecommenderSystem {

  val GS_ROOT = "gs://recommendersystemstorage/data/"

  def parseLine(line: String): Vector = {
    val ratings = line
      .split("\t")
      .zipWithIndex
      .collect{ case (rating, col) if rating.toInt > 0 => (col.toInt, rating.toDouble) }
    Vectors.sparse(1000, ratings)

  }

  def cosineSimilarity(train: SparseVector, test: SparseVector, movieToPredict: Int): (Double, Double) = {
    if (train.indices.contains(movieToPredict-1)) {
      var count = 0
      var dot = 0.0
      var trainSq = 0.0
      var testSq = 0.0
      val activeRating = train.values(train.indices.indexOf(movieToPredict-1))
      test.foreachActive((mid, testRating) => {
        val trainIdx = train.indices.indexOf(mid)
        if (trainIdx > -1) {
          val trainRating = train.values(trainIdx)
          count += 1
          dot += testRating * trainRating
          trainSq += trainRating * trainRating
          testSq += testRating * testRating
        }
      })
      if (count == 1) {
        (1 - 0.25 * math.abs(math.sqrt(trainSq) - math.sqrt(testSq)), activeRating)
      }
      else if (count > 1) {
        (dot / math.sqrt(trainSq * testSq), activeRating)
      }
      else (0.0, 0.0)
    }
    else (0.0, 0.0)
  }

  def main(args: Array[String]): Unit = {

    if (args.length < 2) {
      sys.error("Usage: [training_file.txt] [test_file.txt]")
      sys.exit(0)
    }
    val TRAIN_PATH = GS_ROOT + args(0)
    val TEST_PATH = GS_ROOT + args(1)

    val conf = new SparkConf()
      .setAppName("Recommender System")
      .setMaster("spark://master:7077")

    val sc =
      new SparkContext(conf)

    val train = sc
      .textFile(TRAIN_PATH)
      .zipWithIndex
      .map { case (line, rowIdx) => (rowIdx.toInt + 1, parseLine(line)) }
      .cache()

    val test = sc
      .textFile(TEST_PATH)
      .map { line =>
        line.split(" ") match {
          case Array(uid, mid, rating) =>
            (uid.toInt, mid.toInt, rating.toDouble)
        }
      }

    val testRatings = test
      .collect { case (uid, mid, rating) if rating > 0 => (uid, (mid, rating)) }
      .groupByKey
      .map { case (uid, ratings) => (uid, Vectors.sparse(1000, ratings.toSeq)) }
    val toPredict = test
      .collect { case (uid, mid, rating) if rating == 0 => (uid, (mid, rating)) }

    toPredict
      .collect
      .map(testEx => {
      val activeUid = testEx._1
      val activeMid = testEx._2._1
      // Ratings as SparseVector
      val activeUserRatings = testRatings
        .lookup(activeUid)
        .head
        .toSparse
      // (activeUid, trainUid, similarity) sorted by similarity
      val weights = train
        .map { case (trainUid, trainRatings) => (activeUid, cosineSimilarity(trainRatings.toSparse, activeUserRatings, activeMid), trainUid) }
        .filter(_._2._1 > 0.0)
        .sortBy(_._2._1, ascending=false)
        .collect
      var dot = 0.0
      var weightSum = 0.0
      if (weights.length < 2) {
        (activeUid, activeMid, math.round(activeUserRatings.values.sum / activeUserRatings.values.length))
      }
      else {
        weights.foreach { case (activeUid, (similarity, movieRating), _) => {
          dot += similarity * movieRating
          weightSum += similarity
        } }
        (activeUid, activeMid, math.round(dot / weightSum))
      }})
      .foreach(println)

    sc.stop()

  }
}



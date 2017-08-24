package com.johnny.RecommenderSystemCFLib

import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by Johnny on 8/21/17.
  */
object RecommenderSystem {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setAppName("Recommender System")
      .setMaster("local")
    val sc = new SparkContext(conf)
    val spark = SparkSession
      .builder()
      .config(conf)
      .getOrCreate()

    import spark.implicits._

    val TRAIN_PATH = "./src/main/data/trainsub.txt";
    val TEST5_PATH = "./src/main/data/test5sub.txt";
    val TEST10_PATH = "./src/main/data/test10.txt";
    val TEST20_PATH = "./src/main/data/test20.txt";

    val ratings = sc
      .textFile(TRAIN_PATH)
      .zipWithIndex
      .flatMap { case (line, rowNum) => Rating.parseRating(rowNum.toInt + 1, line) }
      .toDF()

    val test = sc
      .textFile(TEST5_PATH)
      .map(Rating.parseTestRating)

    val test_ratings = test.filter(_.rating > 0)
    val test_predictions = test.filter(_.rating == 0)

    val als = new ALS()
      .setMaxIter(5)
      .setRegParam(0.01)
      .setUserCol("userId")
      .setItemCol("movieId")
      .setRatingCol("rating")

    val model = als.fit(ratings)
    model.setColdStartStrategy("drop")

    val predictions = model.transform(test_predictions.toDF)
    predictions.collect().foreach(row => println(row.toString()))


  }

}

package com.johnny.RecommenderSystemCFLib

/**
  * Created by Johnny on 8/21/17.
  */
object Rating {

  def parseRating(userId: Int, line: String): Seq[Rating] = line
    .split("\t")
    .zipWithIndex
    .collect { case (rating, index) if rating.toInt > 0 => Rating(userId, index.toInt+1, rating.toInt) }
    .to[Vector]

  def parseTestRating(line: String): Rating = {
    val arr = line.split(" ")
    Rating(arr(0).toInt, arr(1).toInt, arr(2).toInt)
  }
}

case class Rating(userId: Int, movieId: Int, rating: Int)
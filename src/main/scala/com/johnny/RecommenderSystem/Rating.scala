package com.johnny.RecommenderSystem

object Rating {

  def parseTraining(userId: Int, line: String): Seq[Rating] = line
    .split("\t")
    .zipWithIndex
    .collect { case (rating, index) if rating.toInt > 0 => Rating(userId+1, index.toInt+1, rating.toInt) }

  def parseTest(line: String): Rating = {
    val arr = line.split(" ")
    Rating(arr(0).toInt, arr(1).toInt, arr(2).toInt)
  }

}

case class Rating(userId: Int, movieId: Int, rating: Int)
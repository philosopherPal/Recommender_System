import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import java.io.{File, PrintWriter}
import scala.util.Random
import org.apache.spark.rdd.RDD
import scala.math._
import scala.collection.mutable.ListBuffer
import scala.collection.immutable.List
import scala.collection.immutable.Set
import scala.collection.Map
import org.apache.spark.mllib.linalg.{SparseVector, Vectors}
object UserBasedCF {
  def main(args: Array[String]): Unit = {
    val start_time = System.currentTimeMillis()
    val conf = new SparkConf().setAppName("Small").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val inputcsv = sc.textFile(args(0)).filter(row => !row.contains("userID"))
    //var test = sc.textFile("/Users/pallavitaneja/Desktop/inf553-data_mining/Assignment_3/Data/video_small_testing_num.csv")
    val trainInput = inputcsv
      .map(line => {
        val info = line.split(",")
        (info(0).toInt, info(1).toInt)
      })
    val trainRatings = inputcsv
      .map(line => {
        val info = line.split(",")
        ((info(0).toInt, info(1).toInt), info(2).toDouble)
      })
    val test = sc.textFile(args(1)).filter(row => !row.contains("userID"))
      .map(line => {
        val info = line.split(",")
        (info(0).toInt, info(1).toInt)
      })
    val testRating = sc.textFile(args(1)).filter(row => !row.contains("userID"))
      .map(line => {
        val info = line.split(",")
        ((info(0).toInt, info(1).toInt), info(2).toDouble)
      })

    // in the training dataset, all the (userid, productid) pairs that also includes in testing dataset should be substracted
    // there exists (userid, productid) that have different rating values between training and testing dataset, so we cannot substracted based on ratings
    val subt = trainInput.subtract(test)
    //  after substraction, rejoin the ratings into substrated training dataset
    val train = subt.map(x => {
      ((x._1, x._2), 1)
    }).join(trainRatings).map(x => {
      (x._1._1, (x._1._2, x._2._2))
    }).persist()  //  (userID, (productID, ratings))  training data without intersection of testing data
    //var train = sc.textFile("/Users/pallavitaneja/Desktop/inf553-data_mining/Assignment_3/Data/video_small_num.csv")

    //    var header = test.first()
    //    test = test.filter(line => line != header).cache()
    //val testingList = test.collect().toList
    //    var testRating = test.map(_.split(",")).map(line => ((line(0).toInt, line(1).toInt), line(2).toDouble))
    //    var header1 = train.first()
    //    train = train.filter(line => line != header1).cache()
    //    val actual_users = train.map(_.split(",")).map(line => (line(0).toInt, line(1).toInt)).sortByKey()
    //    val actual_ratings = train.map(_.split(",")).map(line => ((line(0).toInt, line(1).toInt), line(2).toDouble)).sortByKey()
    //    val train_users = actual_users.subtract(test_users).sortByKey().map { case (user, product) => ((user, product), 1) }
    val train_ratings = train.map { case x => (x._1, (x._2._1)) }.groupByKey().sortByKey() map { case (k, v) => (k, v.toSet) }
    //    // train_ratings.foreach(println)
    //    val train_ratings1 = actual_ratings.join(train_users).map { case ((user, product), (r1, r2)) => (user, (product, r1)) }.groupByKey().sortByKey().persist()
    val train_ratings2 = train.map { case x=> ((x._1, x._2._1), (x._2._2)) }.sortByKey().persist()
    val train_ratings3 = train.map { case x => ((x._1), (x._2._2, 1)) }.reduceByKey((x, y) => (x._1 + y._1, x._2 + y._2)).mapValues(y => 1.0 * y._1 / y._2)
    //train_ratings3.foreach(println)
    val avgRatingMap = train_ratings3.collectAsMap()
    var setRatingMap = train.collectAsMap()
    //Map[userId,(Pid,Rating)]
    var productMap = train_ratings.collectAsMap()
    //Map[userId, Set(ProductId )]
    var userRatingMap = train_ratings2.collectAsMap()
    //Map[(userId,ProductId),Rating]
    var weightSet: Map[Int, Map[Int, Double]] = Map() //Map[(Int,Int) , weight] for itembased
    for ((user1, ps1) <- productMap) {
      var correlatedUser: Map[Int, Double] = Map()
      for ((user2, ps2) <- productMap) {
        if (user1 < user2) {
          val s = ps1.intersect(ps2)
          //          s1 ++ userRatingMap(user1,ps1)
          //          s2 ++ userRatingMap(user2,ps2)
          if (s.size > 1) {

            val pearsonCorr = pearsonWt(user1, user2, s, userRatingMap)
            if (pearsonCorr != 0.0) {
              correlatedUser += (user2 -> pearsonCorr)
            }


          }
        }
        if (correlatedUser.size > 0) {
          weightSet += (user1 -> correlatedUser)
        }
      }
    } //println(productMap)
    //println(weightSet.size)
    var predictionMap: Map[(Int, Int), Double] = Map() // [(user, product) predictionValue)]
    val testingList = test.collect().toList
    for (item <- testingList) {
      var predictions = prediction(productMap, item._1, item._2, weightSet, avgRatingMap, userRatingMap)
      predictionMap += ((item._1, item._2) -> predictions)
    }

    var predictionRating = sc.parallelize(predictionMap.toList)
    val outFileName = "src/main/Pallavi_Taneja_UserBasedCF.txt"
    val pwrite = new PrintWriter(new File(outFileName))
    var outputList = predictionRating.map(item => {
      //var arr = item._1.toArray
      (item._1._1, item._1._2, item._2)
    }).collect()
    outputList = outputList.sortWith(cmp)
    for (i <- outputList) {
      pwrite.write(i._1+","+ i._2 +","+ i._3+"\n")
    }
    pwrite.close()

    val rateDiff = predictionRating.join(testRating).map { case ((user, product), (r1, r2)) =>
      math.abs(r1 - r2)
    }
    var range1 = rateDiff.filter { case (diff) => (diff) >= 0 && diff < 1 }.count()
    var range2 = rateDiff.filter { case (diff) => (diff) >= 1 && diff < 2 }.count()
    var range3 = rateDiff.filter { case (diff) => (diff) >= 2 && diff < 3 }.count()
    var range4 = rateDiff.filter { case (diff) => (diff) >= 3 && diff < 4 }.count()
    var range5 = rateDiff.filter { case (diff) => diff >= 4 && diff <= 5 }.count()
    var range6 = rateDiff.filter { case (diff) => diff > 5 || diff < 0 }.count()

    var MSE = predictionRating.join(testRating).map { case ((user, product), (r1, r2)) => {
      val err = r1 - r2
      err * err
    }
    }.mean()

    val RMSE = math.sqrt(MSE);
    println(">=0 and <1:" + range1)
    println(">=1 and <2:" + range2)
    println(">=2 and <3:" + range3)
    println(">=3 and <4:" + range4)
    println(">=4:" + range5)
   // println("Outliers:" + range6)
    println("RMSE = " + RMSE)


    // println(setMap)

    //val avgUserRating =
    //println(train_ratings.count())
    //train_ratings.foreach(println)


    val end_time = System.currentTimeMillis()
    println("Time: " + (end_time - start_time) / 1000 + " secs")
  }

  def pearsonWt(user1: Int, user2: Int, s: Set[Int], userRatingMap: scala.collection.Map[(Int, Int), Double]): Double = {
    var numerator = 0.0
    var denominator1 = 0.0
    var denominator2 = 0.0
    var meanUser1 = 0.0
    var meanUser2 = 0.0
    var sumUser1 = 0.0
    var sumUser2 = 0.0
    for (item <- s) {
      sumUser1 += userRatingMap(user1, item)
      sumUser2 += userRatingMap(user2, item)
    }
    meanUser1 = sumUser1 / (s.size)
    meanUser2 = sumUser2 / (s.size)
    for (item <- s) {
      numerator += (userRatingMap(user1, item) - meanUser1) * (userRatingMap(user2, item) - meanUser2)
      denominator1 += math.pow((userRatingMap(user1, item) - meanUser1), 2)
      denominator2 += math.pow((userRatingMap(user2, item) - meanUser2), 2)
    }

    var denominator = math.pow(denominator1, 0.5) * math.pow(denominator2, 0.5)
    var pcWeightUV: Double = 0.0
    if (denominator != 0) {
      pcWeightUV = numerator / denominator
    }

    pcWeightUV
  }

  def prediction(productMap: Map[Int, Set[Int]], uid: Int, pid: Int, weightSet: Map[Int, Map[Int, Double]],
                 avgRating: Map[Int, Double], userRatingMap: Map[(Int, Int), Double]): Double = {


    var numerator: Double = 0.0
    var denominator: Double = 0.0
    var l: Int = 0
    var m: Int = 0
    val u1Avg = avgRating(uid)
    var pav: Double = u1Avg
    try {
      val users: Set[Int] = productMap(pid)
      // users that rate this product
      for (ru <- users) {
        l = ru
        if (weightSet.contains(uid)) {
          if (weightSet(uid).contains(ru)) {
            numerator += (userRatingMap((ru, pid)) - avgRating(ru)) * weightSet(uid)(ru)
            denominator += abs(weightSet(uid)(ru))
          }
        } else {
          if (weightSet.contains(ru)) {
            if (weightSet(ru).contains(uid)) {
              numerator += (userRatingMap((ru, pid)) - avgRating(ru)) * weightSet(ru)(uid)
              denominator += abs(weightSet(ru)(uid))
            }
          }
        }
      }


      if (denominator != 0) {
        pav = u1Avg + numerator / denominator
      }
      else {
        pav = u1Avg
      }
    }catch {
      case e : NoSuchElementException => {

      }
    }
    pav
  }
  def cmp(e1:(Int, Int, Double), e2:(Int, Int, Double)): Boolean = {
    var res:Boolean = false
    if (e1._1 != e2._1) {
      res = e1._1 < e2._1
    } else {
      res = e1._2 < e2._2
    }

    res
  }
}




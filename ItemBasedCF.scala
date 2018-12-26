import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import scala.util.Random
import org.apache.spark.rdd.RDD
import scala.collection.Map
import java.io.{File, PrintWriter}
import scala.collection.mutable.ListBuffer
import scala.collection.immutable.List
import scala.collection.immutable.Set
import scala.math._
import org.apache.spark.mllib.linalg.{SparseVector, Vectors}
object ItemBasedCF {
  def main(args: Array[String]): Unit = {
    val start_time = System.currentTimeMillis()
    val conf = new SparkConf().setAppName("Small").setMaster("local")
    val sc = new SparkContext(conf)
    val inputcsv = sc.textFile(args(0)).filter(row => !row.contains("userID"))

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


    val subt = trainInput.subtract(test)

    val train = subt.map(x => {
      ((x._1, x._2), 1)
    }).join(trainRatings).map(x => {
      (x._1._1, (x._1._2, x._2._2))
    }).persist()  //  (userID, (productID, ratings))  training data without intersection of testing data
    //train.foreach(println)
    val productRating = train.map{case (x) => ((x._2._1),(x._2._2))}
    val avgProductRating = productRating.mapValues(value => (value, 1)) // map entry with a count of 1
      .reduceByKey {
      case ((sumL, countL), (sumR, countR)) =>
        (sumL + sumR, countL + countR)
    }
      .mapValues {
        case (sum , count) => sum / count
      }.collectAsMap()

    //avgProductRating.foreach(println)
    var userRating = train.map(x => ((x._1,x._2._1),x._2._2))
    var userRatingMap = userRating.collectAsMap()//[(userId, Pid),Rating]

    val csv = sc.textFile(args(0))
    val header = csv.first()
    val productList = train.map{case(x)=>(x._2._1,x._1)}.groupByKey().sortByKey().map { case (productId, userId) => (productId, userId.toList) }
    val productSet = train.map{case (x) => (x._1,x._2._1)}.groupByKey().sortByKey().map{case(userId, productId) => (userId, productId.toSet)}
    //data.foreach(println)
    //val data1=data.groupBy()
    //var productList = data.map(inp => (inp(1).toInt, inp(0).toInt)).groupByKey().sortByKey().map { case (productId, userId) => (productId, userId.toList) }
    var list = productList.collect().toList
    val userSet = train.map{case(x)=>(x._2._1,x._1)}.groupByKey().sortByKey().map { case (productId, userId) => (productId, userId.toSet) }

    //productList.foreach(println)
    val userSetMap = userSet.collectAsMap() // ProductMap[Pid,Set(Uid)]
    val productSetMap = productSet.collectAsMap()
    val users = train.map((inp => inp._1))
    val m: Int = users.count().toInt
    var hashFuncList = new ListBuffer[HashFunctions]
    val numHashes: Int = 600
    for (i <- 0 to numHashes - 1) {
      val a = new Random().nextInt(m)
      if (gcd(a, m) == 1) {
        val b = new Random().nextInt(m)
        //println(b)
        hashFuncList += new HashFunctions(a, b, m)
      }

    }
    var hashArray = hashFuncList.toArray
    val signatures = productList.map(record => {
      val sigLb: ListBuffer[Int] = new ListBuffer[Int]
      for (h <- 0 to hashArray.length - 1) {
        val lb: ListBuffer[Int] = new ListBuffer[Int]
        for (u <- record._2) {
          lb += hashArray(h).hash(u)
        }
        sigLb += hashArray(h).minhash(lb.toList)
      }
      (record._1, sigLb.toList)
    })

    val rows = 4
    val candidates = signatures.flatMap(x => {
      x._2.grouped(rows).zipWithIndex.map(y => {
        ((y._2, y._1), x._1)
      }).toList
    }).groupByKey().filter(_._2.size > 1).map(_._2.toSet).distinct().flatMap(_.subsets(2)).distinct()
    //candidates.foreach(println)
    // val candidateSet = candidates.collect().toList
    var candidateTuple = candidates.map(x =>{
      var y = x.toArray

      (y(0),y(1))
    })
    var camdidateTup = candidateTuple.collect()
    var weightSet: Map[(Int,Int), Double] = Map()
    for((i1,i2) <- camdidateTup){
      val s  = userSetMap(i1).intersect(userSetMap(i2))
      if(s.size > 1){
        val pearsonCorr = pearsonWt(i1, i2, s, userRatingMap)
        if (pearsonCorr != 0.0) {
          weightSet += ((i1 , i2) -> pearsonCorr)
        }
      }
    }
    // println(weightSet)

    var predictionMap: Map[(Int, Int), Double] = Map() // [(user, product) predictionValue)]
    val testingList = test.collect().toList
    for (item <- testingList) {
      var predictions = prediction(item._1, item._2, productSetMap, weightSet, avgProductRating, userRatingMap)
      predictionMap += ((item._1, item._2) -> predictions)
    }

    var predictionRating = sc.parallelize(predictionMap.toList)
    //    val rateDiff = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
    //      math.abs(r1 - r2)}
    val outFileName = "src/main/Pallavi_Taneja_ItemBasedCF.txt"
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
    //println("Outliers:" + range6)
    println("RMSE = " + RMSE)

    val end_time = System.currentTimeMillis()
    println("Time: " + (end_time - start_time) / 1000 + " secs")

  }
  def pearsonWt(i1: Int, i2: Int, corratedUser: Set[Int], userRatingMap: Map[(Int, Int), Double]) : Double ={
    var res:Double =0.0
    var numerator = 0.0
    var denominator1 = 0.0
    var denominator2 = 0.0
    var meanI1 = 0.0
    var meanI2 = 0.0
    var sumI1 = 0.0
    var sumI2 = 0.0
    for(u <- corratedUser){
      sumI1 += userRatingMap(u , i1)
      sumI2 += userRatingMap(u , i2)
    }
    meanI1 = sumI1/corratedUser.size
    meanI2 = sumI2/corratedUser.size
    for(u <- corratedUser){
      numerator += (userRatingMap(u,i1) - meanI1 ) * (userRatingMap(u,i2) - meanI2)
      denominator1 += pow((userRatingMap(u,i1) - meanI1),2)
      denominator2 += pow((userRatingMap(u,i2) - meanI2),2)
    }
    if(denominator1 != 0 && denominator2 != 0){
      res = numerator * 1.0 / (sqrt(denominator1) * sqrt(denominator2))
    }
    res
  }
  def prediction(uid:Int, pid:Int, productSetMap:Map[Int,Set[Int]], weightSetMap:Map[(Int,Int),Double], avgRating:Map[Int,Double], userRatingMap:Map[(Int,Int),Double]): Double ={
    val avgI1 = avgRating(pid)
    var numerator: Double = 0.0
    var denominator: Double = 0.0
    var predict: Double = avgI1
    val productSet:Set[Int] = productSetMap(uid)
    for(item <- productSet){
      if(weightSetMap.contains((pid,item)))
      {
        numerator += userRatingMap(uid,item) * weightSetMap((pid,item))
        denominator += abs(weightSetMap((pid,item)))
      }
      else{
        if(weightSetMap.contains((item,pid))) {
          numerator += userRatingMap(uid,item) * weightSetMap((item,pid))
          denominator += abs(weightSetMap((item,pid)))
        }
      }

    }
    if (denominator != 0) {
      predict = numerator / denominator
    }
    else {
      predict = avgI1
    }
    predict

  }
  class HashFunctions(a: Int, b: Int, m: Int) extends Serializable {
    var ha = a
    var hb = b
    var hm = m

    def hash(x: Int): Int = {
      (ha * x + hb) % hm
    }

    def minhash(v: List[Int]): Int = {
      v.min
    }
  }
  def gcd(a: Int, b: Int): Int = {
    if (b == 0) a else gcd(b, a % b)
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


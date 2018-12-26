import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import scala.util.Random
import org.apache.spark.rdd.RDD
import java.io.{File, PrintWriter}
import scala.collection.mutable.ListBuffer
import scala.collection.immutable.List
import scala.collection.immutable.Set
import org.apache.spark.mllib.linalg.{SparseVector, Vectors}
object JaccardLSH {
  def main(args: Array[String]): Unit = {
    val start_time = System.currentTimeMillis()
    val conf = new SparkConf().setAppName("Small").setMaster("local")
    val sc = new SparkContext(conf)
    val csv = sc.textFile(args(0))
    val header = csv.first()
    val data = csv.filter(row => row != header)
    //val data1=data.groupBy()
    var productList = data.map(line => line.split(",")).map(inp => (inp(1).toInt, inp(0).toInt)).groupByKey().sortByKey().map { case (productId, userId) => (productId, userId.toList) }
    var list = productList.collect().toList
    //    productList.foreach(println)

    val users = data.map(line =>line.split(",").map(inp=>inp(0)))
    val m : Int = users.count().toInt
    var hashFuncList = new  ListBuffer[HashFunctions]
    val numHashes: Int = 600
    for(i <- 0 to numHashes-1){
      val a = new Random().nextInt(m)
      if(gcd(a,m)==1)
      {
        val b=new Random().nextInt(m)
        //println(b)
        hashFuncList += new HashFunctions(a,b,m)
      }

    }
    var hashArray = hashFuncList.toArray
    val signatures = productList.map(record => {
      val sigLb:ListBuffer[Int] = new ListBuffer[Int]
      for (h <- 0 to hashArray.length - 1) {
        val lb:ListBuffer[Int] = new ListBuffer[Int]
        for (u <- record._2) {
          lb += hashArray(h).hash(u)
        }
        sigLb += hashArray(h).minhash(lb.toList)
      }
      (record._1, sigLb.toList)
    })
    val productMap = productList.collectAsMap()
    //signatures.foreach(println)

    //    for(i <- 0 to hashArray.length-1){
    //      hashArray(i) * productList._2._2(i)
    //    }
    //((rows,band),candidatePairs)
    val rows = 4
    val candidates = signatures.flatMap(x=>{
      x._2.grouped(rows).zipWithIndex.map(y=>
      {((y._2,y._1),x._1)
      }).toList
    }).groupByKey().filter(_._2.size>1).map(_._2.toSet).distinct().flatMap(_.subsets(2)).distinct()
    //    candidates.foreach(println)
    var result1 = candidates.map(row=>{
      var cand = row.toList
      var cand0 = productMap(cand(0))
      var cand1 = productMap(cand(1))
      var jaccard = cand0.intersect(cand1).size * 1.0 / cand0.union(cand1).distinct.size
      (row, jaccard)
      //      }).filter(_._2 >= 0.5).map(_._1)
    }).filter(_._2 >= 0.5)
    var result = result1.map(_._1)



    //var result = candidates
//    val res_count = result.count()
//    //    println(res_count)
//    // result.take(10).foreach(println)
//    val GroundTruth = sc.textFile("/Users/pallavitaneja/Desktop/inf553-data_mining/Assignment_3/Data/video_small_ground_truth_jaccard.csv")
//    val header1 = GroundTruth.first()
//    val GTData = GroundTruth.filter(row => row != header1)
//    var truthSet = GTData.map(row => {
//      var x = row.split(",")
//      Set(x(0).toInt, x(1).toInt)
//    })
//    val num_truth = truthSet.count()
//    //    println(num_truth)
//    val tp = truthSet.intersection(result).count()
//    println("candCount:  "+candidates.count())
//    println("TP:"+tp)
//    println("GTCount:   "+num_truth)
//    println("JaccCandcount:   "+res_count)
//    println("precision:   "+ tp*1.0/res_count)
//    println("recall:    "+ tp*1.0/num_truth)
    val outFileName = "src/main/Pallavi_Taneja_SimilarProducts_Jaccard.txt"
    val pwrite = new PrintWriter(new File(outFileName))
    var outputList = result1.map(item => {
      var arr = item._1.toArray
      (arr(0), arr(1), item._2)
    }).collect()
    outputList = outputList.sortWith(cmp)
    for (i <- outputList) {
      pwrite.write(i._1+","+ i._2 +","+ i._3+"\n")
    }
    pwrite.close()
    val end_time = System.currentTimeMillis()
    println("Time: " + (end_time - start_time) / 1000 + " secs")
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

  class HashFunctions(a: Int, b: Int, m: Int) extends Serializable {
    var ha = a
    var hb = b
    var hm = m

    def hash(x: Int) : Int = {
      (ha * x + hb) % hm
    }

    def minhash(v: List[Int]): Int = {
      v.min
    }
  }
  def gcd(a: Int,b: Int): Int= {
    if(b ==0) a else gcd(b, a%b)
  }
}


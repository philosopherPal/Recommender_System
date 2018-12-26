import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import java.io.{File, PrintWriter}
object ModelBasedCF{

  def main(args: Array[String])={
    val start_time = System.currentTimeMillis()
    val conf = new SparkConf().setAppName("Small").setMaster("local")
    val sc = new SparkContext(conf)
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)
    var test = sc.textFile(args(1))
    var header = test.first()
    test = test.filter(line => line!=header).cache()
    var test_users =test.map(_.split(",")).map(line => (line(0).toInt,line(1).toInt)).sortByKey()


    var train = sc.textFile(args(0))
    var header1 = train.first()
    train = train.filter(line => line!=header1).cache()
    var actual_users =train.map(_.split(",")).map(line => (line(0).toInt,line(1).toInt)).sortByKey()
    val actual_ratings = train.map(_.split(",")).map(line => ((line(0).toInt,line(1).toInt),line(2).toDouble)).sortByKey()

    var train_users = actual_users.subtract(test_users).sortByKey().map{case (user,product)=>((user,product),1)}

    var train_ratings = actual_ratings.join(train_users).map{ case ((user,product),(r1,r2)) => Rating(user.toInt,product.toInt,r1.toDouble)}

    val ratings = train.map(_.split(',') match { case Array(userId,productId,ratings,l) =>
      Rating(userId.toInt, productId.toInt, ratings.toDouble)})

    val ratings1 =test.map(_.split(',') match { case Array(userId,productId,ratings) =>
      Rating(userId.toInt, productId.toInt, ratings.toDouble)})
    // Build the recommendation model using ALS
    val rank = 2
    val numIterations = 10
    val model = ALS.train(train_ratings, rank, numIterations, 0.36, 1, 4)

    // Evaluate the model on rating data

    val predictions =
      model.predict(test_users).map { case Rating(user,product,rating) =>
        ((user, product), rating)
      }
    val maxPred = predictions.map(_._2).max()
    val minPred = predictions.map(_._2).min()
    val scales = predictions.map(x =>{
      var scaling = ((x._2 - minPred) / (maxPred - minPred)) * 4 + 1
      (x._1, scaling)
    })

    val pred_users = predictions.map{ case ((userId,productId),rating) => (userId,productId)}
    //println(pred_users.count())

    //val predAvgRating = predictions.map{case ((user,movie),rating) => (user,(rating,1))}.reduceByKey((x,y) => (x._1+y._1,x._2+y._2)).mapValues(y=> 1.0*y._1/y._2).sortByKey();
    //val missingUsers = test_users.subtract(pred_users).join(predAvgRating).map{ case (user,(product,rate)) => ((user,product),rate)}
    var ratesAndPreds = ratings1.map { case Rating(user, product, rating) =>
      ((user, product), rating)
    }.join(scales)
    // .union(missingUsers)).sortByKey()

    //    var pred_avg_rating = ratesAndPreds.map{case ((user,product),(r1,r2))=>r2}.mean()
    //    var Outliers1= ratesAndPreds.filter{case ((user,product),(r1,r2) ) => r2>5}
    //    //Outliers = Outliers.map{case ((user,product),(r1,r2) ) => ((user,product),(r1,math.abs(r2-r1+pred_avg_rating)))}
    //    Outliers1 = Outliers1.map{case ((user,product),(r1,r2) ) => ((user,product),(r1,(pred_avg_rating)))}
    //    var Outliers2= ratesAndPreds.filter{case ((user,product),(r1,r2) ) => r2<0}
    //    Outliers2 = Outliers1.map{case ((user,product),(r1,r2) ) => ((user,product),(r1,(pred_avg_rating)))}
    //    ratesAndPreds = ratesAndPreds.filter{case ((user,product),(r1,r2) ) => r2>=0 && r2<=5}
    //    ratesAndPreds = ratesAndPreds.union(Outliers1)
    //    ratesAndPreds = ratesAndPreds.union(Outliers2)
    val rateDiff = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
      math.abs(r1 - r2)}
    // ratesAndPreds.foreach(println)
    //val result = ratesAndPreds.map{case ((user,product),(r1,r2) )=> user+","+product+","+r2}.persist()

    val result = ratesAndPreds.map{case ((user,product),(r1,r2) )=> user+","+product+","+r2}persist()
    // result.collect().foreach(println)
    //result.coalesce(1).foreach(println)//.saveAsTextFile("Result")
    var range1 = rateDiff.filter{ case (diff) => (diff)>=0 && diff<1}.count()
    var range2 = rateDiff.filter{ case (diff) => (diff)>=1 && diff<2}.count()
    var range3 = rateDiff.filter{ case (diff) => (diff)>=2 && diff<3}.count()
    var range4 = rateDiff.filter{ case (diff) => (diff)>=3 && diff<4}.count()
    var range5 = rateDiff.filter{ case (diff) => diff>=4 && diff<=5}.count()
    var range6 = rateDiff.filter{case (diff) => diff>5 || diff<0}.count()

    val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
      val err = (r1 - r2)
      err * err
    }.mean();
    val RMSE = math.sqrt(MSE);
    println(">=0 and <1:"+range1)
    println(">=1 and <2:"+range2)
    println(">=2 and <3:"+range3)
    println(">=3 and <4:"+range4)
    println(">=4:"+range5)
    //println("Outliers:"+range6)
    println("RMSE = "+ RMSE)
    val end_time = System.currentTimeMillis()
    println("Time: " + (end_time - start_time)/1000 + " secs")
    val outFileName = "src/main/Pallavi_Taneja_ModelBasedCF.txt"
    val pwrite = new PrintWriter(new File(outFileName))
    var outputList = ratesAndPreds.map(item => {
      //var arr = item._1.toArray
      (item._1._1, item._1._2, item._2._2)
    }).collect()
    outputList = outputList.sortWith(cmp)
    for (i <- outputList) {
      pwrite.write(i._1+","+ i._2 +","+ i._3+"\n")
    }
    pwrite.close()

    //    val out = result.sortBy(x => (x(0),x(1)))
    //    for (o <- out.collect()){
    //      pwrite.write(o+"\n")
    //    }
    //    pwrite.close()

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


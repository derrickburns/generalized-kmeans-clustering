package com.massivedatascience.clusterer.util

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{Logging, SparkContext}

import scala.reflect.ClassTag

trait SparkHelper extends Logging {

  /**
   * Names and RDD, caches data in memory, and ensures that it is in computed synchronously
   * @param name name to apply to RDD
   * @param data rdd
   * @tparam T type of data in RDD
   * @return the input RDD
   */
  def sync[T](name: String, data: RDD[T], synchronous: Boolean = true): RDD[T] = {
    data.setName(name).cache()
    if (synchronous) {
      val count = data.count()
      logInfo(s"have $count items of RDD ${data.name}")
    }
    data
  }

  def exchange[T](name: String, from: RDD[T])(f: RDD[T] => RDD[T]): RDD[T] = {
    val to = f(from)
    from.unpersist()
    to.setName(name).persist()
  }

  def withCached[T, Q](
    names: Seq[String],
    rdds: Seq[RDD[T]])(f: Seq[RDD[T]] => Q): Q = {

    rdds.zip(names).foreach { case (r, n) => sync(n, r, true)}
    val result = f(rdds)
    rdds.foreach(_.unpersist())
    result
  }


  def withCached[T, Q](
    name: String, v: RDD[T],
    blocking: Boolean = false,
    synchronous: Boolean = true)(f: RDD[T] => Q): Q = {

    sync(name, v, synchronous)
    val result = f(v)
    if (v.getStorageLevel != StorageLevel.NONE) v.unpersist(blocking)
    result
  }

  def withNamed[T, Q](name: String, v: RDD[T])(f: RDD[T] => Q): Q = {
    noSync(name, v)
    f(v)
  }

  def withBroadcast[T: ClassTag, Q](v: T)(f: Broadcast[T] => Q)(implicit sc: SparkContext): Q = {
    val broadcast = sc.broadcast(v)
    val result = f(broadcast)
    broadcast.unpersist()
    result
  }

  def noSync[T](name: String, data: RDD[T]): RDD[T] = data.setName(name)
}



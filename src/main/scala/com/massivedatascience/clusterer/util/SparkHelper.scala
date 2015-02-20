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
  def sync[T](name: String, data: RDD[T], synchronous: Boolean = true)(implicit sc: SparkContext): RDD[T] = {
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
    name: String, v: RDD[T],
    blocking: Boolean = false,
    synchronous: Boolean = true)(f: RDD[T] => Q)(implicit sc: SparkContext): Q = {

    sync(name, v, synchronous)
    val result = f(v)
    if (v.getStorageLevel != StorageLevel.NONE) v.unpersist(blocking)
    result
  }

  def withNamed[T, Q](name: String, v: RDD[T])(f: RDD[T] => Q)(implicit sc: SparkContext): Q = {
    noSync(name, v)
    f(v)
  }

  def withBroadcast[T: ClassTag, Q](v: T)(f: Broadcast[T] => Q)(implicit sc: SparkContext): Q = {
    val broadcast = sc.broadcast(v)
    val result = f(broadcast)
    broadcast.unpersist()
    result
  }

  def noSync[T](name: String, data: RDD[T])(implicit sc: SparkContext): RDD[T] = data.setName(name)

}



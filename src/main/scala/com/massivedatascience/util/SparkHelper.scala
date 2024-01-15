/*
 * Licensed to the Massive Data Science and Derrick R. Burns under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Massive Data Science and Derrick R. Burns licenses this file to You under the
 * Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.massivedatascience.util

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.SparkContext

import scala.reflect.ClassTag

trait SparkHelper {

  /**
   * Names and RDD, caches data in memory, and ensures that it is in computed synchronously
   * @param name name to apply to RDD
   * @param data rdd
   * @tparam T type of data in RDD
   * @return the input RDD
   */
  protected def sync[T](name: String, data: RDD[T], synchronous: Boolean = true): RDD[T] = {
    data.setName(name).cache()
    if (synchronous) {
      data.foreachPartition(p => None)
    }
    data: RDD[T]
  }

  protected def exchange[T](name: String, from: RDD[T])(f: RDD[T] => RDD[T]): RDD[T] = {
    val to: RDD[T] = sync[T](name, f(from))
    from.unpersist()
    to: RDD[T]
  }

  protected def withCached[T, Q](
    names: Seq[String],
    rdds: Seq[RDD[T]])(f: Seq[RDD[T]] => Q): Q = {

    rdds.zip(names).foreach { case (r, n) => sync[T](n, r) }
    val result = f(rdds)
    rdds.foreach(_.unpersist())
    result
  }

  protected def sideEffect[T](v: T)(f: T => Unit): T = {
    f(v)
    v
  }

  protected def withCached[T, Q](
    name: String, v: RDD[T],
    blocking: Boolean = false,
    synchronous: Boolean = true)(f: RDD[T] => Q): Q = {

    sync(name, v, synchronous)
    val result = f(v)
    if (v.getStorageLevel != StorageLevel.NONE) v.unpersist(blocking)
    result
  }

  protected def withNamed[T, Q](name: String, v: RDD[T])(f: RDD[T] => Q): Q = {
    noSync(name, v)
    f(v)
  }

  protected def withBroadcast[T: ClassTag, Q](v: T)(f: Broadcast[T] => Q)(implicit sc: SparkContext): Q = {
    val broadcast = sc.broadcast(v)
    val result = f(broadcast)
    broadcast.unpersist()
    result
  }

  protected def noSync[T](name: String, data: RDD[T]): RDD[T] = data.setName(name)
}


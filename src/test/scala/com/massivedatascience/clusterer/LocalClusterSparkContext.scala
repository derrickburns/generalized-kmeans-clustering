/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * This code is a modified version of the original Spark 1.0.2 implementation.
 */

package com.massivedatascience.clusterer

import org.scalatest.{ Suite, BeforeAndAfterAll }

import org.apache.spark.{ SparkConf, SparkContext }

trait LocalClusterSparkContext extends BeforeAndAfterAll {
  self: Suite =>
  @transient var sc: SparkContext = _

  override def beforeAll() {

    val conf = new SparkConf()
    conf.set("spark.broadcast.compress", "false")
    conf.set("spark.shuffle.compress", "false")
    conf.set("spark.shuffle.spill.compress", "false")
    conf.set("spark.master", "local")
    conf.setAppName("test-cluster")
    conf.set("spark.akka.frameSize", "1") 
    sc = new SparkContext(conf)
    super.beforeAll()
  }

  override def afterAll() {
    if (sc != null) {
      sc.stop()
    }
    super.afterAll()
  }
}
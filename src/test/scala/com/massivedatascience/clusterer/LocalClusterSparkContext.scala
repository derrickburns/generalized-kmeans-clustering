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
import org.apache.spark.sql.SparkSession

object SharedSparkContext {
  @volatile private var _sparkContext: SparkContext = _
  @volatile private var _sparkSession: SparkSession = _
  private val lock = new Object()
  
  def getOrCreate(): SparkContext = {
    if (_sparkContext == null || _sparkContext.isStopped) {
      lock.synchronized {
        if (_sparkContext == null || _sparkContext.isStopped) {
          // Stop any existing context first
          if (_sparkContext != null && !_sparkContext.isStopped) {
            _sparkContext.stop()
          }
          
          val conf = new SparkConf()
            .setMaster("local[2]")  // Use 2 cores for better parallelism
            .setAppName("test-cluster")
            .set("spark.broadcast.compress", "false")
            .set("spark.shuffle.compress", "false") 
            .set("spark.shuffle.spill.compress", "false")
            .set("spark.rpc.message.maxSize", "1")  // Updated from deprecated spark.akka.frameSize
            .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .set("spark.sql.warehouse.dir", System.getProperty("java.io.tmpdir"))
            .set("spark.driver.host", "127.0.0.1")  // Use explicit IP to avoid hostname resolution warnings
            .set("spark.driver.bindAddress", "127.0.0.1")
            .set("spark.ui.enabled", "false")  // Disable UI for tests
            .set("spark.sql.adaptive.enabled", "false")  // Disable adaptive query execution
            .set("spark.sql.adaptive.coalescePartitions.enabled", "false")
            .set("spark.sql.execution.arrow.pyspark.enabled", "false")  // Disable Arrow for tests
            .set("spark.sql.execution.arrow.sparkr.enabled", "false")   // Disable Arrow for R
            .set("spark.hadoop.fs.file.impl.disable.cache", "true")     // Disable file system cache for tests
            .set("spark.sql.codegen.wholeStage", "false")               // Disable code generation for tests
            .set("spark.sql.codegen.fallback", "true")                  // Enable fallback for code generation
            .set("spark.sql.catalogImplementation", "in-memory")        // Use in-memory catalog for tests
            .set("spark.eventLog.enabled", "false")                     // Disable event logging for tests
            .set("spark.dynamicAllocation.enabled", "false")            // Disable dynamic allocation
            .set("spark.serializer.objectStreamReset", "1")             // Reset object stream frequently
            
          _sparkContext = new SparkContext(conf)
          _sparkSession = SparkSession.builder().config(conf).getOrCreate()
        }
      }
    }
    _sparkContext
  }
  
  def getSparkSession(): SparkSession = {
    getOrCreate()  // Ensure context is created
    _sparkSession
  }
  
  def stop(): Unit = {
    lock.synchronized {
      if (_sparkSession != null) {
        _sparkSession.stop()
        _sparkSession = null
      }
      if (_sparkContext != null && !_sparkContext.isStopped) {
        _sparkContext.stop()
        _sparkContext = null
      }
    }
  }
}

trait LocalClusterSparkContext extends BeforeAndAfterAll {
  self: Suite =>
  @transient var sc: SparkContext = _

  override def beforeAll() {
    sc = SharedSparkContext.getOrCreate()
    super.beforeAll()
  }

  override def afterAll() {
    // Don't stop the shared context here - let the JVM shutdown hook handle it
    super.afterAll()
  }
}
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
 * This code is a modified version of the original Spark 1.0.2 K-Means implementation.
 */

package com.massivedatascience.clusterer

import com.massivedatascience.linalg.WeightedVector
import com.massivedatascience.transforms.Embedding
import org.apache.spark.mllib.linalg.Vectors
import org.scalatest.FunSuite
import com.massivedatascience.clusterer.TestingUtils._


class EmbeddingSuite extends FunSuite with LocalSparkContext {

  test("apply method") {
    Embedding(Embedding.SYMMETRIZING_KL_EMBEDDING)
    Embedding(Embedding.DENSE_EMBEDDING)
    Embedding(Embedding.HAAR_EMBEDDING)
    Embedding(Embedding.HIGH_DIMENSIONAL_RI)
    Embedding(Embedding.MEDIUM_DIMENSIONAL_RI)
    Embedding(Embedding.LOW_DIMENSIONAL_RI)
    Embedding(Embedding.IDENTITY_EMBEDDING)
  }

  test("symmetrized embedding")  {

    val e = Embedding(Embedding.SYMMETRIZING_KL_EMBEDDING)
    val v1 = WeightedVector(Vectors.dense(4.0), 8.0)

    val expected = Vectors.dense(5.38629436111989)

    val embedded = e.embed(v1)

    assert( embedded.weight == 8.0 )
    assert( embedded.homogeneous ~== expected absTol 1E-5)

  }

  test("random indexing embedding")  {
    val e = Embedding(Embedding.LOW_DIMENSIONAL_RI)
    val v1 = WeightedVector(Vectors.sparse(10, Seq( (1, 2.0), (3, 6.0))), 3.0)

    val expected = Vectors.sparse(64, Seq( ( 10, 6.0), (11, -2.0), (13, -6.0), (14, 2.0)))
    val embedded = e.embed(v1)

    assert( embedded.weight == 3.0 )
    assert( embedded.homogeneous ~== expected absTol 1E-5)
  }

}

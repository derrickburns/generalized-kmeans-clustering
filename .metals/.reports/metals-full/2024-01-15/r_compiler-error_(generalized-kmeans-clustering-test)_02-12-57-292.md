file://<WORKSPACE>/src/test/scala/com/massivedatascience/clusterer/EmbeddingSuite.scala
### scala.reflect.internal.Types$TypeError: illegal cyclic reference involving class EmbeddingSuite

occurred in the presentation compiler.

action parameters:
offset: 1215
uri: file://<WORKSPACE>/src/test/scala/com/massivedatascience/clusterer/EmbeddingSuite.scala
text:
```scala
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
import org.scalatest._
import funsuite._
import com.holdenkarau.spark.testing.LocalSparkContext

class EmbeddingSuite extends AnyFunS@@uite with LocalSparkContext {

  test("apply method") {
    Embedding(Embedding.SYMMETRIZING_KL_EMBEDDING)
    Embedding(Embedding.DENSE_EMBEDDING)
    Embedding(Embedding.HAAR_EMBEDDING)
    Embedding(Embedding.HIGH_DIMENSIONAL_RI)
    Embedding(Embedding.MEDIUM_DIMENSIONAL_RI)
    Embedding(Embedding.LOW_DIMENSIONAL_RI)
    Embedding(Embedding.IDENTITY_EMBEDDING)
  }

  test("symmetrized embedding") {
    val e = Embedding(Embedding.SYMMETRIZING_KL_EMBEDDING)
    val v1 = WeightedVector(Vectors.dense(4.0), 8.0)

    val expected = Vectors.dense(5.38629436111989)
    val embedded = e.embed(v1)

    assert(embedded.weight == 8.0)
    assert(embedded.homogeneous ~== expected absTol 1E-5)
  }

  test("low dimensional random indexing embedding") {
    val e = Embedding(Embedding.LOW_DIMENSIONAL_RI)
    val v1 = WeightedVector(Vectors.sparse(10, Seq((1, 2.0), (3, 6.0))), 3.0)

    val expected = Vectors.sparse(64, Seq((10, 6.0), (11, -2.0), (13, -6.0), (14, 2.0)))
    val embedded = e.embed(v1)

    assert(embedded.weight == 3.0)
    assert(embedded.homogeneous ~== expected absTol 1E-5)
  }

  test("medium dimensional random indexing embedding") {
    val e = Embedding(Embedding.MEDIUM_DIMENSIONAL_RI)
    val v1 = WeightedVector(Vectors.sparse(10, Seq((1, 2.0), (3, 6.0))), 3.0)

    val expected = Vectors.sparse(256, Seq((3, 6.0), (6, -2.0), (13, -2.0), (15, -6.0), (17, -2.0),
      (19, -6.0), (21, 6.0), (23, 2.0), (26, 8.0), (28, 2.0), (31, -6.0)))
    val embedded = e.embed(v1)

    assert(embedded.weight == 3.0)
    assert(embedded.homogeneous ~== expected absTol 1E-5)
  }

}

```



#### Error stacktrace:

```

```
#### Short summary: 

scala.reflect.internal.Types$TypeError: illegal cyclic reference involving class EmbeddingSuite
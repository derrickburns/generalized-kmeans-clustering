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
 */

package com.massivedatascience.clusterer

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

class AssignmentPlanSuite extends AnyFunSuite with Matchers {

  test("CrossJoinAssignmentPlan should be created correctly") {
    val plan = CrossJoinAssignmentPlan(
      divergence = "squaredEuclidean",
      rowIdProvider = RowIdProvider.monotonic(),
      featuresCol = "features",
      predictionCol = "prediction"
    )

    assert(plan.divergence == "squaredEuclidean")
    assert(plan.featuresCol == "features")
    assert(plan.predictionCol == "prediction")
    assert(plan.rowIdProvider == MonotonicRowIdProvider)
  }

  test("RDDMapAssignmentPlan should be created correctly") {
    val plan = RDDMapAssignmentPlan(
      divergence = "kl",
      featuresCol = "data",
      predictionCol = "cluster"
    )

    assert(plan.divergence == "kl")
    assert(plan.featuresCol == "data")
    assert(plan.predictionCol == "cluster")
  }

  test("UDFAssignmentPlan should be created correctly") {
    val plan = UDFAssignmentPlan(
      divergence = "squaredEuclidean",
      featuresCol = "features",
      predictionCol = "prediction"
    )

    assert(plan.divergence == "squaredEuclidean")
    assert(plan.featuresCol == "features")
    assert(plan.predictionCol == "prediction")
  }

  test("ConditionalAssignmentPlan should be created correctly") {
    val defaultPlan = RDDMapAssignmentPlan("squaredEuclidean")
    val plan        = ConditionalAssignmentPlan(
      defaultPlan = defaultPlan,
      featuresCol = "features",
      predictionCol = "prediction"
    )

    assert(plan.defaultPlan == defaultPlan)
    assert(plan.featuresCol == "features")
    assert(plan.predictionCol == "prediction")
  }

  test("AssignmentPlan.crossJoin factory should create CrossJoinAssignmentPlan") {
    val plan = AssignmentPlan.crossJoin()

    assert(plan.isInstanceOf[CrossJoinAssignmentPlan])
    assert(plan.featuresCol == "features")
    assert(plan.predictionCol == "prediction")

    val crossJoinPlan = plan.asInstanceOf[CrossJoinAssignmentPlan]
    assert(crossJoinPlan.divergence == "squaredEuclidean")
    assert(crossJoinPlan.rowIdProvider == MonotonicRowIdProvider)
  }

  test("AssignmentPlan.crossJoin factory should accept custom parameters") {
    val customProvider = RowIdProvider.fromColumn("id")
    val plan           = AssignmentPlan.crossJoin(
      featuresCol = "data",
      predictionCol = "cluster",
      rowIdProvider = customProvider
    )

    val crossJoinPlan = plan.asInstanceOf[CrossJoinAssignmentPlan]
    assert(crossJoinPlan.featuresCol == "data")
    assert(crossJoinPlan.predictionCol == "cluster")
    assert(crossJoinPlan.rowIdProvider == customProvider)
  }

  test("AssignmentPlan.rddMap factory should create RDDMapAssignmentPlan") {
    val plan = AssignmentPlan.rddMap()

    assert(plan.isInstanceOf[RDDMapAssignmentPlan])
    assert(plan.featuresCol == "features")
    assert(plan.predictionCol == "prediction")

    val rddMapPlan = plan.asInstanceOf[RDDMapAssignmentPlan]
    assert(rddMapPlan.divergence == "squaredEuclidean")
  }

  test("AssignmentPlan.rddMap factory should accept custom divergence") {
    val plan = AssignmentPlan.rddMap(divergence = "kl")

    val rddMapPlan = plan.asInstanceOf[RDDMapAssignmentPlan]
    assert(rddMapPlan.divergence == "kl")
  }

  test("AssignmentPlan.udf factory should create UDFAssignmentPlan") {
    val plan = AssignmentPlan.udf()

    assert(plan.isInstanceOf[UDFAssignmentPlan])
    assert(plan.featuresCol == "features")
    assert(plan.predictionCol == "prediction")
  }

  test("AssignmentPlan.auto factory should create ConditionalAssignmentPlan") {
    val plan = AssignmentPlan.auto()

    assert(plan.isInstanceOf[ConditionalAssignmentPlan])

    val conditionalPlan = plan.asInstanceOf[ConditionalAssignmentPlan]
    assert(conditionalPlan.defaultPlan.isInstanceOf[RDDMapAssignmentPlan])
  }

  test("AssignmentPlan should support pattern matching") {
    val plan: AssignmentPlan = AssignmentPlan.crossJoin()

    val result = plan match {
      case CrossJoinAssignmentPlan(div, _, feat, pred) =>
        s"CrossJoin: $div, $feat -> $pred"
      case RDDMapAssignmentPlan(div, feat, pred)       =>
        s"RDDMap: $div, $feat -> $pred"
      case UDFAssignmentPlan(div, feat, pred)          =>
        s"UDF: $div, $feat -> $pred"
      case ConditionalAssignmentPlan(_, feat, pred)    =>
        s"Conditional: $feat -> $pred"
    }

    assert(result == "CrossJoin: squaredEuclidean, features -> prediction")
  }

  test("Different plan types should not be equal") {
    val plan1 = AssignmentPlan.crossJoin()
    val plan2 = AssignmentPlan.rddMap()
    val plan3 = AssignmentPlan.udf()

    assert(plan1 != plan2)
    assert(plan2 != plan3)
    assert(plan1 != plan3)
  }

  test("Same plan types with same parameters should be equal") {
    val plan1 = AssignmentPlan.rddMap(divergence = "kl", featuresCol = "data")
    val plan2 = AssignmentPlan.rddMap(divergence = "kl", featuresCol = "data")

    assert(plan1 == plan2)
  }

  test("AssignmentPlan should be serializable") {
    val plan: AssignmentPlan = AssignmentPlan.crossJoin()

    // This will throw if not serializable
    val stream = new java.io.ByteArrayOutputStream()
    val oos    = new java.io.ObjectOutputStream(stream)
    oos.writeObject(plan)
    oos.close()

    val bytes = stream.toByteArray
    assert(bytes.nonEmpty)
  }

  test("ConditionalAssignmentPlan should contain default plan") {
    val defaultPlan = AssignmentPlan.rddMap(divergence = "kl")
    val plan        = ConditionalAssignmentPlan(defaultPlan)

    assert(plan.defaultPlan == defaultPlan)
    assert(plan.defaultPlan.asInstanceOf[RDDMapAssignmentPlan].divergence == "kl")
  }

  // Note: Interpreter tests require full Spark setup and are deferred to integration tests
  // These tests focus on the ADT structure and factory methods
}

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

package com.massivedatascience.clusterer.ml.df

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

/** Tests for pairwise constraint framework.
  *
  * Validates:
  *   - MustLink and CannotLink constraints
  *   - ConstraintSet indexing and lookup
  *   - Constraint validation and violation detection
  *   - Connected component computation
  *   - Constraint penalties
  */
class ConstraintsSuite extends AnyFunSuite with Matchers {

  // ========== MustLink Tests ==========

  test("MustLink canonical form has smaller index first") {
    val ml1 = MustLink(1L, 2L)
    ml1.canonical shouldBe MustLink(1L, 2L)

    val ml2 = MustLink(5L, 3L)
    ml2.canonical shouldBe MustLink(3L, 5L)
  }

  test("MustLink preserves weight") {
    val ml = MustLink(1L, 2L, 0.5)
    ml.weight shouldBe 0.5
    ml.canonical.weight shouldBe 0.5
  }

  // ========== CannotLink Tests ==========

  test("CannotLink canonical form has smaller index first") {
    val cl1 = CannotLink(1L, 2L)
    cl1.canonical shouldBe CannotLink(1L, 2L)

    val cl2 = CannotLink(5L, 3L)
    cl2.canonical shouldBe CannotLink(3L, 5L)
  }

  test("CannotLink preserves weight") {
    val cl = CannotLink(1L, 2L, 0.7)
    cl.weight shouldBe 0.7
    cl.canonical.weight shouldBe 0.7
  }

  // ========== ConstraintSet Basic Tests ==========

  test("ConstraintSet.empty has no constraints") {
    val cs = ConstraintSet.empty
    cs.isEmpty shouldBe true
    cs.size shouldBe 0
    cs.numMustLink shouldBe 0
    cs.numCannotLink shouldBe 0
  }

  test("ConstraintSet from sequence") {
    val constraints = Seq(
      MustLink(1L, 2L),
      MustLink(2L, 3L),
      CannotLink(1L, 4L)
    )
    val cs = ConstraintSet(constraints)

    cs.isEmpty shouldBe false
    cs.size shouldBe 3
    cs.numMustLink shouldBe 2
    cs.numCannotLink shouldBe 1
  }

  test("ConstraintSet.fromPairs creates correct constraints") {
    val cs = ConstraintSet.fromPairs(
      mustLinks = Seq((1L, 2L), (3L, 4L)),
      cannotLinks = Seq((1L, 5L))
    )

    cs.numMustLink shouldBe 2
    cs.numCannotLink shouldBe 1
    cs.hasMustLink(1L, 2L) shouldBe true
    cs.hasCannotLink(1L, 5L) shouldBe true
  }

  // ========== ConstraintSet Lookup Tests ==========

  test("mustLinkPartners returns correct partners") {
    val cs = ConstraintSet.fromPairs(
      mustLinks = Seq((1L, 2L), (1L, 3L), (4L, 5L)),
      cannotLinks = Seq.empty
    )

    cs.mustLinkPartners(1L) should contain theSameElementsAs Set(2L, 3L)
    cs.mustLinkPartners(2L) should contain theSameElementsAs Set(1L)
    cs.mustLinkPartners(4L) should contain theSameElementsAs Set(5L)
    cs.mustLinkPartners(99L) shouldBe empty
  }

  test("cannotLinkPartners returns correct partners") {
    val cs = ConstraintSet.fromPairs(
      mustLinks = Seq.empty,
      cannotLinks = Seq((1L, 2L), (1L, 3L))
    )

    cs.cannotLinkPartners(1L) should contain theSameElementsAs Set(2L, 3L)
    cs.cannotLinkPartners(2L) should contain theSameElementsAs Set(1L)
    cs.cannotLinkPartners(99L) shouldBe empty
  }

  test("hasMustLink checks constraint existence") {
    val cs = ConstraintSet.fromPairs(
      mustLinks = Seq((1L, 2L)),
      cannotLinks = Seq.empty
    )

    cs.hasMustLink(1L, 2L) shouldBe true
    cs.hasMustLink(2L, 1L) shouldBe true // Symmetric
    cs.hasMustLink(1L, 3L) shouldBe false
  }

  test("hasCannotLink checks constraint existence") {
    val cs = ConstraintSet.fromPairs(
      mustLinks = Seq.empty,
      cannotLinks = Seq((1L, 4L))
    )

    cs.hasCannotLink(1L, 4L) shouldBe true
    cs.hasCannotLink(4L, 1L) shouldBe true // Symmetric
    cs.hasCannotLink(1L, 2L) shouldBe false
  }

  // ========== Weight Lookup Tests ==========

  test("mustLinkWeight returns correct weight") {
    val cs = ConstraintSet(Seq(MustLink(1L, 2L, 0.8)))

    cs.mustLinkWeight(1L, 2L) shouldBe 0.8
    cs.mustLinkWeight(2L, 1L) shouldBe 0.8 // Symmetric
    cs.mustLinkWeight(1L, 3L) shouldBe 0.0 // No constraint
  }

  test("cannotLinkWeight returns correct weight") {
    val cs = ConstraintSet(Seq(CannotLink(1L, 4L, 0.6)))

    cs.cannotLinkWeight(1L, 4L) shouldBe 0.6
    cs.cannotLinkWeight(4L, 1L) shouldBe 0.6 // Symmetric
    cs.cannotLinkWeight(1L, 5L) shouldBe 0.0 // No constraint
  }

  // ========== Assignment Validation Tests ==========

  test("isValidAssignment detects must-link violations") {
    val cs = ConstraintSet.fromPairs(
      mustLinks = Seq((1L, 2L)),
      cannotLinks = Seq.empty
    )

    val assignments = Map(1L -> 0, 2L -> 0)

    // Point 3 can go anywhere (no constraints)
    cs.isValidAssignment(3L, 0, assignments) shouldBe true
    cs.isValidAssignment(3L, 1, assignments) shouldBe true

    // Point with must-link partner in same cluster = valid
    cs.isValidAssignment(1L, 0, Map(2L -> 0)) shouldBe true

    // Point with must-link partner in different cluster = invalid
    cs.isValidAssignment(1L, 1, Map(2L -> 0)) shouldBe false
  }

  test("isValidAssignment detects cannot-link violations") {
    val cs = ConstraintSet.fromPairs(
      mustLinks = Seq.empty,
      cannotLinks = Seq((1L, 2L))
    )

    // Point with cannot-link partner in different cluster = valid
    cs.isValidAssignment(1L, 1, Map(2L -> 0)) shouldBe true

    // Point with cannot-link partner in same cluster = invalid
    cs.isValidAssignment(1L, 0, Map(2L -> 0)) shouldBe false
  }

  test("isValidAssignment handles unassigned partners") {
    val cs = ConstraintSet.fromPairs(
      mustLinks = Seq((1L, 2L)),
      cannotLinks = Seq((1L, 3L))
    )

    // Partners not yet assigned - should be valid
    val emptyAssignments = Map.empty[Long, Int]
    cs.isValidAssignment(1L, 0, emptyAssignments) shouldBe true
  }

  // ========== Violation Penalty Tests ==========

  test("computeViolationPenalty with no violations") {
    val cs = ConstraintSet.fromPairs(
      mustLinks = Seq((1L, 2L)),
      cannotLinks = Seq((1L, 3L))
    )

    // Valid assignment: 1 and 2 together, 3 separate
    val assignments = Map(1L -> 0, 2L -> 0, 3L -> 1)
    cs.computeViolationPenalty(assignments) shouldBe 0.0
  }

  test("computeViolationPenalty detects must-link violations") {
    val cs = ConstraintSet(Seq(MustLink(1L, 2L, 1.0)))

    // Must-link violation: 1 and 2 in different clusters
    val assignments = Map(1L -> 0, 2L -> 1)
    cs.computeViolationPenalty(assignments) shouldBe 1.0
  }

  test("computeViolationPenalty detects cannot-link violations") {
    val cs = ConstraintSet(Seq(CannotLink(1L, 2L, 1.5)))

    // Cannot-link violation: 1 and 2 in same cluster
    val assignments = Map(1L -> 0, 2L -> 0)
    cs.computeViolationPenalty(assignments) shouldBe 1.5
  }

  test("computeViolationPenalty sums multiple violations") {
    val cs = ConstraintSet(Seq(
      MustLink(1L, 2L, 1.0),
      CannotLink(3L, 4L, 2.0)
    ))

    // Both constraints violated
    val assignments = Map(1L -> 0, 2L -> 1, 3L -> 0, 4L -> 0)
    cs.computeViolationPenalty(assignments) shouldBe 3.0
  }

  // ========== Connected Components Tests ==========

  test("mustLinkComponents finds connected components") {
    val cs = ConstraintSet.fromPairs(
      mustLinks = Seq((1L, 2L), (2L, 3L), (4L, 5L)),
      cannotLinks = Seq.empty
    )

    val components = cs.mustLinkComponents()

    // 1, 2, 3 should be in same component
    components(1L) shouldBe components(2L)
    components(2L) shouldBe components(3L)

    // 4, 5 should be in same component but different from 1-3
    components(4L) shouldBe components(5L)
    components(4L) should not be components(1L)
  }

  test("mustLinkComponents handles isolated points") {
    val cs = ConstraintSet.fromPairs(
      mustLinks = Seq((1L, 2L)),
      cannotLinks = Seq((3L, 4L))  // 3 and 4 are isolated from must-links
    )

    val components = cs.mustLinkComponents()

    // 1 and 2 in same component
    components(1L) shouldBe components(2L)

    // 3 and 4 should be in separate components (no must-link between them)
    components.get(3L) should not be components.get(4L)
  }

  // ========== Satisfiability Tests ==========

  test("isSatisfiable returns true for valid constraints") {
    val cs = ConstraintSet.fromPairs(
      mustLinks = Seq((1L, 2L), (2L, 3L)),
      cannotLinks = Seq((1L, 4L))
    )

    cs.isSatisfiable shouldBe true
  }

  test("isSatisfiable detects conflicting constraints") {
    val cs = ConstraintSet.fromPairs(
      mustLinks = Seq((1L, 2L)),
      cannotLinks = Seq((1L, 2L))  // Conflict: same pair has both ML and CL
    )

    cs.isSatisfiable shouldBe false
  }

  test("isSatisfiable detects transitive conflicts") {
    val cs = ConstraintSet.fromPairs(
      mustLinks = Seq((1L, 2L), (2L, 3L)),  // 1-2-3 must be together
      cannotLinks = Seq((1L, 3L))           // 1 and 3 cannot be together
    )

    cs.isSatisfiable shouldBe false
  }

  // ========== LinearConstraintPenalty Tests ==========

  test("LinearConstraintPenalty computes must-link penalty") {
    val cs = ConstraintSet(Seq(MustLink(1L, 2L, 1.0)))
    val penalty = new LinearConstraintPenalty(mustLinkWeight = 2.0, cannotLinkWeight = 1.0)

    // No violation
    penalty.computeAssignmentPenalty(1L, 0, Map(2L -> 0), cs) shouldBe 0.0

    // Must-link violation
    penalty.computeAssignmentPenalty(1L, 1, Map(2L -> 0), cs) shouldBe 2.0
  }

  test("LinearConstraintPenalty computes cannot-link penalty") {
    val cs = ConstraintSet(Seq(CannotLink(1L, 2L, 1.0)))
    val penalty = new LinearConstraintPenalty(mustLinkWeight = 1.0, cannotLinkWeight = 3.0)

    // No violation
    penalty.computeAssignmentPenalty(1L, 1, Map(2L -> 0), cs) shouldBe 0.0

    // Cannot-link violation
    penalty.computeAssignmentPenalty(1L, 0, Map(2L -> 0), cs) shouldBe 3.0
  }

  test("LinearConstraintPenalty handles unassigned partners") {
    val cs = ConstraintSet(Seq(MustLink(1L, 2L, 1.0)))
    val penalty = new LinearConstraintPenalty()

    // Partner not assigned - no penalty
    penalty.computeAssignmentPenalty(1L, 0, Map.empty, cs) shouldBe 0.0
  }

  // ========== ExponentialConstraintPenalty Tests ==========

  test("ExponentialConstraintPenalty returns 0 for no violations") {
    val cs = ConstraintSet(Seq(MustLink(1L, 2L, 1.0)))
    val penalty = new ExponentialConstraintPenalty(alpha = 1.0)

    // No violation
    penalty.computeAssignmentPenalty(1L, 0, Map(2L -> 0), cs) shouldBe 0.0
  }

  test("ExponentialConstraintPenalty grows exponentially") {
    val cs = ConstraintSet(Seq(MustLink(1L, 2L, 1.0)))
    val penalty = new ExponentialConstraintPenalty(alpha = 1.0)

    // Violation penalty = exp(1*1) - 1 = e - 1 ≈ 1.718
    val p = penalty.computeAssignmentPenalty(1L, 1, Map(2L -> 0), cs)
    math.abs(p - (math.E - 1)) should be < 1e-10
  }
}

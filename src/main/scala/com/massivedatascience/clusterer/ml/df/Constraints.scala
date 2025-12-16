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

import scala.collection.mutable

/** Pairwise constraint between data points.
  *
  * Constraints guide clustering by specifying which points should (must-link) or should not
  * (cannot-link) be in the same cluster.
  */
sealed trait Constraint extends Serializable {
  def i: Long
  def j: Long
  def weight: Double

  /** Canonical form with smaller index first. */
  def canonical: Constraint
}

/** Must-link constraint: points i and j should be in the same cluster.
  *
  * @param i
  *   first point index
  * @param j
  *   second point index
  * @param weight
  *   constraint strength (default: 1.0)
  */
case class MustLink(i: Long, j: Long, weight: Double = 1.0) extends Constraint {
  override def canonical: MustLink =
    if (i <= j) this else MustLink(j, i, weight)
}

/** Cannot-link constraint: points i and j should be in different clusters.
  *
  * @param i
  *   first point index
  * @param j
  *   second point index
  * @param weight
  *   constraint strength (default: 1.0)
  */
case class CannotLink(i: Long, j: Long, weight: Double = 1.0) extends Constraint {
  override def canonical: CannotLink =
    if (i <= j) this else CannotLink(j, i, weight)
}

/** Collection of constraints with efficient lookup.
  *
  * Provides O(1) lookup for constraints involving a specific point, and methods to check constraint
  * violations.
  */
class ConstraintSet(constraints: Seq[Constraint]) extends Serializable {

  // Index constraints by point for efficient lookup
  private val mustLinkIndex: Map[Long, Set[Long]] = {
    val builder = mutable.Map.empty[Long, mutable.Set[Long]]
    constraints.collect { case ml: MustLink =>
      builder.getOrElseUpdate(ml.i, mutable.Set.empty) += ml.j
      builder.getOrElseUpdate(ml.j, mutable.Set.empty) += ml.i
    }
    builder.view.mapValues(_.toSet).toMap
  }

  private val cannotLinkIndex: Map[Long, Set[Long]] = {
    val builder = mutable.Map.empty[Long, mutable.Set[Long]]
    constraints.collect { case cl: CannotLink =>
      builder.getOrElseUpdate(cl.i, mutable.Set.empty) += cl.j
      builder.getOrElseUpdate(cl.j, mutable.Set.empty) += cl.i
    }
    builder.view.mapValues(_.toSet).toMap
  }

  // Constraint weights for penalty calculation
  private val mustLinkWeights: Map[(Long, Long), Double] =
    constraints.collect { case ml: MustLink =>
      val key = if (ml.i <= ml.j) (ml.i, ml.j) else (ml.j, ml.i)
      key -> ml.weight
    }.toMap

  private val cannotLinkWeights: Map[(Long, Long), Double] =
    constraints.collect { case cl: CannotLink =>
      val key = if (cl.i <= cl.j) (cl.i, cl.j) else (cl.j, cl.i)
      key -> cl.weight
    }.toMap

  /** Get all must-link partners for a point. */
  def mustLinkPartners(pointId: Long): Set[Long] =
    mustLinkIndex.getOrElse(pointId, Set.empty)

  /** Get all cannot-link partners for a point. */
  def cannotLinkPartners(pointId: Long): Set[Long] =
    cannotLinkIndex.getOrElse(pointId, Set.empty)

  /** Check if must-link constraint exists between two points. */
  def hasMustLink(i: Long, j: Long): Boolean = {
    val partners = mustLinkIndex.getOrElse(i, Set.empty)
    partners.contains(j)
  }

  /** Check if cannot-link constraint exists between two points. */
  def hasCannotLink(i: Long, j: Long): Boolean = {
    val partners = cannotLinkIndex.getOrElse(i, Set.empty)
    partners.contains(j)
  }

  /** Get weight of must-link constraint (0 if none). */
  def mustLinkWeight(i: Long, j: Long): Double = {
    val key = if (i <= j) (i, j) else (j, i)
    mustLinkWeights.getOrElse(key, 0.0)
  }

  /** Get weight of cannot-link constraint (0 if none). */
  def cannotLinkWeight(i: Long, j: Long): Double = {
    val key = if (i <= j) (i, j) else (j, i)
    cannotLinkWeights.getOrElse(key, 0.0)
  }

  /** Check if assigning point to cluster would violate hard constraints.
    *
    * @param pointId
    *   point being assigned
    * @param clusterId
    *   target cluster
    * @param assignments
    *   current point-to-cluster mapping
    * @return
    *   true if assignment is valid (no violations)
    */
  def isValidAssignment(
      pointId: Long,
      clusterId: Int,
      assignments: Map[Long, Int]
  ): Boolean = {
    // Check must-link: partners must be in same cluster
    val mlPartners = mustLinkPartners(pointId)
    val mlViolated = mlPartners.exists { partnerId =>
      assignments.get(partnerId).exists(_ != clusterId)
    }
    if (mlViolated) return false

    // Check cannot-link: partners must be in different clusters
    val clPartners = cannotLinkPartners(pointId)
    val clViolated = clPartners.exists { partnerId =>
      assignments.get(partnerId).contains(clusterId)
    }
    !clViolated
  }

  /** Compute total constraint violation penalty for a set of assignments.
    *
    * @param assignments
    *   point-to-cluster mapping
    * @return
    *   weighted sum of violations
    */
  def computeViolationPenalty(assignments: Map[Long, Int]): Double = {
    var penalty = 0.0

    // Must-link violations: same pair in different clusters
    for {
      ((i, j), weight) <- mustLinkWeights
      ci               <- assignments.get(i)
      cj               <- assignments.get(j)
      if ci != cj
    } {
      penalty += weight
    }

    // Cannot-link violations: same pair in same cluster
    for {
      ((i, j), weight) <- cannotLinkWeights
      ci               <- assignments.get(i)
      cj               <- assignments.get(j)
      if ci == cj
    } {
      penalty += weight
    }

    penalty
  }

  /** Find connected components of must-link constraints.
    *
    * Points in the same component must be in the same cluster.
    *
    * @return
    *   mapping from point ID to component ID
    */
  def mustLinkComponents(): Map[Long, Int] = {
    val allPoints   = (mustLinkIndex.keys ++ cannotLinkIndex.keys).toSet
    val visited     = mutable.Set.empty[Long]
    val components  = mutable.Map.empty[Long, Int]
    var componentId = 0

    for (point <- allPoints if !visited.contains(point)) {
      // BFS to find connected component
      val queue = mutable.Queue(point)
      while (queue.nonEmpty) {
        val current = queue.dequeue()
        if (!visited.contains(current)) {
          visited += current
          components(current) = componentId
          queue ++= mustLinkPartners(current).filterNot(visited.contains)
        }
      }
      componentId += 1
    }

    components.toMap
  }

  /** Number of must-link constraints. */
  def numMustLink: Int = mustLinkWeights.size

  /** Number of cannot-link constraints. */
  def numCannotLink: Int = cannotLinkWeights.size

  /** Total number of constraints. */
  def size: Int = numMustLink + numCannotLink

  /** Check if there are any constraints. */
  def isEmpty: Boolean = size == 0

  /** Check if constraints are satisfiable (no conflicting must-link/cannot-link). */
  def isSatisfiable: Boolean = {
    // Check if any must-link component has internal cannot-link
    val components = mustLinkComponents()
    !cannotLinkWeights.keys.exists { case (i, j) =>
      components.get(i) == components.get(j) && components.contains(i)
    }
  }
}

object ConstraintSet {

  /** Create empty constraint set. */
  def empty: ConstraintSet = new ConstraintSet(Seq.empty)

  /** Create from sequence of constraints. */
  def apply(constraints: Seq[Constraint]): ConstraintSet =
    new ConstraintSet(constraints)

  /** Create from separate must-link and cannot-link pairs.
    *
    * @param mustLinks
    *   pairs of point indices that must be together
    * @param cannotLinks
    *   pairs of point indices that must be apart
    */
  def fromPairs(
      mustLinks: Seq[(Long, Long)],
      cannotLinks: Seq[(Long, Long)]
  ): ConstraintSet = {
    val constraints = mustLinks.map { case (i, j) => MustLink(i, j) } ++
      cannotLinks.map { case (i, j) => CannotLink(i, j) }
    new ConstraintSet(constraints)
  }
}

/** Constraint penalty calculator for soft constraint enforcement.
  *
  * Computes penalty terms to add to the clustering objective function.
  */
trait ConstraintPenalty extends Serializable {

  /** Compute penalty for assigning a point to a cluster.
    *
    * @param pointId
    *   point being assigned
    * @param clusterId
    *   target cluster
    * @param assignments
    *   current assignments (may be partial)
    * @param constraints
    *   constraint set
    * @return
    *   penalty value (0 = no violation)
    */
  def computeAssignmentPenalty(
      pointId: Long,
      clusterId: Int,
      assignments: Map[Long, Int],
      constraints: ConstraintSet
  ): Double
}

/** Linear penalty: sum of violated constraint weights.
  *
  * @param mustLinkWeight
  *   penalty multiplier for must-link violations
  * @param cannotLinkWeight
  *   penalty multiplier for cannot-link violations
  */
class LinearConstraintPenalty(
    val mustLinkWeight: Double = 1.0,
    val cannotLinkWeight: Double = 1.0
) extends ConstraintPenalty {

  override def computeAssignmentPenalty(
      pointId: Long,
      clusterId: Int,
      assignments: Map[Long, Int],
      constraints: ConstraintSet
  ): Double = {
    var penalty = 0.0

    // Must-link violations
    for (partnerId <- constraints.mustLinkPartners(pointId)) {
      assignments.get(partnerId) match {
        case Some(partnerCluster) if partnerCluster != clusterId =>
          penalty += mustLinkWeight * constraints.mustLinkWeight(pointId, partnerId)
        case _                                                   => // No violation
      }
    }

    // Cannot-link violations
    for (partnerId <- constraints.cannotLinkPartners(pointId)) {
      assignments.get(partnerId) match {
        case Some(partnerCluster) if partnerCluster == clusterId =>
          penalty += cannotLinkWeight * constraints.cannotLinkWeight(pointId, partnerId)
        case _                                                   => // No violation
      }
    }

    penalty
  }
}

/** Exponential penalty: encourages strict constraint satisfaction.
  *
  * penalty = exp(α * violations) - 1
  *
  * @param alpha
  *   exponential growth rate
  */
class ExponentialConstraintPenalty(val alpha: Double = 1.0) extends ConstraintPenalty {

  private val linearPenalty = new LinearConstraintPenalty()

  override def computeAssignmentPenalty(
      pointId: Long,
      clusterId: Int,
      assignments: Map[Long, Int],
      constraints: ConstraintSet
  ): Double = {
    val linear = linearPenalty.computeAssignmentPenalty(
      pointId,
      clusterId,
      assignments,
      constraints
    )
    if (linear == 0.0) 0.0 else math.exp(alpha * linear) - 1.0
  }
}

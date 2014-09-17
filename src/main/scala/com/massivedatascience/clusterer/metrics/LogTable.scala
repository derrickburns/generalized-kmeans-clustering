package com.massivedatascience.clusterer.metrics

import com.massivedatascience.clusterer.base.{Zero, One}

trait LogTable {
  private val logTable = new Array[Double](4096 * 1000)

  def fastLog(d: Double): Double = {
    if (d == Zero || d == One) {
      Zero
    } else {
      val x = d.toInt
      if (x < logTable.length && x.asInstanceOf[Double] == d) {
        if (logTable(x) == Zero) logTable(x) = Math.log(x)
        logTable(x)
      } else {
        Math.log(x)
      }
    }
  }
}

package com.massivedatascience.clusterer

trait HasLog extends Serializable {
  def log(x: Double) : Double
}

trait GeneralLog extends HasLog {
  @inline
  override def log(x: Double) : Double = if (x == 0.0 || x == 1.0) 0.0 else Math.log(x)
}

trait DiscreteLog extends HasLog{
  private val logTable = new Array[Double](4096 * 1000)

  override def log(d: Double): Double = {
    if (d == 0.0 || d == 1.0) {
      0.0
    } else {
      if (d < logTable.length ) {
        val x = d.toInt
        if (x.toDouble == d) {
          if (logTable(x) == 0.0) logTable(x) = Math.log(x)
          logTable(x)
        } else {
          Math.log(d)
        }
      }  else {
        Math.log(d)
      }
    }
  }
}
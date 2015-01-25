package com.massivedatascience.clusterer


object HaarWavelet {


  def average(input: Array[Double]) : Array[Double] = {
    val len = input.length >> 1
    val result = new Array[Double](len)
    var i = 0
    while(i < len) {
      result(i) = (input(2*i) + input(2*i+1))/2.0
      i = i + 1
    }
    result
  }

  class HaarWavelet(size: Int) extends Serializable {

    val tmp = new Array[Double](size)

    /**
     * in place Haar transform
     *
     * @param result  input to transform
     * @param halfLength half the length of array
     */
    def haar(input: Array[Double], result: Array[Double], halfLength: Int) : Array[Double] = {
      var i = 0
      while (i != halfLength) {
        val j = i << 1
        tmp(i) = input(j)
        tmp(halfLength + i) = input(j + 1)
        i = i + 1
      }

      i = 0
      while (i != halfLength) {
        val j = halfLength + i
        val l = tmp(i) / 2.0
        val r = tmp(j) / 2.0
        result(i) = l + r
        result(j) = l - r
        i = i + 1
      }
      result
    }

    /**
     * in place inverse Haar transform
     *
     * @param result input to transform
     * @param halfLength half the length of the coordinates
     */
    def inverseHaar(input: Array[Double], result: Array[Double], halfLength: Int) : Array[Double]= {
      var i = 0
      while (i != halfLength) {
        val j = halfLength + i
        val l = input(i)
        val r = input(j)
        tmp(i) = l + r
        tmp(j) = l - r
        i = i + 1
      }

      i = 0
      while (i != halfLength) {
        val j = i << 1
        result(j) = tmp(i)
        result(j + 1) = tmp(halfLength + i)
        i = i + 1
      }
      result
    }

    def transform(values: Array[Double]) = {
      var start = values.length
      val result = values.clone()
      while (start != 0) {
        println(result.mkString("[", ",", "]"))
        require(start == 1 || (start & 1) == 0, "length of input must be power of 2")
        start = start >> 1
        haar(result, result, start)
      }

      start = 1
      while (start < values.length) {
        inverseHaar(result, result, start)
        start = start << 1
        println(result.mkString("[", ",", "]"))
      }
      result
    }
  }


  def main(args : Array[ String ] ) : Unit = {

    val y = Array( 3.0, 1.0, 0.0, 4.0, 8.0, 6.0, 9.0, 9.0 )
    val x = new HaarWavelet(y.length)

    val h1 = x.haar(y, y.clone(), y.length >> 1)
    val h2 = x.inverseHaar(h1, h1.clone(), y.length >> 1)

    println( h2.mkString("[", ",", "]"))

  }

}

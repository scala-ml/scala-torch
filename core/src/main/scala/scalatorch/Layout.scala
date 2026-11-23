package scalatorch

private[scalatorch] final class Layout private (
    val shape: Shape, val strides: Vector[Long], val offset: Long
):
  def index(linear: Int): Int =
    var rest = linear.toLong
    var out = offset
    var d = shape.rank - 1
    while d >= 0 do
      out += (rest % shape.dims(d)) * strides(d)
      rest /= shape.dims(d)
      d -= 1
    out.toInt

  /** The elements in logical order, read from `source`. Same positions as `index(0), index(1), ...`, but found by stepping
    * (one addition per element) instead of a division and a remainder per axis and element; rows with stride 1 are copied as
    * blocks. Broadcast axes (stride 0) and sliced views included. */
  def gatherFloats(source: Array[Float]): Array[Float] =
    val total = shape.numel.toInt
    val out = new Array[Float](total)
    val rank = shape.rank
    if total == 0 then return out
    if rank == 0 then { out(0) = source(offset.toInt); return out }
    val sizes = new Array[Int](rank)
    val steps = new Array[Long](rank)
    var d = 0
    while d < rank do { sizes(d) = shape.dims(d).toInt; steps(d) = strides(d); d += 1 }
    val inner = sizes(rank - 1)
    val innerStep = steps(rank - 1)
    val counter = new Array[Int](rank)
    var base = offset
    var o = 0
    while o < total do
      if innerStep == 1L then System.arraycopy(source, base.toInt, out, o, inner)
      else
        var p = base
        var j = 0
        while j < inner do { out(o + j) = source(p.toInt); p += innerStep; j += 1 }
      o += inner
      var axis = rank - 2
      var carry = true
      while axis >= 0 && carry do
        counter(axis) += 1
        base += steps(axis)
        if counter(axis) == sizes(axis) then
          base -= steps(axis) * sizes(axis)
          counter(axis) = 0
          axis -= 1
        else carry = false
    out

  def isContiguous: Boolean =
    if shape.numel == 0 then true
    else
      var expected = 1L
      var d = shape.rank - 1
      while d >= 0 do
        if shape.dims(d) > 1 && strides(d) != expected then return false
        expected *= shape.dims(d)
        d -= 1
      true

private[scalatorch] object Layout:
  def checked(shape: Shape, strides: Vector[Long], offset: Long, length: Long): Layout =
    require(strides.size == shape.rank, "stride rank mismatch")
    require(offset >= 0 && offset <= length, "invalid offset")
    if shape.numel > 0 then
      try
        var low = offset
        var high = offset
        for (dim, stride) <- shape.dims.zip(strides) do
          val delta = Math.multiplyExact(dim - 1, stride)
          if delta < 0 then low = Math.addExact(low, delta)
          else high = Math.addExact(high, delta)
        require(low >= 0 && high < length, "layout exceeds storage bounds")
      catch case _: ArithmeticException => throw IllegalArgumentException("layout offset overflow")
    new Layout(shape, strides, offset)

  def contiguous(shape: Shape, length: Long): Layout =
    // Empty layouts never address an element, so unused strides need not overflow.
    if shape.numel == 0 then return checked(shape, Vector.fill(shape.rank)(0L), 0L, length)
    var stride = 1L
    val strides = Array.fill(shape.rank)(0L)
    try
      for d <- (0 until shape.rank).reverse do
        strides(d) = stride
        stride = Math.multiplyExact(stride, shape.dims(d))
    catch case _: ArithmeticException => throw IllegalArgumentException("stride overflow")
    checked(shape, strides.toVector, 0L, length)

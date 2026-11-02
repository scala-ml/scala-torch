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

package scalatorch

final class Tensor[A] private[scalatorch] (
    private[scalatorch] val storage: Storage[A], private[scalatorch] val layout: Layout
):
  def shape: Shape = layout.shape
  val dtype: DType[A] = storage.dtype
  val device: Device = Device.Cpu

  private[scalatorch] def logicalArray: Array[A] =
    val out = storage.allocate(shape.numel.toInt)
    if layout.isContiguous then
      System.arraycopy(storage.data,layout.offset.toInt,out,0,out.length)
    else
      var i = 0
      while i < out.length do
        out(i) = storage.data(layout.index(i))
        i += 1
    out

  private def copyAs[B](expected: DType[B]): Array[B] =
    require(dtype == expected, s"expected $expected, found $dtype")
    logicalArray.asInstanceOf[Array[B]]
  def toFloatArray: Array[Float] = copyAs(DType.Float32)
  def toDoubleArray: Array[Double] = copyAs(DType.Float64)
  def toLongArray: Array[Long] = copyAs(DType.Int64)
  def toBooleanArray: Array[Boolean] = copyAs(DType.Bool)

  private def view(shape: Shape, strides: Vector[Long], offset: Long): Tensor[A] =
    require(shape.numel <= Int.MaxValue, "CPU tensor exceeds Int indexing limit")
    new Tensor(storage, Layout.checked(shape, strides, offset, storage.length))

  private def checkDim(dim: Int): Unit = require(dim >= 0 && dim < shape.rank, "dimension out of range")

  def transpose(dim0: Int, dim1: Int): Tensor[A] =
    checkDim(dim0); checkDim(dim1)
    def swap[B](v: Vector[B]) = v.updated(dim0, v(dim1)).updated(dim1, v(dim0))
    view(Shape.from(swap(shape.dims)), swap(layout.strides), layout.offset)

  def reshape(target: Shape): Tensor[A] =
    require(target.numel == shape.numel, "reshape changes element count")
    require(layout.isContiguous, "reshape requires contiguous layout; call contiguous first")
    val packed = Layout.contiguous(target, shape.numel)
    view(target, packed.strides, layout.offset)

  def contiguous: Tensor[A] =
    if layout.isContiguous then this
    else new Tensor(new Storage(logicalArray, dtype), Layout.contiguous(shape, shape.numel))

  def broadcastTo(target: Shape): Tensor[A] =
    require(target.rank >= shape.rank, "broadcast cannot remove dimensions")
    val padding = target.rank - shape.rank
    val ds = Vector.fill(padding)(1L) ++ shape.dims
    val ss = Vector.fill(padding)(0L) ++ layout.strides
    val strides = ds.indices.map { i =>
      require(ds(i) == target.dims(i) || ds(i) == 1, "incompatible broadcast dimensions")
      if ds(i) == target.dims(i) then ss(i) else 0L
    }.toVector
    view(target, strides, layout.offset)

  /** Half-open slice with explicit bounds. Negative step permits end = -1 sentinel. */
  def slice(dim: Int, start: Long, end: Long, step: Long = 1): Tensor[A] =
    checkDim(dim)
    require(step != 0 && step != Long.MinValue, "invalid slice step")
    val size = shape.dims(dim)
    val count =
      if step > 0 then
        require(start >= 0 && start <= end && end <= size, "invalid slice bounds")
        if start == end then 0L else 1L + (end - start - 1L) / step
      else
        require(start >= 0 && start < size && end >= -1 && end <= start, "invalid reverse slice bounds")
        if start == end then 0L else 1L + (start - end - 1L) / -step
    try
      val target = Shape.from(shape.dims.updated(dim, count))
      val offset = if target.numel == 0 then layout.offset else
        Math.addExact(layout.offset, Math.multiplyExact(start, layout.strides(dim)))
      view(target, layout.strides.updated(dim, Math.multiplyExact(step, layout.strides(dim))), offset)
    catch case _: ArithmeticException => throw IllegalArgumentException("slice offset overflow")

object Tensor:
  private def make[A](values: Array[A], shape: Shape, dtype: DType[A]): Tensor[A] =
    require(shape.numel <= Int.MaxValue, "CPU array exceeds Int indexing limit")
    require(values.length.toLong == shape.numel, "buffer length does not match shape")
    new Tensor(new Storage(values.clone(), dtype), Layout.contiguous(shape, values.length))
  def fromFloats(values: Array[Float], shape: Shape): Tensor[Float] = make(values, shape, DType.Float32)
  def fromDoubles(values: Array[Double], shape: Shape): Tensor[Double] = make(values, shape, DType.Float64)
  def fromLongs(values: Array[Long], shape: Shape): Tensor[Long] = make(values, shape, DType.Int64)
  def fromBooleans(values: Array[Boolean], shape: Shape): Tensor[Boolean] = make(values, shape, DType.Bool)

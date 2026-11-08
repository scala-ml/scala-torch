package scalatorch

/** Deterministic CPU reference operations. Backend selection is explicit. */
object Ops:
  private def cpuFloat(x: Tensor[Float]): Unit =
    require(x.device == Device.Cpu && x.dtype == DType.Float32, "expected CPU Float32 tensor")

  def add(a: Tensor[Float], b: Tensor[Float]): Tensor[Float] =
    cpuFloat(a); cpuFloat(b)
    val rank = math.max(a.shape.rank, b.shape.rank)
    val ad = Vector.fill(rank - a.shape.rank)(1L) ++ a.shape.dims
    val bd = Vector.fill(rank - b.shape.rank)(1L) ++ b.shape.dims
    val shape = Shape.from(ad.zip(bd).map { (x, y) =>
      require(x == y || x == 1 || y == 1, "incompatible broadcast dimensions")
      if x == 1 then y else x
    })
    val av = a.broadcastTo(shape)
    val bv = b.broadcastTo(shape)
    val out = new Array[Float](shape.numel.toInt)
    var i = 0
    while i < out.length do
      out(i) = av.storage.data(av.layout.index(i)) + bv.storage.data(bv.layout.index(i))
      i += 1
    Tensor.fromFloats(out, shape)

  def sum(a: Tensor[Float], dims: Seq[Int], keepDim: Boolean): Tensor[Float] =
    cpuFloat(a)
    require(dims.distinct.size == dims.size, "duplicate reduction dimension")
    require(dims.forall(d => d >= 0 && d < a.shape.rank), "reduction dimension out of range")
    val axes = dims.toSet
    val target = Shape.from(a.shape.dims.zipWithIndex.flatMap { (n, d) =>
      if axes(d) then (if keepDim then Some(1L) else None) else Some(n)
    })
    require(target.numel <= Int.MaxValue, "CPU reduction output exceeds Int indexing limit")
    val out = new Array[Float](target.numel.toInt)
    var i = 0
    while i < a.shape.numel do
      var rest = i.toLong
      var dest = 0L
      var stride = 1L
      var d = a.shape.rank - 1
      while d >= 0 do
        val coord = rest % a.shape.dims(d)
        rest /= a.shape.dims(d)
        if !axes(d) then
          dest += coord * stride
          stride *= a.shape.dims(d)
        d -= 1
      out(dest.toInt) += a.storage.data(a.layout.index(i))
      i += 1
    Tensor.fromFloats(out, target)

  private[scalatorch] def matmulShape(a: Tensor[Float], b: Tensor[Float]): Shape =
    cpuFloat(a); cpuFloat(b)
    require(a.shape.rank == 2 && b.shape.rank == 2, "matmul currently requires rank 2")
    require(a.shape.dims(1) == b.shape.dims(0), "matmul inner dimensions differ")
    val out = Shape.from(Seq(a.shape.dims(0), b.shape.dims(1)))
    require(out.numel <= Int.MaxValue, "CPU output exceeds Int indexing limit")
    out

  def matmul(a: Tensor[Float], b: Tensor[Float]): Tensor[Float] =
    val shape = matmulShape(a, b)
    if shape.numel == 0 then return Tensor.fromFloats(Array.emptyFloatArray, shape)
    require((a.shape.dims ++ b.shape.dims).forall(_ <= Int.MaxValue), "GEMM dimension exceeds Int indexing")
    val m = shape.dims(0).toInt
    val n = shape.dims(1).toInt
    val k = a.shape.dims(1).toInt
    val out = new Array[Float](shape.numel.toInt)
    val left = if a.layout.isContiguous then a.storage.data else a.toFloatArray
    val leftOffset = if a.layout.isContiguous then a.layout.offset.toInt else 0
    val right = if b.layout.isContiguous then b.storage.data else b.toFloatArray
    val rightOffset = if b.layout.isContiguous then b.layout.offset.toInt else 0
    // Each output element keeps the sequential p = 0..k-1 Float accumulation, so results match the
    // scalar triple loop bit for bit; only the loop order, memory access and parallelism change.
    val block = 256
    val columnBlocks = (n + block - 1) / block
    val tasks = m.toLong * columnBlocks
    def run(task: Int): Unit =
      val i = task / columnBlocks
      val j0 = (task % columnBlocks) * block
      val j1 = math.min(n, j0 + block)
      val acc = new Array[Float](j1 - j0)
      var p = 0
      while p < k do
        val x = left(leftOffset + i * k + p)
        val base = rightOffset + p * n + j0
        var j = 0
        while j < acc.length do
          acc(j) += x * right(base + j)
          j += 1
        p += 1
      System.arraycopy(acc, 0, out, i * n + j0, acc.length)
    if tasks > 1 && m.toLong * n * k >= 262144L then
      java.util.stream.IntStream.range(0, tasks.toInt).parallel().forEach(task => run(task))
    else
      var task = 0
      while task < tasks do { run(task); task += 1 }
    Tensor.fromFloats(out, shape)

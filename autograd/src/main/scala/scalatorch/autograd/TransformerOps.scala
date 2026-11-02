package scalatorch.autograd

import scalatorch.*
import com.github.kmizu.numscala.cpu.{MatrixF32, ScalarF32Kernels}

/** First-order Float32 transformer operations with immutable saved forward values. */
private[autograd] object TransformerOps:
  def embedding(table: Variable, indices: Tensor[Long]): Variable =
    val shape = table.data.shape
    require(shape.rank == 2 && shape.dims(1) > 0, "embedding requires a rank-2 table with positive width")
    require(indices.device == Device.Cpu && indices.dtype == DType.Int64, "embedding requires CPU Int64 indices")
    val tokens = indices.toLongArray
    require(tokens.forall(id => id >= 0 && id < shape.dims(0)), "embedding token ID out of bounds")
    val outputShape = Shape.from(indices.shape.dims :+ shape.dims(1))
    require(outputShape.numel <= Int.MaxValue, "embedding output exceeds Int indexing")
    if tokens.isEmpty then
      return Variable.result(Tensor.fromFloats(Array.emptyFloatArray, outputShape), Vector(table -> ((_: Tensor[Float]) =>
        Tensor.fromFloats(new Array[Float](shape.numel.toInt), shape))))
    val rows = shape.dims(0).toInt
    val width = shape.dims(1).toInt
    val ids = tokens.map(_.toInt) // validated before narrowing
    val source = MatrixF32.wrap(table.data.toFloatArray, rows, width)
    val output = MatrixF32.zeros(tokens.length, width)
    ScalarF32Kernels.gatherRowsInto(source, ids, output)
    Variable.result(Tensor.fromFloats(output.data, outputShape), Vector(table -> ((g: Tensor[Float]) =>
      val gradient = MatrixF32.zeros(rows, width)
      ScalarF32Kernels.scatterAddRowsInto(gradient, ids, MatrixF32.wrap(g.toFloatArray, tokens.length, width))
      Tensor.fromFloats(gradient.data, shape))))

  def rmsNorm(input: Variable, weight: Variable, epsilon: Double): Variable =
    require(epsilon.isFinite && epsilon > 0, "RMSNorm epsilon must be finite and positive")
    val shape = input.data.shape
    require(shape.rank >= 1 && shape.dims.last > 0, "RMSNorm requires a positive last dimension")
    require(weight.data.shape == Shape.from(Seq(shape.dims.last)), "RMSNorm weight shape mismatch")
    val x = input.data.toFloatArray
    val scale = weight.data.toFloatArray
    val width = scale.length
    val rows = x.length / width
    val inverse = new Array[Double](rows)
    val output = new Array[Float](x.length)
    var row = 0
    while row < rows do
      val offset = row * width
      var squares = 0.0
      var j = 0
      while j < width do
        val v = x(offset + j).toDouble
        squares += v * v
        j += 1
      val inv = 1.0 / math.sqrt(squares / width + epsilon)
      inverse(row) = inv
      j = 0
      while j < width do
        output(offset + j) = (x(offset + j).toDouble * inv * scale(j)).toFloat
        j += 1
      row += 1
    val inputPullback = (g: Tensor[Float]) =>
      val upstream = g.toFloatArray
      val out = new Array[Float](x.length)
      var r = 0
      while r < rows do
        val offset = r * width
        val inv = inverse(r)
        var dot = 0.0
        var j = 0
        while j < width do
          dot += upstream(offset + j).toDouble * scale(j) * x(offset + j) * inv
          j += 1
        j = 0
        while j < width do
          out(offset + j) = (inv * (upstream(offset + j).toDouble * scale(j) -
            x(offset + j).toDouble * inv * dot / width)).toFloat
          j += 1
        r += 1
      Tensor.fromFloats(out, shape)
    val weightPullback = (g: Tensor[Float]) =>
      val upstream = g.toFloatArray
      val sums = new Array[Double](width)
      var r = 0
      while r < rows do
        var j = 0
        while j < width do
          val index = r * width + j
          sums(j) += upstream(index).toDouble * x(index) * inverse(r)
          j += 1
        r += 1
      Tensor.fromFloats(sums.map(_.toFloat), weight.data.shape)
    Variable.result(Tensor.fromFloats(output, shape), Vector(input -> inputPullback, weight -> weightPullback))

  def silu(input: Variable): Variable =
    val x = input.data.toFloatArray
    val derivatives = new Array[Double](x.length)
    val output = new Array[Float](x.length)
    var i = 0
    while i < x.length do
      val value = x(i).toDouble
      val sigmoid = if value >= 0 then 1.0 / (1.0 + math.exp(-value)) else
        val e = math.exp(value)
        e / (1.0 + e)
      output(i) = (if value == Double.NegativeInfinity then -0.0 else value * sigmoid).toFloat
      derivatives(i) = if sigmoid == 0 then 0.0 else if sigmoid == 1 then 1.0 else
        sigmoid * (1.0 + value * (1.0 - sigmoid))
      i += 1
    Variable.result(Tensor.fromFloats(output, input.data.shape), Vector(input -> ((g: Tensor[Float]) =>
      val upstream = g.toFloatArray
      var j = 0
      while j < upstream.length do
        upstream(j) = (if derivatives(j) == 0 then 0.0 else upstream(j).toDouble * derivatives(j)).toFloat
        j += 1
      Tensor.fromFloats(upstream, input.data.shape))))

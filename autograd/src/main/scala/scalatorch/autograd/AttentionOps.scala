package scalatorch.autograd

import scalatorch.*

/** Correctness-first CPU attention; no dropout, padding mask, or kernel fusion. */
private[autograd] object AttentionOps:
  private def zeros(shape: Shape): Tensor[Float] = Tensor.fromFloats(new Array[Float](shape.numel.toInt), shape)

  def causal(query: Variable, key: Variable, value: Variable, queryStart: Int, maxScores: Long): Variable =
    val qs = query.data.shape; val ks = key.data.shape; val vs = value.data.shape
    require(qs.rank == 4 && ks.rank == 4 && vs.rank == 4, "attention expects (batch,heads,tokens,width)")
    require(queryStart >= 0 && maxScores >= 0, "invalid attention offset or workspace limit")
    require(qs.dims(0) == ks.dims(0) && ks.dims(0) == vs.dims(0), "attention batch mismatch")
    require(qs.dims(1) > 0 && ks.dims(1) > 0 && ks.dims(1) == vs.dims(1) && qs.dims(1) % ks.dims(1) == 0,
      "query heads must be divisible by matching key/value heads")
    require(qs.dims(3) > 0 && qs.dims(3) == ks.dims(3) && vs.dims(3) > 0, "attention width mismatch")
    require(ks.dims(2) == vs.dims(2), "key/value token mismatch")
    val outputShape = Shape.from(Seq(qs.dims(0), qs.dims(1), qs.dims(2), vs.dims(3)))
    require(outputShape.numel <= Int.MaxValue, "attention output exceeds Int indexing")
    if query.data.shape.numel == 0 then
      return Variable.result(zeros(outputShape), Vector(query -> ((_: Tensor[Float]) => zeros(qs)),
        key -> ((_: Tensor[Float]) => zeros(ks)), value -> ((_: Tensor[Float]) => zeros(vs))))
    require(ks.dims(2) > 0, "nonempty queries require at least one key")
    val scoreCount = Shape.from(Seq(qs.dims(0), qs.dims(1), qs.dims(2), ks.dims(2))).numel
    require(scoreCount <= maxScores && scoreCount <= Int.MaxValue, "attention score workspace limit exceeded")
    // Nonempty primitive tensors imply each used dimension and index product fits Int.
    val batches = qs.dims(0).toInt; val qHeads = qs.dims(1).toInt; val qTokens = qs.dims(2).toInt
    val kvHeads = ks.dims(1).toInt; val kTokens = ks.dims(2).toInt
    val width = qs.dims(3).toInt; val valueWidth = vs.dims(3).toInt
    val group = qHeads / kvHeads
    val scale = 1.0 / math.sqrt(width.toDouble)
    val q = query.data.toFloatArray; val k = key.data.toFloatArray; val v = value.data.toFloatArray
    val probabilities = new Array[Double](scoreCount.toInt)
    val output = new Array[Float](outputShape.numel.toInt)
    def visible(token: Int): Int = math.min(kTokens.toLong, queryStart.toLong + token + 1).toInt

    for b <- 0 until batches; h <- 0 until qHeads; t <- 0 until qTokens do
      val row = (b * qHeads + h) * qTokens + t
      val qOffset = row * width
      val kvOffset = (b * kvHeads + h / group) * kTokens
      val base = row * kTokens
      val count = visible(t)
      var maximum = Double.NegativeInfinity
      var j = 0
      while j < count do
        var dot = 0.0
        var d = 0
        while d < width do
          dot += q(qOffset + d).toDouble * k((kvOffset + j) * width + d)
          d += 1
        val score = dot * scale
        probabilities(base + j) = score
        maximum = math.max(maximum, score)
        j += 1
      var denominator = 0.0
      j = 0
      while j < count do
        val p = math.exp(probabilities(base + j) - maximum)
        probabilities(base + j) = p
        denominator += p
        j += 1
      j = 0
      while j < count do
        probabilities(base + j) /= denominator
        j += 1
      var d = 0
      while d < valueWidth do
        var sum = 0.0
        j = 0
        while j < count do
          sum += probabilities(base + j) * v((kvOffset + j) * valueWidth + d)
          j += 1
        output(row * valueWidth + d) = sum.toFloat
        d += 1

    def pullback(gradient: Tensor[Float], target: Int): Tensor[Float] =
      val shape = Seq(qs, ks, vs)(target)
      val sums = new Array[Double](shape.numel.toInt)
      val upstream = gradient.toFloatArray
      val dp = if target == 2 then Array.emptyDoubleArray else new Array[Double](kTokens)
      for b <- 0 until batches; h <- 0 until qHeads; t <- 0 until qTokens do
        val row = (b * qHeads + h) * qTokens + t
        val qOffset = row * width
        val kvOffset = (b * kvHeads + h / group) * kTokens
        val base = row * kTokens
        val count = visible(t)
        var j = 0
        var weighted = 0.0
        if target != 2 then
          while j < count do
            var dot = 0.0
            var d = 0
            while d < valueWidth do
              dot += upstream(row * valueWidth + d).toDouble * v((kvOffset + j) * valueWidth + d)
              d += 1
            dp(j) = dot
            weighted += probabilities(base + j) * dot
            j += 1
        j = 0
        while j < count do
          var d = 0
          if target == 2 then
            while d < valueWidth do
              sums((kvOffset + j) * valueWidth + d) += probabilities(base + j) * upstream(row * valueWidth + d)
              d += 1
          else
            val dz = probabilities(base + j) * (dp(j) - weighted) * scale
            while d < width do
              if target == 0 then sums(qOffset + d) += dz * k((kvOffset + j) * width + d)
              else sums((kvOffset + j) * width + d) += dz * q(qOffset + d)
              d += 1
          j += 1
      Tensor.fromFloats(sums.map(_.toFloat), shape)
    Variable.result(Tensor.fromFloats(output, outputShape), Vector(
      query -> ((g: Tensor[Float]) => pullback(g, 0)), key -> ((g: Tensor[Float]) => pullback(g, 1)),
      value -> ((g: Tensor[Float]) => pullback(g, 2))))

  def rope(input: Variable, positionStart: Int, theta: Double): Variable =
    val shape = input.data.shape
    require(shape.rank == 4 && shape.dims(3) > 0 && shape.dims(3) % 2 == 0, "RoPE expects rank 4 with positive even width")
    require(positionStart >= 0 && theta.isFinite && theta > 0, "invalid rotary position or theta")
    if shape.numel == 0 then return Variable.result(input.data, Vector(input -> ((g: Tensor[Float]) => g)))
    val tokens = shape.dims(2).toInt
    val width = shape.dims(3).toInt
    val half = width / 2
    val cosines = new Array[Double](tokens * half)
    val sines = new Array[Double](tokens * half)
    for t <- 0 until tokens; d <- 0 until half do
      val angle = (positionStart.toLong + t).toDouble * math.pow(theta, -2.0 * d / width)
      require(angle.isFinite, "rotary angle overflow")
      cosines(t * half + d) = math.cos(angle)
      sines(t * half + d) = math.sin(angle)
    def rotate(values: Array[Float], inverse: Boolean): Tensor[Float] =
      val out = new Array[Float](values.length)
      var row = 0
      while row < values.length / width do
        val t = row % tokens
        var d = 0
        while d < half do
          val c = cosines(t * half + d)
          val s = sines(t * half + d) * (if inverse then -1.0 else 1.0)
          val a = values(row * width + d).toDouble
          val b = values(row * width + half + d).toDouble
          out(row * width + d) = (a * c - b * s).toFloat
          out(row * width + half + d) = (b * c + a * s).toFloat
          d += 1
        row += 1
      Tensor.fromFloats(out, shape)
    Variable.result(rotate(input.data.toFloatArray, inverse = false),
      Vector(input -> ((g: Tensor[Float]) => rotate(g.toFloatArray, inverse = true))))

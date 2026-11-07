package scalatorch.models

import scalatorch.*
import scalatorch.nn.{LowRankAdapter, Parameter}
import scala.collection.mutable

/** LyCORIS & Composable LoRA adapter algebra (CP-LoCon, LoHa, LoKr, DoRA, IA3, full diff). */
private[models] object LycorisAdapter:
  final class LayerTensors:
    val fields: mutable.Map[String, Tensor[Float]] = mutable.Map.empty
    var alpha: Option[Double] = None

  val TensorSuffixes: Vector[(String, String)] = Vector(
    ".lora_down.weight" -> "lora_down.weight",
    ".lora_up.weight" -> "lora_up.weight",
    ".lora_mid.weight" -> "lora_mid.weight",
    ".lora_A.weight" -> "lora_down.weight",
    ".lora_B.weight" -> "lora_up.weight",
    ".lora_A.default.weight" -> "lora_down.weight",
    ".lora_B.default.weight" -> "lora_up.weight",
    ".dora_scale" -> "dora_scale",
    ".lora_magnitude_vector.weight" -> "dora_scale",
    ".hada_w1_a" -> "hada_w1_a",
    ".hada_w1_b" -> "hada_w1_b",
    ".hada_w2_a" -> "hada_w2_a",
    ".hada_w2_b" -> "hada_w2_b",
    ".hada_t1" -> "hada_t1",
    ".hada_t2" -> "hada_t2",
    ".lokr_w1" -> "lokr_w1",
    ".lokr_w1_a" -> "lokr_w1_a",
    ".lokr_w1_b" -> "lokr_w1_b",
    ".lokr_w2" -> "lokr_w2",
    ".lokr_w2_a" -> "lokr_w2_a",
    ".lokr_w2_b" -> "lokr_w2_b",
    ".lokr_t2" -> "lokr_t2",
    ".diff" -> "diff",
    ".ia3_w" -> "ia3_w",
    ".weight" -> "weight",
    ".on_input" -> "on_input"
  )

  /** CP Tucker contraction: `torch.einsum('ijhw,io,jk->okhw', t, a, b)` where:
    *   - `t` has shape `(I, J, H, W)`
    *   - `a` has shape `(I, O)` (or `(O, I)` if `transposeA = true`)
    *   - `b` has shape `(J, K)`
    * Returns tensor of shape `(O, K, H, W)`.
    */
  def cp(t: Tensor[Float], a: Tensor[Float], b: Tensor[Float], transposeA: Boolean = false): Tensor[Float] =
    require(t.shape.rank == 4, s"CP core tensor must have rank 4, got ${t.shape}")
    val iDim = t.shape.dims(0).toInt
    val jDim = t.shape.dims(1).toInt
    val hDim = t.shape.dims(2).toInt
    val wDim = t.shape.dims(3).toInt
    val aFlat = if a.shape.rank == 2 then a else a.reshape(Shape.from(Seq(a.shape.dims(0), a.shape.numel / a.shape.dims(0))))
    val bFlat = if b.shape.rank == 2 then b else b.reshape(Shape.from(Seq(b.shape.dims(0), b.shape.numel / b.shape.dims(0))))
    val oDim = if transposeA then
      require(aFlat.shape.dims(1).toInt == iDim, s"CP factor A shape ${a.shape} does not match core dim $iDim")
      aFlat.shape.dims(0).toInt
    else
      require(aFlat.shape.dims(0).toInt == iDim, s"CP factor A shape ${a.shape} does not match core dim $iDim")
      aFlat.shape.dims(1).toInt
    require(bFlat.shape.dims(0).toInt == jDim, s"CP factor B shape ${b.shape} does not match core dim $jDim")
    val kDim = bFlat.shape.dims(1).toInt
    val hw = hDim * wDim
    val tArr = t.toFloatArray
    val aArr = aFlat.toFloatArray
    val bArr = bFlat.toFloatArray
    // First contract over j: temp[i, k, hw] = sum_j t[i, j, hw] * b[j, k]
    val temp = new Array[Float](iDim * kDim * hw)
    var i = 0
    while i < iDim do
      var j = 0
      while j < jDim do
        val tBase = (i * jDim + j) * hw
        val bBase = j * kDim
        var k = 0
        while k < kDim do
          val bVal = bArr(bBase + k)
          val dstBase = (i * kDim + k) * hw
          var p = 0
          while p < hw do
            temp(dstBase + p) += tArr(tBase + p) * bVal
            p += 1
          k += 1
        j += 1
      i += 1
    // Next contract over i: out[o, k, hw] = sum_i a(i, o) * temp[i, k, hw]
    val out = new Array[Float](oDim * kDim * hw)
    var o = 0
    while o < oDim do
      i = 0
      while i < iDim do
        val aVal = if transposeA then aArr(o * iDim + i) else aArr(i * oDim + o)
        val srcBase = (i * kDim) * hw
        val dstBase = (o * kDim) * hw
        var kp = 0
        val totalKp = kDim * hw
        while kp < totalKp do
          out(dstBase + kp) += aVal * temp(srcBase + kp)
          kp += 1
        i += 1
      o += 1
    Tensor.fromFloats(out, Shape.from(Seq(oDim.toLong, kDim.toLong, hDim.toLong, wDim.toLong)))

  /** Contract `mid` `(rUp, rDown, kH, kW)` with `down` `(rDown, inC, ...)` -> `(rUp, inC, kH, kW)`. */
  def contractMidDown(mid: Tensor[Float], down: Tensor[Float]): Tensor[Float] =
    require(mid.shape.rank == 4, s"lora_mid.weight must have rank 4, got ${mid.shape}")
    val rUp = mid.shape.dims(0).toInt
    val rDown = mid.shape.dims(1).toInt
    val kH = mid.shape.dims(2).toInt
    val kW = mid.shape.dims(3).toInt
    val downFlat =
      if down.shape.rank == 2 then down
      else if down.shape.rank == 4 && down.shape.dims(2) == 1L && down.shape.dims(3) == 1L then
        down.reshape(Shape.from(Seq(down.shape.dims(0), down.shape.dims(1))))
      else throw IllegalArgumentException(s"CP LoCon down factor must have 1x1 spatial dims: ${down.shape}")
    require(downFlat.shape.dims(0).toInt == rDown, s"CP LoCon rank mismatch between mid ${mid.shape} and down ${down.shape}")
    val inC = downFlat.shape.dims(1).toInt
    val hw = kH * kW
    val mArr = mid.toFloatArray
    val dArr = downFlat.toFloatArray
    val out = new Array[Float](rUp * inC * hw)
    var i = 0
    while i < rUp do
      var j = 0
      while j < rDown do
        val mBase = (i * rDown + j) * hw
        val dBase = j * inC
        var c = 0
        while c < inC do
          val dVal = dArr(dBase + c)
          val dstBase = (i * inC + c) * hw
          var p = 0
          while p < hw do
            out(dstBase + p) += mArr(mBase + p) * dVal
            p += 1
          c += 1
        j += 1
      i += 1
    Tensor.fromFloats(out, Shape.from(Seq(rUp.toLong, inC.toLong, kH.toLong, kW.toLong)))

  private def matmul2d(a: Tensor[Float], b: Tensor[Float]): Tensor[Float] =
    val a2 = if a.shape.rank == 2 then a else a.reshape(Shape.from(Seq(a.shape.dims(0), a.shape.numel / a.shape.dims(0))))
    val b2 = if b.shape.rank == 2 then b else b.reshape(Shape.from(Seq(b.shape.dims(0), b.shape.numel / b.shape.dims(0))))
    require(a2.shape.dims(1) == b2.shape.dims(0), s"matrix multiplication shape mismatch: ${a.shape} vs ${b.shape}")
    val m = a2.shape.dims(0).toInt
    val k = a2.shape.dims(1).toInt
    val n = b2.shape.dims(1).toInt
    val aArr = a2.toFloatArray
    val bArr = b2.toFloatArray
    val out = new Array[Float](m * n)
    var r = 0
    while r < m do
      var p = 0
      while p < k do
        val aVal = aArr(r * k + p)
        val bBase = p * n
        val oBase = r * n
        var c = 0
        while c < n do
          out(oBase + c) += aVal * bArr(bBase + c)
          c += 1
        p += 1
      r += 1
    Tensor.fromFloats(out, Shape.from(Seq(m.toLong, n.toLong)))

  private def kron(a: Tensor[Float], b: Tensor[Float]): Tensor[Float] =
    val rank = math.max(a.shape.rank, b.shape.rank)
    val aDims =
      if a.shape.rank == rank then a.shape.dims
      else if a.shape.rank == 2 && rank == 4 then Vector(a.shape.dims(0), a.shape.dims(1), 1L, 1L)
      else Vector.fill(rank - a.shape.rank)(1L) ++ a.shape.dims
    val bDims =
      if b.shape.rank == rank then b.shape.dims
      else Vector.fill(rank - b.shape.rank)(1L) ++ b.shape.dims
    val outDims = aDims.zip(bDims).map(_ * _)
    val aArr = a.toFloatArray
    val bArr = b.toFloatArray
    val out = new Array[Float](outDims.product.toInt)
    if rank == 2 then
      val a0 = aDims(0).toInt; val a1 = aDims(1).toInt
      val b0 = bDims(0).toInt; val b1 = bDims(1).toInt
      val o1 = outDims(1).toInt
      for i0 <- 0 until a0; i1 <- 0 until a1 do
        val aVal = aArr(i0 * a1 + i1)
        for j0 <- 0 until b0; j1 <- 0 until b1 do
          out((i0 * b0 + j0) * o1 + (i1 * b1 + j1)) = aVal * bArr(j0 * b1 + j1)
    else if rank == 4 then
      val a0 = aDims(0).toInt; val a1 = aDims(1).toInt; val a2 = aDims(2).toInt; val a3 = aDims(3).toInt
      val b0 = bDims(0).toInt; val b1 = bDims(1).toInt; val b2 = bDims(2).toInt; val b3 = bDims(3).toInt
      val o1 = outDims(1).toInt; val o2 = outDims(2).toInt; val o3 = outDims(3).toInt
      for i0 <- 0 until a0; i1 <- 0 until a1; i2 <- 0 until a2; i3 <- 0 until a3 do
        val aVal = aArr(((i0 * a1 + i1) * a2 + i2) * a3 + i3)
        for j0 <- 0 until b0; j1 <- 0 until b1; j2 <- 0 until b2; j3 <- 0 until b3 do
          val bVal = bArr(((j0 * b1 + j1) * b2 + j2) * b3 + j3)
          val r0 = i0 * b0 + j0
          val r1 = i1 * b1 + j1
          val r2 = i2 * b2 + j2
          val r3 = i3 * b3 + j3
          out(((r0 * o1 + r1) * o2 + r2) * o3 + r3) = aVal * bVal
    else throw IllegalArgumentException(s"unsupported Kronecker rank: $rank")
    Tensor.fromFloats(out, Shape.from(outDims))

  private def eye(n: Int, rank4: Boolean): Tensor[Float] =
    val arr = new Array[Float](n * n)
    var i = 0
    while i < n do
      arr(i * n + i) = 1.0f
      i += 1
    val shape = if rank4 then Shape.from(Seq(n.toLong, n.toLong, 1L, 1L)) else Shape.from(Seq(n.toLong, n.toLong))
    Tensor.fromFloats(arr, shape)

  /** Wrap a full weight delta `deltaW` of shape `baseShape` into a `LowRankAdapter` with unit scale (`alpha / r = 1`). */
  def fromFullDelta(deltaW: Tensor[Float], baseShape: Shape): LowRankAdapter =
    require(deltaW.shape == baseShape, s"adapter delta shape ${deltaW.shape} does not match target $baseShape")
    if baseShape.rank == 2 then
      val outDim = baseShape.dims(0).toInt
      val inDim = baseShape.dims(1).toInt
      if outDim <= inDim then
        LowRankAdapter(Parameter(deltaW, false), Parameter(eye(outDim, rank4 = false), false), outDim.toDouble)
      else
        LowRankAdapter(Parameter(eye(inDim, rank4 = false), false), Parameter(deltaW, false), inDim.toDouble)
    else if baseShape.rank == 4 then
      val outC = baseShape.dims(0).toInt
      val inC = baseShape.dims(1).toInt
      if baseShape.dims(2) == 1L && baseShape.dims(3) == 1L && inC < outC then
        LowRankAdapter(Parameter(eye(inC, rank4 = true), false), Parameter(deltaW, false), inC.toDouble)
      else
        LowRankAdapter(Parameter(deltaW, false), Parameter(eye(outC, rank4 = true), false), outC.toDouble)
    else throw IllegalArgumentException(s"unsupported target rank for full delta: $baseShape")

  private def applyDora(base: Tensor[Float], loraDelta: Tensor[Float], doraScale: Tensor[Float]): Tensor[Float] =
    val outDim = base.shape.dims(0).toInt
    require(doraScale.shape.numel.toInt == outDim, s"DoRA scale length ${doraScale.shape.numel} does not match out_features $outDim")
    val rowSize = (base.shape.numel / outDim).toInt
    val baseArr = base.toFloatArray
    val deltaArr = loraDelta.toFloatArray
    val scaleArr = doraScale.toFloatArray
    val out = new Array[Float](baseArr.length)
    var o = 0
    while o < outDim do
      val offset = o * rowSize
      var normSq = 0.0
      var j = 0
      while j < rowSize do
        val m = (baseArr(offset + j) + deltaArr(offset + j)).toDouble
        normSq += m * m
        j += 1
      val norm = math.max(math.sqrt(normSq), 1e-12)
      val factor = (scaleArr(o).toDouble / norm).toFloat
      j = 0
      while j < rowSize do
        val m = baseArr(offset + j) + deltaArr(offset + j)
        out(offset + j) = m * factor - baseArr(offset + j)
        j += 1
      o += 1
    Tensor.fromFloats(out, base.shape)

  def build(
      base: String,
      layer: LayerTensors,
      target: Parameter,
      maxRank: Long,
      dynamicRank: Option[Int]
  ): LowRankAdapter =
    val t = layer.fields
    val baseShape = target.data.shape
    val doraOpt = t.get("dora_scale")

    if t.contains("lora_down.weight") || t.contains("lora_up.weight") || t.contains("lora_mid.weight") then
      val rawDown = t.getOrElse("lora_down.weight", throw IllegalArgumentException(s"missing LoRA A/down factor: $base"))
      val rawUp = t.getOrElse("lora_up.weight", throw IllegalArgumentException(s"missing LoRA B/up factor: $base"))
      val linearOnPointwise = rawDown.shape.rank == 2 && baseShape.rank == 4 && baseShape.dims.drop(2) == Vector(1L, 1L)
      val down = if linearOnPointwise then rawDown.reshape(Shape.from(rawDown.shape.dims ++ Vector(1L, 1L))) else rawDown
      val up =
        if rawUp.shape.rank == 2 && baseShape.rank == 4 then rawUp.reshape(Shape.from(rawUp.shape.dims ++ Vector(1L, 1L)))
        else rawUp

      t.get("lora_mid.weight") match
        case Some(mid) =>
          require(baseShape.rank == 4 && mid.shape.rank == 4, s"CP LoCon requires 4D convolution target and mid tensor: $base")
          require(mid.shape.dims.drop(2) == baseShape.dims.drop(2), s"CP LoCon mid spatial shape mismatch: $base")
          val rDown = down.shape.dims(0)
          val rUp = up.shape.dims(1)
          require(rDown <= maxRank && rUp <= maxRank && down.shape.dims(1) == baseShape.dims(1) && up.shape.dims(0) == baseShape.dims(0),
            s"CP LoCon factor shape does not match target: $base")
          val effDown = contractMidDown(mid, down)
          val origAlpha = layer.alpha.getOrElse(rDown.toDouble)
          if doraOpt.isEmpty then
            val effAlpha = origAlpha * (rUp.toDouble / rDown.toDouble)
            LowRankAdapter(Parameter(effDown, false), Parameter(up, false), effAlpha)
          else
            val fullW = cp(mid, up, down, transposeA = true)
            val scale = (origAlpha / rDown.toDouble).toFloat
            val scaledW = Tensor.fromFloats(fullW.toFloatArray.map(_ * scale), baseShape)
            fromFullDelta(applyDora(target.data, scaledW, doraOpt.get), baseShape)
        case None =>
          val rank = down.shape.dims(0)
          require(
            rank <= maxRank && up.shape.dims(1) == rank && down.shape.dims(1) == baseShape.dims(1) &&
              up.shape.dims(0) == baseShape.dims(0),
            s"LoRA factor shape does not match target: $base"
          )
          if baseShape.rank == 4 then
            require(
              down.shape.dims.drop(2) == baseShape.dims.drop(2) && up.shape.dims.drop(2) == Vector(1L, 1L),
              s"unsupported convolution LoRA kernel shape: $base"
            )
          val requested = dynamicRank.fold(rank)(n => if n < 0 then math.max(0L, rank + n.toLong) else math.min(rank, n.toLong))
          val (actualDown, actualUp, alpha) =
            if requested == rank then (down, up, layer.alpha.getOrElse(rank.toDouble))
            else if requested == 0 then
              (Tensor.fromFloats(new Array[Float](down.shape.numel.toInt), down.shape), up, layer.alpha.getOrElse(rank.toDouble))
            else
              (
                down.slice(0, 0, requested).contiguous,
                up.slice(1, 0, requested).contiguous,
                layer.alpha.fold(requested.toDouble)(_ * requested.toDouble / rank.toDouble)
              )
          if doraOpt.isEmpty then
            LowRankAdapter(Parameter(actualDown, false), Parameter(actualUp, false), alpha)
          else
            val effRank = actualDown.shape.dims(0).toDouble
            val scale = (alpha / effRank).toFloat
            val prod = matmul2d(actualUp, actualDown)
            val scaledDelta = Tensor.fromFloats(prod.toFloatArray.map(_ * scale), baseShape)
            fromFullDelta(applyDora(target.data, scaledDelta, doraOpt.get), baseShape)

    else if t.contains("hada_w1_a") then
      require(doraOpt.isEmpty, "Composable DoRA requires conventional LoRA tensors")
      val w1a = t("hada_w1_a")
      val w1b = t.getOrElse("hada_w1_b", throw IllegalArgumentException(s"missing hada_w1_b: $base"))
      val w2a = t.getOrElse("hada_w2_a", throw IllegalArgumentException(s"missing hada_w2_a: $base"))
      val w2b = t.getOrElse("hada_w2_b", throw IllegalArgumentException(s"missing hada_w2_b: $base"))
      val rank = w1b.shape.dims(0)
      val scale = (layer.alpha.getOrElse(rank.toDouble) / rank.toDouble).toFloat
      val first = t.get("hada_t1").fold(matmul2d(w1a, w1b))(t1 => cp(t1, w1a, w1b))
      val second = t.get("hada_t2").fold(matmul2d(w2a, w2b))(t2 => cp(t2, w2a, w2b))
      require(first.shape.numel == baseShape.numel && second.shape.numel == baseShape.numel,
        s"LoHa factor product size does not match target $baseShape: $base")
      val fArr = first.toFloatArray
      val sArr = second.toFloatArray
      val out = new Array[Float](fArr.length)
      var i = 0
      while i < fArr.length do
        out(i) = fArr(i) * sArr(i) * scale
        i += 1
      fromFullDelta(Tensor.fromFloats(out, baseShape), baseShape)

    else if t.keysIterator.exists(_.startsWith("lokr_")) then
      require(doraOpt.isEmpty, "Composable DoRA requires conventional LoRA tensors")
      val rank = t.get("lokr_w1_b").orElse(t.get("lokr_w2_b")).map(_.shape.dims(0)).getOrElse(0L)
      val scale = if rank == 0L then 1.0f else (layer.alpha.getOrElse(rank.toDouble) / rank.toDouble).toFloat
      val w1 = t.get("lokr_w1").getOrElse {
        val a = t.getOrElse("lokr_w1_a", throw IllegalArgumentException(s"missing lokr_w1_a: $base"))
        val b = t.getOrElse("lokr_w1_b", throw IllegalArgumentException(s"missing lokr_w1_b: $base"))
        matmul2d(a, b)
      }
      val w2 = t.get("lokr_w2").getOrElse {
        val a = t.getOrElse("lokr_w2_a", throw IllegalArgumentException(s"missing lokr_w2_a: $base"))
        val b = t.getOrElse("lokr_w2_b", throw IllegalArgumentException(s"missing lokr_w2_b: $base"))
        t.get("lokr_t2").fold(matmul2d(a, b))(t2 => cp(t2, a, b))
      }
      val kProd = kron(w1, w2)
      require(kProd.shape.numel == baseShape.numel, s"LoKr Kronecker product shape ${kProd.shape} does not match target $baseShape: $base")
      val arr = kProd.toFloatArray
      if scale != 1.0f then
        var i = 0
        while i < arr.length do
          arr(i) *= scale
          i += 1
      fromFullDelta(Tensor.fromFloats(arr, baseShape), baseShape)

    else if t.contains("diff") then
      require(doraOpt.isEmpty, "Composable DoRA requires conventional LoRA tensors")
      val diff = t("diff")
      require(diff.shape.numel == baseShape.numel, s"Full diff adapter shape ${diff.shape} does not match target $baseShape: $base")
      val scale = layer.alpha.getOrElse(1.0).toFloat
      val arr = diff.toFloatArray.map(_ * scale)
      fromFullDelta(Tensor.fromFloats(arr, baseShape), baseShape)

    else if t.contains("weight") || t.contains("ia3_w") then
      require(doraOpt.isEmpty, "Composable DoRA requires conventional LoRA tensors")
      val w = t.getOrElse("weight", t("ia3_w"))
      val onInput = t.get("on_input").exists(x => x.toFloatArray.headOption.exists(_ != 0.0f))
      val scale = layer.alpha.getOrElse(1.0).toFloat
      val wArr = w.toFloatArray
      val baseArr = target.data.toFloatArray
      val outDim = baseShape.dims(0).toInt
      val inDim = baseShape.dims(1).toInt
      val spatial = (baseShape.numel / (outDim.toLong * inDim.toLong)).toInt
      require(wArr.length == (if onInput then inDim else outDim), s"IA3 vector length ${wArr.length} does not match target $baseShape: $base")
      val out = new Array[Float](baseArr.length)
      for o <- 0 until outDim; c <- 0 until inDim do
        val factor = (if onInput then wArr(c) else wArr(o)) * scale
        val offset = (o * inDim + c) * spatial
        var s = 0
        while s < spatial do
          out(offset + s) = baseArr(offset + s) * factor
          s += 1
      fromFullDelta(Tensor.fromFloats(out, baseShape), baseShape)

    else throw IllegalArgumentException(s"unsupported LoRA/LyCORIS tensor family for $base")

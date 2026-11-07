package scalatorch.models

import scalatorch.*

final case class KohyaHrFixConfig(
    blockNumber: Int = 3,
    downscaleFactor: Double = 2.0,
    startPercent: Double = 0.0,
    endPercent: Double = 0.35,
    downscaleAfterSkip: Boolean = true,
    downscaleMethod: String = "bicubic",
    upscaleMethod: String = "bicubic"
):
  def validate(layersPerLevel: Seq[Int]): Unit =
    val loc = KohyaHrFix.blockLocation(blockNumber, layersPerLevel)
    if loc.isEmpty then
      val maxBlock = layersPerLevel.sum + layersPerLevel.size - 1
      throw IllegalArgumentException(s"Kohya HRFix: block number $blockNumber is out of range for this UNet (max $maxBlock)")

object KohyaHrFix:
  val InterpMethods: Set[String] = Set("bicubic", "nearest-exact", "bilinear", "area", "bislerp")

  private val localState = new ThreadLocal[Option[(KohyaHrFixConfig, Float)]]:
    override def initialValue(): Option[(KohyaHrFixConfig, Float)] = None

  def current: Option[(KohyaHrFixConfig, Float)] = localState.get()

  def withStep[A](cfg: Option[KohyaHrFixConfig], timestep: Float)(body: => A): A =
    val prev = localState.get()
    localState.set(cfg.map((_, timestep)))
    try body
    finally localState.set(prev)

  def blockLocation(blockNumber: Int, layersPerLevel: Seq[Int]): Option[(Int, Int)] =
    var n = blockNumber
    if n < 1 then None
    else
      var level = 0
      while level < layersPerLevel.size do
        val layers = layersPerLevel(level)
        val last = level == layersPerLevel.size - 1
        val span = layers + (if last then 0 else 1)
        if n <= span then return Some((level, n - 1))
        n -= span
        level += 1
      None

  def timestepWindow(startPercent: Double, endPercent: Double): (Double, Double) =
    (999.0 * (1.0 - startPercent), 999.0 * (1.0 - endPercent))

  def isActive(cfg: KohyaHrFixConfig, timestep: Float): Boolean =
    val (tHi, tLo) = timestepWindow(cfg.startPercent, cfg.endPercent)
    timestep >= tLo && timestep <= tHi

  private def cubic(x: Double): Double =
    val a = -0.75
    val v = math.abs(x)
    if v < 1.0 then (a + 2.0) * v * v * v - (a + 3.0) * v * v + 1.0
    else if v < 2.0 then a * v * v * v - 5.0 * a * v * v + 8.0 * a * v - 4.0 * a
    else 0.0

  private def linear(x: Double): Double = math.max(0.0, 1.0 - math.abs(x))

  private def axisWeights(input: Int, output: Int, cubicMode: Boolean): Array[Array[(Int, Double)]] =
    val scale = input.toDouble / output.toDouble
    val support = if cubicMode then 2.0 else 1.0
    Array.tabulate(output) { position =>
      val center = (position + 0.5) * scale
      val start = math.ceil(center - support - 0.5).toInt
      val end = math.floor(center + support - 0.5).toInt + 1
      val weights = (start until end).map { rawIndex =>
        val distance = rawIndex + 0.5 - center
        val idx = rawIndex.max(0).min(input - 1)
        val w = if cubicMode then cubic(distance) else linear(distance)
        (idx, w)
      }.filter(_._2 != 0.0)
      val total = weights.map(_._2).sum
      weights.map { case (idx, w) => (idx, w / total) }.toArray
    }

  def interpolate2d(x: Tensor[Float], targetH: Int, targetW: Int, rawMethod: String): Tensor[Float] =
    val dims = x.shape.dims
    if dims.size != 4 then return x
    val inH = dims(2).toInt
    val inW = dims(3).toInt
    if inH == targetH && inW == targetW then return x
    val batches = dims(0).toInt
    val channels = dims(1).toInt
    val values = x.toFloatArray
    val method = if InterpMethods.contains(rawMethod) then rawMethod else "bicubic"
    val out = method match
      case "nearest-exact" =>
        val res = new Array[Float](batches * channels * targetH * targetW)
        val planeIn = inH * inW
        val planeOut = targetH * targetW
        for b <- 0 until batches; c <- 0 until channels; y <- 0 until targetH; px <- 0 until targetW do
          val sx = math.floor((px + 0.5) * inW.toDouble / targetW).toInt.min(inW - 1)
          val sy = math.floor((y + 0.5) * inH.toDouble / targetH).toInt.min(inH - 1)
          res((b * channels + c) * planeOut + y * targetW + px) = values((b * channels + c) * planeIn + sy * inW + sx)
        res
      case "area" =>
        val res = new Array[Float](batches * channels * targetH * targetW)
        val planeIn = inH * inW
        val planeOut = targetH * targetW
        for b <- 0 until batches; c <- 0 until channels; y <- 0 until targetH; px <- 0 until targetW do
          val y0 = (y.toLong * inH / targetH).toInt
          val y1 = math.max(y0 + 1, (((y.toLong + 1) * inH + targetH - 1) / targetH).toInt.min(inH))
          val x0 = (px.toLong * inW / targetW).toInt
          val x1 = math.max(x0 + 1, (((px.toLong + 1) * inW + targetW - 1) / targetW).toInt.min(inW))
          var sum = 0f
          for sy <- y0 until y1; sx <- x0 until x1 do
            sum += values((b * channels + c) * planeIn + sy * inW + sx)
          res((b * channels + c) * planeOut + y * targetW + px) = sum / ((y1 - y0) * (x1 - x0)).toFloat
        res
      case _ =>
        val cubicMode = method != "bilinear"
        val horiz = axisWeights(inW, targetW, cubicMode)
        val vert = axisWeights(inH, targetH, cubicMode)
        val planeIn = inH * inW
        val planeMid = inH * targetW
        val planeOut = targetH * targetW
        val mid = new Array[Float](batches * channels * planeMid)
        for b <- 0 until batches; c <- 0 until channels; y <- 0 until inH; px <- 0 until targetW do
          val ws = horiz(px)
          var acc = 0.0
          var i = 0
          while i < ws.length do
            val (sx, w) = ws(i)
            acc += values((b * channels + c) * planeIn + y * inW + sx) * w
            i += 1
          mid((b * channels + c) * planeMid + y * targetW + px) = acc.toFloat
        val res = new Array[Float](batches * channels * planeOut)
        for b <- 0 until batches; c <- 0 until channels; y <- 0 until targetH; px <- 0 until targetW do
          val ws = vert(y)
          var acc = 0.0
          var i = 0
          while i < ws.length do
            val (sy, w) = ws(i)
            acc += mid((b * channels + c) * planeMid + sy * targetW + px) * w
            i += 1
          res((b * channels + c) * planeOut + y * targetW + px) = acc.toFloat
        res
    Tensor.fromFloats(out, Shape.from(Seq(batches.toLong, channels.toLong, targetH.toLong, targetW.toLong)))

  def shrinkIfActive(x: Tensor[Float], cfg: KohyaHrFixConfig, timestep: Float): Tensor[Float] =
    val f = cfg.downscaleFactor
    if !isActive(cfg, timestep) || f == 1.0 || x.shape.rank != 4 then x
    else
      val h = math.max(1, math.round(x.shape.dims(2).toDouble / f).toInt)
      val w = math.max(1, math.round(x.shape.dims(3).toDouble / f).toInt)
      interpolate2d(x, h, w, cfg.downscaleMethod)

final case class TokenDownsamplingConfig(
    factor: Double = 1.0,
    factor2: Double = 1.0,
    disableAfter: Double = 1.0
):
  require(factor.isFinite && factor >= 1.0 && factor <= 5.0, "Token Downsampling factor must be finite in [1, 5]")
  require(factor2.isFinite && factor2 >= 1.0 && factor2 <= 5.0, "Token Downsampling factor2 must be finite in [1, 5]")
  require(disableAfter.isFinite && disableAfter >= 0.0 && disableAfter <= 1.0, "Token Downsampling disable_after must be finite in [0, 1]")

  def enabledFor(steps: Int): Boolean =
    val limit = (disableAfter * steps).toInt
    limit > 0 && (factor > 1.0 || factor2 > 1.0)

object TokenDownsampling:
  final case class ActiveState(
      factors: Map[Int, Double],
      active: Boolean,
      stages: Int
  ):
    def factorForSection(section: String): Double =
      if !active then 1.0
      else
        val depthOpt =
          if section.startsWith("down.") then
            section.drop(5).toIntOption.map(idx => 1 << idx)
          else if section == "mid" then
            Some(1 << (stages - 1))
          else if section.startsWith("up.") then
            section.drop(3).toIntOption.map(idx => 1 << (stages - idx - 1))
          else None
        depthOpt.flatMap(factors.get).getOrElse(1.0)

  private val localState = new ThreadLocal[Option[ActiveState]]:
    override def initialValue(): Option[ActiveState] = None

  private val localSpatialShape = new ThreadLocal[Option[(Int, Int)]]:
    override def initialValue(): Option[(Int, Int)] = None

  def current: Option[ActiveState] = localState.get()
  def currentSpatialShape: Option[(Int, Int)] = localSpatialShape.get()

  def withSpatialShape[A](height: Int, width: Int)(body: => A): A =
    val prev = localSpatialShape.get()
    localSpatialShape.set(Some((height, width)))
    try body
    finally localSpatialShape.set(prev)

  def withStep[A](cfg: Option[TokenDownsamplingConfig], sdxl: Boolean, stepOneBased: Int, totalSteps: Int, stages: Int = 4)(body: => A): A =
    val prev = localState.get()
    val state = cfg.map { c =>
      val limit = (c.disableAfter * totalSteps).toInt
      val factors = if sdxl then Map(2 -> c.factor, 4 -> c.factor2) else Map(1 -> c.factor, 2 -> c.factor2)
      ActiveState(factors, stepOneBased <= limit && (c.factor > 1.0 || c.factor2 > 1.0), stages)
    }
    localState.set(state)
    try body
    finally localState.set(prev)

  /** Downsamples 3D token states (B, H*W, C) using nearest-exact interpolation to (B, targetH*targetW, C). */
  def resampleTokens(states: Tensor[Float], height: Int, width: Int, factor: Double): Tensor[Float] =
    val dims = states.shape.dims
    require(dims.size == 3 && height.toLong * width == dims(1), "Token Downsampling requires an unambiguous spatial token layout")
    if factor <= 1.0 then states
    else
      val batch = dims(0).toInt
      val channels = dims(2).toInt
      val targetH = math.max(1, (height / factor).toInt)
      val targetW = math.max(1, (width / factor).toInt)
      if targetH == height && targetW == width then states
      else
        val src = states.toFloatArray
        val outSeq = targetH * targetW
        val out = new Array[Float](batch * outSeq * channels)
        val inSeq = height * width
        for b <- 0 until batch; ty <- 0 until targetH; tx <- 0 until targetW do
          val sy = math.floor((ty + 0.5) * height.toDouble / targetH).toInt.min(height - 1)
          val sx = math.floor((tx + 0.5) * width.toDouble / targetW).toInt.min(width - 1)
          val srcOffset = (b * inSeq + sy * width + sx) * channels
          val dstOffset = (b * outSeq + ty * targetW + tx) * channels
          System.arraycopy(src, srcOffset, out, dstOffset, channels)
        Tensor.fromFloats(out, Shape.from(Seq(batch.toLong, outSeq.toLong, channels.toLong)))

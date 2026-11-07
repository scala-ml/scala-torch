package scalatorch.models

import scalatorch.*

final case class CadsConfig(
    tau1: Double = 0.6,
    tau2: Double = 0.9,
    noiseScale: Double = 0.25,
    mixingFactor: Double = 1.0,
    rescale: Boolean = true,
    applyToUncond: Boolean = false
):
  require(
    tau1.isFinite && tau1 >= 0 && tau1 <= 1 &&
      tau2.isFinite && tau2 >= 0 && tau2 <= 1 &&
      noiseScale.isFinite && noiseScale >= 0 &&
      mixingFactor.isFinite && mixingFactor >= 0 && mixingFactor <= 1,
    "invalid CADS configuration parameters"
  )
  def newState(seed: Long): CadsState = new CadsState(this, seed)

final class CadsState(val config: CadsConfig, seed: Long):
  private val rng = new TorchCpuNormal(seed & 0xFFFFFFFFL)
  def step(
      conditional: Tensor[Float],
      unconditional: Option[Tensor[Float]],
      progress: Double
  ): (Tensor[Float], Option[Tensor[Float]]) =
    val g = Cads.gamma(progress, config.tau1, config.tau2)
    if g >= 1.0 then (conditional, unconditional)
    else
      unconditional match
        case Some(u) if config.applyToUncond =>
          val uNoise = Tensor.fromFloats(rng.normal(u.shape.numel.toInt), u.shape)
          val cNoise = Tensor.fromFloats(rng.normal(conditional.shape.numel.toInt), conditional.shape)
          val uPerturbed = Cads.perturb(u, g, config.noiseScale, config.mixingFactor, config.rescale, uNoise)
          val cPerturbed = Cads.perturb(conditional, g, config.noiseScale, config.mixingFactor, config.rescale, cNoise)
          (cPerturbed, Some(uPerturbed))
        case _ =>
          val cNoise = Tensor.fromFloats(rng.normal(conditional.shape.numel.toInt), conditional.shape)
          val cPerturbed = Cads.perturb(conditional, g, config.noiseScale, config.mixingFactor, config.rescale, cNoise)
          (cPerturbed, unconditional)

object Cads:
  def gamma(progress: Double, tau1: Double, tau2: Double): Double =
    val lo = math.min(tau1, tau2)
    val hi = math.max(tau1, tau2)
    val t = 1.0 - progress
    if t <= lo then 1.0
    else if t >= hi then 0.0
    else (hi - t) / (hi - lo)

  def perturb(
      y: Tensor[Float],
      gamma: Double,
      noiseScale: Double,
      psi: Double,
      rescale: Boolean,
      noise: Tensor[Float]
  ): Tensor[Float] =
    require(y.shape == noise.shape && y.shape.rank == 3, "CADS perturb requires matching rank-3 (B, T, D) tensors")
    if gamma >= 1.0 then y
    else
      val dims = y.shape.dims
      val b = dims(0).toInt
      val planeSize = (dims(1) * dims(2)).toInt
      val yArr = y.toFloatArray
      val nArr = noise.toFloatArray
      val out = new Array[Float](yArr.length)
      val sg = math.sqrt(math.max(gamma, 0.0))
      val s1g = noiseScale * math.sqrt(math.max(1.0 - gamma, 0.0))
      val yh = new Array[Double](planeSize)
      var batchIdx = 0
      while batchIdx < b do
        val base = batchIdx * planeSize
        var sum0 = 0.0
        var sum1 = 0.0
        var i = 0
        while i < planeSize do
          val yv = yArr(base + i).toDouble
          val yhv = sg * yv + s1g * nArr(base + i).toDouble
          yh(i) = yhv
          sum0 += yv
          sum1 += yhv
          i += 1
        if rescale && planeSize > 1 then
          val m0 = sum0 / planeSize
          val m1 = sum1 / planeSize
          var sq0 = 0.0
          var sq1 = 0.0
          i = 0
          while i < planeSize do
            val d0 = yArr(base + i).toDouble - m0
            val d1 = yh(i) - m1
            sq0 += d0 * d0
            sq1 += d1 * d1
            i += 1
          val s0 = math.sqrt(sq0 / (planeSize - 1))
          val s1 = math.max(math.sqrt(sq1 / (planeSize - 1)), 1e-6)
          i = 0
          while i < planeSize do
            val yr = (yh(i) - m1) / s1 * s0 + m0
            out(base + i) = (psi * yr + (1.0 - psi) * yh(i)).toFloat
            i += 1
        else
          i = 0
          while i < planeSize do
            out(base + i) = yh(i).toFloat
            i += 1
        batchIdx += 1
      Tensor.fromFloats(out, y.shape)

final case class HyperTileConfig(
    tileSize: Int = 256,
    swapSize: Int = 2,
    maxDepth: Int = 0,
    scaleDepth: Boolean = false
):
  require(tileSize >= 8 && swapSize >= 1 && maxDepth >= 0 && maxDepth <= 8, "invalid HyperTile configuration")
  def newState(latentH: Int, latentW: Int, seed: Long): HyperTileState =
    new HyperTileState(this, latentH, latentW, seed)

final class HyperTileState(val config: HyperTileConfig, val latentH: Int, val latentW: Int, seed: Long):
  private val rng = new java.util.Random(seed)
  def split(seq: Int): Option[(Int, Int, Int, Int)] =
    HyperTile.split(seq, latentH, latentW, config.tileSize, config.swapSize, config.maxDepth, config.scaleDepth, Some(rng))

object HyperTile:
  private val localState = new ThreadLocal[Option[HyperTileState]]:
    override def initialValue(): Option[HyperTileState] = None
  private val localInSpatial = new ThreadLocal[Boolean]:
    override def initialValue(): Boolean = false

  def current: Option[HyperTileState] =
    if localInSpatial.get() then localState.get() else None

  def currentUnscoped: Option[HyperTileState] =
    localState.get()

  def withState[A](state: Option[HyperTileState])(body: => A): A =
    val prev = localState.get()
    localState.set(state)
    try body
    finally localState.set(prev)

  def inSpatial[A](body: => A): A =
    val prev = localInSpatial.get()
    localInSpatial.set(true)
    try body
    finally localInSpatial.set(prev)

  def randomDivisor(value: Int, minValue: Int, maxOptions: Int = 1, rng: Option[java.util.Random] = None): Int =
    val clampedMin = math.max(1, math.min(minValue, value))
    val divisors = (clampedMin to value).filter(i => value % i == 0).take(math.max(1, maxOptions))
    val ns = divisors.map(i => value / i)
    if ns.size > 1 then
      val idx = rng.fold(0)(_.nextInt(ns.size))
      ns(idx)
    else ns.head

  def split(
      seq: Int,
      latentH: Int,
      latentW: Int,
      tileSize: Int,
      swapSize: Int,
      maxDepth: Int,
      scaleDepth: Boolean,
      rng: Option[java.util.Random] = None
  ): Option[(Int, Int, Int, Int)] =
    val latentTile = math.max(8, tileSize) / 8
    var depth = 0
    var found: Option[(Int, Int, Int, Int)] = None
    var done = false
    while depth <= maxDepth && !done do
      val scale = 1 << depth
      val h = latentH / scale
      val w = latentW / scale
      if h * w == seq && latentH % scale == 0 && latentW % scale == 0 then
        val factor = if scaleDepth then scale else 1
        val nh = randomDivisor(h, latentTile * factor, swapSize, rng)
        val nw = randomDivisor(w, latentTile * factor, swapSize, rng)
        if nh * nw > 1 then found = Some((nh, nw, h, w))
        done = true
      depth += 1
    found

  /** Reshapes (B, H*W, C) -> (B, nh, h/nh, nw, w/nw, C) -> permute(0,1,3,2,4,5) -> (B*nh*nw, (h/nh)*(w/nw), C). */
  def tileTokens(input: Tensor[Float], nh: Int, nw: Int, h: Int, w: Int): Tensor[Float] =
    val dims = input.shape.dims
    val b = dims(0).toInt
    val c = dims(2).toInt
    val th = h / nh
    val tw = w / nw
    val src = input.toFloatArray
    val dst = new Array[Float](src.length)
    var bi = 0
    while bi < b do
      var nhi = 0
      while nhi < nh do
        var nwi = 0
        while nwi < nw do
          val dstBatch = ((bi * nh + nhi) * nw + nwi) * (th * tw * c)
          var thi = 0
          while thi < th do
            val sy = nhi * th + thi
            var twi = 0
            while twi < tw do
              val sx = nwi * tw + twi
              val srcOffset = (bi * (h * w) + sy * w + sx) * c
              val dstOffset = dstBatch + (thi * tw + twi) * c
              System.arraycopy(src, srcOffset, dst, dstOffset, c)
              twi += 1
            thi += 1
          nwi += 1
        nhi += 1
      bi += 1
    Tensor.fromFloats(dst, Shape.from(Seq((b * nh * nw).toLong, (th * tw).toLong, c.toLong)))

  /** Restores (B*nh*nw, (h/nh)*(w/nw), C) -> (B, nh, nw, h/nh, w/nw, C) -> permute(0,1,3,2,4,5) -> (B, H*W, C). */
  def untileTokens(output: Tensor[Float], b: Int, nh: Int, nw: Int, h: Int, w: Int): Tensor[Float] =
    val c = output.shape.dims(2).toInt
    val th = h / nh
    val tw = w / nw
    val src = output.toFloatArray
    val dst = new Array[Float](src.length)
    var bi = 0
    while bi < b do
      var nhi = 0
      while nhi < nh do
        var nwi = 0
        while nwi < nw do
          val srcBatch = ((bi * nh + nhi) * nw + nwi) * (th * tw * c)
          var thi = 0
          while thi < th do
            val dy = nhi * th + thi
            var twi = 0
            while twi < tw do
              val dx = nwi * tw + twi
              val srcOffset = srcBatch + (thi * tw + twi) * c
              val dstOffset = (bi * (h * w) + dy * w + dx) * c
              System.arraycopy(src, srcOffset, dst, dstOffset, c)
              twi += 1
            thi += 1
          nwi += 1
        nhi += 1
      bi += 1
    Tensor.fromFloats(dst, Shape.from(Seq(b.toLong, (h * w).toLong, c.toLong)))

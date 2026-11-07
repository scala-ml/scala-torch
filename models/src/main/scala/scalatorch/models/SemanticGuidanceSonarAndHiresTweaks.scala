package scalatorch.models

import scalatorch.*

final case class SemanticGuidanceConfig(
    prompt: String = "",
    negativePrompt: String = "",
    warmup: Int = 10,
    editGuidanceScale: Double = 1.0,
    tailPercentageThreshold: Double = 0.05,
    momentumScale: Double = 0.3,
    momentumBeta: Double = 0.6
):
  require(
    warmup >= 0 && editGuidanceScale.isFinite && tailPercentageThreshold.isFinite &&
      tailPercentageThreshold >= 0.0 && tailPercentageThreshold <= 1.0 &&
      momentumScale.isFinite && momentumBeta.isFinite,
    "Semantic Guidance: invalid settings"
  )

final case class EncodedSemanticConcept(
    embeds: Tensor[Float],
    pooled: Option[Tensor[Float]],
    strength: Double
)

object SemanticGuidance:
  def parseConcepts(prompt: String, negative: String): Vector[(String, Double)] =
    val out = Vector.newBuilder[(String, Double)]
    for (text, sign) <- Seq((prompt, 1.0), (negative, -1.0)) if text.nonEmpty do
      for rawPart <- text.split(",", -1) do
        val part = rawPart.trim
        val span = DiffusionPrompt.parse(part).head
        out += ((span.text, sign * span.weight))
    out.result()

  /** Inverse standard normal CDF (Acklam's rational approximation, max rel error < 1.15e-9). */
  def zScore(tail: Double): Double =
    if tail <= 0.0 then Double.PositiveInfinity
    else if tail >= 1.0 then Double.NegativeInfinity
    else
      val p = 1.0 - tail
      val a1 = -3.969683028665376e+01
      val a2 = 2.209460984245205e+02
      val a3 = -2.759285104469687e+02
      val a4 = 1.383577518672690e+02
      val a5 = -3.066479806614716e+01
      val a6 = 2.506628277459239e+00
      val b1 = -5.447609879822406e+01
      val b2 = 1.615858368580409e+02
      val b3 = -1.556989798598866e+02
      val b4 = 6.680131188771972e+01
      val b5 = -1.328068155288572e+01
      val c1 = -7.784894002430293e-03
      val c2 = -3.223964580411365e-01
      val c3 = -2.400758277161838e+00
      val c4 = -2.549732539343734e+00
      val c5 = 4.374664141464968e+00
      val c6 = 2.938163982698783e+00
      val d1 = 7.784695709041462e-03
      val d2 = 3.224671290700398e-01
      val d3 = 2.445134137142996e+00
      val d4 = 3.754408661907416e+00
      val pLow = 0.02425
      val pHigh = 1.0 - pLow
      if p < pLow then
        val q = math.sqrt(-2.0 * math.log(p))
        (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
          ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
      else if p <= pHigh then
        val q = p - 0.5
        val r = q * q
        (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
          (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
      else
        val q = math.sqrt(-2.0 * math.log(1.0 - p))
        -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
          ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

  def editDirection(concept: Array[Float], uncond: Array[Float], strength: Double, scale: Double, z: Double): Array[Float] =
    require(concept.length == uncond.length, "Semantic Guidance concept/uncond length mismatch")
    val n = concept.length
    val thr =
      if !z.isFinite then z
      else
        var sum = 0.0
        var i = 0
        while i < n do
          sum += concept(i)
          i += 1
        val mean = sum / math.max(1, n)
        var sq = 0.0
        i = 0
        while i < n do
          val d = concept(i) - mean
          sq += d * d
          i += 1
        val std = math.sqrt(sq / math.max(1, n - 1))
        mean + z * std
    val out = new Array[Float](n)
    var i = 0
    while i < n do
      val ed = strength * (concept(i).toDouble - uncond(i).toDouble)
      if math.abs(ed) > thr then out(i) = (ed * scale).toFloat
      i += 1
    out

  def applyToConditioning(
      pos: Tensor[Float],
      uncond: Tensor[Float],
      emptyChunk: Tensor[Float],
      concepts: Vector[EncodedSemanticConcept],
      cfg: SemanticGuidanceConfig,
      posAdded: Option[DiffusionAddedConditioning] = None,
      negAdded: Option[DiffusionAddedConditioning] = None,
      additionTimeWidth: Int = 256
  ): (Tensor[Float], Option[DiffusionAddedConditioning]) =
    if concepts.isEmpty then (pos, posAdded)
    else
      val z = zScore(cfg.tailPercentageThreshold)
      val scale = cfg.editGuidanceScale
      val targetTokens = uncond.shape.dims(1).toInt
      val targetChunks = targetTokens / 77
      val uncondArr = uncond.toFloatArray
      val posArr = pos.toFloatArray.clone()
      for c <- concepts do
        val cTokens = c.embeds.shape.dims(1).toInt
        if cTokens > targetTokens then
          throw IllegalArgumentException("Semantic Guidance concept is longer than the prompt; shorten it")
        val padded =
          if cTokens < targetTokens then ClipPromptEncoding.padTo(c.embeds, targetChunks, emptyChunk, 67108864L)
          else c.embeds
        val dir = editDirection(padded.toFloatArray, uncondArr, c.strength, scale, z)
        var i = 0
        while i < posArr.length do
          posArr(i) += dir(i)
          i += 1
      val nextPos = Tensor.fromFloats(posArr, pos.shape)
      val nextPosAdded = (posAdded, negAdded) match
        case (Some(pa), Some(na)) if concepts.head.pooled.isDefined && additionTimeWidth >= 2 =>
          val pooledLen = pa.pooledText.shape.numel.toInt
          val totalWidth = pooledLen.toLong + pa.timeIds.shape.dims(1) * additionTimeWidth
          val uncondVec = na.features(1L, additionTimeWidth, totalWidth, 16777216L).toFloatArray
          val sizePart = uncondVec.drop(pooledLen)
          val newPooled = pa.pooledText.toFloatArray.clone()
          for c <- concepts do
            val cPool = c.pooled.get.toFloatArray
            val cVec = new Array[Float](uncondVec.length)
            System.arraycopy(cPool, 0, cVec, 0, pooledLen)
            System.arraycopy(sizePart, 0, cVec, pooledLen, sizePart.length)
            val dir = editDirection(cVec, uncondVec, c.strength, scale, z)
            var k = 0
            while k < pooledLen do
              newPooled(k) += dir(k)
              k += 1
          Some(DiffusionAddedConditioning(Tensor.fromFloats(newPooled, pa.pooledText.shape), pa.timeIds))
        case _ => posAdded
      (nextPos, nextPosAdded)

final case class SonarConfig(
    sampler: String = "Euler a",
    momentum: Double = 0.95,
    momentumHist: Double = 0.75,
    momentumHistInit: String = "zero",
    momentumSign: String = "pos",
    refImg: String = "",
    refMeth: String = "linear",
    refHgf: Double = 0.2,
    refMinStep: Double = 0.0,
    refMaxStep: Double = 0.75,
    upscale: Boolean = false,
    upscaler: String = "Lanczos",
    upscaleRatio: Double = 1.0,
    upscaleWidth: Int = 0,
    upscaleHeight: Int = 0
):
  require(Sonar.Samplers.contains(sampler), s"Sonar: unknown sampler '$sampler' (use ${Sonar.Samplers.mkString(", ")})")
  require(Sonar.Signs.contains(momentumSign), s"Sonar: unknown momentum sign '$momentumSign'")
  require(Sonar.HistoryInits.contains(momentumHistInit), s"Sonar: unknown momentum hist init '$momentumHistInit'")
  require(Sonar.RefMethods.contains(refMeth), s"Sonar: unknown ref meth '$refMeth'")
  require(
    momentum.isFinite && momentum >= 0.0 && momentum <= 1.0 &&
      momentumHist.isFinite && momentumHist >= 0.0 && momentumHist <= 1.0 &&
      refHgf.isFinite && refHgf >= 0.0 && refHgf <= 1.0,
    "Sonar: momentum, momentum_hist, and ref_hgf must be between 0 and 1"
  )
  require(refMinStep.isFinite && refMinStep >= 0.0 && refMaxStep.isFinite && refMaxStep >= 0.0, "Sonar: the reference step bounds must be >= 0")
  require(upscaleRatio.isFinite && upscaleRatio >= 1.0 && upscaleWidth >= 0 && upscaleHeight >= 0, "Sonar: the upscale ratio must be >= 1 and the size >= 0")

  def effectiveSampler: String = if sampler == "Naive" then "Euler" else sampler
  def hasUpscale: Boolean = upscale && upscaler != "None" && (upscaleRatio > 1.0 || upscaleWidth > 0 || upscaleHeight > 0)

object Sonar:
  val Samplers: Seq[String] = Vector("Euler a", "Euler", "Naive")
  val Signs: Set[String] = Set("pos", "neg", "rand")
  val HistoryInits: Set[String] = Set("zero", "rand_init", "rand_new")
  val RefMethods: Set[String] = Set("linear", "euler")
  val NaiveNoise: Float = 1e-5f
  val RefStrengthDivisor: Double = 10.0

  def stepBound(value: Double, steps: Int): Int =
    if value > 1.0 then value.toInt else math.round(value * steps).toInt

  def momentumStep(
      x: Array[Float],
      d: Array[Float],
      dt: Double,
      momentum: Double,
      momentumHist: Double,
      sign: String,
      history: Option[Array[Float]]
  ): (Array[Float], Array[Float]) =
    val p = 1.0 - momentum
    val n = x.length
    val corrected = new Array[Float](n)
    var i = 0
    if sign == "pos" then
      while i < n do
        val h = history.fold(0.0)(arr => arr(i).toDouble)
        corrected(i) = ((1.0 - p) * d(i) + p * h).toFloat
        i += 1
    else if sign == "neg" then
      while i < n do
        val h = history.fold(0.0)(arr => arr(i).toDouble)
        corrected(i) = ((1.0 + p) * d(i) - p * h).toFloat
        i += 1
    else throw IllegalArgumentException(s"unknown momentum sign $sign")

    val nextX = new Array[Float](n)
    i = 0
    while i < n do
      nextX(i) = (x(i) + corrected(i) * dt).toFloat
      i += 1

    val q = 1.0 - momentumHist
    val nextHist = history match
      case None => corrected
      case Some(prev) =>
        val hOut = new Array[Float](n)
        var k = 0
        if sign == "pos" then
          while k < n do
            hOut(k) = ((1.0 - q) * prev(k) + q * corrected(k)).toFloat
            k += 1
        else
          while k < n do
            hOut(k) = ((1.0 + q) * prev(k) - q * corrected(k)).toFloat
            k += 1
        hOut
    (nextX, nextHist)

  def normalizeReference(latent: Tensor[Float]): Tensor[Float] =
    val dims = latent.shape.dims
    val batch = dims(0).toInt
    val channels = dims(1).toInt
    val plane = dims(2).toInt * dims(3).toInt
    val arr = latent.toFloatArray
    val out = new Array[Float](arr.length)
    for b <- 0 until batch; c <- 0 until channels do
      val base = (b * channels + c) * plane
      var sum = 0.0
      var i = 0
      while i < plane do
        sum += arr(base + i)
        i += 1
      val mean = sum / math.max(1, plane)
      var sq = 0.0
      i = 0
      while i < plane do
        val diff = arr(base + i) - mean
        sq += diff * diff
        i += 1
      val std = math.max(1e-8, math.sqrt(sq / math.max(1, plane - 1)))
      i = 0
      while i < plane do
        out(base + i) = ((arr(base + i) - mean) / std).toFloat
        i += 1
    Tensor.fromFloats(out, latent.shape)

  def guidanceStep(
      x: Array[Float],
      shape: Shape,
      method: String,
      reference: Tensor[Float],
      strength: Double,
      denoised: Array[Float],
      sigma: Double,
      sigmaNext: Double
  ): Array[Float] =
    val dims = shape.dims
    val batch = dims(0).toInt
    val elemPerBatch = (dims(1) * dims(2) * dims(3)).toInt
    val refArr = reference.toFloatArray
    val out = new Array[Float](x.length)
    val srcForStats = if method == "euler" then denoised else x
    for b <- 0 until batch do
      val base = b * elemPerBatch
      val refBase = math.min(b, reference.shape.dims(0).toInt - 1) * elemPerBatch
      var sum = 0.0
      var i = 0
      while i < elemPerBatch do
        sum += srcForStats(base + i)
        i += 1
      val mean = sum / math.max(1, elemPerBatch)
      var sq = 0.0
      i = 0
      while i < elemPerBatch do
        val diff = srcForStats(base + i) - mean
        sq += diff * diff
        i += 1
      val std = math.sqrt(sq / math.max(1, elemPerBatch - 1))
      i = 0
      if method == "euler" then
        val stepFactor = (sigmaNext - sigma) * strength
        while i < elemPerBatch do
          val shifted = refArr(refBase + i) * std + mean
          val d = (x(base + i) - shifted) / sigma
          out(base + i) = (x(base + i) + d * stepFactor).toFloat
          i += 1
      else
        while i < elemPerBatch do
          val shifted = refArr(refBase + i) * std + mean
          out(base + i) = ((1.0 - strength) * x(base + i) + strength * shifted).toFloat
          i += 1
    out

  def upscaleTarget(width: Int, height: Int, cfg: SonarConfig): Option[(Int, Int, Boolean)] =
    if !cfg.upscale || cfg.upscaler == "None" then None
    else
      val w0 = cfg.upscaleWidth
      val h0 = cfg.upscaleHeight
      if w0 == 0 && h0 == 0 then
        if cfg.upscaleRatio == 1.0 then None
        else Some((math.round(width * cfg.upscaleRatio).toInt, math.round(height * cfg.upscaleRatio).toInt, false))
      else
        val w = if w0 == 0 then math.round(width.toDouble * h0 / height).toInt else w0
        val h = if h0 == 0 then math.round(height.toDouble * w0 / width).toInt else h0
        Some((w, h, true))

final class SonarStepper(
    val cfg: SonarConfig,
    val steps: Int,
    val seed: Long,
    val ancestral: Boolean,
    val reference: Option[Tensor[Float]] = None
):
  private val naive = cfg.sampler == "Naive"
  private var calls = 0
  private var history: Option[Array[Float]] = None
  private var started = false
  private val minStep = Sonar.stepBound(cfg.refMinStep, steps)
  private val maxStep = Sonar.stepBound(cfg.refMaxStep, steps)
  private val strength = cfg.refHgf / Sonar.RefStrengthDivisor
  private val rng = new TorchCpuNormal(seed & 0xffffffffffffL)
  private val dirRng = new java.util.Random(seed & 0xffffffffffffL)

  private def direction(): String =
    if cfg.momentumSign == "rand" then
      if dirRng.nextDouble() < 0.5 then "pos" else "neg"
    else cfg.momentumSign

  private def move(x: Array[Float], d: Array[Float], dt: Double): Array[Float] =
    if cfg.momentum < 1.0 then
      val (nx, nh) = Sonar.momentumStep(x, d, dt, cfg.momentum, cfg.momentumHist, direction(), history)
      history = Some(nh)
      nx
    else
      Array.tabulate(x.length)(i => (x(i) + d(i) * dt).toFloat)

  def step(
      sample: Tensor[Float],
      modelOutput: Tensor[Float],
      sigma: Double,
      sigmaNext: Double,
      prediction: DiffusionPrediction,
      ancestralNoise: Option[Tensor[Float]] = None
  ): Tensor[Float] =
    val x = sample.toFloatArray
    val m = modelOutput.toFloatArray
    val n = x.length
    val denoised = new Array[Float](n)
    var i = 0
    prediction match
      case DiffusionPrediction.Epsilon =>
        while i < n do
          denoised(i) = (x(i) - sigma * m(i)).toFloat
          i += 1
      case DiffusionPrediction.VPrediction =>
        val c1 = -sigma / math.sqrt(sigma * sigma + 1.0)
        val c2 = 1.0 / (sigma * sigma + 1.0)
        while i < n do
          denoised(i) = (m(i) * c1 + x(i) * c2).toFloat
          i += 1
      case DiffusionPrediction.Sample if !ancestral =>
        System.arraycopy(m, 0, denoised, 0, n)
      case _ =>
        throw IllegalArgumentException(s"Sonar does not support prediction type '$prediction'")

    if !started then
      history = cfg.momentumHistInit match
        case "zero"      => None
        case "rand_init" => Some(x.clone())
        case "rand_new"  => Some(rng.normal(n))
      started = true

    val d = Array.tabulate(n)(idx => ((x(idx) - denoised(idx)) / sigma).toFloat)
    val nxt =
      if ancestral then
        val up = if sigma <= 0.0 then 0.0 else math.sqrt(sigmaNext * sigmaNext * (sigma * sigma - sigmaNext * sigmaNext) / (sigma * sigma))
        val down = math.sqrt(math.max(0.0, sigmaNext * sigmaNext - up * up))
        val moved = move(x, d, down - sigma)
        ancestralNoise match
          case Some(z) if up > 0.0 =>
            val zArr = z.toFloatArray
            Array.tabulate(n)(idx => (moved(idx) + zArr(idx) * up).toFloat)
          case _ => moved
      else
        var moved = move(x, d, sigmaNext - sigma)
        if reference.isDefined && naive && strength != 0.0 && calls >= minStep && calls <= maxStep then
          moved = Sonar.guidanceStep(moved, sample.shape, cfg.refMeth, reference.get, strength, denoised, sigma, sigmaNext)
        if naive then
          val nz = rng.normal(n)
          var k = 0
          while k < n do
            moved(k) = moved(k) + nz(k) * Sonar.NaiveNoise
            k += 1
        moved
    calls += 1
    Tensor.fromFloats(nxt, sample.shape)

final case class HiresSeedPlan(
    seeds: Vector[Long],
    subseeds: Vector[Long],
    strength: Double,
    count: Int,
    resizeWidth: Int,
    resizeHeight: Int,
    promptsResolved: Boolean = false,
    seedOffset: Int = 0
):
  def batch(start: Int, size: Int, repeat: Int): (Vector[Long], Vector[Long]) =
    require(start >= 0 && size >= 1 && start + size <= seeds.size && repeat >= 0 && repeat < count, "Hires Fix Tweaks: invalid batch slice or repeat")
    val increment = if strength == 0.0 then repeat.toLong else 0L
    (
      seeds.slice(start, start + size).map(_ + increment),
      subseeds.slice(start, start + size).map(_ + repeat.toLong)
    )
  def seedFor(baseSeed: Long, index: Int = 0): Long =
    if seeds.nonEmpty && index < seeds.size && seeds(index) > 0 then seeds(index) + seedOffset
    else baseSeed + seedOffset
  def subseedFor(baseSubseed: Long, index: Int = 0): Long =
    if subseeds.nonEmpty && index < subseeds.size && subseeds(index) >= 0 then subseeds(index)
    else baseSubseed
  def subseedStrength(baseStrength: Double): Double = if strength > 0.0 then strength else baseStrength
  def repeat: Int = count
  def seedResizeFromW: Int = resizeWidth
  def seedResizeFromH: Int = resizeHeight
  def resizeFrom: Option[(Int, Int)] = Option.when(resizeWidth > 0 && resizeHeight > 0)((resizeWidth, resizeHeight))

object HiresFixTweaks:
  def wireSeed(seed: Long): worker.NativeJson =
    if seed > 9007199254740991L then worker.NativeJson.Str(seed.toString)
    else worker.NativeJson.Number(BigDecimal(seed))

  def seedPlan(
      batchCount: Int = 1,
      seedEnabled: Boolean = false,
      seed: Long = 0L,
      extras: Boolean = false,
      subseed: Long = 0L,
      strength: Double = 0.0,
      resizeWidth: Int = 0,
      resizeHeight: Int = 0,
      promptsResolved: Boolean = false,
      baseSeeds: Vector[Long],
      baseSubseeds: Vector[Long],
      baseStrength: Double
  ): HiresSeedPlan =
    require(batchCount >= 1 && batchCount <= 16, "Hires Fix Tweaks: batch_count is outside its integer range")
    require(seed >= -1L && subseed >= -1L, "Hires Fix Tweaks: seed/subseed is outside its integer range")
    require(resizeWidth >= 0 && resizeWidth <= 16384 && resizeWidth % 8 == 0, "Hires Fix Tweaks: resize_width must be a multiple of 8 in 0..16384")
    require(resizeHeight >= 0 && resizeHeight <= 16384 && resizeHeight % 8 == 0, "Hires Fix Tweaks: resize_height must be a multiple of 8 in 0..16384")
    require(strength.isFinite && strength >= 0.0 && strength <= 1.0 && baseStrength.isFinite && baseStrength >= 0.0 && baseStrength <= 1.0,
      "Hires Fix Tweaks: variation strength must be finite in [0, 1]")
    require(baseSeeds.nonEmpty && baseSeeds.size == baseSubseeds.size && baseSeeds.forall(_ >= 0L) && baseSubseeds.forall(_ >= 0L),
      "Hires Fix Tweaks:invalid base seeds")
    var seeds = baseSeeds
    var subs = baseSubseeds
    var effStrength = baseStrength
    var w = 0
    var h = 0
    if seedEnabled then
      effStrength = if extras then strength else 0.0
      val rng = new java.security.SecureRandom()
      if seed != 0L then
        val resolved = if seed == -1L then rng.nextLong(4294967294L) else seed
        seeds = Vector.tabulate(baseSeeds.size)(i => resolved + (if effStrength == 0.0 then i.toLong else 0L))
      if extras then
        if subseed != 0L then
          val resolvedSub = if subseed == -1L then rng.nextLong(4294967294L) else subseed
          subs = Vector.tabulate(baseSubseeds.size)(i => resolvedSub + i.toLong)
        if resizeWidth > 0 && resizeHeight > 0 then
          w = resizeWidth
          h = resizeHeight
    val inc = (batchCount - 1).toLong
    require(
      seeds.forall(s => BigInt(s) + (if effStrength == 0.0 then inc else 0L) <= Long.MaxValue) &&
        subs.forall(s => BigInt(s) + inc <= Long.MaxValue),
      "Hires Fix Tweaks: batch or repeat seeds exceed the signed 64-bit range"
    )
    HiresSeedPlan(seeds, subs, effStrength, batchCount, w, h, promptsResolved)

  /** Draw seeded noise at (resizeWidth/8, resizeHeight/8), vary with subseed if strength > 0, and center-crop/paste onto target shape. */
  def resizedNoise(
      targetShape: Shape,
      seed: Long,
      subseed: Option[Long],
      strength: Double,
      resizeWidth: Int,
      resizeHeight: Int,
      factor: Int,
      draw: (Long, Shape) => Tensor[Float]
  ): Tensor[Float] =
    val dims = targetShape.dims
    val channels = dims(1)
    val targetH = dims(2).toInt
    val targetW = dims(3).toInt
    val srcH = if resizeWidth > 0 && resizeHeight > 0 then resizeHeight / factor else targetH
    val srcW = if resizeWidth > 0 && resizeHeight > 0 then resizeWidth / factor else targetW
    val srcShape = Shape.from(Seq(1L, channels, srcH.toLong, srcW.toLong))
    var noise = draw(seed, srcShape)
    if strength > 0.0 && subseed.isDefined then
      noise = VariationNoise.slerp(noise, draw(subseed.get, srcShape), strength)
    if srcH == targetH && srcW == targetW then noise
    else
      val result = draw(seed, targetShape).toFloatArray.clone()
      val nArr = noise.toFloatArray
      val dy = math.floorDiv(targetH - srcH, 2)
      val dx = math.floorDiv(targetW - srcW, 2)
      val rows = if dy >= 0 then srcH else srcH + 2 * dy
      val cols = if dx >= 0 then srcW else srcW + 2 * dx
      if rows > 0 && cols > 0 then
        val dstY0 = math.max(dy, 0)
        val dstX0 = math.max(dx, 0)
        val srcY0 = math.max(-dy, 0)
        val srcX0 = math.max(-dx, 0)
        for c <- 0 until channels.toInt; r <- 0 until rows do
          val dstOffset = (c * targetH + (dstY0 + r)) * targetW + dstX0
          val srcOffset = (c * srcH + (srcY0 + r)) * srcW + srcX0
          System.arraycopy(nArr, srcOffset, result, dstOffset, cols)
      Tensor.fromFloats(result, targetShape)

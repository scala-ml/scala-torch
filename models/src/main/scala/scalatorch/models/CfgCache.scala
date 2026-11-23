package scalatorch.models

import scalatorch.{Shape,Tensor}

/** FasterCache CFG-Cache (Lv et al., ICLR 2025, arXiv:2410.19355) + Adaptive Guidance (Castillo et al., arXiv:2312.12487):
  * caches the CFG noise difference `eps_c - eps_u` after warmup steps and skips the unconditional U-Net pass
  * for `interval - 1` steps (or dynamically whenever `cos(eps_c, eps_u) >= adaptiveThreshold`), optionally
  * emphasizing high-frequency residual structure late in sampling (`hfWeight`).
  */
final case class CfgCacheConfig(
    start: Double = 0.25,
    interval: Int = 3,
    adaptiveThreshold: Double = 0.0,
    hfWeight: Double = 0.0
):
  require(start.isFinite && start >= 0.0 && start <= 1.0, "CFG-Cache start must be in 0..1")
  require(interval >= 2 && interval <= 20, "CFG-Cache interval must be in 2..20")
  require(adaptiveThreshold.isFinite && adaptiveThreshold >= 0.0 && adaptiveThreshold <= 1.0,
    "CFG-Cache adaptive_threshold must be in 0..1")
  require(hfWeight.isFinite && hfWeight >= 0.0 && hfWeight <= 0.5, "CFG-Cache hf_weight must be in 0..0.5")

final class CfgCacheState(val config: CfgCacheConfig, val totalSteps: Int):
  private var cachedDelta: Option[Tensor[Float]] = None
  private var lastFullStep: Int = -1
  private var lastCosine: Double = -1.0
  private var stepIdx: Int = 0
  private var spatialWarmupUntil: Int = 0
  private var deferredUntil: Int = 0
  var fullSteps: Int = 0
  var cachedSteps: Int = 0

  def currentStep: Int = stepIdx
  def cosineSimilarity: Double = lastCosine
  def deferUntil(minStep: Int): Unit = deferredUntil = math.max(deferredUntil, minStep)

  def withStep[A](step: Int)(body: => A): A =
    val prev = stepIdx
    stepIdx = step
    try body
    finally stepIdx = prev

  def shouldReuse(stepIndex: Int, sampleShape: Shape): Boolean =
    if cachedDelta.exists(_.shape != sampleShape) then
      cachedDelta = None
      spatialWarmupUntil = math.max(spatialWarmupUntil, stepIndex + 2)
      return false
    if stepIndex < math.max(spatialWarmupUntil, deferredUntil) then return false
    if DeepCache.current.exists(_.isScheduledFullStep(stepIndex)) then return false
    cachedDelta match
      case None => false
      case Some(_) =>
        val warmupSteps = math.max(1, math.ceil(totalSteps * config.start).toInt)
        if stepIndex < warmupSteps then false
        else if config.adaptiveThreshold > 0.0 && lastCosine >= config.adaptiveThreshold then true
        else (stepIndex - lastFullStep) < config.interval

  def record(stepIndex: Int, neg: Tensor[Float], pos: Tensor[Float]): Unit =
    if deferredUntil > 0 && stepIndex < deferredUntil - 1 then return ()
    require(neg.shape == pos.shape, "CFG-Cache branch shape mismatch")
    val nArr = neg.toFloatArray
    val pArr = pos.toFloatArray
    val delta = new Array[Float](pArr.length)
    var dot = 0.0
    var nn = 0.0
    var np = 0.0
    var i = 0
    while i < pArr.length do
      val nv = nArr(i).toDouble
      val pv = pArr(i).toDouble
      delta(i) = (pv - nv).toFloat
      dot += nv * pv
      nn += nv * nv
      np += pv * pv
      i += 1
    lastCosine = dot / (math.sqrt(nn * np) + 1e-12)
    cachedDelta = Some(Tensor.fromFloats(delta, pos.shape))
    lastFullStep = stepIndex
    fullSteps += 1

  /** Synthesizes `neg_approx = pos - delta_eff` so every CFG combination/rescaling path works identically. */
  def applyCached(stepIndex: Int, pos: Tensor[Float]): Tensor[Float] =
    val deltaTensor = cachedDelta.getOrElse(throw IllegalStateException("CFG-Cache used before recording"))
    require(deltaTensor.shape == pos.shape, "CFG-Cache cached shape mismatch")
    val pArr = pos.toFloatArray
    val dArr = deltaTensor.toFloatArray
    val progress = stepIndex.toDouble / math.max(1, totalSteps - 1)
    val effDelta =
      if config.hfWeight <= 0.0 || progress < 0.5 then dArr
      else CfgCache.boostHighFreq(dArr, pos.shape, config.hfWeight.toFloat)
    cachedSteps += 1
    Tensor.fromFloats(Array.tabulate(pArr.length)(i => pArr(i) - effDelta(i)), pos.shape)

object CfgCache:
  private val active = new ThreadLocal[Option[CfgCacheState]]:
    override def initialValue(): Option[CfgCacheState] = None

  def current: Option[CfgCacheState] = active.get()

  def withConfig[A](config: Option[(CfgCacheConfig, Int)])(body: => A): A =
    val prev = active.get()
    val state = config.map((cfg, total) => new CfgCacheState(cfg, total))
    active.set(state)
    try body
    finally active.set(prev)

  /** Applies 3x3 separable [0.25, 0.5, 0.25] low-pass filter per NCHW channel with clamp padding
    * and amplifies the high-frequency component: `lp + (1 + weight) * (x - lp)`.
    */
  private[models] def boostHighFreq(data: Array[Float], shape: Shape, weight: Float): Array[Float] =
    val dims = shape.dims
    if dims.size != 4 || weight == 0f then return data
    val batch = dims(0).toInt
    val channels = dims(1).toInt
    val h = dims(2).toInt
    val w = dims(3).toInt
    if h <= 0 || w <= 0 then return data
    val plane = h * w
    val tmp = new Array[Float](plane)
    val out = new Array[Float](data.length)
    val scale = 1f + weight
    var bc = 0
    val totalPlanes = batch * channels
    while bc < totalPlanes do
      val base = bc * plane
      var y = 0
      while y < h do
        val row = base + y * w
        val tmpRow = y * w
        var x = 0
        while x < w do
          val xm1 = if x > 0 then x - 1 else 0
          val xp1 = if x + 1 < w then x + 1 else w - 1
          tmp(tmpRow + x) = 0.25f * data(row + xm1) + 0.5f * data(row + x) + 0.25f * data(row + xp1)
          x += 1
        y += 1
      y = 0
      while y < h do
        val ym1 = (if y > 0 then y - 1 else 0) * w
        val y0 = y * w
        val yp1 = (if y + 1 < h then y + 1 else h - 1) * w
        val outRow = base + y0
        var x = 0
        while x < w do
          val lp = 0.25f * tmp(ym1 + x) + 0.5f * tmp(y0 + x) + 0.25f * tmp(yp1 + x)
          val orig = data(outRow + x)
          out(outRow + x) = lp + scale * (orig - lp)
          x += 1
        y += 1
      bc += 1
    out

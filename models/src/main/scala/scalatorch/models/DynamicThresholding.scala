package scalatorch.models

import scalatorch.*

final case class DynamicThresholdingConfig(
    mimicScale: Double = 7.0,
    thresholdPercentile: Double = 1.0,
    mimicMode: String = "Constant",
    mimicScaleMin: Double = 0.0,
    cfgMode: String = "Constant",
    cfgScaleMin: Double = 0.0,
    schedVal: Double = 1.0,
    separateFeatureChannels: Boolean = true,
    scalingStartpoint: String = "MEAN",
    variabilityMeasure: String = "AD",
    interpolatePhi: Double = 1.0,
    trainingSigmas: Vector[Float] = Vector.empty
):
  require(
    mimicScale.isFinite && mimicScale >= 0 &&
      thresholdPercentile.isFinite && thresholdPercentile >= 0 && thresholdPercentile <= 1 &&
      mimicScaleMin.isFinite && mimicScaleMin >= 0 &&
      cfgScaleMin.isFinite && cfgScaleMin >= 0 &&
      schedVal.isFinite &&
      interpolatePhi.isFinite && interpolatePhi >= 0 && interpolatePhi <= 1,
    "invalid DynamicThresholding numeric parameters"
  )
  require(DynamicThresholding.Modes.contains(mimicMode), s"unsupported DynamicThresholding mimic_mode: $mimicMode")
  require(DynamicThresholding.Modes.contains(cfgMode), s"unsupported DynamicThresholding cfg_mode: $cfgMode")
  require(Set("MEAN", "ZERO").contains(scalingStartpoint.toUpperCase(java.util.Locale.ROOT)), s"unsupported DynamicThresholding scaling_startpoint: $scalingStartpoint")
  require(Set("AD", "STD").contains(variabilityMeasure.toUpperCase(java.util.Locale.ROOT)), s"unsupported DynamicThresholding variability_measure: $variabilityMeasure")

  /** Rewrites the conditional CFG branch in model-output space so `u + g * (c_new - u)` equals the dynamically thresholded prediction. */
  def rewriteConditional(
      sample: Tensor[Float],
      uncond: Tensor[Float],
      cond: Tensor[Float],
      timestep: Float,
      guidanceScale: Float,
      prediction: DiffusionPrediction = DiffusionPrediction.Epsilon,
      sigmas: Vector[Float] = trainingSigmas
  ): Tensor[Float] =
    if guidanceScale <= 1.0f then cond
    else
      val zTensor = if sample.shape == uncond.shape then sample else UNetExecution.controlInput(sample)
      require(zTensor.shape == uncond.shape && cond.shape == uncond.shape, "DynamicThresholding tensor shape mismatch")
      val table = if sigmas.nonEmpty then sigmas else DynamicThresholding.defaultTrainingSigmas
      val tv = math.min(math.max(timestep.toDouble, 0.0), (table.size - 1).toDouble)
      val lo = math.floor(tv).toInt
      val hi = math.min(table.size - 1, math.ceil(tv).toInt)
      val w = tv - lo
      def alphaAt(idx: Int): Double =
        val s = table(idx).toDouble
        1.0 / (1.0 + s * s)
      val a = alphaAt(lo) * (1.0 - w) + alphaAt(hi) * w
      val sa = math.sqrt(a)
      val s1a = math.sqrt(math.max(1.0 - a, 1e-8))
      val z = zTensor.toFloatArray
      val u = uncond.toFloatArray
      val c = cond.toFloatArray
      val n = u.length
      val x0u = new Array[Float](n)
      val x0c = new Array[Float](n)
      val isV = prediction == DiffusionPrediction.VPrediction
      var i = 0
      while i < n do
        if isV then
          x0u(i) = (sa * z(i) - s1a * u(i)).toFloat
          x0c(i) = (sa * z(i) - s1a * c(i)).toFloat
        else
          x0u(i) = ((z(i) - s1a * u(i)) / sa).toFloat
          x0c(i) = ((z(i) - s1a * c(i)) / sa).toFloat
        i += 1
      val frac = 1.0 - math.min(math.max(timestep.toDouble, 0.0), 999.0) / 999.0
      val x0r = DynamicThresholding.dynthresh(
        Tensor.fromFloats(x0c, cond.shape),
        Tensor.fromFloats(x0u, uncond.shape),
        guidanceScale.toDouble,
        frac,
        this
      ).toFloatArray
      val cNew = new Array[Float](n)
      val g = guidanceScale.toDouble
      i = 0
      while i < n do
        val r = if isV then (sa * z(i) - x0r(i)) / s1a else (z(i) - sa * x0r(i)) / s1a
        cNew(i) = (u(i) + (r - u(i)) / g).toFloat
        i += 1
      Tensor.fromFloats(cNew, cond.shape)

object DynamicThresholding:
  val Modes: Set[String] = Set(
    "Constant",
    "Linear Down",
    "Half Cosine Down",
    "Cosine Down",
    "Linear Up",
    "Half Cosine Up",
    "Cosine Up",
    "Power Up",
    "Power Down",
    "Linear Repeating",
    "Cosine Repeating",
    "Sawtooth"
  )

  private[models] lazy val defaultTrainingSigmas: Vector[Float] =
    DiffusionNoiseSchedule.linear(1000, 0.00085, 0.012, scaled = true).floatTrainingSigmas

  def interpretScale(scale: Double, mode: String, min: Double, frac: Double, schedVal: Double = 1.0): Double =
    var s = scale - min
    mode match
      case "Constant" => ()
      case "Linear Down" => s *= 1.0 - frac
      case "Half Cosine Down" => s *= math.cos(frac)
      case "Cosine Down" => s *= math.cos(frac * 1.5707)
      case "Linear Up" => s *= frac
      case "Half Cosine Up" => s *= 1.0 - math.cos(frac)
      case "Cosine Up" => s *= 1.0 - math.cos(frac * 1.5707)
      case "Power Up" => s *= math.pow(frac, schedVal)
      case "Power Down" => s *= 1.0 - math.pow(frac, schedVal)
      case "Linear Repeating" =>
        val portion = (frac * schedVal) % 1.0
        s *= (if portion < 0.5 then (0.5 - portion) * 2.0 else (portion - 0.5) * 2.0)
      case "Cosine Repeating" =>
        s *= math.cos(frac * 6.28318 * schedVal) * 0.5 + 0.5
      case "Sawtooth" =>
        s *= (frac * schedVal) % 1.0
      case _ => ()
    s + min

  private def absmax(values: Array[Double], offset: Int, length: Int): Double =
    var m = 0.0
    var i = 0
    while i < length do
      val a = math.abs(values(offset + i))
      if a > m then m = a
      i += 1
    m

  private def quantileAbs(values: Array[Double], offset: Int, length: Int, q: Double): Double =
    val copy = new Array[Double](length)
    var i = 0
    while i < length do
      copy(i) = math.abs(values(offset + i))
      i += 1
    java.util.Arrays.sort(copy)
    if length == 1 then copy(0)
    else
      val pos = math.min(math.max(q, 0.0), 1.0) * (length - 1)
      val lo = math.floor(pos).toInt
      val hi = math.min(length - 1, math.ceil(pos).toInt)
      val w = pos - lo
      copy(lo) * (1.0 - w) + copy(hi) * w

  private def std(values: Array[Double], offset: Int, length: Int): Double =
    if length <= 1 then 0.0
    else
      var sum = 0.0
      var i = 0
      while i < length do
        sum += values(offset + i)
        i += 1
      val mean = sum / length
      var sq = 0.0
      i = 0
      while i < length do
        val d = values(offset + i) - mean
        sq += d * d
        i += 1
      math.sqrt(sq / (length - 1))

  def dynthresh(
      cond: Tensor[Float],
      uncond: Tensor[Float],
      cfgScale: Double,
      frac: Double = 0.0,
      config: DynamicThresholdingConfig = DynamicThresholdingConfig()
  ): Tensor[Float] =
    require(cond.shape == uncond.shape && cond.shape.rank == 4, "dynthresh requires matching rank-4 (B, C, H, W) tensors")
    val dims = cond.shape.dims
    val b = dims(0).toInt
    val c = dims(1).toInt
    val spatial = (dims(2) * dims(3)).toInt
    val total = b * c * spatial
    val mimicScale = interpretScale(config.mimicScale, config.mimicMode, config.mimicScaleMin, frac, config.schedVal)
    val effectiveCfg = interpretScale(cfgScale, config.cfgMode, config.cfgScaleMin, frac, config.schedVal)
    val pct = config.thresholdPercentile
    val isStd = config.variabilityMeasure.equalsIgnoreCase("STD")
    val isZero = config.scalingStartpoint.equalsIgnoreCase("ZERO")
    val phi = config.interpolatePhi

    val condArr = cond.toFloatArray
    val uncondArr = uncond.toFloatArray
    val mimFlat = new Array[Double](total)
    val cfgFlat = new Array[Double](total)
    val mimC = new Array[Double](total)
    val cfgC = new Array[Double](total)
    val mimMeans = new Array[Double](b * c)
    val cfgMeans = new Array[Double](b * c)

    var plane = 0
    while plane < b * c do
      val base = plane * spatial
      var sumMim = 0.0
      var sumCfg = 0.0
      var i = 0
      while i < spatial do
        val idx = base + i
        val uVal = uncondArr(idx).toDouble
        val rel = condArr(idx).toDouble - uVal
        val mVal = uVal + rel * mimicScale
        val cVal = uVal + rel * effectiveCfg
        mimFlat(idx) = mVal
        cfgFlat(idx) = cVal
        sumMim += mVal
        sumCfg += cVal
        i += 1
      val mMean = sumMim / spatial
      val cMean = sumCfg / spatial
      mimMeans(plane) = mMean
      cfgMeans(plane) = cMean
      i = 0
      while i < spatial do
        val idx = base + i
        mimC(idx) = mimFlat(idx) - mMean
        cfgC(idx) = cfgFlat(idx) - cMean
        i += 1
      plane += 1

    val mimRefs = new Array[Double](b * c)
    val cfgRefs = new Array[Double](b * c)
    if config.separateFeatureChannels then
      plane = 0
      while plane < b * c do
        val base = plane * spatial
        if isStd then
          mimRefs(plane) = std(mimC, base, spatial)
          cfgRefs(plane) = std(cfgC, base, spatial)
        else
          mimRefs(plane) = absmax(mimC, base, spatial)
          cfgRefs(plane) = quantileAbs(cfgC, base, spatial, pct)
        plane += 1
    else
      val mRef = if isStd then std(mimC, 0, total) else absmax(mimC, 0, total)
      val cRef = if isStd then std(cfgC, 0, total) else quantileAbs(cfgC, 0, total, pct)
      java.util.Arrays.fill(mimRefs, mRef)
      java.util.Arrays.fill(cfgRefs, cRef)

    val out = new Array[Float](total)
    plane = 0
    while plane < b * c do
      val base = plane * spatial
      val mRef = mimRefs(plane)
      val cRef = math.max(cfgRefs(plane), 1e-12)
      val maxRef = math.max(math.max(mRef, cRef), 1e-12)
      val cMean = cfgMeans(plane)
      var i = 0
      while i < spatial do
        val idx = base + i
        val res =
          if isZero then
            cfgFlat(idx) * (mRef / cRef)
          else if isStd then
            (cfgC(idx) / cRef) * mRef + cMean
          else
            val clamped = math.max(-maxRef, math.min(maxRef, cfgC(idx)))
            (clamped / maxRef) * mRef + cMean
        val blended = if phi != 1.0 then res * phi + cfgFlat(idx) * (1.0 - phi) else res
        out(idx) = blended.toFloat
        i += 1
      plane += 1
    Tensor.fromFloats(out, cond.shape)

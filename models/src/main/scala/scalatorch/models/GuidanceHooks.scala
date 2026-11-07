package scalatorch.models

import scalatorch.*

final case class ApgConfig(
    momentum: Double = 0.5,
    adaptiveMomentum: Double = 0.18,
    normThreshold: Double = 15.0,
    eta: Double = 1.0,
    guidanceLimiter: Boolean = false,
    guidanceSigmaStart: Double = 5.42,
    guidanceSigmaEnd: Double = 0.28
):
  require(
    momentum.isFinite && adaptiveMomentum.isFinite && normThreshold.isFinite &&
      eta.isFinite && guidanceSigmaStart.isFinite && guidanceSigmaEnd.isFinite,
    "invalid APG parameters"
  )

final case class PagConfig(
    scale: Double = 3.0,
    adaptiveScale: Double = 0.0,
    layers: Vector[String] = Vector("mid")
):
  require(scale.isFinite && scale >= 0 && adaptiveScale.isFinite && adaptiveScale >= 0 && layers.nonEmpty, "invalid PAG parameters")

  def effectiveScale(timestep: Float): Float =
    val s = if adaptiveScale > 0.0 then
      math.max(0.0, scale - adaptiveScale * (1000.0 - timestep.toDouble))
    else scale
    s.toFloat

object Pag:
  private val sectionHolder = new ThreadLocal[String]:
    override def initialValue(): String = ""
  private val layersHolder = new ThreadLocal[Set[String]]:
    override def initialValue(): Set[String] = Set.empty

  private def normalizeLayer(raw: String): String =
    val t = raw.trim.toLowerCase(java.util.Locale.ROOT)
    if t == "middle" || t == "mid_block" then "mid"
    else if t == "down_blocks" then "down"
    else if t == "up_blocks" then "up"
    else if t.startsWith("down_blocks.") then "down." + t.stripPrefix("down_blocks.")
    else if t.startsWith("up_blocks.") then "up." + t.stripPrefix("up_blocks.")
    else t

  def currentSection: String = sectionHolder.get()

  def isCurrentLayerPerturbed: Boolean =
    val active = layersHolder.get()
    if active.isEmpty then false
    else
      val sec = sectionHolder.get()
      if sec.isEmpty then false
      else active.contains(sec) || (sec.startsWith("down.") && active.contains("down")) || (sec.startsWith("up.") && active.contains("up"))

  def withSection[A](section: String)(body: => A): A =
    val prev = sectionHolder.get()
    sectionHolder.set(section)
    try body
    finally sectionHolder.set(prev)

  def withPerturbed[A](layers: Vector[String])(body: => A): A =
    val prev = layersHolder.get()
    layersHolder.set(layers.map(normalizeLayer).toSet)
    try body
    finally layersHolder.set(prev)

final class GuidanceHookState(
    val apg: Option[ApgConfig],
    val useMahiro: Boolean,
    val trainingSigmas: Vector[Float],
    val latentModifier: Option[LatentModifierConfig] = None,
    val latentModifierSeed: Long = 0L
):
  private var runningAverage: Option[Array[Double]] = None
  private val lmRng = new TorchCpuNormal(latentModifierSeed & 0xffffffffL)

  def rewriteConditional(
      sample: Tensor[Float],
      uncond: Tensor[Float],
      cond: Tensor[Float],
      timestep: Float,
      guidanceScale: Float,
      prediction: DiffusionPrediction = DiffusionPrediction.Epsilon
  ): Tensor[Float] =
    if guidanceScale <= 1.0f || (apg.isEmpty && !useMahiro && latentModifier.isEmpty) then cond
    else
      val zTensor = if sample.shape == uncond.shape then sample else UNetExecution.controlInput(sample)
      require(zTensor.shape == uncond.shape && cond.shape == uncond.shape, "GuidanceHookState tensor shape mismatch")
      val table = if trainingSigmas.nonEmpty then trainingSigmas else DynamicThresholding.defaultTrainingSigmas
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
      val sigma = s1a / sa

      val z = zTensor.toFloatArray
      val u = uncond.toFloatArray
      val c = cond.toFloatArray
      val n = u.length
      val x0u = new Array[Float](n)
      val x0c = new Array[Float](n)
      val xk = new Array[Float](n)
      val isV = prediction == DiffusionPrediction.VPrediction
      var i = 0
      while i < n do
        xk(i) = (z(i) / sa).toFloat
        if isV then
          x0u(i) = (sa * z(i) - s1a * u(i)).toFloat
          x0c(i) = (sa * z(i) - s1a * c(i)).toFloat
        else
          x0u(i) = ((z(i) - s1a * u(i)) / sa).toFloat
          x0c(i) = ((z(i) - s1a * c(i)) / sa).toFloat
        i += 1

      val g = guidanceScale.toDouble
      val x0Guided = apg match
        case Some(cfg) =>
          val limited = cfg.guidanceLimiter && (
            (cfg.guidanceSigmaStart >= 0.0 && sigma > cfg.guidanceSigmaStart) ||
              (cfg.guidanceSigmaEnd >= 0.0 && sigma <= cfg.guidanceSigmaEnd)
          )
          if limited then x0c.clone()
          else
            val condDiff = Array.tabulate(n)(idx => (xk(idx) - x0c(idx)).toDouble)
            val uncondDiff = Array.tabulate(n)(idx => (xk(idx) - x0u(idx)).toDouble)
            val diff = Array.tabulate(n)(idx => condDiff(idx) - uncondDiff(idx))
            val mom =
              if runningAverage.isEmpty || runningAverage.get.length != n || tv >= 999.0 then
                runningAverage = Some(diff.clone())
                diff
              else
                val m = GuidanceHooks.apgMomentum(cfg.momentum, cfg.adaptiveMomentum, tv)
                val prev = runningAverage.get
                var k = 0
                while k < n do
                  prev(k) = diff(k) + m * prev(k)
                  k += 1
                prev
            val guided = GuidanceHooks.projectAndGuide(condDiff, mom, uncond.shape, g, cfg.eta, cfg.normThreshold)
            Array.tabulate(n)(idx => (xk(idx) - guided(idx)).toFloat)
        case None =>
          Array.tabulate(n)(idx => (x0u(idx) + g * (x0c(idx) - x0u(idx))).toFloat)

      val x0Mahiro = if useMahiro then
        GuidanceHooks.mahiro(
          Tensor.fromFloats(x0c, cond.shape),
          Tensor.fromFloats(x0u, uncond.shape),
          Tensor.fromFloats(x0Guided, cond.shape),
          g
        ).toFloatArray
      else x0Guided

      val x0Final = latentModifier match
        case Some(lmCfg) =>
          val prog = 1.0 - math.min(math.max(tv, 0.0), 999.0) / 999.0
          LatentModifier.modifyLatents(Tensor.fromFloats(x0Mahiro, cond.shape), lmCfg, prog, Some(lmRng)).toFloatArray
        case None => x0Mahiro

      val cNew = new Array[Float](n)
      i = 0
      while i < n do
        val r = if isV then (sa * z(i) - x0Final(i)) / s1a else (z(i) - sa * x0Final(i)) / s1a
        cNew(i) = (u(i) + (r - u(i)) / g).toFloat
        i += 1
      Tensor.fromFloats(cNew, cond.shape)

object GuidanceHooks:
  def apgMomentum(momentum: Double, adaptive: Double, t: Double): Double =
    var s = momentum
    if adaptive > 0.0 then
      val delta = math.pow(adaptive, 4.0) * (1000.0 - t)
      if momentum < 0.0 then
        s = math.min(0.0, s + (-momentum) * delta)
      else
        s = math.max(0.0, s - momentum * delta)
    s

  private[models] def projectAndGuide(
      cond: Array[Double],
      diffIn: Array[Double],
      shape: Shape,
      scale: Double,
      eta: Double,
      normThreshold: Double
  ): Array[Double] =
    val b = shape.dims(0).toInt
    val elemPerItem = (shape.numel / b).toInt
    val out = new Array[Double](cond.length)
    var item = 0
    while item < b do
      val base = item * elemPerItem
      var diffScale = 1.0
      if normThreshold > 0.0 then
        var normSq = 0.0
        var i = 0
        while i < elemPerItem do
          val v = diffIn(base + i)
          normSq += v * v
          i += 1
        val norm = math.max(math.sqrt(normSq), 1e-12)
        diffScale = math.min(1.0, normThreshold / norm)
      var condNormSq = 0.0
      var i = 0
      while i < elemPerItem do
        val c = cond(base + i)
        condNormSq += c * c
        i += 1
      val condNorm = math.max(math.sqrt(condNormSq), 1e-12)
      var dot = 0.0
      i = 0
      while i < elemPerItem do
        val d = diffIn(base + i) * diffScale
        val v1 = cond(base + i) / condNorm
        dot += d * v1
        i += 1
      i = 0
      while i < elemPerItem do
        val d = diffIn(base + i) * diffScale
        val v1 = cond(base + i) / condNorm
        val par = dot * v1
        val orth = d - par
        out(base + i) = cond(base + i) + (scale - 1.0) * (orth + eta * par)
        i += 1
      item += 1
    out

  def mahiro(condX0: Tensor[Float], uncondX0: Tensor[Float], cfgX0: Tensor[Float], scale: Double): Tensor[Float] =
    require(condX0.shape == uncondX0.shape && condX0.shape == cfgX0.shape && condX0.shape.rank == 4, "mahiro requires matching rank-4 tensors")
    val dims = condX0.shape.dims
    val b = dims(0).toInt
    val c = dims(1).toInt
    val spatial = (dims(2) * dims(3)).toInt
    val cArr = condX0.toFloatArray
    val uArr = uncondX0.toFloatArray
    val gArr = cfgX0.toFloatArray
    val total = cArr.length
    val leap = new Array[Double](total)
    val normU = new Array[Double](total)
    val normM = new Array[Double](total)
    var i = 0
    while i < total do
      val l = cArr(i).toDouble * scale
      val ul = uArr(i).toDouble * scale
      val m = 0.5 * (l + gArr(i).toDouble)
      leap(i) = l
      normU(i) = math.sqrt(math.abs(ul)) * math.signum(ul)
      normM(i) = math.sqrt(math.abs(m)) * math.signum(m)
      i += 1
    var simSum = 0.0
    var item = 0
    while item < b do
      val itemBase = item * c * spatial
      var sp = 0
      while sp < spatial do
        var dot = 0.0
        var nuSq = 0.0
        var nmSq = 0.0
        var ch = 0
        while ch < c do
          val idx = itemBase + ch * spatial + sp
          val uVal = normU(idx)
          val mVal = normM(idx)
          dot += uVal * mVal
          nuSq += uVal * uVal
          nmSq += mVal * mVal
          ch += 1
        val denom = math.max(math.sqrt(nuSq) * math.sqrt(nmSq), 1e-8)
        simSum += dot / denom
        sp += 1
      item += 1
    val sim = simSum / (b * spatial)
    val simsc = 2.0 * (sim + 1.0)
    val out = new Array[Float](total)
    i = 0
    while i < total do
      out(i) = ((simsc * gArr(i).toDouble + (4.0 - simsc) * leap(i)) / 4.0).toFloat
      i += 1
    Tensor.fromFloats(out, condX0.shape)

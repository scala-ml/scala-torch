package scalatorch.models

import scalatorch.*
import scalatorch.autograd.Variable

private[models] object SamplerPrediction:
  def toOriginalEuler(sigma: Double, prediction: DiffusionPrediction, modelOutput: Variable, sample: Variable): Variable =
    prediction match
      case DiffusionPrediction.Epsilon => sample - modelOutput * sigma.toFloat
      case DiffusionPrediction.Sample => modelOutput
      case DiffusionPrediction.VPrediction =>
        val alpha = 1.0 / math.hypot(sigma, 1.0)
        sample * (1.0 / (sigma * sigma + 1.0)).toFloat - modelOutput * (sigma * alpha).toFloat

  def toOriginalVp(sigmaEuler: Double, prediction: DiffusionPrediction, modelOutput: Variable, sample: Variable): Variable =
    val alpha = 1.0 / math.hypot(sigmaEuler, 1.0)
    val sigmaVp = sigmaEuler * alpha
    prediction match
      case DiffusionPrediction.Epsilon => (sample - modelOutput * sigmaVp.toFloat) * (1.0 / alpha).toFloat
      case DiffusionPrediction.Sample => modelOutput
      case DiffusionPrediction.VPrediction => sample * alpha.toFloat - modelOutput * sigmaVp.toFloat

  def fromOriginalEuler(sigma: Double, prediction: DiffusionPrediction, denoised: Variable, sample: Variable, fallback: Variable): Variable =
    if sigma <= 1e-6 then fallback
    else
      prediction match
        case DiffusionPrediction.Epsilon => (sample - denoised) * (1.0 / sigma).toFloat
        case DiffusionPrediction.Sample => denoised
        case DiffusionPrediction.VPrediction =>
          val hypot = math.hypot(sigma, 1.0)
          (sample * (1.0 / (sigma * sigma + 1.0)).toFloat - denoised) * (hypot / sigma).toFloat

  def fromOriginalVp(sigmaEuler: Double, prediction: DiffusionPrediction, denoised: Variable, sample: Variable, fallback: Variable): Variable =
    val alpha = 1.0 / math.hypot(sigmaEuler, 1.0)
    val sigmaVp = sigmaEuler * alpha
    if sigmaVp <= 1e-6 then fallback
    else
      prediction match
        case DiffusionPrediction.Epsilon => (sample - denoised * alpha.toFloat) * (1.0 / sigmaVp).toFloat
        case DiffusionPrediction.Sample => denoised
        case DiffusionPrediction.VPrediction => (sample * alpha.toFloat - denoised) * (1.0 / sigmaVp).toFloat

  def scaleEulerInput(sigma: Double, sample: Variable): Variable =
    sample * (1.0 / math.hypot(sigma, 1.0)).toFloat

  def sigmaToTimestep(noise: DiffusionNoiseSchedule, sigma: Double): Float =
    noise.sigmaToT(sigma).toFloat

  def sigmaToTimestep(trainingSigmas: Vector[Float], sigma: Double): Float =
    if trainingSigmas.size <= 1 then 0f
    else
      val logs = trainingSigmas.map(s => math.log(s.toDouble).toFloat)
      val target = math.log(math.max(sigma, 1e-10))
      if target <= logs.head.toDouble then 0f
      else if target >= logs.last.toDouble then (logs.size - 1).toFloat
      else
        var lo = 0
        var hi = logs.size - 1
        while hi - lo > 1 do
          val mid = lo + (hi - lo) / 2
          if logs(mid).toDouble <= target then lo = mid else hi = mid
        val w = math.max(0.0, math.min(1.0, (target - logs(lo).toDouble) / (logs(hi).toDouble - logs(lo).toDouble)))
        ((1.0 - w) * lo + w * hi).toFloat

/** Linear Multistep (Adams-Bashforth) scheduler in Euler sigma space. */
final case class LmsScheduler(
    schedule: EulerSchedule,
    prediction: DiffusionPrediction = DiffusionPrediction.Epsilon,
    order: Int = 4,
    maxElements: Long = 134217728L,
    startStepIndex: Int = 0,
    fullSigmasOverride: Option[Vector[Double]] = None
):
  require(order >= 1 && order <= 8 && maxElements >= 0, "invalid LMS configuration")
  private val derivatives = scala.collection.mutable.ArrayBuffer.empty[Variable]
  private val allSigmas: Vector[Double] = fullSigmasOverride.getOrElse(schedule.sigmas)

  private def lmsCoefficient(stepIndex: Int, currentOrder: Int, u: Int): Double =
    val fullIdx = startStepIndex + stepIndex
    val tStart = allSigmas(fullIdx)
    val tEnd = allSigmas(fullIdx + 1)
    if math.abs(tEnd - tStart) < 1e-12 then 0.0
    else
      def poly(tau: Double): Double =
        var prod = 1.0
        var k = 0
        while k < currentOrder do
          if k != u then
            val sk = allSigmas(fullIdx - k)
            val su = allSigmas(fullIdx - u)
            val denom = su - sk
            if math.abs(denom) > 1e-12 then prod *= (tau - sk) / denom
          k += 1
        prod
      // 8-point Gauss-Legendre quadrature on [tStart, tEnd] (exact for degree <= 15)
      val nodes = Array(-0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498,
        0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975363)
      val weights = Array(0.1012285362903763, 0.2223810344533745, 0.3137066458778873, 0.3626837833783620,
        0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763)
      val half = 0.5 * (tEnd - tStart)
      val mid = 0.5 * (tEnd + tStart)
      var sum = 0.0
      var i = 0
      while i < nodes.length do
        sum += weights(i) * poly(mid + half * nodes(i))
        i += 1
      half * sum

  def scaleModelInput(index: Int, sample: Variable): Variable =
    SamplerPrediction.scaleEulerInput(schedule.sigmas(index), sample)

  def step(index: Int, modelOutput: Variable, sample: Variable): EulerStep =
    require(index >= 0 && index < schedule.timesteps.size, "LMS step index out of range")
    val sigma = schedule.sigmas(index)
    val clean = SamplerPrediction.toOriginalEuler(sigma, prediction, modelOutput, sample)
    val d = (sample - clean) * (1.0 / sigma).toFloat
    derivatives.prepend(d)
    if derivatives.size > order then derivatives.remove(derivatives.size - 1)
    val coeffOrder = math.min(startStepIndex + index + 1, order)
    val numTerms = math.min(coeffOrder, derivatives.size)
    var acc = sample
    var u = 0
    while u < numTerms do
      val c = lmsCoefficient(index, coeffOrder, u).toFloat
      acc = acc + derivatives(u) * c
      u += 1
    EulerStep(acc, clean)

final case class SecondOrderPrediction(
    sample: Variable,
    midpointSample: Variable,
    midpointSigma: Double,
    derivative: Variable,
    predictedOriginal: Variable,
    sigmaDown: Double,
    sigmaUp: Double,
    requiresSecondEvaluation: Boolean
)

/** KDPM2 and KDPM2 Ancestral Discrete Scheduler in Euler sigma space, matching Diffusers 0.33.1. */
final case class Kdpm2Scheduler(
    schedule: EulerSchedule,
    prediction: DiffusionPrediction = DiffusionPrediction.Epsilon,
    ancestral: Boolean = false,
    maxElements: Long = 134217728L
):
  def scaleModelInput(index: Int, sample: Variable): Variable =
    SamplerPrediction.scaleEulerInput(schedule.sigmas(index), sample)

  def scaleMidpointInput(midpointSigma: Double, sample: Variable): Variable =
    SamplerPrediction.scaleEulerInput(midpointSigma, sample)

  def predict(index: Int, modelOutput: Variable, sample: Variable, noise: Option[Variable] = None): SecondOrderPrediction =
    val sigma = schedule.sigmas(index)
    val next = schedule.sigmas(index + 1)
    val clean = SamplerPrediction.toOriginalEuler(sigma, prediction, modelOutput, sample)
    val d = (sample - clean) * (1.0 / sigma).toFloat
    if next <= 0.0 then
      val prev = sample + d * (next - sigma).toFloat
      SecondOrderPrediction(sample, prev, 0.0, d, clean, 0.0, 0.0, requiresSecondEvaluation = false)
    else if !ancestral then
      // KDPM2DiscreteScheduler: sigmas_interpol[k] = sqrt(sigma * next)
      val midSigma = math.exp(0.5 * (math.log(sigma) + math.log(next))).toFloat.toDouble
      val midSample = sample + d * (midSigma - sigma).toFloat
      SecondOrderPrediction(sample, midSample, midSigma, d, clean, next, 0.0, requiresSecondEvaluation = true)
    else
      // KDPM2AncestralDiscreteScheduler: matches scheduling_k_dpm_2_ancestral_discrete.py lines 303-319 & 509-520
      def upAt(k: Int): Double =
        val sk = schedule.sigmas(k).toFloat.toDouble
        val sn = schedule.sigmas(k + 1).toFloat.toDouble
        if sn <= 0.0 then 0.0
        else math.sqrt(math.max(0.0, sn * sn * (sk * sk - sn * sn) / (sk * sk))).toFloat.toDouble
      def downAt(k: Int): Double =
        val sn = schedule.sigmas(k + 1).toFloat.toDouble
        val uk = upAt(k)
        if sn <= 0.0 then 0.0
        else math.sqrt(math.max(0.0, sn * sn - uk * uk)).toFloat.toDouble
      def interpAt(k: Int): Double =
        if k >= schedule.timesteps.size - 1 then 0.0
        else
          val sk = schedule.sigmas(k).toFloat.toDouble
          val dk = downAt(k)
          math.exp(0.5 * (math.log(sk) + math.log(math.max(dk, 1e-12)))).toFloat.toDouble
      val idx = if index == 0 then 0 else index
      // Wait: self.sigmas_interpol = cat([interp[:1], interp[1:].repeat_interleave(2), interp[-1:]])
      // At step_index = 2 * index: index 0 -> interp(0); index >= 1 -> interp(index)!
      // Wait: in Stage 2 (step_index = 2 * index + 1), step_index - 1 = 2 * index, so it ALSO indexes 2 * index!
      // Wait: why did Python print interp = [4.0861, 1.6156, 1.6156, 0.6952, 0.6952, 0.0292, 0.0292, 0.0, ...] earlier?!
      val sigmaInterpol = interpAt(idx)
      val sigmaUp = upAt(idx)
      val sigmaDown = downAt(idx)
      if sigmaInterpol <= 0.0 then
        val prev = sample + d * (-sigma).toFloat
        SecondOrderPrediction(sample, prev, 0.0, d, clean, 0.0, 0.0, requiresSecondEvaluation = false)
      else
        val midSample = sample + d * (sigmaInterpol - sigma).toFloat
        SecondOrderPrediction(sample, midSample, sigmaInterpol, d, clean, sigmaDown, sigmaUp, requiresSecondEvaluation = true)

  def correct(index: Int, first: SecondOrderPrediction, midOutput: Variable, noise: Option[Variable]): EulerStep =
    if !first.requiresSecondEvaluation then EulerStep(first.midpointSample, first.predictedOriginal)
    else
      val sigma = schedule.sigmas(index)
      val clean2 = SamplerPrediction.toOriginalEuler(first.midpointSigma, prediction, midOutput, first.midpointSample)
      val d2 = (first.midpointSample - clean2) * (1.0 / first.midpointSigma).toFloat
      val base = first.sample + d2 * (first.sigmaDown - sigma).toFloat
      val result = if ancestral && first.sigmaUp > 0.0 && noise.isDefined then
        base + noise.get * first.sigmaUp.toFloat
      else base
      EulerStep(result, clean2)

/** DPM++ 2S a (DPMSolverSinglestepScheduler) in VP space (1 model evaluation per step, alternating order 1/2),
  * while retaining Euler-space predict/correct methods for backward compatibility with direct unit tests.
  */
final case class DpmPlusPlus2Sa(
    schedule: EulerSchedule,
    prediction: DiffusionPrediction = DiffusionPrediction.Epsilon,
    maxElements: Long = 134217728L,
    totalStepsOverride: Option[Int] = None,
    startStepIndex: Int = 0
):
  private val totalSteps: Int = totalStepsOverride.getOrElse(schedule.timesteps.size)
  private val orderList: Vector[Int] =
    val base =
      if totalSteps <= 0 then Vector.empty[Int]
      else if totalSteps % 2 == 0 then Vector.tabulate(totalSteps / 2 - 1)(_ => Vector(1, 2)).flatten ++ Vector(1, 1)
      else Vector.tabulate(totalSteps / 2)(_ => Vector(1, 2)).flatten :+ 1
    if base.nonEmpty then base.updated(base.size - 1, 1) else base
  private var m0: Option[Variable] = None
  private var m1: Option[Variable] = None
  private var savedSample: Option[Variable] = None

  /** Single-evaluation VP step matching Diffusers DPMSolverSinglestepScheduler.step. */
  def stepVp(index: Int, modelOutput: Variable, sample: Variable): DpmPlusPlusStep =
    require(index >= 0 && index < schedule.timesteps.size, "DPM++ 2S a step index out of range")
    val s = schedule.sigmas(index)
    val next = schedule.sigmas(index + 1)
    val x0Pred = SamplerPrediction.toOriginalVp(s, prediction, modelOutput, sample)
    m1 = m0
    m0 = Some(x0Pred)
    val stepIndex = math.min(orderList.size - 1, math.max(0, startStepIndex + index))
    var order = if orderList.nonEmpty then orderList(stepIndex) else 1
    if order == 2 && (m1.isEmpty || index < 1) then order = 1
    if order == 1 || savedSample.isEmpty then savedSample = Some(sample)
    if next == 0.0 then DpmPlusPlusStep(x0Pred, x0Pred)
    else if order == 1 then
      val alphaS = 1.0 / math.hypot(s, 1.0)
      val sigmaS = s * alphaS
      val alphaT = 1.0 / math.hypot(next, 1.0)
      val sigmaT = next * alphaT
      val h = math.log(s) - math.log(next)
      val prev = savedSample.get * (sigmaT / sigmaS).toFloat - x0Pred * (alphaT * math.expm1(-h)).toFloat
      DpmPlusPlusStep(prev, x0Pred)
    else
      val s1 = schedule.sigmas(index - 1)
      val alphaS1 = 1.0 / math.hypot(s1, 1.0)
      val sigmaS1 = s1 * alphaS1
      val alphaT = 1.0 / math.hypot(next, 1.0)
      val sigmaT = next * alphaT
      val h = math.log(s1) - math.log(next)
      val h0 = math.log(s1) - math.log(s)
      val r0 = h0 / h
      val d0 = m1.get
      val d1 = (m0.get - m1.get) * (1.0 / r0).toFloat
      val coeff = (alphaT * math.expm1(-h)).toFloat
      val prev = savedSample.get * (sigmaT / sigmaS1).toFloat - d0 * coeff - d1 * (0.5f * coeff)
      DpmPlusPlusStep(prev, x0Pred)

  def scaleModelInput(index: Int, sample: Variable): Variable =
    SamplerPrediction.scaleEulerInput(schedule.sigmas(index), sample)

  def scaleMidpointInput(midpointSigma: Double, sample: Variable): Variable =
    SamplerPrediction.scaleEulerInput(midpointSigma, sample)

  def predict(index: Int, modelOutput: Variable, sample: Variable): SecondOrderPrediction =
    val sigma = schedule.sigmas(index)
    val next = schedule.sigmas(index + 1)
    val clean = SamplerPrediction.toOriginalEuler(sigma, prediction, modelOutput, sample)
    val d = (sample - clean) * (1.0 / sigma).toFloat
    if next <= 0.0 then
      val prev = clean
      SecondOrderPrediction(sample, prev, 0.0, d, clean, 0.0, 0.0, requiresSecondEvaluation = false)
    else
      val up = math.min(next, math.sqrt(math.max(0.0, next * next * (sigma * sigma - next * next) / (sigma * sigma))))
      val down = math.sqrt(math.max(1e-12, next * next - up * up))
      val t = -math.log(sigma)
      val tNext = -math.log(down)
      val h = tNext - t
      val s = t + 0.5 * h
      val midSigma = math.exp(-s)
      val midSample = sample * (midSigma / sigma).toFloat - clean * math.expm1(-0.5 * h).toFloat
      SecondOrderPrediction(sample, midSample, midSigma, d, clean, down, up, requiresSecondEvaluation = true)

  def correct(index: Int, first: SecondOrderPrediction, midOutput: Variable, noise: Variable): EulerStep =
    if !first.requiresSecondEvaluation then EulerStep(first.midpointSample, first.predictedOriginal)
    else
      val sigma = schedule.sigmas(index)
      val clean2 = SamplerPrediction.toOriginalEuler(first.midpointSigma, prediction, midOutput, first.midpointSample)
      val h = -math.log(first.sigmaDown) - (-math.log(sigma))
      val base = first.sample * (first.sigmaDown / sigma).toFloat - clean2 * math.expm1(-h).toFloat
      val res = if first.sigmaUp > 0.0 then base + noise * first.sigmaUp.toFloat else base
      EulerStep(res, clean2)

/** DPM++ SDE (Stochastic Two-Stage) in Euler sigma space, matching Diffusers DPMSolverSDEScheduler. */
final case class DpmPlusPlusSde(
    schedule: EulerSchedule,
    prediction: DiffusionPrediction = DiffusionPrediction.Epsilon,
    maxElements: Long = 134217728L
):
  def scaleModelInput(index: Int, sample: Variable): Variable =
    SamplerPrediction.scaleEulerInput(schedule.sigmas(index), sample)

  def scaleMidpointInput(midpointSigma: Double, sample: Variable): Variable =
    SamplerPrediction.scaleEulerInput(midpointSigma, sample)

  def predict(index: Int, modelOutput: Variable, sample: Variable, noise1: Variable): SecondOrderPrediction =
    val sigma = schedule.sigmas(index)
    val next = schedule.sigmas(index + 1)
    val clean = SamplerPrediction.toOriginalEuler(sigma, prediction, modelOutput, sample)
    val d = (sample - clean) * (1.0 / sigma).toFloat
    if next <= 0.0 then
      SecondOrderPrediction(sample, clean, 0.0, d, clean, 0.0, 0.0, requiresSecondEvaluation = false)
    else
      val t = -math.log(sigma)
      val tNext = -math.log(next)
      val deltaTime = tNext - t
      val tProposed = t + 0.5 * deltaTime
      val sigmaTo = math.exp(-tProposed)
      val sigmaUp = math.min(sigmaTo, math.sqrt(math.max(0.0, sigmaTo * sigmaTo * (sigma * sigma - sigmaTo * sigmaTo) / (sigma * sigma))))
      val sigmaDown = math.sqrt(math.max(1e-12, sigmaTo * sigmaTo - sigmaUp * sigmaUp))
      val ancestralT = -math.log(sigmaDown)
      val midSample = sample * (sigmaDown / sigma).toFloat - clean * math.expm1(t - ancestralT).toFloat + noise1 * sigmaUp.toFloat
      SecondOrderPrediction(sample, midSample, sigmaTo, d, clean, next, deltaTime, requiresSecondEvaluation = true)

  def correct(index: Int, first: SecondOrderPrediction, midOutput: Variable, noise2: Variable): EulerStep =
    if !first.requiresSecondEvaluation then EulerStep(first.midpointSample, first.predictedOriginal)
    else
      val sigma = schedule.sigmas(index)
      val next = first.sigmaDown
      val t = -math.log(sigma)
      val clean2 = SamplerPrediction.toOriginalEuler(first.midpointSigma, prediction, midOutput, first.midpointSample)
      val sigmaUp = math.min(next, math.sqrt(math.max(0.0, next * next * (sigma * sigma - next * next) / (sigma * sigma))))
      val sigmaDown = math.sqrt(math.max(1e-12, next * next - sigmaUp * sigmaUp))
      val ancestralT = -math.log(sigmaDown)
      val base = first.sample * (sigmaDown / sigma).toFloat - clean2 * math.expm1(t - ancestralT).toFloat + noise2 * sigmaUp.toFloat
      EulerStep(base, clean2)

/** DPM++ 2M SDE, DPM++ 2M SDE Heun, and DPM++ 3M SDE in VP space (matching Diffusers DPMSolverMultistepScheduler sde-dpmsolver++). */
final case class DpmPlusPlusMultistepSde(
    sigmas: Vector[Double],
    prediction: DiffusionPrediction = DiffusionPrediction.Epsilon,
    order: Int = 2,
    heun: Boolean = false,
    maxElements: Long = 134217728L,
    totalStepsOverride: Option[Int] = None,
    startStepIndex: Int = 0
):
  require(order == 2 || order == 3, "DPM++ Multistep SDE order must be 2 or 3")
  private val totalSteps: Int = totalStepsOverride.getOrElse(sigmas.size - 1)
  private val history = scala.collection.mutable.ArrayBuffer.empty[(Double, Variable)]
  private var lowerOrderNums: Int = 0

  def step(index: Int, modelOutput: Variable, sample: Variable, noise: Variable): DpmPlusPlusStep =
    require(index >= 0 && index < sigmas.size - 1, "DPM++ SDE step index out of range")
    val s = sigmas(index)
    val next = sigmas(index + 1)
    val stepIndex = startStepIndex + index
    val lowerOrderFinal = (stepIndex == totalSteps - 1) || (next == 0.0)
    val lowerOrderSecond = (stepIndex == totalSteps - 2) && (totalSteps < 15)
    val original = SamplerPrediction.toOriginalVp(s, prediction, modelOutput, sample)
    history.prepend((s, original))
    if history.size > order then history.remove(history.size - 1)
    val result =
      if next == s then DpmPlusPlusStep(sample, original)
      else if next == 0.0 then DpmPlusPlusStep(original, original)
      else
        val alphaS = 1.0 / math.hypot(s, 1.0)
        val sigmaS = s * alphaS
        val alphaT = 1.0 / math.hypot(next, 1.0)
        val sigmaT = next * alphaT
        val h = math.log(s) - math.log(next)
        val xCoeff = ((sigmaT / sigmaS) * math.exp(-h)).toFloat
        val d0Coeff = (alphaT * (1.0 - math.exp(-2.0 * h))).toFloat
        val noiseStd = (sigmaT * math.sqrt(math.max(0.0, 1.0 - math.exp(-2.0 * h)))).toFloat
        if lowerOrderNums < 1 || lowerOrderFinal || history.size < 2 || history(1)._1 <= s then
          val prev = sample * xCoeff + original * d0Coeff + noise * noiseStd
          DpmPlusPlusStep(prev, original)
        else if order == 2 || lowerOrderNums < 2 || lowerOrderSecond || history.size < 3 || history(2)._1 <= history(1)._1 then
          val s1 = history(1)._1
          val m0 = original
          val m1 = history(1)._2
          val h0 = math.log(s1) - math.log(s)
          val r0 = h0 / h
          val d1 = (m0 - m1) * (1.0 / r0).toFloat
          val d1Coeff =
            if heun then (alphaT * ((1.0 - math.exp(-2.0 * h)) / (-2.0 * h) + 1.0)).toFloat
            else (0.5 * alphaT * (1.0 - math.exp(-2.0 * h))).toFloat
          val prev = sample * xCoeff + m0 * d0Coeff + d1 * d1Coeff + noise * noiseStd
          DpmPlusPlusStep(prev, original)
        else
          val s1 = history(1)._1
          val s2 = history(2)._1
          val m0 = original
          val m1 = history(1)._2
          val m2 = history(2)._2
          val h0 = math.log(s1) - math.log(s)
          val h1 = math.log(s2) - math.log(s1)
          val r0 = h0 / h
          val r1 = h1 / h
          val d1_0 = (m0 - m1) * (1.0 / r0).toFloat
          val d1_1 = (m1 - m2) * (1.0 / r1).toFloat
          val d1 = d1_0 + (d1_0 - d1_1) * (r0 / (r0 + r1)).toFloat
          val d2 = (d1_0 - d1_1) * (1.0 / (r0 + r1)).toFloat
          val d1Coeff = (alphaT * ((1.0 - math.exp(-2.0 * h)) / (-2.0 * h) + 1.0)).toFloat
          val twoH = 2.0 * h
          val d2Coeff = (alphaT * ((1.0 - math.exp(-twoH) - twoH) / (twoH * twoH) - 0.5)).toFloat
          val prev = sample * xCoeff + m0 * d0Coeff + d1 * d1Coeff + d2 * d2Coeff + noise * noiseStd
          DpmPlusPlusStep(prev, original)
    if lowerOrderNums < order then lowerOrderNums += 1
    result

/** UniPC (Unified Predictor-Corrector) Multistep Scheduler in VP space, matching Diffusers UniPCMultistepScheduler (bh2, predict_x0=True). */
final case class UniPcScheduler(
    sigmas: Vector[Double],
    prediction: DiffusionPrediction = DiffusionPrediction.Epsilon,
    order: Int = 2,
    maxElements: Long = 134217728L,
    totalStepsOverride: Option[Int] = None,
    startStepIndex: Int = 0
):
  private val totalSteps: Int = totalStepsOverride.getOrElse(sigmas.size - 1)
  private val history = scala.collection.mutable.ArrayBuffer.empty[(Double, Variable)]
  private var lastSample: Option[Variable] = None
  private var thisOrder: Int = 1
  private var lowerOrderNums: Int = 0

  def step(index: Int, modelOutput: Variable, sample: Variable): DpmPlusPlusStep =
    require(index >= 0 && index < sigmas.size - 1, "UniPC step index out of range")
    val s = sigmas(index)
    val next = sigmas(index + 1)
    val stepIndex = startStepIndex + index
    val modelT = SamplerPrediction.toOriginalVp(s, prediction, modelOutput, sample)

    // 1. UniC corrector (when step_index > 0 and last_sample is defined)
    val correctedSample =
      if stepIndex > 0 && lastSample.isDefined && history.nonEmpty && history.head._1 > s && s > 0.0 then
        val x = lastSample.get
        val s0 = history.head._1
        val m0 = history.head._2
        val alphaT = 1.0 / math.hypot(s, 1.0)
        val sigmaT = s * alphaT
        val alphaS0 = 1.0 / math.hypot(s0, 1.0)
        val sigmaS0 = s0 * alphaS0
        val h = math.log(s0) - math.log(s)
        val hh = -h
        val hPhi1 = math.expm1(hh)
        val hPhiK = hPhi1 / hh - 1.0
        val bH = hPhi1
        val xTBase = x * (sigmaT / sigmaS0).toFloat - m0 * (alphaT * hPhi1).toFloat
        val d1T = modelT - m0
        if thisOrder == 1 || history.size < 2 || history(1)._1 <= s0 then
          xTBase - d1T * (alphaT * bH * 0.5).toFloat
        else
          val s1 = history(1)._1
          val m1 = history(1)._2
          val rk = (math.log(s0) - math.log(s1)) / h
          val d1_0 = (m1 - m0) * (1.0 / rk).toFloat
          val b0 = hPhiK / bH
          val hPhi2 = hPhiK / hh - 0.5
          val b1 = hPhi2 * 2.0 / bH
          val rho0 = (b1 - b0) / (rk - 1.0)
          val rho1 = b0 - rho0
          val corrRes = d1_0 * rho0.toFloat + d1T * rho1.toFloat
          xTBase - corrRes * (alphaT * bH).toFloat
      else sample

    // 2. Push model_output_convert to history
    history.prepend((s, modelT))
    if history.size > order then history.remove(history.size - 1)

    // 3. Determine this_order for UniP (and next step's UniC)
    val remOrder = math.min(order, totalSteps - stepIndex)
    thisOrder = math.max(1, math.min(remOrder, lowerOrderNums + 1))
    lastSample = Some(correctedSample)

    // 4. UniP predictor
    val nextSample =
      if next == s then correctedSample
      else if next == 0.0 then modelT
      else
        val alphaS0 = 1.0 / math.hypot(s, 1.0)
        val sigmaS0 = s * alphaS0
        val alphaT = 1.0 / math.hypot(next, 1.0)
        val sigmaT = next * alphaT
        val h = math.log(s) - math.log(next)
        val hh = -h
        val hPhi1 = math.expm1(hh)
        val bH = hPhi1
        val xTBase = correctedSample * (sigmaT / sigmaS0).toFloat - modelT * (alphaT * hPhi1).toFloat
        if thisOrder == 1 || history.size < 2 || history(1)._1 <= s then
          xTBase
        else
          val s1 = history(1)._1
          val m1 = history(1)._2
          val rk = (math.log(s) - math.log(s1)) / h
          val d1 = (m1 - modelT) * (1.0 / rk).toFloat
          xTBase - d1 * (alphaT * bH * 0.5).toFloat

    if lowerOrderNums < order then lowerOrderNums += 1
    DpmPlusPlusStep(nextSample, modelT)

/** LCM (Latent Consistency Model) Schedule and Scheduler in VP space, matching Diffusers LCMScheduler. */
final class LcmSchedule private (
    val layout: EulerSchedule,
    private[models] val alphas: Vector[Double],
    private[models] val nextAlphas: Vector[Double]
)

object LcmSchedule:
  def apply(noise: DiffusionNoiseSchedule, steps: Int, originalInferenceSteps: Int = 50): LcmSchedule =
    val count = noise.betas.size
    require(steps > 0 && steps <= 10000, "invalid LCM step count")
    val k = math.max(1, count / originalInferenceSteps)
    val lcmOriginSteps = Vector.tabulate(originalInferenceSteps)(i => math.min(count - 1, (i + 1) * k - 1)).reverse
    val times = Vector.tabulate(steps) { i =>
      val idx = math.min(lcmOriginSteps.size - 1, math.floor(i.toDouble * lcmOriginSteps.size / math.max(1, steps)).toInt)
      lcmOriginSteps(idx)
    }
    val trainingAlphas = noise.floatTrainingSigmas.map(s => 1.0 / (1.0 + s.toDouble * s.toDouble))
    val alphas = times.map(trainingAlphas)
    val nextAlphas = Vector.tabulate(times.size) { i =>
      if i + 1 < times.size then trainingAlphas(times(i + 1)) else trainingAlphas(times(i))
    }
    val sigmas = times.map(i => noise.floatTrainingSigmas(i).toDouble) :+ 0.0
    new LcmSchedule(EulerSchedule(times.map(_.toFloat), sigmas, 1.0), alphas, nextAlphas)

final case class LcmScheduler(
    schedule: LcmSchedule,
    prediction: DiffusionPrediction = DiffusionPrediction.Epsilon,
    sigmaData: Double = 0.5,
    timestepScaling: Double = 10.0
):
  def step(index: Int, modelOutput: Variable, sample: Variable, noise: Option[Variable] = None): DpmPlusPlusStep =
    require(index >= 0 && index < schedule.alphas.size, "LCM step index out of range")
    val alpha = schedule.alphas(index)
    val nextAlpha = schedule.nextAlphas(index)
    val sqrtAlpha = math.sqrt(alpha).toFloat
    val sqrtOneMinusAlpha = math.sqrt(math.max(0.0, 1.0 - alpha)).toFloat
    val x0Pred = prediction match
      case DiffusionPrediction.Epsilon => (sample - modelOutput * sqrtOneMinusAlpha) * (1.0f / sqrtAlpha)
      case DiffusionPrediction.Sample => modelOutput
      case DiffusionPrediction.VPrediction => sample * sqrtAlpha - modelOutput * sqrtOneMinusAlpha
    val scaledT = schedule.layout.timesteps(index).toDouble * timestepScaling
    val cSkip = (sigmaData * sigmaData) / (scaledT * scaledT + sigmaData * sigmaData)
    val cOut = scaledT / math.sqrt(scaledT * scaledT + sigmaData * sigmaData)
    val denoised = sample * cSkip.toFloat + x0Pred * cOut.toFloat
    val prev = if index < schedule.alphas.size - 1 && noise.isDefined then
      denoised * math.sqrt(nextAlpha).toFloat + noise.get * math.sqrt(math.max(0.0, 1.0 - nextAlpha)).toFloat
    else denoised
    DpmPlusPlusStep(prev, denoised)

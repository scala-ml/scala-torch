package scalatorch.models

import scalatorch.autograd.Variable

enum DiffusionPrediction:
  case Epsilon, Sample, VPrediction

enum DiffusionTimestepSpacing:
  case Linspace, Leading, Trailing

/** Immutable training coefficients, accumulated in Double independently of tensor precision. */
final class DiffusionNoiseSchedule private (val betas: Vector[Double],val sigmas: Vector[Double],
    private[models] val floatTrainingSigmas: Vector[Float]):
  lazy val alphasCumprod: Vector[Double] = sigmas.map(s => 1.0 / (1.0 + s * s))
  /** Rescales betas so terminal SNR approaches zero while keeping terminal sigma finite in Float32. */
  def withZeroTerminalSnr: DiffusionNoiseSchedule =
    var alpha=1.0
    val sqrtAlphaBars=betas.map { b =>
      alpha*=1.0-b
      math.sqrt(alpha)
    }
    val s0=sqrtAlphaBars.head
    val sT=sqrtAlphaBars.last
    val denom=math.max(s0-sT,1e-12)
    val rescaledBars=sqrtAlphaBars.map { s =>
      val r=(s-sT)*s0/denom
      math.max(r*r,1e-7)
    }
    val newBetas=Vector.tabulate(rescaledBars.size) { i =>
      val a=if i==0 then rescaledBars(0) else rescaledBars(i)/rescaledBars(i-1)
      math.min(math.max(1.0-a,1e-7),1.0-1e-7)
    }
    DiffusionNoiseSchedule.fromBetas(newBetas)

  /** NUI DDIM selects training sigmas starting at index 1 and may return extra steps. */
  def ddimEuler(steps: Int): EulerSchedule =
    require(steps>0 && steps<=10000 && sigmas.size>1,"invalid DDIM schedule dimensions")
    val stride=math.max(sigmas.size/steps,1)
    val indices=(1 until sigmas.size by stride).reverse
    fromEulerLevels(indices.map(index=>floatTrainingSigmas(index).toDouble).toVector).copy(nominalInferenceSteps=Some(steps))

  private[models] def betaPpf(q: Double,alpha: Double=0.6,beta: Double=0.6): Double =
    val logBeta=org.apache.commons.numbers.gamma.LogBeta.value(alpha,beta)
    if q<=0.0 then 0.0
    else if q>=1.0 then 1.0
    else
      var lo=0.0;var hi=1.0
      var x=alpha/(alpha+beta)
      var iter=0;var done=false
      while iter<40 && !done do
        val f=org.apache.commons.numbers.gamma.RegularizedBeta.value(x,alpha,beta)-q
        if math.abs(f)<1e-10 then done=true
        else
          if f<0.0 then lo=x else hi=x
          val pdf=math.exp((alpha-1.0)*math.log(math.max(x,1e-15))+(beta-1.0)*math.log(math.max(1.0-x,1e-15))-logBeta)
          val nx=x-f/math.max(pdf,1e-15)
          x=if nx>lo && nx<hi then nx else 0.5*(lo+hi)
          iter+=1
      x

  private[models] def sigmaToT(sigma: Double): Double =
    if floatTrainingSigmas.size<=1 then 0.0
    else
      val logs=floatTrainingSigmas.map(s => math.log(s.toDouble).toFloat)
      val target=math.log(math.max(sigma,1e-10))
      if target<=logs.head.toDouble then 0.0
      else if target>=logs.last.toDouble then (logs.size-1).toDouble
      else
        var lo=0
        var hi=logs.size-1
        while hi-lo>1 do
          val mid=lo+(hi-lo)/2
          if logs(mid).toDouble<=target then lo=mid else hi=mid
        val w=math.max(0.0,math.min(1.0,(target-logs(lo).toDouble)/(logs(hi).toDouble-logs(lo).toDouble)))
        (1.0-w)*lo+w*hi

  /** Diffusers discrete Euler-space schedule (Heun, LMS, KDPM2, KDPM2 a, DPM++ SDE). */
  def discreteSchedule(steps: Int,spacing: DiffusionTimestepSpacing=DiffusionTimestepSpacing.Linspace,
      stepsOffset: Int=0,conversion: String="none",roundKarrasTimesteps: Boolean=false,
      alpha: Double=0.6,beta: Double=0.6): EulerSchedule =
    val base=euler(steps,spacing,stepsOffset)
    val inSigmas=base.sigmas.dropRight(1)
    val sigmaMin=inSigmas.last
    val sigmaMax=inSigmas.head
    val (times,levels)=conversion match
      case "karras" =>
        val rho=7.0
        val minInv=math.pow(sigmaMin,1.0/rho)
        val maxInv=math.pow(sigmaMax,1.0/rho)
        val sigs=Vector.tabulate(steps) { i =>
          val ramp=if steps==1 then 0.0 else i.toDouble/(steps-1)
          math.pow(maxInv+ramp*(minInv-maxInv),rho)
        }
        val ts=sigs.map { s =>
          val t=sigmaToT(s)
          (if roundKarrasTimesteps then math.rint(t) else t).toFloat
        }
        (ts,sigs)
      case "exponential" =>
        val high=math.log(sigmaMax)
        val low=math.log(sigmaMin)
        val sigs=Vector.tabulate(steps) { i =>
          val ramp=if steps==1 then 0.0 else i.toDouble/(steps-1)
          math.exp(high+ramp*(low-high))
        }
        val ts=sigs.map(s => sigmaToT(s).toFloat)
        (ts,sigs)
      case "beta" =>
        val sigs=Vector.tabulate(steps) { i =>
          val q=if steps==1 then 1.0 else 1.0-i.toDouble/(steps-1)
          sigmaMin+betaPpf(q,alpha,beta)*(sigmaMax-sigmaMin)
        }
        val ts=sigs.map(s => sigmaToT(s).toFloat)
        (ts,sigs)
      case _ => (base.timesteps,inSigmas)
    val floatLevels=levels.map(_.toFloat.toDouble)
    val initial=if spacing==DiffusionTimestepSpacing.Leading then math.hypot(floatLevels.head,1.0) else floatLevels.head
    EulerSchedule(times,floatLevels :+ 0.0,initial)

  /** NUI Beta schedule selects deduplicated discrete training sigmas via regularized incomplete beta inversion. */
  def betaEuler(steps: Int,alpha: Double=0.6,beta: Double=0.6): EulerSchedule =
    require(steps>0 && steps<=10000,"invalid Beta inference step count")
    require(alpha.isFinite && alpha>0 && beta.isFinite && beta>0,"Beta schedule alpha and beta must be finite and positive")
    val total=floatTrainingSigmas.size-1
    val levels=Vector.newBuilder[Double]
    var last = -1
    var i=0
    while i<steps do
      val t=1.0-i.toDouble/steps
      val idx=math.min(math.max(math.rint(betaPpf(t,alpha,beta)*total).toInt,0),total)
      if idx!=last then
        levels+=floatTrainingSigmas(idx).toDouble
        last=idx
      i+=1
    fromEulerLevels(levels.result()).copy(nominalInferenceSteps=Some(steps))

  /** NUI KL Optimal spans n+1 positive sigmas, including a nonzero terminal value. */
  def klOptimalEuler(steps: Int): EulerSchedule =
    require(steps>0 && steps<=10000,"invalid KL Optimal inference step count")
    val low=math.atan(sigmas.head)
    val high=math.atan(sigmas.last)
    val levels=Vector.tabulate(steps+1) { i =>
      val fraction=i.toDouble/steps
      math.tan(fraction*low+(1-fraction)*high)
    }
    fromEulerLevels(levels.dropRight(1),levels.last)

  /** Align Your Steps uses NUI's SD1/SDXL reference ratios, log-interpolated in sigma space. */
  def alignYourStepsEuler(steps: Int,sdxl: Boolean): EulerSchedule =
    require(steps>0 && steps<=10000,"invalid Align Your Steps inference step count")
    val divisors=if sdxl then Vector(1.0,2.314,3.875,6.701,10.89,16.954,26.333,38.46,62.457,129.336)
      else Vector(1.0,2.257,3.785,5.418,7.749,10.469,15.176,22.415,36.629,96.151)
    val base=divisors.map(value=>sigmas.last/value) :+ 0.029
    val levels=Vector.tabulate(steps) { i =>
      val position=if steps==1 then (base.size-1).toDouble else i.toDouble*(base.size-1)/(steps-1)
      val left=math.floor(position).toInt
      val right=math.min(left+1,base.size-1)
      val fraction=position-left
      math.exp(math.log(base(left))*(1-fraction)+math.log(base(right))*fraction)
    }
    fromEulerLevels(levels)

  /** NUI Normal has the same logarithmic timestep spacing as Uniform. */
  def normalEuler(steps: Int): EulerSchedule = nuiUniformEuler(steps)

  /** NUI Simple selects discrete training sigmas at evenly spaced indexes. */
  def simpleEuler(steps: Int): EulerSchedule =
    require(steps>0 && steps<=10000,"invalid Simple inference step count")
    val levels=Vector.tabulate(steps) { i =>
      val index=sigmas.size-1-((i.toDouble*sigmas.size/steps).toInt)
      floatTrainingSigmas(index).toDouble
    }
    fromEulerLevels(levels)

  /** NUI SGM Uniform omits the final point of an n+1 timestep grid. */
  def sgmUniformEuler(steps: Int): EulerSchedule =
    require(steps>0 && steps<=10000,"invalid SGM Uniform inference step count")
    val levels=Vector.tabulate(steps) { i =>
      val t=(sigmas.size-1).toDouble*(steps-i)/steps
      val low=math.floor(t).toInt;val high=math.min(low+1,sigmas.size-1)
      val fraction=t-low
      math.exp(math.log(sigmas(low))*(1-fraction)+math.log(sigmas(high))*fraction)
    }
    fromEulerLevels(levels)

  /** NUI Uniform uses log-interpolated training sigmas, including the maximum for one step. */
  def nuiUniformEuler(steps: Int): EulerSchedule =
    require(steps>0 && steps<=10000,"invalid NUI Euler inference step count")
    val times=Vector.tabulate(steps) { i =>
      (if steps==1 then betas.size-1.0 else (betas.size-1.0)*(steps-1-i)/(steps-1)).toFloat
    }
    val levels=times.map { value =>
      val lo=math.floor(value.toDouble).toInt
      val hi=math.min(lo+1,sigmas.size-1)
      val fraction=value.toDouble-lo
      math.exp(math.log(sigmas(lo))*(1-fraction)+math.log(sigmas(hi))*fraction)
    }
    EulerSchedule(times,levels :+ 0.0,levels.head)

  /** Karras noise levels with log-linear inversion of the training sigma table. */
  def karrasEuler(steps: Int,rho: Double=7.0): EulerSchedule =
    require(steps>0 && steps<=10000,"invalid Karras inference step count")
    require(rho.isFinite && rho>0,"Karras rho must be finite and positive")
    val low=math.pow(sigmas.head,1/rho)
    val high=math.pow(sigmas.last,1/rho)
    require(low.isFinite && high.isFinite && low>0 && high>0,"unrepresentable Karras powers")
    val levels=Vector.tabulate(steps) { i =>
      if i==0 then sigmas.last
      else if i==steps-1 then sigmas.head
      else math.pow(high+(low-high)*i/(steps-1),rho)
    }
    fromEulerLevels(levels)

  /** NUI's default exponential sigma schedule, with log-linear timestep inversion. */
  def exponentialEuler(steps: Int): EulerSchedule =
    require(steps>0 && steps<=10000,"invalid exponential inference step count")
    val high=math.log(sigmas.last)
    val low=math.log(sigmas.head)
    val levels=Vector.tabulate(steps) { i =>
      if i==0 then sigmas.last
      else if i==steps-1 then sigmas.head
      else math.exp(high+(low-high)*i/(steps-1))
    }
    fromEulerLevels(levels)

  private def fromEulerLevels(levels: Vector[Double],terminal: Double=0.0): EulerSchedule =
    val logs=sigmas.map(math.log)
    val times=levels.map { sigma =>
      require(sigma.isFinite && sigma>0,"invalid Karras sigma")
      val target=math.log(sigma)
      if target<=logs.head then 0f
      else if target>=logs.last then (logs.size-1).toFloat
      else
        var lo=0
        var hi=logs.size-1
        while hi-lo>1 do
          val mid=lo+(hi-lo)/2
          if logs(mid)<=target then lo=mid else hi=mid
        (lo+(target-logs(lo))/(logs(hi)-logs(lo))).toFloat
    }
    EulerSchedule(times,levels :+ terminal,levels.head)

  def euler(steps: Int,spacing: DiffusionTimestepSpacing=DiffusionTimestepSpacing.Linspace,
      stepsOffset: Int=0): EulerSchedule =
    require(steps>0 && steps<=10000,"invalid Euler inference step count")
    // Checkpoints retain steps_offset when switching samplers/spacing. Only
    // leading timesteps consume it; linspace and trailing ignore it.
    val count=betas.size
    val times=Vector.tabulate(steps) { i =>
      val t: Double=spacing match
        case DiffusionTimestepSpacing.Linspace => if steps==1 then 0.0 else (count-1).toDouble*(steps-1-i)/(steps-1)
        case DiffusionTimestepSpacing.Leading => ((steps-1-i).toLong*(count/steps)+stepsOffset.toLong).toDouble
        case DiffusionTimestepSpacing.Trailing => math.rint(count.toDouble-i*count.toDouble/steps)-1
      require(t>= -1 && t<=count-1,"Euler timestep outside training schedule")
      t.toFloat
    }
    val levels=times.map { t =>
      val lower=math.max(0,math.floor(t.toDouble).toInt)
      val upper=math.min(lower+1,count-1)
      sigmas(lower)+(sigmas(upper)-sigmas(lower))*math.max(0.0,t.toDouble-lower)
    }
    val initial=if spacing==DiffusionTimestepSpacing.Leading then math.hypot(levels.head,1) else levels.head
    EulerSchedule(times,levels :+ 0.0,initial)

object DiffusionNoiseSchedule:
  private val maxSteps=1000000
  def fromBetas(betas: Vector[Double]): DiffusionNoiseSchedule =
    fromBetasWithFloat(betas,betas.map(_.toFloat))
  private def fromBetasWithFloat(betas: Vector[Double],floatBetas: Vector[Float]): DiffusionNoiseSchedule =
    require(betas.nonEmpty && betas.size<=maxSteps,"invalid training schedule length")
    require(betas.forall(b => b.isFinite && b>0 && b<1),"betas must be finite in (0,1)")
    var alpha=1.0
    val sigmas=betas.map { beta =>
      alpha*=1-beta
      val sigma=math.sqrt((1-alpha)/alpha)
      require(sigma.isFinite && sigma.toFloat>0 && sigma.toFloat.isFinite && (1/sigma).toFloat.isFinite,
        "training sigma is not representable")
      sigma
    }
    var floatAlpha=1.0
    val floatSigmas=floatBetas.zipWithIndex.map { (beta,index) =>
      floatAlpha*=(1f-beta).toDouble
      val a=floatAlpha.toFloat
      val sigma=math.sqrt(((1f-a)/a).toDouble).toFloat
      if sigma.isFinite && sigma>0f then sigma else sigmas(index).toFloat
    }
    new DiffusionNoiseSchedule(betas,sigmas,floatSigmas)
  def linear(trainingSteps: Int,betaStart: Double,betaEnd: Double,scaled: Boolean=false): DiffusionNoiseSchedule =
    require(trainingSteps>0 && trainingSteps<=maxSteps,"invalid training schedule length")
    require(betaStart.isFinite && betaEnd.isFinite && betaStart>0 && betaStart<=betaEnd && betaEnd<1,
      "invalid linear beta bounds")
    val start=if scaled then math.sqrt(betaStart) else betaStart
    val end=if scaled then math.sqrt(betaEnd) else betaEnd
    val doubles=Vector.tabulate(trainingSteps) { i =>
      val value=if trainingSteps==1 then start else start+(end-start)*i/(trainingSteps-1)
      if scaled then value*value else value
    }
    val floatStart=start.toFloat;val floatEnd=end.toFloat
    val step=if trainingSteps==1 then 0f else ((floatEnd.toDouble-floatStart.toDouble)/(trainingSteps-1)).toFloat
    val floats=Vector.tabulate(trainingSteps) { i =>
      val value=if i<trainingSteps/2 then floatStart+i.toFloat*step else floatEnd-(trainingSteps-1-i).toFloat*step
      if trainingSteps==1 then (if scaled then floatStart*floatStart else floatStart)
      else if scaled then value*value else value
    }
    fromBetasWithFloat(doubles,floats)

/** Explicit model timesteps and descending positive sigmas with a nonnegative terminal sigma. */
final case class EulerSchedule(timesteps: Vector[Float],sigmas: Vector[Double],initialNoiseSigma: Double,
    nominalInferenceSteps: Option[Int]=None):
  require(timesteps.nonEmpty && timesteps.size<=1000000 && sigmas.size==timesteps.size+1,"invalid Euler schedule length")
  require(nominalInferenceSteps.forall(n => n>0 && n<=10000),"invalid nominal Euler inference step count")
  require(timesteps.forall(t => t.isFinite && t>= -1) && timesteps.zip(timesteps.drop(1)).forall((a,b) => a>=b),
    "Euler timesteps must be finite and descending")
  require(sigmas.last.isFinite && sigmas.last>=0 && sigmas.last.toFloat.isFinite && sigmas.dropRight(1).forall(s => s.isFinite && s>0 && s.toFloat>0 && s.toFloat.isFinite &&
    (1/s).toFloat.isFinite) && sigmas.zip(sigmas.drop(1)).forall((a,b) => a>=b),"invalid Euler sigmas")
  require(initialNoiseSigma.isFinite && initialNoiseSigma.toFloat>0 && initialNoiseSigma.toFloat.isFinite,
    "invalid Euler initial noise scale")

final case class EulerStep(previous: Variable,predictedOriginal: Variable)

/** Stateless CPU Euler update, indexed explicitly so branches/retries do not mutate a shared cursor. */
final case class EulerScheduler(schedule: EulerSchedule,prediction: DiffusionPrediction=DiffusionPrediction.Epsilon,
    maxElements: Long=134217728L):
  require(maxElements>=0,"negative Euler allocation limit")
  private def sigma(index: Int): Double =
    require(index>=0 && index<schedule.timesteps.size,"Euler step index out of range")
    schedule.sigmas(index)
  private def check(input: Variable): Unit =
    require(input.data.shape.numel<=maxElements && input.data.shape.numel<=Int.MaxValue,"Euler tensor exceeds allocation limit")
  private def pair(a: Variable,b: Variable): Unit =
    require(a.data.shape==b.data.shape,"Euler tensor shape mismatch")
    check(a);check(b)
  def scaleModelInput(index: Int,sample: Variable): Variable =
    val s=sigma(index);check(sample)
    sample*(1/math.hypot(s,1)).toFloat
  def addNoise(index: Int,clean: Variable,noise: Variable): Variable =
    val s=sigma(index);pair(clean,noise)
    clean+noise*s.toFloat
  def step(index: Int,modelOutput: Variable,sample: Variable): EulerStep =
    val s=sigma(index);pair(modelOutput,sample)
    val original=prediction match
      case DiffusionPrediction.Epsilon => sample-modelOutput*s.toFloat
      case DiffusionPrediction.Sample => modelOutput
      case DiffusionPrediction.VPrediction => modelOutput*(-s/math.hypot(s,1)).toFloat+sample*(1/(s*s+1)).toFloat
    val derivative=(sample-original)*(1/s).toFloat
    EulerStep(sample+derivative*(schedule.sigmas(index+1)-s).toFloat,original)

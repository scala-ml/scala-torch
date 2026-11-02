package scalatorch.models

import scalatorch.autograd.Variable

enum DiffusionPrediction:
  case Epsilon, Sample, VPrediction

enum DiffusionTimestepSpacing:
  case Linspace, Leading, Trailing

/** Immutable training coefficients, accumulated in Double independently of tensor precision. */
final class DiffusionNoiseSchedule private (val betas: Vector[Double],val sigmas: Vector[Double]):
  /** Karras noise levels with log-linear inversion of the training sigma table. */
  def karrasEuler(steps: Int,rho: Double=7.0): EulerSchedule =
    require(steps>0 && steps<=betas.size,"invalid Karras inference step count")
    require(rho.isFinite && rho>0,"Karras rho must be finite and positive")
    val low=math.pow(sigmas.head,1/rho)
    val high=math.pow(sigmas.last,1/rho)
    require(low.isFinite && high.isFinite && low>0 && high>0,"unrepresentable Karras powers")
    val levels=Vector.tabulate(steps) { i =>
      if i==0 then sigmas.last
      else if i==steps-1 then sigmas.head
      else math.pow(high+(low-high)*i/(steps-1),rho)
    }
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
    EulerSchedule(times,levels :+ 0.0,levels.head)

  def euler(steps: Int,spacing: DiffusionTimestepSpacing=DiffusionTimestepSpacing.Linspace,
      stepsOffset: Int=0): EulerSchedule =
    require(steps>0 && steps<=betas.size,"invalid Euler inference step count")
    // Checkpoints retain steps_offset when switching samplers/spacing. Only
    // leading timesteps consume it; linspace and trailing ignore it.
    val count=betas.size
    val times=Vector.tabulate(steps) { i =>
      val t: Double=spacing match
        case DiffusionTimestepSpacing.Linspace => if steps==1 then 0.0 else (count-1).toDouble*(steps-1-i)/(steps-1)
        case DiffusionTimestepSpacing.Leading => ((steps-1-i).toLong*(count/steps)+stepsOffset.toLong).toDouble
        case DiffusionTimestepSpacing.Trailing => math.rint(count.toDouble-i*count.toDouble/steps)-1
      require(t>=0 && t<=count-1,"Euler timestep outside training schedule")
      t.toFloat
    }
    val levels=times.map { t =>
      val lower=math.floor(t.toDouble).toInt
      val upper=math.min(lower+1,count-1)
      sigmas(lower)+(sigmas(upper)-sigmas(lower))*(t.toDouble-lower)
    }
    val initial=if spacing==DiffusionTimestepSpacing.Leading then math.hypot(levels.head,1) else levels.head
    EulerSchedule(times,levels :+ 0.0,initial)

object DiffusionNoiseSchedule:
  private val maxSteps=1000000
  def fromBetas(betas: Vector[Double]): DiffusionNoiseSchedule =
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
    new DiffusionNoiseSchedule(betas,sigmas)
  def linear(trainingSteps: Int,betaStart: Double,betaEnd: Double,scaled: Boolean=false): DiffusionNoiseSchedule =
    require(trainingSteps>0 && trainingSteps<=maxSteps,"invalid training schedule length")
    require(betaStart.isFinite && betaEnd.isFinite && betaStart>0 && betaStart<=betaEnd && betaEnd<1,
      "invalid linear beta bounds")
    val start=if scaled then math.sqrt(betaStart) else betaStart
    val end=if scaled then math.sqrt(betaEnd) else betaEnd
    fromBetas(Vector.tabulate(trainingSteps) { i =>
      val value=if trainingSteps==1 then start else start+(end-start)*i/(trainingSteps-1)
      if scaled then value*value else value
    })

/** Explicit model timesteps and descending positive sigmas followed by a terminal zero. */
final case class EulerSchedule(timesteps: Vector[Float],sigmas: Vector[Double],initialNoiseSigma: Double):
  require(timesteps.nonEmpty && timesteps.size<=1000000 && sigmas.size==timesteps.size+1,"invalid Euler schedule length")
  require(timesteps.forall(t => t.isFinite && t>=0) && timesteps.zip(timesteps.drop(1)).forall((a,b) => a>=b),
    "Euler timesteps must be finite and descending")
  require(sigmas.last==0 && sigmas.dropRight(1).forall(s => s.isFinite && s>0 && s.toFloat>0 && s.toFloat.isFinite &&
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

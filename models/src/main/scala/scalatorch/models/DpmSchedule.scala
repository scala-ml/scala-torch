package scalatorch.models

/** VP solver schedule: unit initial noise, explicit model timesteps and terminal zero. */
final case class DpmSchedule(timesteps: Vector[Float],sigmas: Vector[Double]):
  private[models] val layout=EulerSchedule(timesteps,sigmas,1.0)
  require(sigmas.zip(sigmas.drop(1)).forall((a,b) => a>=b),"DPM sigmas must not increase")

object DpmSchedule:
  /** Ten evaluations at the discrete indices published in the AYS quickstart.
    * This internal experimental schedule is not the NUI interpolated AYS schedule.
    * The last published index is a nonzero-noise endpoint, not a model evaluation.
    */
  /** Evaluation steps for NVIDIA Align Your Steps (`steps == 10` matches exact published SD1.5/SDXL indices;
    * other step counts log-interpolate cleanly across the 11 published paper sigmas with nonzero terminal sigma).
    */
  private[models] def paperAys(noise: DiffusionNoiseSchedule,sdxl: Boolean,steps: Int=10): DpmSchedule =
    require(steps>0 && steps<=10000,"invalid AYS inference step count")
    require(noise.betas.size==1000,"paper AYS indices require 1000 training timesteps")
    val expected=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true).betas
    require(noise.betas.zip(expected).forall((a,b)=>math.abs(a-b)<=1e-7),
      "paper AYS indices require the standard SD scaled-linear beta schedule")
    val indices=if sdxl then Vector(999,845,730,587,443,310,193,116,53,13,0)
      else Vector(999,850,736,645,545,455,343,233,124,24,0)
    if steps==10 then return DpmSchedule(indices.dropRight(1).map(_.toFloat),indices.map(noise.sigmas))
    val base=indices.map(noise.sigmas)
    val levels=Vector.tabulate(steps+1) { i =>
      val pos=i.toDouble*(base.size-1)/steps
      val lo=math.floor(pos).toInt
      val hi=math.min(lo+1,base.size-1)
      val frac=pos-lo
      math.exp(math.log(base(lo))*(1.0-frac)+math.log(base(hi))*frac)
    }
    val training=noise.floatTrainingSigmas
    val logs=training.map(s => math.log(s.toDouble).toFloat)
    val times=levels.dropRight(1).map { sigma =>
      if logs.size==1 then 0f
      else
        val target=math.log(sigma)
        var lo=0;var hi=logs.size-1
        while hi-lo>1 do
          val mid=(lo+hi)/2
          if logs(mid).toDouble<=target then lo=mid else hi=mid
        val weight=math.max(0.0,math.min(1.0,(target-logs(lo))/(logs(hi)-logs(lo))))
        math.rint(lo+weight).toFloat
    }
    DpmSchedule(times,levels)

  private[models] def paperAysZeroTerminal(noise: DiffusionNoiseSchedule,sdxl: Boolean,steps: Int=10): DpmSchedule =
    val s=paperAys(noise,sdxl,steps)
    DpmSchedule(s.timesteps,s.sigmas.dropRight(1) :+ 0.0)

  /** Diffusers leading spacing used by the DDIM schedule fallback. */
  def leading(noise: DiffusionNoiseSchedule,steps: Int,stepsOffset: Int=0): DpmSchedule =
    require(steps>0 && steps<=10000,"invalid DPM inference step count")
    val stride=noise.betas.size/(steps+1)
    val times=Vector.tabulate(steps) { i =>
      val index=(steps-i).toLong*stride+stepsOffset
      require(index>=0 && index<noise.betas.size,"DPM timestep outside training schedule")
      index.toInt
    }
    DpmSchedule(times.map(_.toFloat),times.map(noise.sigmas) :+ 0.0)

  /** Diffusers trailing spacing used by NUI's Simple and SGM Uniform fallbacks. */
  def trailing(noise: DiffusionNoiseSchedule,steps: Int): DpmSchedule =
    require(steps>0 && steps<=10000,"invalid DPM inference step count")
    val count=noise.betas.size
    val times=Vector.tabulate(steps)(i => math.rint(count.toDouble-i*count.toDouble/steps).toInt-1)
    DpmSchedule(times.map(_.toFloat),times.map(i => noise.sigmas(math.max(0,i))) :+ 0.0)

  /** Diffusers DPM linspace uses steps+1 rounded points and drops the zero endpoint. */
  def uniform(noise: DiffusionNoiseSchedule,steps: Int): DpmSchedule =
    require(steps>0 && steps<=10000,"invalid DPM inference step count")
    val times=Vector.tabulate(steps)(i => math.rint((noise.betas.size-1).toDouble*(steps-i)/steps).toInt)
    DpmSchedule(times.map(_.toFloat),times.map(noise.sigmas) :+ 0.0)

  /** Linear/scaled-linear training schedule profile used by NUI's DPM++ Karras path. */
  def karras(noise: DiffusionNoiseSchedule,steps: Int): DpmSchedule =
    val schedule=noise.karrasEuler(steps)
    DpmSchedule(schedule.timesteps.map(t => math.rint(t.toDouble).toFloat),schedule.sigmas)

  /** Beta schedule profile for DPM++. */
  def beta(noise: DiffusionNoiseSchedule,steps: Int,alpha: Double=0.6,beta: Double=0.6): DpmSchedule =
    require(steps>0 && steps<=10000,"invalid DPM inference step count")
    require(alpha.isFinite && alpha>0 && beta.isFinite && beta>0,"Beta schedule alpha and beta must be finite and positive")
    // A1111/NUI Beta: Beta(alpha,beta) quantiles of the timestep index, repeated steps removed (same as Euler).
    val e=noise.betaEuler(steps,alpha,beta)
    DpmSchedule(e.timesteps.map(t=>math.rint(t.toDouble).toFloat),e.sigmas)

  /** Diffusers converts exponential model timesteps to int64 by truncation. */
  def exponential(noise: DiffusionNoiseSchedule,steps: Int): DpmSchedule =
    require(steps>0 && steps<=10000,"invalid DPM inference step count")
    val training=noise.floatTrainingSigmas
    val logs=training.map(s => math.log(s.toDouble).toFloat)
    val high=math.log(training.last.toDouble);val low=math.log(training.head.toDouble)
    val levels=Vector.tabulate(steps)(i => math.exp(if steps==1 then high else high+(low-high)*i/(steps-1)))
    val times=levels.map { sigma =>
      if logs.size==1 then 0f
      else
        val target=math.log(sigma)
        var lo=0;var hi=logs.size-1
        while hi-lo>1 do
          val mid=(lo+hi)/2
          if logs(mid).toDouble<=target then lo=mid else hi=mid
        val weight=math.max(0.0,math.min(1.0,(target-logs(lo))/(logs(hi)-logs(lo))))
        (lo+weight).toInt.toFloat
    }
    DpmSchedule(times,levels :+ 0.0)

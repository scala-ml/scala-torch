package scalatorch.models

/** VP solver schedule: unit initial noise, explicit model timesteps and terminal zero. */
final case class DpmSchedule(timesteps: Vector[Float],sigmas: Vector[Double]):
  private[models] val layout=EulerSchedule(timesteps,sigmas,1.0)
  require(sigmas.zip(sigmas.drop(1)).forall((a,b) => a>=b),"DPM sigmas must not increase")

object DpmSchedule:
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
    val training=noise.floatTrainingSigmas
    val logs=training.map(s => math.log(s.toDouble).toFloat)
    val sigmaMin=training.head.toDouble
    val sigmaMax=training.last.toDouble
    val levels=Vector.tabulate(steps) { i =>
      val q=if steps==1 then 1.0 else 1.0-i.toDouble/(steps-1)
      sigmaMin+noise.betaPpf(q,alpha,beta)*(sigmaMax-sigmaMin)
    }
    val times=levels.map { sigma =>
      if logs.size==1 then 0f
      else
        val target=math.log(math.max(sigma,1e-10))
        var lo=0;var hi=logs.size-1
        while hi-lo>1 do
          val mid=(lo+hi)/2
          if logs(mid).toDouble<=target then lo=mid else hi=mid
        val weight=math.max(0.0,math.min(1.0,(target-logs(lo))/(logs(hi)-logs(lo))))
        (lo+weight).toInt.toFloat
    }
    DpmSchedule(times,levels.map(_.toFloat.toDouble) :+ 0.0)

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

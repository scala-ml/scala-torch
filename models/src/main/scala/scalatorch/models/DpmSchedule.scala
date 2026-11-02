package scalatorch.models

/** VP solver schedule: unit initial noise, explicit model timesteps and terminal zero. */
final case class DpmSchedule(timesteps: Vector[Float],sigmas: Vector[Double]):
  private[models] val layout=EulerSchedule(timesteps,sigmas,1.0)
  require(sigmas.zip(sigmas.drop(1)).forall((a,b) => a>b),"DPM sigmas must strictly decrease")

object DpmSchedule:
  /** Diffusers DPM linspace uses steps+1 rounded points and drops the zero endpoint. */
  def uniform(noise: DiffusionNoiseSchedule,steps: Int): DpmSchedule =
    require(steps>0 && steps<=noise.betas.size,"invalid DPM inference step count")
    val times=Vector.tabulate(steps)(i => math.rint((noise.betas.size-1).toDouble*(steps-i)/steps).toInt)
    DpmSchedule(times.map(_.toFloat),times.map(noise.sigmas) :+ 0.0)

  /** Linear/scaled-linear training schedule profile used by NUI's DPM++ Karras path. */
  def karras(noise: DiffusionNoiseSchedule,steps: Int): DpmSchedule =
    val schedule=noise.karrasEuler(steps)
    DpmSchedule(schedule.timesteps.map(t => math.rint(t.toDouble).toFloat),schedule.sigmas)

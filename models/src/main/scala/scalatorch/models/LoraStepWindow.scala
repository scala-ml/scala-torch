package scalatorch.models

/** NUI LoRA step windows use one-based steps and a half-open stop boundary. */
final case class LoraStepWindow(start: Option[Double]=None,stop: Option[Double]=None):
  require((start.toVector++stop.toVector).forall(_.isFinite),"LoRA step bounds must be finite")
  def active(step: Int,total: Int): Boolean =
    require(total>0 && step>=1 && step<=total,"invalid LoRA step position")
    def boundary(value: Double): Double =
      if value>0 && value<1 then math.rint(value*total) else if value<0 then math.ceil(value) else math.floor(value)
    start.forall(value=>step.toDouble>=boundary(value)) && stop.forall(value=>step.toDouble<boundary(value))

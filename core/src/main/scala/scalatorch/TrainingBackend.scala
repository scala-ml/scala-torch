package scalatorch

/** Explicit first-order pullbacks. Inputs are borrowed; each result is independently owned. */
trait TrainingBackend[T] extends FloatBackend[T]:
  def siluBackward(value: T,gradient: T): T
  def rmsNormInputBackward(value: T,weight: T,gradient: T,epsilon: Double = 1e-6): T
  def rmsNormWeightBackward(value: T,gradient: T,epsilon: Double = 1e-6): T

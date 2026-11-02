package scalatorch

/** Explicit first-order pullbacks. Inputs are borrowed; each result is independently owned. */
trait TrainingBackend[T] extends FloatBackend[T]:
  def embeddingBackward(tableShape: Shape,indices: Tensor[Long],gradient: T): T
  def ropeBackward(gradient: T,positionStart: Int = 0,theta: Double = 10000.0): T
  def siluBackward(value: T,gradient: T): T
  def rmsNormInputBackward(value: T,weight: T,gradient: T,epsilon: Double = 1e-6): T
  def rmsNormWeightBackward(value: T,gradient: T,epsilon: Double = 1e-6): T

package scalatorch

/** Explicit first-order pullbacks. Inputs are borrowed; each result is independently owned. */
trait TrainingBackend[T] extends FloatBackend[T]:
  /** Validation may transfer a small status flag, never the tensor contents. */
  def allFinite(value: T): Boolean
  def crossEntropy(logits: T,targets: Tensor[Long],reduction: LossReduction = LossReduction.Mean,ignoreIndex: Long = -100L): T
  def crossEntropyBackward(logits: T,targets: Tensor[Long],gradient: T,
      reduction: LossReduction = LossReduction.Mean,ignoreIndex: Long = -100L): T
  /** Caller owns all three independently releasable outputs (query, key, value). */
  def causalAttentionBackward(query: T,key: T,value: T,gradient: T,queryStart: Int = 0,
      maxScoreElements: Long = 16777216L): (T,T,T)
  def embeddingBackward(tableShape: Shape,indices: Tensor[Long],gradient: T): T
  def ropeBackward(gradient: T,positionStart: Int = 0,theta: Double = 10000.0): T
  def siluBackward(value: T,gradient: T): T
  def rmsNormInputBackward(value: T,weight: T,gradient: T,epsilon: Double = 1e-6): T
  def rmsNormWeightBackward(value: T,gradient: T,epsilon: Double = 1e-6): T

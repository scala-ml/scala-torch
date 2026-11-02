package scalatorch.nn

import scalatorch.*
import scalatorch.autograd.Variable

final case class RMSNorm(weight: Parameter, epsilon: Double = 1e-6) derives Module:
  require(weight.data.shape.rank == 1 && weight.data.shape.numel > 0, "RMSNorm weight must be a nonempty vector")
  require(epsilon.isFinite && epsilon > 0, "RMSNorm epsilon must be finite and positive")
  def apply(input: Variable): Variable = input.rmsNorm(weight.value, epsilon)

object RMSNorm:
  def init(width: Int, epsilon: Double = 1e-6): RMSNorm =
    require(width > 0, "RMSNorm width must be positive")
    RMSNorm(Parameter(Tensor.fromFloats(Array.fill(width)(1f), Shape.from(Seq(width.toLong)))), epsilon)

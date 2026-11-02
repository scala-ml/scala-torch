package scalatorch.nn

import scalatorch.*
import scalatorch.autograd.Variable

final case class Embedding(weight: Parameter) derives Module:
  require(weight.data.shape.rank == 2 && weight.data.shape.dims(1) > 0, "Embedding weight must have shape (vocabulary, width)")
  def apply(indices: Tensor[Long]): Variable = weight.value.embedding(indices)

object Embedding:
  def init(vocabulary: Int, width: Int, seed: Long): Embedding =
    require(vocabulary > 0 && width > 0 && vocabulary.toLong * width <= Int.MaxValue, "invalid Embedding dimensions")
    val random = new java.util.Random(seed)
    val values = Array.fill(vocabulary * width)(random.nextGaussian().toFloat)
    Embedding(Parameter(Tensor.fromFloats(values, Shape.from(Seq(vocabulary.toLong, width.toLong)))))

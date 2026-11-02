package scalatorch.nn

import scalatorch.*
import scalatorch.autograd.Variable

/** Additive update x Aᵀ Bᵀ * alpha/rank. No dropout or merged weight storage. */
final case class LowRankAdapter(a: Parameter, b: Parameter, alpha: Double) derives Module:
  require(a.data.shape.rank == 2 && b.data.shape.rank == 2, "adapter factors must be matrices")
  require(a.data.shape.dims.forall(_ > 0) && b.data.shape.dims.forall(_ > 0) &&
    a.data.shape.dims(0) == b.data.shape.dims(1), "invalid adapter factor dimensions")
  private val scale = LowRankAdapter.scale(alpha, a.data.shape.dims(0))
  def apply(input: Variable): Variable =
    input.matmul(a.value.transpose(0, 1)).matmul(b.value.transpose(0, 1)) * scale

object LowRankAdapter:
  private def scale(alpha: Double, rank: Long): Float =
    val value = (alpha / rank).toFloat
    require(alpha.isFinite && alpha > 0 && value.isFinite && value > 0, "invalid adapter scale")
    value

  def init(in: Int, out: Int, rank: Int, alpha: Double, seed: Long): LowRankAdapter =
    require(in > 0 && out > 0 && rank > 0 && in.toLong * rank <= Int.MaxValue &&
      out.toLong * rank <= Int.MaxValue, "invalid adapter dimensions")
    scale(alpha, rank)
    val random = new java.util.Random(seed)
    val bound = 1.0 / math.sqrt(in.toDouble)
    val a = Tensor.fromFloats(Array.fill(in * rank)(((random.nextDouble() * 2 - 1) * bound).toFloat),
      Shape.from(Seq(rank.toLong, in.toLong)))
    val b = Tensor.fromFloats(new Array[Float](out * rank), Shape.from(Seq(out.toLong, rank.toLong)))
    LowRankAdapter(Parameter(a), Parameter(b), alpha)

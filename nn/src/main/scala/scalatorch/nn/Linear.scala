package scalatorch.nn

import scalatorch.*
import scalatorch.autograd.Variable

final case class Linear(weight: Parameter, bias: Option[Parameter], lora: Option[LowRankAdapter] = None) derives Module:
  require(weight.data.shape.rank == 2, "Linear weight must have shape (out, in)")
  bias.foreach(p => require(p.data.shape == Shape.from(Seq(weight.data.shape.dims.head)), "Linear bias shape mismatch"))
  lora.foreach(a => require(a.a.data.shape.dims(1) == weight.data.shape.dims(1) &&
    a.b.data.shape.dims(0) == weight.data.shape.dims(0), "Linear adapter shape mismatch"))
  def apply(input: Variable): Variable =
    val base = input.matmul(weight.value.transpose(0, 1))
    val result = lora.fold(base)(a => base + a(input))
    bias.fold(result)(p => result + p.value)

  /** Freeze independent parameter handles; immutable weight storage may be shared. */
  def withLoRA(rank: Int, alpha: Double, seed: Long): Linear =
    require(lora.isEmpty, "Linear already has an adapter")
    val adapter = LowRankAdapter.init(weight.data.shape.dims(1).toInt, weight.data.shape.dims(0).toInt, rank, alpha, seed)
    Linear(Parameter(weight.data, false), bias.map(p => Parameter(p.data, false)), Some(adapter))

object Linear:
  def init(in: Int, out: Int, seed: Long, bias: Boolean = true): Linear =
    require(in > 0 && out > 0 && in.toLong * out <= Int.MaxValue, "invalid Linear dimensions")
    val random = new java.util.Random(seed)
    val bound = 1.0 / math.sqrt(in.toDouble)
    def sample(n: Int) = Array.fill(n)(((random.nextDouble() * 2 - 1) * bound).toFloat)
    val weight = Parameter(Tensor.fromFloats(sample(in * out), Shape.from(Seq(out.toLong, in.toLong))))
    val offset = if bias then Some(Parameter(Tensor.fromFloats(sample(out), Shape.from(Seq(out.toLong))))) else None
    Linear(weight, offset)

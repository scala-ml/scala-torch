package scalatorch.nn

import scalatorch.*
import scalatorch.autograd.Variable

/** Final-axis LayerNorm for token features, with optional per-feature scale and bias. */
final case class LayerNorm(width: Int,weight: Option[Parameter],bias: Option[Parameter],
    epsilon: Double = 1e-5) derives Module:
  require(width>0 && epsilon.isFinite && epsilon>0,"invalid LayerNorm width/epsilon")
  Seq(weight,bias).flatten.foreach(p =>
    require(p.data.shape==Shape.from(Seq(width.toLong)),"LayerNorm affine shape mismatch"))
  def apply(input: Variable): Variable =
    require(input.data.shape.rank>=1 && input.data.shape.dims.last==width,"LayerNorm input width mismatch")
    val normalized=input.layerNorm(epsilon)
    val scaled=weight.fold(normalized)(p => normalized*p.value)
    bias.fold(scaled)(p => scaled+p.value)

object LayerNorm:
  def init(width: Int,epsilon: Double = 1e-5,affine: Boolean = true): LayerNorm =
    require(width>0 && epsilon.isFinite && epsilon>0,"invalid LayerNorm width/epsilon")
    def parameter(value: Float) =
      if affine then Some(Parameter(Tensor.fromFloats(Array.fill(width)(value),Shape.from(Seq(width.toLong)))))
      else None
    LayerNorm(width,parameter(1f),parameter(0f),epsilon)

package scalatorch.nn

import scalatorch.*
import scalatorch.autograd.Variable

/** Group normalization with optional independent per-channel scale and bias. */
final case class GroupNorm(groups: Int,channels: Int,weight: Option[Parameter],bias: Option[Parameter],
    epsilon: Double = 1e-5) derives Module:
  require(groups>0 && channels>0 && channels%groups==0,"invalid GroupNorm channels/groups")
  require(epsilon.isFinite && epsilon>0,"GroupNorm epsilon must be finite and positive")
  Seq(weight,bias).flatten.foreach(p =>
    require(p.data.shape==Shape.from(Seq(channels.toLong)),"GroupNorm affine shape mismatch"))
  def apply(input: Variable): Variable =
    require(input.data.shape.rank>=2 && input.data.shape.dims(1)==channels,"GroupNorm input channel mismatch")
    val normalized=input.groupNorm(groups,epsilon)
    val affineShape=Shape.from(Seq(1L,channels.toLong)++Seq.fill(input.data.shape.rank-2)(1L))
    val scaled=weight.fold(normalized)(p => normalized*p.value.reshape(affineShape))
    bias.fold(scaled)(p => scaled+p.value.reshape(affineShape))

object GroupNorm:
  def init(channels: Int,groups: Int,epsilon: Double = 1e-5,affine: Boolean = true): GroupNorm =
    require(groups>0 && channels>0 && channels%groups==0,"invalid GroupNorm channels/groups")
    require(epsilon.isFinite && epsilon>0,"GroupNorm epsilon must be finite and positive")
    def parameter(value: Float) =
      if affine then Some(Parameter(Tensor.fromFloats(Array.fill(channels)(value),Shape.from(Seq(channels.toLong)))))
      else None
    GroupNorm(groups,channels,parameter(1f),parameter(0f),epsilon)

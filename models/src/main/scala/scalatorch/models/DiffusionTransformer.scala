package scalatorch.models

import scalatorch.*
import scalatorch.autograd.Variable
import scalatorch.nn.*

/** Zero-dropout GEGLU projection, using Gaussian-error GELU on the gate half. */
final case class DiffusionFeedForward(projection: Linear,output: Linear) derives Module:
  val inputWidth: Long=projection.weight.data.shape.dims(1)
  private val projected=projection.weight.data.shape.dims.head
  private val hidden=projected/2
  val outputWidth: Long=output.weight.data.shape.dims.head
  require(inputWidth>0 && projected>0 && projected%2==0 && outputWidth>0 &&
    output.weight.data.shape.dims(1)==hidden,"incompatible GEGLU projection widths")
  def apply(input: Variable): Variable =
    require(input.data.shape.rank>=1 && input.data.shape.dims.last==inputWidth,"GEGLU input width mismatch")
    val rows=input.data.shape.numel/inputWidth
    val packed=projection(input.reshape(Shape.from(Seq(rows,inputWidth))))
    val gated=packed.narrow(1,0,hidden)*packed.narrow(1,hidden,hidden).gelu()
    output(gated).reshape(Shape.from(input.data.shape.dims.updated(input.data.shape.rank-1,outputWidth)))

/** Pre-norm token Transformer with self-attention, cross-attention and GEGLU residuals. */
final case class DiffusionTransformerBlock(norm1: LayerNorm,attn1: DiffusionAttention,
    norm2: LayerNorm,attn2: DiffusionAttention,norm3: LayerNorm,feedForward: DiffusionFeedForward) derives Module:
  private val width=norm1.width.toLong
  require(norm2.width==width && norm3.width==width && feedForward.inputWidth==width &&
    feedForward.outputWidth==width,"Transformer norm/feed-forward width mismatch")
  require(attn1.toQ.weight.data.shape.dims(1)==width && attn1.toK.weight.data.shape.dims(1)==width &&
    attn2.toQ.weight.data.shape.dims(1)==width,"Transformer attention width mismatch")
  def apply(input: Variable,context: Variable): Variable =
    require(input.data.shape.rank==3 && input.data.shape.dims(2)==width,"Transformer input must be matching token features")
    val self=input+attn1(norm1(input))
    val cross=self+attn2(norm2(self),Some(context))
    cross+feedForward(norm3(cross))

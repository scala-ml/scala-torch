package scalatorch.models

import scalatorch.*
import scalatorch.autograd.Variable
import scalatorch.nn.*

/** Group-normalized image Transformer with 1x1 projections and an image-space residual. */
final case class DiffusionSpatialTransformer(norm: GroupNorm,projIn: Conv2d,
    blocks: Vector[DiffusionTransformerBlock],projOut: Conv2d) derives Module:
  private val channels=norm.channels.toLong
  private val inner=projIn.weight.data.shape.dims.head
  require(projIn.weight.data.shape==Shape.from(Seq(inner,channels,1L,1L)) &&
    projIn.config==Conv2dConfig(),"spatial input projection must be a matching 1x1 convolution")
  require(projOut.weight.data.shape==Shape.from(Seq(channels,inner,1L,1L)) &&
    projOut.config==Conv2dConfig(),"spatial output projection must restore image channels")
  require(blocks.nonEmpty && blocks.forall(_.norm1.width==inner),"spatial Transformer stack width mismatch")
  private val contextWidth=blocks.head.attn2.toK.weight.data.shape.dims(1)
  require(blocks.forall(_.attn2.toK.weight.data.shape.dims(1)==contextWidth),"spatial context widths differ")

  def apply(input: Variable,context: Variable): Variable =
    val shape=input.data.shape
    require(shape.rank==4 && shape.dims(1)==channels,"spatial Transformer requires matching NCHW input")
    require(context.data.shape.rank==3 && context.data.shape.dims(0)==shape.dims(0) &&
      context.data.shape.dims(2)==contextWidth,"spatial Transformer context batch/width mismatch")
    val projected=projIn(norm(input))
    val batch=shape.dims(0);val height=shape.dims(2);val width=shape.dims(3)
    val tokens=projected.reshape(Shape.from(Seq(batch,inner,height*width))).transpose(1,2)
    val transformed=TokenDownsampling.withSpatialShape(height.toInt,width.toInt) { HyperTile.inSpatial { blocks.foldLeft(tokens)((value,block) => block(value,context)) } }
    val image=transformed.transpose(1,2).reshape(Shape.from(Seq(batch,inner,height,width)))
    input+projOut(image)

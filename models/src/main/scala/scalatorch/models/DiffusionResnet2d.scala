package scalatorch.models

import scalatorch.*
import scalatorch.autograd.Variable
import scalatorch.nn.*

/** Shape-preserving diffusion residual block with additive timestep conditioning.
  * SiLU preactivation, zero dropout, and optional channel-changing shortcut.
  */
final case class DiffusionResnet2d(norm1: GroupNorm,conv1: Conv2d,timeEmbProj: Option[Linear],
    norm2: GroupNorm,conv2: Conv2d,convShortcut: Option[Conv2d] = None,outputScale: Double = 1.0) derives Module:
  private val inChannels=norm1.channels
  private val hiddenChannels=norm2.channels
  private val outChannels=conv2.weight.data.shape.dims.head
  require(outputScale.isFinite && outputScale>0 && (1.0/outputScale).toFloat.isFinite &&
    (1.0/outputScale).toFloat>0,"residual scale must have a positive finite Float32 inverse")
  private def checkConv(c: Conv2d,in: Long,out: Long,kernel: Int): Unit =
    require(c.weight.data.shape==Shape.from(Seq(out,in,kernel.toLong,kernel.toLong)) &&
      c.config==Conv2dConfig(padding=(kernel/2,kernel/2)),"incompatible diffusion convolution")
  checkConv(conv1,inChannels,hiddenChannels,3)
  checkConv(conv2,hiddenChannels,outChannels,3)
  convShortcut match
    case Some(c) => checkConv(c,inChannels,outChannels,1)
    case None => require(inChannels==outChannels,"channel-changing residual requires a shortcut")
  timeEmbProj.foreach { projection =>
    require(projection.weight.data.shape.dims.head==hiddenChannels &&
      projection.weight.data.shape.dims(1)>0,"timestep projection width mismatch")
  }

  def apply(input: Variable,timeEmbedding: Option[Variable] = None): Variable =
    require(input.data.shape.rank==4 && input.data.shape.dims(1)==inChannels,"residual input must be NCHW with matching channels")
    require(timeEmbedding.isDefined==timeEmbProj.isDefined,"timestep embedding presence mismatch")
    timeEmbedding.foreach { time =>
      require(time.data.shape==Shape.from(Seq(input.data.shape.dims.head,timeEmbProj.get.weight.data.shape.dims(1))),
        "timestep embedding batch/width mismatch")
    }
    val hidden=conv1(norm1(input).silu())
    val conditioned=timeEmbProj.fold(hidden) { projection =>
      hidden+projection(timeEmbedding.get.silu()).reshape(Shape.from(Seq(input.data.shape.dims.head,hiddenChannels.toLong,1L,1L)))
    }
    val residual=conv2(norm2(conditioned).silu())
    val shortcut=convShortcut.fold(input)(_(input))
    (shortcut+residual)*(1.0/outputScale).toFloat

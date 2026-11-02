package scalatorch.models

import scalatorch.*
import scalatorch.autograd.Variable
import scalatorch.nn.*

/** Group-normalized dense image self-attention with an image residual. */
final case class VaeAttention2d(norm: GroupNorm,attention: DiffusionAttention) derives Module:
  private val channels=norm.channels.toLong
  require(attention.toQ.weight.data.shape==Shape.from(Seq(channels,channels)) &&
    attention.toK.weight.data.shape==Shape.from(Seq(channels,channels)),"VAE attention channel mismatch")
  private[models] def preflight(shape: Shape): Unit =
    require(shape.rank==4 && shape.dims(1)==channels && shape.dims(2)>0 && shape.dims(3)>0,"VAE attention expects matching NCHW input")
    val tokens=Math.multiplyExact(shape.dims(2),shape.dims(3))
    val scores=Shape.from(Seq(shape.dims.head,attention.heads.toLong,tokens,tokens)).numel
    require(scores<=attention.maxScoreElements && scores<=Int.MaxValue,"VAE attention score budget exceeded")
  def apply(input: Variable): Variable =
    val shape=input.data.shape
    preflight(shape)
    val tokens=norm(input).reshape(Shape.from(Seq(shape.dims.head,channels,shape.dims(2)*shape.dims(3)))).transpose(1,2)
    input+attention(tokens).transpose(1,2).reshape(shape)

final case class VaeMidBlock(first: DiffusionResnet2d,attention: Option[VaeAttention2d],second: DiffusionResnet2d) derives Module:
  private val channels=first.norm1.channels.toLong
  require(VaeGraph.residualChannels(first,channels)==channels && VaeGraph.residualChannels(second,channels)==channels,
    "VAE middle residuals must preserve channels")
  attention.foreach(a => require(a.norm.channels==channels,"VAE middle attention channel mismatch"))
  def apply(input: Variable): Variable =
    val hidden=first(input)
    second(attention.fold(hidden)(_(hidden)))

final case class VaeDownBlock(layers: Vector[DiffusionResnet2d],downsample: Option[DiffusionDownsample2d]) derives Module:
  require(layers.nonEmpty,"empty VAE down block")
final case class VaeUpBlock(layers: Vector[DiffusionResnet2d],upsample: Option[DiffusionUpsample2d]) derives Module:
  require(layers.nonEmpty,"empty VAE up block")

/** Raw posterior moments (mean then log variance), before sampling or diffusion scaling. */
final case class VaeEncoder(convIn: Conv2d,downBlocks: Vector[VaeDownBlock],midBlock: VaeMidBlock,
    normOut: GroupNorm,convOut: Conv2d,quantConv: Option[Conv2d]=None,maxFeatureElements: Long=134217728L) derives Module:
  require(maxFeatureElements>=0 && downBlocks.nonEmpty && downBlocks.size<=8,"invalid VAE encoder depth/budget")
  require(downBlocks.zipWithIndex.forall((b,i) => b.downsample.isDefined==(i<downBlocks.size-1)),"VAE downsampling must separate stages")
  VaeGraph.preserving(convIn,3)
  private val outputChannels=convOut.weight.data.shape.dims.head
  require(outputChannels%2==0,"VAE moments must contain equal mean/log-variance channels")
  val latentChannels: Long=outputChannels/2
  private var current=convIn.weight.data.shape.dims.head
  downBlocks.foreach { block =>
    block.layers.foreach(r => current=VaeGraph.residualChannels(r,current))
    block.downsample.foreach { d =>
      require(d.conv.weight.data.shape.dims(1)==current,"VAE downsample channel mismatch")
      current=d.conv.weight.data.shape.dims.head
    }
  }
  require(midBlock.first.norm1.channels==current && normOut.channels==current,"VAE encoder middle/output norm mismatch")
  VaeGraph.preserving(convOut,3,Some(current))
  quantConv.foreach(c => VaeGraph.projection(c,outputChannels))

  def apply(input: Variable): Variable =
    var shape=VaeGraph.input(input,convIn,maxFeatureElements)
    shape=VaeGraph.convShape(convIn,shape,maxFeatureElements)
    downBlocks.foreach { b =>
      b.layers.foreach(r => shape=VaeGraph.residualShape(r,shape,maxFeatureElements))
      b.downsample.foreach(d => shape=VaeGraph.downShape(d,shape,maxFeatureElements))
    }
    shape=VaeGraph.midShape(midBlock,shape,maxFeatureElements)
    shape=VaeGraph.convShape(convOut,shape,maxFeatureElements)
    quantConv.foreach(c => shape=VaeGraph.convShape(c,shape,maxFeatureElements))
    var hidden=convIn(input)
    downBlocks.foreach { b =>
      b.layers.foreach(r => hidden=r(hidden))
      b.downsample.foreach(d => hidden=d(hidden))
    }
    val moments=convOut(normOut(midBlock(hidden)).silu())
    quantConv.fold(moments)(_(moments))

/** Reconstructs raw image values from unscaled latents; no pixel clamp or pipeline scaling. */
final case class VaeDecoder(convIn: Conv2d,midBlock: VaeMidBlock,upBlocks: Vector[VaeUpBlock],
    normOut: GroupNorm,convOut: Conv2d,postQuantConv: Option[Conv2d]=None,maxFeatureElements: Long=134217728L) derives Module:
  require(maxFeatureElements>=0 && upBlocks.nonEmpty && upBlocks.size<=8,"invalid VAE decoder depth/budget")
  require(upBlocks.zipWithIndex.forall((b,i) => b.upsample.isDefined==(i<upBlocks.size-1)),"VAE upsampling must separate stages")
  VaeGraph.preserving(convIn,3)
  val latentChannels: Long=convIn.weight.data.shape.dims(1)
  postQuantConv.foreach(c => VaeGraph.projection(c,latentChannels))
  private var current=convIn.weight.data.shape.dims.head
  require(midBlock.first.norm1.channels==current,"VAE decoder middle channel mismatch")
  upBlocks.foreach { b =>
    b.layers.foreach(r => current=VaeGraph.residualChannels(r,current))
    b.upsample.foreach { u =>
      require(u.conv.weight.data.shape.dims(1)==current,"VAE upsample channel mismatch")
      current=u.conv.weight.data.shape.dims.head
    }
  }
  require(normOut.channels==current,"VAE decoder output norm mismatch")
  VaeGraph.preserving(convOut,3,Some(current))

  def apply(input: Variable): Variable =
    var shape=VaeGraph.input(input,convIn,maxFeatureElements)
    postQuantConv.foreach(c => shape=VaeGraph.convShape(c,shape,maxFeatureElements))
    shape=VaeGraph.convShape(convIn,shape,maxFeatureElements)
    shape=VaeGraph.midShape(midBlock,shape,maxFeatureElements)
    upBlocks.foreach { b =>
      b.layers.foreach(r => shape=VaeGraph.residualShape(r,shape,maxFeatureElements))
      b.upsample.foreach(u => shape=VaeGraph.upShape(u,shape,maxFeatureElements))
    }
    VaeGraph.convShape(convOut,shape,maxFeatureElements)
    var hidden=midBlock(convIn(postQuantConv.fold(input)(_(input))))
    upBlocks.foreach { b =>
      b.layers.foreach(r => hidden=r(hidden))
      b.upsample.foreach { u =>
        val s=hidden.data.shape
        hidden=u(hidden,(s.dims(2)*2).toInt,(s.dims(3)*2).toInt)
      }
    }
    convOut(normOut(hidden).silu())

/** Structural checks and allocation preflight shared by encoder/decoder. Limits are per image tensor, not peak memory. */
private[models] object VaeGraph:
  def preserving(c: Conv2d,kernel: Int,in: Option[Long]=None): Unit =
    require(c.weight.data.shape.dims.drop(2)==Vector(kernel.toLong,kernel.toLong) && c.config==Conv2dConfig(padding=(kernel/2,kernel/2)),
      "VAE convolution must preserve spatial shape")
    in.foreach(n => require(c.weight.data.shape.dims(1)==n,"VAE convolution input channel mismatch"))
  def projection(c: Conv2d,channels: Long): Unit =
    preserving(c,1,Some(channels))
    require(c.weight.data.shape.dims.head==channels,"VAE quant projection must preserve channels")
  def residualChannels(r: DiffusionResnet2d,in: Long): Long =
    require(r.timeEmbProj.isEmpty && r.norm1.channels==in,"VAE residual must be unconditioned with matching channels")
    r.conv2.weight.data.shape.dims.head
  private def checked(shape: Shape,limit: Long): Shape =
    require(shape.rank==4 && shape.dims(2)>0 && shape.dims(3)>0 && shape.dims(2)<=Int.MaxValue && shape.dims(3)<=Int.MaxValue,
      "invalid VAE spatial dimensions")
    require(shape.numel<=limit && shape.numel<=Int.MaxValue,"VAE feature allocation budget exceeded")
    shape
  def input(x: Variable,c: Conv2d,limit: Long): Shape =
    val shape=checked(x.data.shape,limit)
    require(shape.dims(1)==c.weight.data.shape.dims(1),"VAE input channel mismatch")
    shape
  def convShape(c: Conv2d,shape: Shape,limit: Long): Shape =
    checked(Shape.from(shape.dims.updated(1,c.weight.data.shape.dims.head)),limit)
  def residualShape(r: DiffusionResnet2d,shape: Shape,limit: Long): Shape =
    convShape(r.conv1,shape,limit)
    convShape(r.conv2,shape,limit)
  def midShape(mid: VaeMidBlock,shape: Shape,limit: Long): Shape =
    val first=residualShape(mid.first,shape,limit)
    mid.attention.foreach(_.preflight(first))
    residualShape(mid.second,first,limit)
  def downShape(d: DiffusionDownsample2d,shape: Shape,limit: Long): Shape =
    val h=shape.dims(2);val w=shape.dims(3)
    val symmetric=d.conv.config.padding==(1,1)
    if !symmetric then checked(Shape.from(shape.dims.updated(2,h+1).updated(3,w+1)),limit)
    val next=Shape.from(shape.dims.updated(1,d.conv.weight.data.shape.dims.head)
      .updated(2,if symmetric then (h+1)/2 else h/2).updated(3,if symmetric then (w+1)/2 else w/2))
    checked(next,limit)
  def upShape(u: DiffusionUpsample2d,shape: Shape,limit: Long): Shape =
    val resized=checked(Shape.from(shape.dims.updated(2,shape.dims(2)*2).updated(3,shape.dims(3)*2)),limit)
    convShape(u.conv,resized,limit)

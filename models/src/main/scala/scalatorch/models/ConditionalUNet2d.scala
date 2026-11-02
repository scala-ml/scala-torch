package scalatorch.models

import scalatorch.*
import scalatorch.autograd.Variable
import scalatorch.nn.*
import scala.collection.mutable.ArrayBuffer

final case class DiffusionUNetLayer(resnet: DiffusionResnet2d,attention: Option[DiffusionSpatialTransformer] = None) derives Module:
  attention.foreach(a => require(a.norm.channels==resnet.conv2.weight.data.shape.dims.head,"U-Net layer attention channel mismatch"))
  def apply(input: Variable,time: Variable,context: Variable): Variable =
    val hidden=resnet(input,Some(time))
    attention.fold(hidden)(_(hidden,context))

final case class DiffusionDownBlock(layers: Vector[DiffusionUNetLayer],downsample: Option[DiffusionDownsample2d]) derives Module:
  require(layers.nonEmpty,"U-Net down block must contain residual layers")
final case class DiffusionUpBlock(layers: Vector[DiffusionUNetLayer],upsample: Option[DiffusionUpsample2d]) derives Module:
  require(layers.nonEmpty,"U-Net up block must contain residual layers")
final case class DiffusionMidBlock(first: DiffusionResnet2d,attention: DiffusionSpatialTransformer,
    second: DiffusionResnet2d) derives Module:
  def apply(input: Variable,time: Variable,context: Variable): Variable =
    second(attention(first(input,Some(time)),context),Some(time))

final case class DiffusionTimeSettings(flipSinToCos: Boolean = true,frequencyShift: Double = 0.0,
    phaseScale: Double = 1.0,maxPeriod: Double = 10000.0,maxElements: Long = 16777216L):
  require(frequencyShift.isFinite && phaseScale.isFinite && maxPeriod.isFinite && maxPeriod>=1 && maxElements>=0,
    "invalid U-Net timestep settings")
object DiffusionTimeSettings:
  given Module[DiffusionTimeSettings] with
    def namedParameters(value: DiffusionTimeSettings) = Vector.empty

/** CPU conditional image U-Net with additive time conditioning and exact LIFO skip accounting. */
final case class ConditionalUNet2d(convIn: Conv2d,timeEmbedding: DiffusionTimeEmbedding,
    downBlocks: Vector[DiffusionDownBlock],midBlock: DiffusionMidBlock,upBlocks: Vector[DiffusionUpBlock],
    normOut: GroupNorm,convOut: Conv2d,timeSettings: DiffusionTimeSettings = DiffusionTimeSettings()) derives Module:
  private val inputChannels=convIn.weight.data.shape.dims(1)
  private val timeWidth=timeEmbedding.linear2.weight.data.shape.dims.head
  private val featureWidth=timeEmbedding.linear1.weight.data.shape.dims(1).toInt
  private val contextWidth=midBlock.attention.blocks.head.attn2.toK.weight.data.shape.dims(1)
  require(featureWidth>=2 && featureWidth/2-timeSettings.frequencyShift>0,"invalid U-Net timestep frequency denominator")
  private def preserving(c: Conv2d): Unit =
    require(c.weight.data.shape.dims.drop(2)==Vector(3L,3L) && c.config==Conv2dConfig(padding=(1,1)),
      "U-Net input/output convolutions must preserve spatial shape")
  preserving(convIn);preserving(convOut)
  require(downBlocks.nonEmpty && downBlocks.size==upBlocks.size,"U-Net resolution counts differ")
  require(downBlocks.zipWithIndex.forall((b,i) => b.downsample.isDefined==(i<downBlocks.size-1)) &&
    upBlocks.zipWithIndex.forall((b,i) => b.upsample.isDefined==(i<upBlocks.size-1)),"U-Net resampling must separate resolution stages")

  private def validate(): Unit =
    def attention(a: DiffusionSpatialTransformer,channels: Long): Unit =
      require(a.norm.channels==channels && a.blocks.forall(_.attn2.toK.weight.data.shape.dims(1)==contextWidth),
        "U-Net attention channel/context mismatch")
    def residual(r: DiffusionResnet2d,channels: Long): Long =
      require(r.norm1.channels==channels,"U-Net residual input channel mismatch")
      require(r.timeEmbProj.exists(_.weight.data.shape.dims(1)==timeWidth),"U-Net residual timestep width mismatch")
      r.conv2.weight.data.shape.dims.head
    def layer(l: DiffusionUNetLayer,channels: Long): Long =
      val out=residual(l.resnet,channels)
      l.attention.foreach(attention(_,out))
      out
    def sampling(c: Conv2d,channels: Long): Long =
      require(c.weight.data.shape.dims(1)==channels,"U-Net resampling channel mismatch")
      c.weight.data.shape.dims.head
    var channels=convIn.weight.data.shape.dims.head
    var level=0
    val skips=ArrayBuffer((channels,level))
    downBlocks.foreach { block =>
      block.layers.foreach { l => channels=layer(l,channels);skips+=((channels,level)) }
      block.downsample.foreach { d => channels=sampling(d.conv,channels);level+=1;skips+=((channels,level)) }
    }
    channels=residual(midBlock.first,channels)
    attention(midBlock.attention,channels)
    channels=residual(midBlock.second,channels)
    upBlocks.foreach { block =>
      block.layers.foreach { l =>
        require(skips.nonEmpty,"U-Net consumes more skips than it produces")
        val (skipChannels,skipLevel)=skips.remove(skips.size-1)
        require(skipLevel==level,"U-Net skip resolution mismatch")
        channels=layer(l,channels+skipChannels)
      }
      block.upsample.foreach { u =>
        require(skips.nonEmpty && skips.last._2==level-1,"U-Net upsampling target resolution mismatch")
        level-=1
        channels=sampling(u.conv,channels)
      }
    }
    require(skips.isEmpty,"U-Net leaves unused skip tensors")
    require(normOut.channels==channels && convOut.weight.data.shape.dims(1)==channels,"U-Net output channel mismatch")
  validate()

  def apply(sample: Variable,timesteps: Tensor[Float],context: Variable): Variable =
    val shape=sample.data.shape
    require(shape.rank==4 && shape.dims(1)==inputChannels,"U-Net sample must have matching NCHW channels")
    val batch=shape.dims.head
    require(context.data.shape.rank==3 && context.data.shape.dims.head==batch &&
      context.data.shape.dims(2)==contextWidth && context.data.shape.dims(1)>0,"U-Net context batch/width mismatch")
    require(timesteps.shape.rank==1 && (timesteps.shape.dims.head==1 || timesteps.shape.dims.head==batch),
      "U-Net requires one timestep or one per sample")
    val times=if timesteps.shape.dims.head==batch then timesteps else timesteps.broadcastTo(Shape.from(Seq(batch)))
    val features=DiffusionTimesteps.sinusoidal(times,featureWidth,timeSettings.flipSinToCos,timeSettings.frequencyShift,
      timeSettings.phaseScale,timeSettings.maxPeriod,timeSettings.maxElements)
    val time=timeEmbedding(Variable(features,requiresGrad=false))
    var hidden=convIn(sample)
    val skips=ArrayBuffer(hidden)
    downBlocks.foreach { block =>
      block.layers.foreach { layer => hidden=layer(hidden,time,context);skips+=hidden }
      block.downsample.foreach { down => hidden=down(hidden);skips+=hidden }
    }
    hidden=midBlock(hidden,time,context)
    upBlocks.foreach { block =>
      block.layers.foreach { layer =>
        val skip=skips.remove(skips.size-1)
        hidden=layer(Variable.cat(Seq(hidden,skip),1),time,context)
      }
      block.upsample.foreach { up =>
        val target=skips.last.data.shape
        hidden=up(hidden,target.dims(2).toInt,target.dims(3).toInt)
      }
    }
    convOut(normOut(hidden).silu())

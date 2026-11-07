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

/** Ordered ControlNet additions matching the U-Net input/downsample/resnet skips and middle output. */
final case class ControlNetResiduals(down: Vector[Tensor[Float]],middle: Tensor[Float]):
  require(down.nonEmpty,"ControlNet must provide at least the input residual")

/** Control residuals computed for the positive and optional negative CFG contexts. */
final case class ControlNetGuidanceResiduals(positive: ControlNetResiduals,negative: Option[ControlNetResiduals])

/** Device-resident ControlNet residuals; the owner closes every tensor. */
final case class CudaControlNetResiduals(down: Vector[scalatorch.ffm.CudaTensor],middle: scalatorch.ffm.CudaTensor)
    extends AutoCloseable:
  def close(): Unit =
    var failure: Throwable=null
    (down :+ middle).reverse.foreach { tensor =>
      try tensor.close()
      catch case error: Throwable => if failure==null then failure=error else failure.addSuppressed(error)
    }
    if failure!=null then throw failure

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
    normOut: GroupNorm,convOut: Conv2d,timeSettings: DiffusionTimeSettings = DiffusionTimeSettings(),
    additionEmbedding: Option[DiffusionTimeEmbedding]=None,additionTimeWidth: Int=0) derives Module:
  private val inputChannels=convIn.weight.data.shape.dims(1)
  private val timeWidth=timeEmbedding.linear2.weight.data.shape.dims.head
  require(if additionEmbedding.isDefined then additionTimeWidth>=2 else additionTimeWidth==0,"incomplete U-Net additional embedding")
  additionEmbedding.foreach(a => require(a.linear2.weight.data.shape.dims.head==timeWidth,"U-Net addition output width mismatch"))
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
    downBlocks.zipWithIndex.foreach { (block,blockIdx) => Pag.withSection(s"down.$blockIdx") {
      block.layers.foreach { l => channels=layer(l,channels);skips+=((channels,level)) }
      block.downsample.foreach { d => channels=sampling(d.conv,channels);level+=1;skips+=((channels,level)) }
    }
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

  private[models] def timestepFeatures(shape: Shape,timesteps: Tensor[Float],context: Shape): Tensor[Float] =
    require(shape.rank==4 && shape.dims(1)==inputChannels,"U-Net sample must have matching NCHW channels")
    val batch=shape.dims.head
    require(context.rank==3 && context.dims.head==batch &&
      context.dims(2)==contextWidth && context.dims(1)>0,"U-Net context batch/width mismatch")
    require(timesteps.shape.rank==1 && (timesteps.shape.dims.head==1 || timesteps.shape.dims.head==batch),
      "U-Net requires one timestep or one per sample")
    val times=if timesteps.shape.dims.head==batch then timesteps else timesteps.broadcastTo(Shape.from(Seq(batch)))
    DiffusionTimesteps.sinusoidal(times,featureWidth,timeSettings.flipSinToCos,timeSettings.frequencyShift,
      timeSettings.phaseScale,timeSettings.maxPeriod,timeSettings.maxElements)

  private[models] def addedFeatures(batch: Long,added: Option[DiffusionAddedConditioning]): Option[Tensor[Float]] =
    require(added.isDefined==additionEmbedding.isDefined,"U-Net additional conditioning presence mismatch")
    added.map(_.features(batch,additionTimeWidth,additionEmbedding.get.linear1.weight.data.shape.dims(1),timeSettings.maxElements))

  def apply(sample: Variable,timesteps: Tensor[Float],context: Variable,
      added: Option[DiffusionAddedConditioning]=None,control: Option[ControlNetResiduals]=None): Variable =
    val features=timestepFeatures(sample.data.shape,timesteps,context.data.shape)
    val extra=addedFeatures(sample.data.shape.dims.head,added)
    val baseTime=timeEmbedding(Variable(features,requiresGrad=false))
    val time=extra.fold(baseTime)(x => baseTime+additionEmbedding.get(Variable(x,requiresGrad=false)))
    var hidden=FooocusInpaint.current.fold(convIn(sample))(fi => Variable(FooocusInpaint.applyConvInResidual(sample.data,convIn(sample).data,fi),requiresGrad=false))
    val skips=ArrayBuffer(hidden)
    val kohya=KohyaHrFix.current
    val kohyaLoc=kohya.flatMap { case (cfg,_) => KohyaHrFix.blockLocation(cfg.blockNumber,downBlocks.map(_.layers.size)) }
    downBlocks.zipWithIndex.foreach { (block,blockIdx) => Pag.withSection(s"down.$blockIdx") {
      block.layers.zipWithIndex.foreach { (layer,layerIdx) =>
        hidden=layer(hidden,time,context)
        if kohyaLoc.contains((blockIdx,layerIdx)) then
          val (cfg,t)=kohya.get
          if !cfg.downscaleAfterSkip then hidden=Variable(KohyaHrFix.shrinkIfActive(hidden.data,cfg,t),requiresGrad=false)
          skips+=hidden
          if cfg.downscaleAfterSkip then hidden=Variable(KohyaHrFix.shrinkIfActive(hidden.data,cfg,t),requiresGrad=false)
        else skips+=hidden
      }
      block.downsample.foreach { down =>
        hidden=down(hidden)
        if kohyaLoc.contains((blockIdx,block.layers.size)) then
          val (cfg,t)=kohya.get
          if !cfg.downscaleAfterSkip then hidden=Variable(KohyaHrFix.shrinkIfActive(hidden.data,cfg,t),requiresGrad=false)
          skips+=hidden
          if cfg.downscaleAfterSkip then hidden=Variable(KohyaHrFix.shrinkIfActive(hidden.data,cfg,t),requiresGrad=false)
        else skips+=hidden
      }
    }
    }
    control.foreach { residuals =>
      require(residuals.down.size==skips.size,
        s"ControlNet down residual count mismatch: expected ${skips.size}, got ${residuals.down.size}")
      residuals.down.indices.foreach { index =>
        val base=skips(index)
        val residual=residuals.down(index)
        require(residual.device==Device.Cpu && residual.dtype==DType.Float32 && residual.shape==base.data.shape,
          s"ControlNet down residual shape/device mismatch at index $index")
        skips(index)=base+Variable(residual,requiresGrad=false)
      }
    }
    hidden=Pag.withSection("mid")(midBlock(hidden,time,context))
    control.foreach { residuals =>
      require(residuals.middle.device==Device.Cpu && residuals.middle.dtype==DType.Float32 &&
        residuals.middle.shape==hidden.data.shape,"ControlNet middle residual shape/device mismatch")
      hidden=hidden+Variable(residuals.middle,requiresGrad=false)
    }
    val freeU=FreeU.current
    upBlocks.zipWithIndex.foreach { (block,stageIndex) => Pag.withSection(s"up.$stageIndex") {
      kohya.foreach { case (cfg,_) =>
        val startIdx=skips.size-block.layers.size
        val (_,maxH,maxW)=(startIdx until skips.size).map { k =>
          val d=skips(k).data.shape.dims
          (d(2)*d(3),d(2).toInt,d(3).toInt)
        }.maxBy(_._1)
        for k <- startIdx until skips.size do
          val d=skips(k).data.shape.dims
          if d(2).toInt!=maxH || d(3).toInt!=maxW then
            skips(k)=Variable(KohyaHrFix.interpolate2d(skips(k).data,maxH,maxW,cfg.upscaleMethod),requiresGrad=false)
        val hd=hidden.data.shape.dims
        if hd(2).toInt!=maxH || hd(3).toInt!=maxW then
          hidden=Variable(KohyaHrFix.interpolate2d(hidden.data,maxH,maxW,cfg.upscaleMethod),requiresGrad=false)
      }
      block.layers.foreach { layer =>
        val skip=skips.remove(skips.size-1)
        val combined=freeU match
          case Some(cfg) if (stageIndex==0 || stageIndex==1) && !hidden.requiresGrad && !skip.requiresGrad =>
            val (scaledHidden,filteredSkip)=FreeU(cfg,stageIndex,hidden.data,skip.data)
            Variable.cat(Seq(Variable(scaledHidden,false),Variable(filteredSkip,false)),1)
          case _ => Variable.cat(Seq(hidden,skip),1)
        hidden=layer(combined,time,context)
      }
      block.upsample.foreach { up =>
        val target=skips.last.data.shape
        hidden=up(hidden,target.dims(2).toInt,target.dims(3).toInt)
      }
    } }
    val baseOut=convOut(normOut(hidden).silu())
    Llul.current.filter(ctx => ctx.isActive && ctx.cfg.applyTo.contains("out") && !hidden.requiresGrad) match
      case Some(ctx) =>
        Llul.cutRegion(hidden.data,ctx.cfg,ctx.fx,ctx.fy) match
          case Some(part) =>
            val extra=convOut(normOut(Variable(part,requiresGrad=false)).silu()).data
            Variable(Llul.mixRegion(baseOut.data,extra,ctx.cfg,ctx.fx,ctx.fy),requiresGrad=false)
          case None => baseOut
      case None => baseOut

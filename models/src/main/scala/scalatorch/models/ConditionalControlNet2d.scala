package scalatorch.models

import scalatorch.*
import scalatorch.autograd.Variable
import scalatorch.nn.*

/** The convolutional image-to-latent stem used by standard Diffusers ControlNet. */
final case class ControlNetConditioningEmbedding(convIn: Conv2d,blocks: Vector[Conv2d],convOut: Conv2d) derives Module:
  def apply(image: Variable): Variable =
    var hidden=convIn(image).silu()
    blocks.foreach(conv=>hidden=conv(hidden).silu())
    convOut(hidden)

/** CPU Float32 ControlNet that emits additions in the base U-Net's exact skip order. */
final case class ConditionalControlNet2d(config: ControlNetConfig,convIn: Conv2d,
    timeEmbedding: DiffusionTimeEmbedding,downBlocks: Vector[DiffusionDownBlock],midBlock: DiffusionMidBlock,
    conditionEmbedding: ControlNetConditioningEmbedding,downProjections: Vector[Conv2d],
    middleProjection: Conv2d,maxElements: Long=134217728L,
    additionEmbedding: Option[DiffusionTimeEmbedding]=None) derives Module:
  require(maxElements>=0,"negative ControlNet attention budget")
  require(additionEmbedding.isDefined==config.unet.additionInputWidth.isDefined,"ControlNet additional embedding mismatch")
  require(downBlocks.size==config.unet.channels.size && downProjections.nonEmpty,
    "ControlNet block count does not match its configuration")
  require(downProjections.size==1+downBlocks.map(_.layers.size).sum+downBlocks.dropRight(1).size,
    "ControlNet projection count does not match its down path")
  private val c=config.unet
  private val featureWidth=c.channels.head
  private val timeWidth=c.timeWidth
  private val contextWidth=c.contextWidth
  require(convIn.weight.data.shape==Shape.from(Seq(c.channels.head.toLong,c.inChannels.toLong,3,3)),
    "ControlNet input convolution shape mismatch")
  require(conditionEmbedding.convOut.weight.data.shape.dims.head==featureWidth,
    "ControlNet conditioning stem output width mismatch")

  private def checkImage(sample: Tensor[Float],context: Tensor[Float],image: Tensor[Float]): Unit =
    require(sample.device==Device.Cpu && sample.dtype==DType.Float32 && sample.shape.rank==4 && sample.shape.dims(1)==c.inChannels,
      "ControlNet sample must be CPU Float32 NCHW with four channels")
    require(context.device==Device.Cpu && context.dtype==DType.Float32 && context.shape.rank==3 &&
      context.shape.dims.head==sample.shape.dims.head && context.shape.dims(2)==contextWidth,
      "ControlNet text context batch/width mismatch")
    val height=sample.shape.dims(2);val width=sample.shape.dims(3)
    require(height>0 && width>0 && height<=Int.MaxValue/8 && width<=Int.MaxValue/8,
      "ControlNet latent dimensions exceed bounds")
    require(image.device==Device.Cpu && image.dtype==DType.Float32 &&
      image.shape==Shape.from(Seq(sample.shape.dims.head,config.conditioningChannels,height*8,width*8)),
      "ControlNet image must be CPU Float32 RGB at eight times the latent dimensions")
    require(image.toFloatArray.forall(_.isFinite),"ControlNet image contains nonfinite values")

  private[models] def orderedImage(image: Tensor[Float]): Tensor[Float] =
    if config.channelOrder=="rgb" then image
    else
      val dims=image.shape.dims
      val plane=Math.multiplyExact(dims(2),dims(3)).toInt
      val values=image.toFloatArray
      val reordered=new Array[Float](values.length)
      var batch=0
      while batch<dims.head.toInt do
        val base=batch*3*plane
        System.arraycopy(values,base+2*plane,reordered,base,plane)
        System.arraycopy(values,base+plane,reordered,base+plane,plane)
        System.arraycopy(values,base,reordered,base+2*plane,plane)
        batch+=1
      Tensor.fromFloats(reordered,image.shape)

  def apply(sample: Tensor[Float],timesteps: Tensor[Float],context: Tensor[Float],image: Tensor[Float],
      scales: Vector[Float]=Vector.empty,added: Option[Tensor[Float]]=None): ControlNetResiduals =
    checkImage(sample,context,image)
    require(added.isDefined==additionEmbedding.isDefined,"ControlNet additional conditioning presence mismatch")
    require(scales.isEmpty || scales.size==downProjections.size+1,"ControlNet scale count mismatch")
    require(scales.forall(_.isFinite),"ControlNet scales must be finite")
    require(timesteps.shape.rank==1 && (timesteps.shape.dims.head==1 || timesteps.shape.dims.head==sample.shape.dims.head),
      "ControlNet timestep batch mismatch")
    val times=if timesteps.shape.dims.head==sample.shape.dims.head then timesteps
      else timesteps.broadcastTo(Shape.from(Seq(sample.shape.dims.head)))
    val features=DiffusionTimesteps.sinusoidal(times,featureWidth,c.flipSinToCos,c.frequencyShift,
      phaseScale=1.0,maxPeriod=10000.0,maxElements=maxElements)
    val baseTime=timeEmbedding(Variable(features,requiresGrad=false))
    val time=additionEmbedding.fold(baseTime)(a => baseTime+a(Variable(added.get,requiresGrad=false)))
    val text=Variable(context,requiresGrad=false)
    var hidden=convIn(Variable(sample,requiresGrad=false))+
      conditionEmbedding(Variable(orderedImage(image),requiresGrad=false))
    val activations=scala.collection.mutable.ArrayBuffer.empty[Variable]
    activations+=hidden
    downBlocks.foreach { block =>
      block.layers.foreach { layer => hidden=layer(hidden,time,text);activations+=hidden }
      block.downsample.foreach { down => hidden=down(hidden);activations+=hidden }
    }
    hidden=midBlock(hidden,time,text)
    require(activations.size==downProjections.size,"ControlNet runtime skip count mismatch")
    val residuals=activations.iterator.zip(downProjections.iterator).zipWithIndex.map { case ((activation,projection),index) =>
      val result=projection(activation)
      if scales.isEmpty then result.data else (result*scales(index)).data
    }.toVector
    val middle=middleProjection(hidden)
    ControlNetResiduals(residuals,if scales.isEmpty then middle.data else (middle*scales.last).data)

/** One already-loaded ControlNet and its normalized NUI request settings. */
final case class DiffusionControlNetRequest(model: ConditionalControlNet2d,image: Tensor[Float],weight: Double=1.0,
    guidanceStart: Double=0.0,guidanceEnd: Double=1.0,controlMode: String="Balanced",
    resizeMode: Int=1,sourceCompatibleResize: Boolean=false,preprocessedImage: Option[Tensor[Float]]=None,
    hiresSoftEligible: Boolean=false,hiresSoftInjection: Boolean=false):
  require(weight.isFinite && weight.abs<=Float.MaxValue,"ControlNet weight must be finite Float32")
  require(guidanceStart.isFinite && guidanceEnd.isFinite && guidanceStart>=0 && guidanceEnd<=1 && guidanceStart<=guidanceEnd,
    "ControlNet guidance interval must satisfy 0 <= start <= end <= 1")
  require(Set("Balanced","My prompt is more important","ControlNet is more important")(controlMode),
    "unsupported ControlNet control mode")
  require(resizeMode>=0 && resizeMode<=2,"ControlNet resize mode must be 0, 1 or 2")
  def conditionalOnly: Boolean=controlMode=="ControlNet is more important"
  /** NUI's soft modes decay residuals from shallow to deep blocks by 0.825. */
  def scales(progress: Double,progressEnd: Double = -1.0): Option[Vector[Float]]=
    require(progress.isFinite && progress>=0 && progress<=1,"ControlNet progress must be in [0,1]")
    val end=if progressEnd == -1.0 then progress else progressEnd
    require(end.isFinite && end>=progress && end<=1,"ControlNet progress end must be in [progress,1]")
    if weight==0 || progress<guidanceStart || end>guidanceEnd || (progressEnd == -1.0 && progress>=guidanceEnd) then None
    else
      val count=model.downProjections.size+1
      val soft=controlMode!="Balanced" || hiresSoftInjection
      Some(Vector.tabulate(count) { index =>
        val decay=if soft then math.pow(0.825,12-index) else 1.0
        (weight*decay).toFloat
      })

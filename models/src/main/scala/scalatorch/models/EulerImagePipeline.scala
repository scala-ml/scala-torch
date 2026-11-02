package scalatorch.models

import scalatorch.*
import scalatorch.autograd.{Autograd,Variable}
import java.util.concurrent.CancellationException

final case class DiffusionImageRequest(prompt: String,negativePrompt: String,width: Int,height: Int,seed: Long,
    guidanceScale: Float=7.5f,truncatePrompt: Boolean=false,clipSkip: Int=1,sdxlSize: Option[SdxlImageSize]=None)
/** SDXL original size and crop; target size is the requested output dimensions. */
final case class SdxlImageSize(originalHeight: Int,originalWidth: Int,cropTop: Int=0,cropLeft: Int=0):
  require(originalHeight>0 && originalWidth>0 && cropTop>=0 && cropLeft>=0,"invalid SDXL original size or crop")
  private[models] def timeIds(height: Int,width: Int): Tensor[Float] =
    Tensor.fromFloats(Array(originalHeight.toFloat,originalWidth.toFloat,cropTop.toFloat,cropLeft.toFloat,height.toFloat,width.toFloat),Shape.from(Seq(1L,6L)))
/** Single NCHW image in [-1,1]; optional standard-normal posterior noise for replay. */
final case class DiffusionImageSource(image: Tensor[Float],strength: Double,posteriorNoise: Option[Tensor[Float]]=None)
final case class DiffusionImageProgress(completedSteps: Int,totalSteps: Int,timestep: Float,latents: Tensor[Float])

/** Single-image Euler/DPM++ sampling with explicit U-Net and VAE placement. */
final class EulerImagePipeline(val tokenizer: ClipFastTokenizer,val textEncoder: ClipTextModel,
    val unet: ConditionalUNet2d,val vae: VaeModel,maxImagePixels: Long=16777216L,maxLatentElements: Long=16777216L,
    val vaeExecution: VaeExecution=VaeExecution.Cpu,val unetExecution: UNetExecution=UNetExecution.Cpu,
    val sdxlConditioning: Option[SdxlTextConditioning]=None):
  require(maxImagePixels>=0 && maxLatentElements>=0,"negative image pipeline allocation limit")
  require(tokenizer.vocabSize<=textEncoder.tokenEmbedding.weight.data.shape.dims.head &&
    tokenizer.maxLength<=textEncoder.positionEmbedding.weight.data.shape.dims.head,"pipeline tokenizer/text encoder mismatch")
  require(sdxlConditioning.fold(textEncoder.finalNorm.width)(_.contextWidth)==unet.midBlock.attention.blocks.head.attn2.toK.weight.data.shape.dims(1),
    "pipeline text context width mismatch")
  require(sdxlConditioning.isDefined==unet.additionEmbedding.isDefined,"pipeline SDXL addition/model mismatch")
  sdxlConditioning.foreach { text =>
    unet.addedFeatures(1L,Some(DiffusionAddedConditioning(
      Tensor.fromFloats(new Array[Float](text.pooledWidth),Shape.from(Seq(1L,text.pooledWidth.toLong))),SdxlImageSize(1,1).timeIds(1,1))))
  }
  require(unet.convIn.weight.data.shape.dims(1)==vae.config.latentChannels &&
    unet.convOut.weight.data.shape.dims.head==vae.config.latentChannels,"pipeline latent channel mismatch")
  private val factor=1 << (vae.decoder.upBlocks.size-1)

  /** Noise is unscaled standard-normal data; supplied noise overrides seed. Guidance is enabled above one. */
  def generate(request: DiffusionImageRequest,schedule: EulerSchedule,
      noise: Option[Tensor[Float]]=None,onProgress: DiffusionImageProgress => Unit=_ => (),
      cancelled: () => Boolean=() => false,
      prediction: DiffusionPrediction=DiffusionPrediction.Epsilon,source: Option[DiffusionImageSource]=None): Tensor[Float] =
    generateImpl(request,schedule,noise,onProgress,cancelled,prediction,false,source)

  /** VP DPM++ 2M: unit initial noise and no Euler model-input scaling. History is local to each call. */
  def generateDpm(request: DiffusionImageRequest,schedule: DpmSchedule,
      noise: Option[Tensor[Float]]=None,onProgress: DiffusionImageProgress => Unit=_ => (),
      cancelled: () => Boolean=() => false,
      prediction: DiffusionPrediction=DiffusionPrediction.Epsilon,source: Option[DiffusionImageSource]=None): Tensor[Float] =
    generateImpl(request,schedule.layout,noise,onProgress,cancelled,prediction,true,source)

  private def generateImpl(request: DiffusionImageRequest,fullSchedule: EulerSchedule,
      noise: Option[Tensor[Float]],onProgress: DiffusionImageProgress => Unit,
      cancelled: () => Boolean,prediction: DiffusionPrediction,dpm: Boolean,source: Option[DiffusionImageSource]): Tensor[Float] = Autograd.noGrad {
    def checkCancelled(): Unit =
      if cancelled() || Thread.currentThread().isInterrupted then throw new CancellationException("image generation cancelled")
    checkCancelled()
    require(sdxlConditioning.nonEmpty || request.sdxlSize.isEmpty,"SDXL size conditioning requires an SDXL model")
    require(request.clipSkip>=1 && request.clipSkip<=textEncoder.layers.size+1,"CLIP skip exceeds text encoder layer count")
    require(request.guidanceScale.isFinite && request.guidanceScale>=0,"invalid image guidance scale")
    require(request.width>0 && request.height>0 && request.width%factor==0 && request.height%factor==0,
      "image dimensions must be positive multiples of the VAE scale")
    val pixels=BigInt(request.width)*request.height
    require(pixels<=maxImagePixels && pixels*vae.config.outChannels<=Int.MaxValue,"image exceeds allocation limit")
    val latentShape=Shape.from(Seq(1,vae.config.latentChannels,request.height/factor,request.width/factor))
    require(latentShape.numel<=maxLatentElements && latentShape.numel<=Int.MaxValue,"latent exceeds allocation limit")
    noise.foreach { value =>
      require(value.shape==latentShape && value.device==Device.Cpu && value.dtype==DType.Float32,"initial noise shape/device mismatch")
      require(value.toFloatArray.forall(_.isFinite),"initial noise has nonfinite values")
    }
    val schedule=source.fold(fullSchedule) { value =>
      require(value.strength.isFinite && value.strength>=0 && value.strength<=1,"image strength must be in 0..1")
      require(value.image.shape==Shape.from(Seq(1L,vae.config.inChannels.toLong,request.height.toLong,request.width.toLong)),
        "source image shape mismatch")
      require(value.image.toFloatArray.forall(x => x.isFinite && x>= -1f && x<=1f),"source image must be finite and in [-1,1]")
      value.posteriorNoise.foreach { n =>
        require(n.shape==latentShape && n.toFloatArray.forall(_.isFinite),"posterior noise shape/value mismatch")
      }
      val steps=math.max(1,(fullSchedule.timesteps.size*value.strength).toInt)
      val start=fullSchedule.timesteps.size-steps
      EulerSchedule(fullSchedule.timesteps.drop(start),fullSchedule.sigmas.drop(start),fullSchedule.initialNoiseSigma)
    }
    val random=new java.util.Random(request.seed)
    def randomNoise(): Tensor[Float]=Tensor.fromFloats(Array.fill(latentShape.numel.toInt)(random.nextGaussian().toFloat),latentShape)
    val clean=source.map { value =>
      checkCancelled()
      val moments=vaeExecution.encode(vae.encoder,value.image,cancelled)
      val posterior=VaePosterior(Variable(moments,requiresGrad=false))
      checkCancelled()
      val latent=vae.toDiffusionLatents(posterior.sample(value.posteriorNoise.getOrElse(randomNoise())))
      require(latent.data.toFloatArray.forall(_.isFinite),"VAE encoder produced nonfinite latents")
      latent
    }
    def encode(text: String): Variable =
      checkCancelled()
      val ids=tokenizer.batch(Vector(text),truncate=request.truncatePrompt).ids
      checkCancelled()
      textEncoder(ids,skipLastLayers=request.clipSkip-1)
    val sdxl=sdxlConditioning.map(_.encode(request.prompt,request.negativePrompt,request.clipSkip,cancelled))
    val conditional=sdxl.fold(encode(request.prompt))(p => Variable(p.positive,requiresGrad=false))
    val unconditional=if request.guidanceScale>1 then Some(sdxl.fold(encode(request.negativePrompt))(p => Variable(p.negative,requiresGrad=false))) else None
    val positiveAdded=sdxl.map { pair =>
      val size=request.sdxlSize.getOrElse(SdxlImageSize(request.height,request.width))
      DiffusionAddedConditioning(pair.pooled,size.timeIds(request.height,request.width))
    }
    val negativeAdded=if request.guidanceScale>1 then sdxl.map(pair =>
      DiffusionAddedConditioning(pair.negativePooled,positiveAdded.get.timeIds)) else None
    checkCancelled()
    val initial=noise.getOrElse(randomNoise())
    var latent=clean.fold(Variable(initial,requiresGrad=false)*schedule.initialNoiseSigma.toFloat) { encoded =>
      val sigma=schedule.sigmas.head
      val alpha=if dpm then 1/math.hypot(1,sigma) else 1.0
      encoded*alpha.toFloat+Variable(initial,requiresGrad=false)*(sigma*alpha).toFloat
    }
    val scheduler=EulerScheduler(schedule,prediction,maxElements=maxLatentElements)
    val dpmSolver=if dpm then Some(DpmPlusPlus2M(schedule.sigmas,prediction,maxLatentElements)) else None
    var history: Option[Variable]=None
    unetExecution.withPredictor(unet,conditional.data,unconditional.map(_.data),cancelled,positiveAdded,negativeAdded) { predict =>
      for i <- schedule.timesteps.indices do
        checkCancelled()
        val input=if dpm then latent else scheduler.scaleModelInput(i,latent)
        val time=Tensor.fromFloats(Array(schedule.timesteps(i)),Shape.from(Seq(1)))
        val predicted=Variable(predict(input.data,time,request.guidanceScale),requiresGrad=false)
        checkCancelled()
        dpmSolver match
          case Some(solver) =>
            val result=solver.step(i,predicted,latent,history)
            latent=result.previous
            history=Some(result.predictedOriginal)
          case None => latent=scheduler.step(i,predicted,latent).previous
        require(latent.data.toFloatArray.forall(_.isFinite),"denoising produced nonfinite latents")
        checkCancelled()
        onProgress(DiffusionImageProgress(i+1,schedule.timesteps.size,schedule.timesteps(i),latent.data))
    }
    checkCancelled()
    val decoderInput=latent*(1/vae.config.scalingFactor).toFloat
    val image=try
      val decoded=vaeExecution.decode(vae.decoder,decoderInput.data,cancelled)
      require(decoded.toFloatArray.forall(_.isFinite),"VAE produced nonfinite image values")
      decoded
    catch
      case e: CancellationException => throw e
      case scala.util.control.NonFatal(e) => throw new VaeDecodeFailure(decoderInput.data,e)
    checkCancelled()
    image
  }

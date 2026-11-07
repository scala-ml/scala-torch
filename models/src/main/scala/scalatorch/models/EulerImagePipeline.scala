package scalatorch.models

import scalatorch.*
import scalatorch.autograd.{Autograd,Variable}
import java.util.concurrent.CancellationException

final case class DiffusionImageRequest(prompt: String,negativePrompt: String,width: Int,height: Int,seed: Long,
    guidanceScale: Float=7.5f,truncatePrompt: Boolean=false,clipSkip: Int=1,sdxlSize: Option[SdxlImageSize]=None,
    controlNet: Option[DiffusionControlNetRequest]=None,controlNets: Vector[DiffusionControlNetRequest]=Vector.empty,
    variationSeed: Option[Long]=None,variationStrength: Double=0.0,torchCpuNoise: Boolean=false,guidanceRescale: Float=0f,
    cfgSchedule: Option[CfgSchedule]=None,resharpen: Option[Resharpen]=None,latentMirroring: Option[LatentMirroring]=None,
    colorGrading: Option[LatentColorGrading]=None,antiBurn: Option[AntiBurn]=None,vectorscope: Option[LatentVectorscope]=None,
    latentHookSeed: Option[Long]=None,latentHookBatchSize: Int=1,latentHookBatchIndex: Int=0,detailDaemon: Option[DetailDaemon]=None,
    loraStepFilter: Option[(Int,Int,scalatorch.nn.LowRankAdapter)=>Boolean]=None,
    dynamicThresholding: Option[DynamicThresholdingConfig]=None,
    tiling: Option[scalatorch.nn.CircularConvConfig]=None,
    cads: Option[CadsConfig]=None,
    hyperTile: Option[HyperTileConfig]=None,
    apg: Option[ApgConfig]=None,
    mahiro: Boolean=false,
    pag: Option[PagConfig]=None,
    trainingSigmas: Vector[Float]=Vector.empty,
    promptScheduleSteps: Option[(Int,Option[Int])]=None,
    weightGradient: Boolean=false,
    negPip: Boolean=false,
    kohyaHrFix: Option[KohyaHrFixConfig]=None,
    tokenDownsampling: Option[TokenDownsamplingConfig]=None,
    latentModifier: Option[LatentModifierConfig]=None,
    softInpainting: Option[SoftInpaintingConfig]=None,
    multiDiffusion: Option[MultiDiffusionConfig]=None,
    semanticGuidance: Option[SemanticGuidanceConfig]=None,
    sonar: Option[SonarConfig]=None,
    sonarReferenceImage: Option[Tensor[Float]]=None,
    noiseResizeFrom: Option[(Int,Int)]=None,
    cdTuner: Option[CdTunerConfig]=None,
    isHiresPass: Boolean=false,
    onStepObservation: Option[(Int,Int,Tensor[Float],Tensor[Float],Tensor[Float])=>Unit]=None,
    initialNoiseOverride: Option[Tensor[Float]]=None,
    llul: Option[LlulConfig]=None,
    agentAttention: Option[AgentAttentionConfig]=None,
    incantations: Option[IncantationsConfig]=None,
    characteristicGuidance: Option[CharacteristicGuidanceConfig]=None,
    clipIo: Option[ClipIoConfig]=None,
    promptFusion: Boolean=false,
    neutralPrompt: Option[NeutralPromptConfig]=None,
    composableLora: Option[ComposableLoraConfig]=None,
    namedLoraAdapters: Map[String,Vector[scalatorch.nn.LowRankAdapter]]=Map.empty,
    aestheticGradients: Option[AestheticGradientsConfig]=None,
    sag: Option[SagConfig]=None,
    reference: Option[ReferenceConfig]=None,
    revision: Option[RevisionConfig]=None,
    ipAdapter: Option[IpAdapterConfig]=None,
    controlLLLite: Option[ControlLLLiteConfig]=None,
    fooocusInpaint: Option[FooocusInpaintConfig]=None,
    compile: Option[CompileConfig]=None,
    controlNetPaths: Map[String,String]=Map.empty,
    onWarning: String=>Unit=_ => ())
/** SDXL original size and crop; target size is the requested output dimensions. */
final case class SdxlImageSize(originalHeight: Int,originalWidth: Int,cropTop: Int=0,cropLeft: Int=0):
  require(originalHeight>0 && originalWidth>0 && cropTop>=0 && cropLeft>=0,"invalid SDXL original size or crop")
  private[models] def timeIds(height: Int,width: Int): Tensor[Float] =
    Tensor.fromFloats(Array(originalHeight.toFloat,originalWidth.toFloat,cropTop.toFloat,cropLeft.toFloat,height.toFloat,width.toFloat),Shape.from(Seq(1L,6L)))
/** Single NCHW image in [-1,1]; optional standard-normal posterior noise for replay. */
final case class DiffusionImageSource(image: Tensor[Float],strength: Double,posteriorNoise: Option[Tensor[Float]]=None,
    mask: Option[Tensor[Float]]=None,originalImage: Option[Tensor[Float]]=None,finalOverlay: Boolean=true,
    maskedImage: Option[Tensor[Float]]=None,latentTransform: Option[Tensor[Float] => Tensor[Float]]=None):
  require(mask.isDefined==originalImage.isDefined,"inpaint mask and original image must be supplied together")
  require(maskedImage.isEmpty || mask.isDefined,"masked image pixels require an inpaint mask")
/** Already-scaled diffusion latents for a Hires/img2img second pass; no VAE encoding is performed. */
final case class DiffusionLatentSource(latents: Tensor[Float],strength: Double)
final case class DiffusionImageProgress(completedSteps: Int,totalSteps: Int,timestep: Float,latents: Tensor[Float])

object EulerImagePipeline:
  private[models] def shouldTileDecode(inputHeight: Int,inputWidth: Int,sampleHeight: Int,sampleWidth: Int,
      factor: Int,configured: Boolean): Boolean =
    require(inputHeight>0 && inputWidth>0 && sampleHeight>0 && sampleWidth>0 && factor>0 &&
      sampleHeight%factor==0 && sampleWidth%factor==0,"invalid VAE decode tile geometry")
    configured || inputHeight>sampleHeight/factor || inputWidth>sampleWidth/factor

  private[models] def concatenateInpaintChannels(sample: Tensor[Float],mask: Tensor[Float],
      maskedImageLatents: Tensor[Float]): Tensor[Float] =
    val dims=sample.shape.dims
    require(sample.device==Device.Cpu && sample.dtype==DType.Float32 && dims.size==4 &&
      mask.device==Device.Cpu && mask.dtype==DType.Float32 &&
      mask.shape==Shape.from(Seq(dims(0),1L,dims(2),dims(3))) &&
      maskedImageLatents.device==Device.Cpu && maskedImageLatents.dtype==DType.Float32 &&
      maskedImageLatents.shape==sample.shape,"nine-channel inpaint condition shape/device mismatch")
    Variable.cat(Seq(Variable(sample),Variable(mask),Variable(maskedImageLatents)),1).data

/** Single-image Euler/DPM++ sampling with explicit U-Net and VAE placement. */
final class EulerImagePipeline(val tokenizer: ClipFastTokenizer,val textEncoder: ClipTextModel,
    val unet: ConditionalUNet2d,val vae: VaeModel,maxImagePixels: Long=16777216L,maxLatentElements: Long=16777216L,
    val vaeExecution: VaeExecution=VaeExecution.Cpu,val unetExecution: UNetExecution=UNetExecution.Cpu,
    val sdxlConditioning: Option[SdxlTextConditioning]=None,val vaeTiling: Boolean=false):
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
  private val latentChannels=vae.config.latentChannels
  private val unetInputChannels=unet.convIn.weight.data.shape.dims(1)
  private val nineChannelInput=unetInputChannels==2*latentChannels+1
  private def encodeVae(image: Tensor[Float],cancelled: () => Boolean): Tensor[Float]=
    if vaeTiling then vaeExecution.encodeTiled(vae.encoder,image,vae.config.sampleHeight,vae.config.sampleWidth,cancelled)
    else vaeExecution.encode(vae.encoder,image,cancelled)
  private def decodeVae(latents: Tensor[Float],cancelled: () => Boolean): Tensor[Float]=
    if (vaeTiling || vae.config.sampleHeight>=512) && EulerImagePipeline.shouldTileDecode(latents.shape.dims(2).toInt,latents.shape.dims(3).toInt,
        vae.config.sampleHeight,vae.config.sampleWidth,factor,vaeTiling) then
      vaeExecution.decodeTiled(vae.decoder,latents,vae.config.sampleHeight,vae.config.sampleWidth,cancelled)
    else vaeExecution.decode(vae.decoder,latents,cancelled)
  /** Decode diffusion-scaled latents, preserving cancellation and recovery information. */
  def decodeLatents(latents: Tensor[Float],cancelled: () => Boolean=() => false): Tensor[Float] =
    val decoderInput=(Variable(latents,requiresGrad=false)*(1/vae.config.scalingFactor).toFloat).data
    val image=try
      val decoded=decodeVae(decoderInput,cancelled)
      require(decoded.toFloatArray.forall(_.isFinite),"VAE produced nonfinite image values")
      decoded
    catch
      case e: CancellationException => throw e
      case scala.util.control.NonFatal(e) => throw new VaeDecodeFailure(decoderInput,e)
    if cancelled() || Thread.currentThread().isInterrupted then throw new CancellationException("VAE decode cancelled")
    image
  require((unetInputChannels==latentChannels || nineChannelInput) &&
    unet.convOut.weight.data.shape.dims.head==vae.config.latentChannels,
    "U-Net must accept latent channels or latent+mask+masked-image channels and output VAE latent channels")
  private val factor=1 << (vae.decoder.upBlocks.size-1)
  private lazy val singleClipConditioning=new SingleClipTextConditioning(tokenizer,textEncoder)

  /** Noise is unscaled standard-normal data; supplied noise overrides seed. Guidance is enabled above one. */
  def generate(request: DiffusionImageRequest,schedule: EulerSchedule,
      noise: Option[Tensor[Float]]=None,onProgress: DiffusionImageProgress => Unit=_ => (),
      cancelled: () => Boolean=() => false,
      prediction: DiffusionPrediction=DiffusionPrediction.Epsilon,source: Option[DiffusionImageSource]=None,
      latentSource: Option[DiffusionLatentSource]=None): Tensor[Float] =
    generateImpl(request,schedule,noise,onProgress,cancelled,prediction,false,false,source,latentSource,false)

  /** Runs the same Euler pass but returns final diffusion latents without VAE decoding. */
  def generateLatents(request: DiffusionImageRequest,schedule: EulerSchedule,
      noise: Option[Tensor[Float]]=None,onProgress: DiffusionImageProgress => Unit=_ => (),
      cancelled: () => Boolean=() => false,prediction: DiffusionPrediction=DiffusionPrediction.Epsilon,
      source: Option[DiffusionImageSource]=None): Tensor[Float] =
    generateImpl(request,schedule,noise,onProgress,cancelled,prediction,false,false,source,None,true)

  /** Heun sampling supports image or latent output and both image and latent sources. */
  def generateHeun(request: DiffusionImageRequest,schedule: EulerSchedule,
      noise: Option[Tensor[Float]]=None,onProgress: DiffusionImageProgress => Unit=_ => (),
      cancelled: () => Boolean=() => false,prediction: DiffusionPrediction=DiffusionPrediction.Epsilon,
      source: Option[DiffusionImageSource]=None,latentSource: Option[DiffusionLatentSource]=None,
      returnLatents: Boolean=false): Tensor[Float] =
    generateImpl(request,schedule,noise,onProgress,cancelled,prediction,false,false,source,latentSource,returnLatents,heun=true)

  /** Euler ancestral sampling uses a request-local seeded noise stream at each stochastic step. */
  def generateAncestral(request: DiffusionImageRequest,schedule: EulerSchedule,
      noise: Option[Tensor[Float]]=None,onProgress: DiffusionImageProgress => Unit=_ => (),
      cancelled: () => Boolean=() => false,prediction: DiffusionPrediction=DiffusionPrediction.Epsilon,
      source: Option[DiffusionImageSource]=None,latentSource: Option[DiffusionLatentSource]=None): Tensor[Float] =
    generateImpl(request,schedule,noise,onProgress,cancelled,prediction,false,true,source,latentSource,false)

  def generateAncestralLatents(request: DiffusionImageRequest,schedule: EulerSchedule,
      noise: Option[Tensor[Float]]=None,onProgress: DiffusionImageProgress => Unit=_ => (),
      cancelled: () => Boolean=() => false,prediction: DiffusionPrediction=DiffusionPrediction.Epsilon,
      source: Option[DiffusionImageSource]=None): Tensor[Float] =
    generateImpl(request,schedule,noise,onProgress,cancelled,prediction,false,true,source,None,true)

  /** VP DPM++ 2M: unit initial noise and no Euler model-input scaling. History is local to each call. */
  def generateDpm(request: DiffusionImageRequest,schedule: DpmSchedule,
      noise: Option[Tensor[Float]]=None,onProgress: DiffusionImageProgress => Unit=_ => (),
      cancelled: () => Boolean=() => false,
      prediction: DiffusionPrediction=DiffusionPrediction.Epsilon,source: Option[DiffusionImageSource]=None,
      latentSource: Option[DiffusionLatentSource]=None): Tensor[Float] =
    generateImpl(request,schedule.layout,noise,onProgress,cancelled,prediction,true,false,source,latentSource,false)

  /** Runs the same DPM++ pass but returns final diffusion latents without VAE decoding. */
  def generateDpmLatents(request: DiffusionImageRequest,schedule: DpmSchedule,
      noise: Option[Tensor[Float]]=None,onProgress: DiffusionImageProgress => Unit=_ => (),
      cancelled: () => Boolean=() => false,prediction: DiffusionPrediction=DiffusionPrediction.Epsilon,
      source: Option[DiffusionImageSource]=None): Tensor[Float] =
    generateImpl(request,schedule.layout,noise,onProgress,cancelled,prediction,true,false,source,None,true)

  /** Deterministic DDIM in VP coordinates. */
  def generateDdim(request: DiffusionImageRequest,schedule: DdimSchedule,clipSample: Boolean,clipRange: Float,
      eta: Double=0.0,onProgress: DiffusionImageProgress => Unit=_ => (),cancelled: () => Boolean=() => false,
      prediction: DiffusionPrediction=DiffusionPrediction.Epsilon,source: Option[DiffusionImageSource]=None,
      latentSource: Option[DiffusionLatentSource]=None,returnLatents: Boolean=false): Tensor[Float] =
    generateImpl(request,schedule.layout,None,onProgress,cancelled,prediction,false,false,source,latentSource,returnLatents,
      Some(new DdimScheduler(schedule,prediction,clipSample,clipRange)),eta)

  def generateLms(request: DiffusionImageRequest,schedule: EulerSchedule,
      noise: Option[Tensor[Float]]=None,onProgress: DiffusionImageProgress => Unit=_ => (),
      cancelled: () => Boolean=() => false,prediction: DiffusionPrediction=DiffusionPrediction.Epsilon,
      source: Option[DiffusionImageSource]=None,latentSource: Option[DiffusionLatentSource]=None,
      returnLatents: Boolean=false): Tensor[Float] =
    generateImpl(request,schedule,noise,onProgress,cancelled,prediction,false,false,source,latentSource,returnLatents,samplerKind="LMS")

  def generateKdpm2(request: DiffusionImageRequest,schedule: EulerSchedule,ancestral: Boolean=false,
      noise: Option[Tensor[Float]]=None,onProgress: DiffusionImageProgress => Unit=_ => (),
      cancelled: () => Boolean=() => false,prediction: DiffusionPrediction=DiffusionPrediction.Epsilon,
      source: Option[DiffusionImageSource]=None,latentSource: Option[DiffusionLatentSource]=None,
      returnLatents: Boolean=false): Tensor[Float] =
    generateImpl(request,schedule,noise,onProgress,cancelled,prediction,false,false,source,latentSource,returnLatents,samplerKind=if ancestral then "DPM2 a" else "DPM2")

  def generateDpm2Sa(request: DiffusionImageRequest,schedule: DpmSchedule,
      noise: Option[Tensor[Float]]=None,onProgress: DiffusionImageProgress => Unit=_ => (),
      cancelled: () => Boolean=() => false,prediction: DiffusionPrediction=DiffusionPrediction.Epsilon,
      source: Option[DiffusionImageSource]=None,latentSource: Option[DiffusionLatentSource]=None,
      returnLatents: Boolean=false): Tensor[Float] =
    generateImpl(request,schedule.layout,noise,onProgress,cancelled,prediction,true,false,source,latentSource,returnLatents,samplerKind="DPM++ 2S a")

  def generateDpmSde(request: DiffusionImageRequest,schedule: EulerSchedule,
      noise: Option[Tensor[Float]]=None,onProgress: DiffusionImageProgress => Unit=_ => (),
      cancelled: () => Boolean=() => false,prediction: DiffusionPrediction=DiffusionPrediction.Epsilon,
      source: Option[DiffusionImageSource]=None,latentSource: Option[DiffusionLatentSource]=None,
      returnLatents: Boolean=false): Tensor[Float] =
    generateImpl(request,schedule,noise,onProgress,cancelled,prediction,false,false,source,latentSource,returnLatents,samplerKind="DPM++ SDE")

  def generateDpmMultistepSde(request: DiffusionImageRequest,schedule: DpmSchedule,order: Int=2,heunVariant: Boolean=false,
      noise: Option[Tensor[Float]]=None,onProgress: DiffusionImageProgress => Unit=_ => (),
      cancelled: () => Boolean=() => false,prediction: DiffusionPrediction=DiffusionPrediction.Epsilon,
      source: Option[DiffusionImageSource]=None,latentSource: Option[DiffusionLatentSource]=None,
      returnLatents: Boolean=false): Tensor[Float] =
    val kind=if order==3 then "DPM++ 3M SDE" else if heunVariant then "DPM++ 2M SDE Heun" else "DPM++ 2M SDE"
    generateImpl(request,schedule.layout,noise,onProgress,cancelled,prediction,true,false,source,latentSource,returnLatents,samplerKind=kind)

  def generateUniPc(request: DiffusionImageRequest,schedule: DpmSchedule,
      noise: Option[Tensor[Float]]=None,onProgress: DiffusionImageProgress => Unit=_ => (),
      cancelled: () => Boolean=() => false,prediction: DiffusionPrediction=DiffusionPrediction.Epsilon,
      source: Option[DiffusionImageSource]=None,latentSource: Option[DiffusionLatentSource]=None,
      returnLatents: Boolean=false): Tensor[Float] =
    generateImpl(request,schedule.layout,noise,onProgress,cancelled,prediction,true,false,source,latentSource,returnLatents,samplerKind="UniPC")

  def generateLcm(request: DiffusionImageRequest,schedule: LcmSchedule,
      noise: Option[Tensor[Float]]=None,onProgress: DiffusionImageProgress => Unit=_ => (),
      cancelled: () => Boolean=() => false,prediction: DiffusionPrediction=DiffusionPrediction.Epsilon,
      source: Option[DiffusionImageSource]=None,latentSource: Option[DiffusionLatentSource]=None,
      returnLatents: Boolean=false): Tensor[Float] =
    generateImpl(request,schedule.layout,noise,onProgress,cancelled,prediction,true,false,source,latentSource,returnLatents,samplerKind="LCM",lcmSchedule=Some(schedule))

  /** DDIM inversion + optional randomness mixing for `img2imgalt`. */
  def invertImageForAlt(request: DiffusionImageRequest,image: Tensor[Float],alt: Img2ImgAltConfig,
      schedule: DiffusionNoiseSchedule,defaultSteps: Int,prediction: DiffusionPrediction=DiffusionPrediction.Epsilon,cancelled: () => Boolean=() => false): Tensor[Float] = Autograd.noGrad {
    val invSteps=if alt.inversionSteps>0 then alt.inversionSteps else defaultSteps
    val moments=encodeVae(image,cancelled)
    val x0=vae.toDiffusionLatents(VaePosterior(Variable(moments,requiresGrad=false)).mean).data
    val posTxt=if alt.overridePrompt.nonEmpty then alt.overridePrompt else request.prompt
    val negTxt=if alt.overrideNegative.nonEmpty then alt.overrideNegative else request.negativePrompt
    if request.promptFusion && (PromptFusion.isActive(posTxt) || PromptFusion.isActive(negTxt)) then
      throw new UnsupportedOperationException("Prompt Fusion schedules are not supported during img2img Alternative inversion")
    val guided=math.abs(alt.decodeCfgScale-1.0)>1e-6
    val (cond,uncond,posAdded,negAdded)=sdxlConditioning match
      case Some(sc) =>
        val p=sc.encode(posTxt,negTxt,request.clipSkip,cancelled)
        val size=request.sdxlSize.getOrElse(SdxlImageSize(request.height,request.width))
        val pa=Some(DiffusionAddedConditioning(p.pooled,size.timeIds(request.height,request.width)))
        val na=if guided then Some(DiffusionAddedConditioning(p.negativePooled,pa.get.timeIds)) else None
        (p.positive,if guided then Some(p.negative) else None,pa,na)
      case None =>
        val s=singleClipConditioning.encode(posTxt,negTxt,request.clipSkip,request.truncatePrompt,encodeNegative=guided,cancelled=cancelled)
        (s.positive,s.negative,None,None)
    val inverted=unetExecution.withPredictor(unet,cond,uncond,cancelled,posAdded,negAdded,request.controlNet.toVector++request.controlNets) { predict =>
      Img2ImgAlt.invertLatents(x0,invSteps,alt.decodeCfgScale,schedule,(x,t,isCond) => {
        val time=Tensor.fromFloats(Array(t.toFloat),Shape.from(Seq(1)))
        val raw=if isCond || uncond.isEmpty then predict(x,time,1.0f,0.0,0f,None,None,1.0,None)
          else predict(x,time,1.0f,0.0,0f,None,None,1.0,None,conditionalOverride=uncond,positiveAddedOverride=negAdded)
        if prediction==DiffusionPrediction.Epsilon then raw
        else
          val alphaBar=math.max(1e-7,schedule.alphasCumprod(t).toDouble)
          val sa=math.sqrt(alphaBar).toFloat
          val s1a=math.sqrt(math.max(1e-7,1.0-alphaBar)).toFloat
          val xArr=x.toFloatArray; val rArr=raw.toFloatArray
          Tensor.fromFloats(Array.tabulate(xArr.length) { idx =>
            if prediction==DiffusionPrediction.VPrediction then sa*rArr(idx)+s1a*xArr(idx)
            else (xArr(idx)-sa*rArr(idx))/s1a
          },raw.shape)
      },cancelled)
    }
    if alt.randomness<=0.0 then inverted
    else
      val noiseData=if request.torchCpuNoise then new TorchCpuNormal(request.seed).normal(x0.shape.numel.toInt)
        else
          val rng=new java.util.Random(request.seed)
          Array.fill(x0.shape.numel.toInt)(rng.nextGaussian().toFloat)
      Img2ImgAlt.mixRandomness(inverted,Tensor.fromFloats(noiseData,x0.shape),alt.randomness)
  }

  private def generateImpl(request: DiffusionImageRequest,fullSchedule: EulerSchedule,
      noise: Option[Tensor[Float]],onProgress: DiffusionImageProgress => Unit,
      cancelled: () => Boolean,prediction: DiffusionPrediction,dpm: Boolean,ancestral: Boolean,source: Option[DiffusionImageSource],
      latentSource: Option[DiffusionLatentSource],returnLatents: Boolean,ddim: Option[DdimScheduler]=None,ddimEta: Double=0.0,
      heun: Boolean=false,samplerKind: String="Euler",lcmSchedule: Option[LcmSchedule]=None): Tensor[Float] = Autograd.noGrad {
    def checkCancelled(): Unit =
      if cancelled() || Thread.currentThread().isInterrupted then throw new CancellationException("image generation cancelled")
    checkCancelled()
    require(!ancestral || prediction!=DiffusionPrediction.Sample,"Euler ancestral sample prediction is not supported")
    require(sdxlConditioning.nonEmpty || request.sdxlSize.isEmpty,"SDXL size conditioning requires an SDXL model")
    require(request.clipSkip>=1 && request.clipSkip<=textEncoder.layers.size+1,"CLIP skip exceeds text encoder layer count")
    require(request.guidanceScale.isFinite && request.guidanceScale>=0,"invalid image guidance scale")
    require(request.variationStrength.isFinite && request.variationStrength>=0 && request.variationStrength<=1 &&
      (request.variationStrength==0 || request.variationSeed.isDefined),"invalid variation seed or strength")
    require(request.width>0 && request.height>0 && request.width%factor==0 && request.height%factor==0,
      "image dimensions must be positive multiples of the VAE scale")
    require(source.isEmpty || latentSource.isEmpty,"image source and latent source are mutually exclusive")
    require(!returnLatents || latentSource.isEmpty,"latent-return passes cannot use a latent source")
    val pixels=BigInt(request.width)*request.height
    require(pixels<=maxImagePixels && pixels*vae.config.outChannels<=Int.MaxValue,"image exceeds allocation limit")
    val latentShape=Shape.from(Seq(1,vae.config.latentChannels,request.height/factor,request.width/factor))
    require(latentShape.numel<=maxLatentElements && latentShape.numel<=Int.MaxValue,"latent exceeds allocation limit")
    noise.foreach { value =>
      require(value.shape==latentShape && value.device==Device.Cpu && value.dtype==DType.Float32,"initial noise shape/device mismatch")
      require(value.toFloatArray.forall(_.isFinite),"initial noise has nonfinite values")
    }
    val sourceStrength=source.map(_.strength).orElse(latentSource.map(_.strength))
    val schedule=sourceStrength.fold(fullSchedule) { strength =>
      require(strength.isFinite && strength>=0 && strength<=1,"image strength must be in 0..1")
      source.foreach { value =>
      val dims=value.image.shape.dims
      require(dims.size==4 && dims(0)==1 && dims(1)==vae.config.inChannels && dims(2)>0 && dims(3)>0 &&
        dims(2)%factor==0 && dims(3)%factor==0 && dims(2)*dims(3)<=maxImagePixels &&
        (value.latentTransform.isDefined || (dims(2)==request.height && dims(3)==request.width)),"source image shape mismatch")
      val posteriorShape=Shape.from(Seq(1L,latentChannels.toLong,dims(2)/factor,dims(3)/factor))
      require(posteriorShape.numel<=maxLatentElements && posteriorShape.numel<=Int.MaxValue,"source latent exceeds allocation limit")
      require(value.image.toFloatArray.forall(x => x.isFinite && x>= -1f && x<=1f),"source image must be finite and in [-1,1]")
      value.posteriorNoise.foreach { n =>
        require(n.shape==posteriorShape && n.toFloatArray.forall(_.isFinite),"posterior noise shape/value mismatch")
      }
      value.mask.foreach { mask =>
        require(mask.device==Device.Cpu && mask.dtype==DType.Float32 &&
          mask.shape==Shape.from(Seq(1L,1L,request.height.toLong,request.width.toLong)),"inpaint mask shape/device mismatch")
        require(mask.toFloatArray.forall(x => x.isFinite && x>=0f && x<=1f),"inpaint mask values must be finite in [0,1]")
      }
      value.originalImage.foreach { original =>
        require(original.device==Device.Cpu && original.dtype==DType.Float32 &&
          original.shape==Shape.from(Seq(1L,vae.config.inChannels.toLong,request.height.toLong,request.width.toLong)) &&
          original.toFloatArray.forall(x => x.isFinite && x>= -1f && x<=1f),"inpaint original image mismatch")
      }
      value.maskedImage.foreach { masked =>
        require(value.mask.isDefined && masked.device==Device.Cpu && masked.dtype==DType.Float32 &&
          masked.shape==value.image.shape && masked.shape.dims(1)==3 &&
          masked.toFloatArray.forall(x => x.isFinite && x>= -1f && x<=1f),"inpaint masked image mismatch")
      }
      if nineChannelInput && value.mask.isDefined then
        require(value.maskedImage.isDefined,"nine-channel inpainting requires masked image pixels")
      if !nineChannelInput && value.latentTransform.isEmpty then
        require(value.maskedImage.isEmpty,"masked image pixels are only valid for nine-channel U-Nets")
      }
      latentSource.foreach { value =>
        require(value.latents.shape==latentShape && value.latents.device==Device.Cpu && value.latents.dtype==DType.Float32,
          "latent source shape/device mismatch")
        require(value.latents.toFloatArray.forall(_.isFinite),"latent source must contain finite values")
      }
      // Diffusers computes the img2img start from the requested count, even
      // when a custom sigma schedule (notably DDIM) has a different length.
      val nominal=fullSchedule.nominalInferenceSteps.getOrElse(fullSchedule.timesteps.size)
      val steps=math.max(1,(nominal*strength).toInt)
      val start=math.max(0,nominal-steps)
      require(start<fullSchedule.timesteps.size,"img2img strength leaves no scheduler steps")
      EulerSchedule(fullSchedule.timesteps.drop(start),fullSchedule.sigmas.drop(start),fullSchedule.initialNoiseSigma)
    }
    val random=new java.util.Random(request.seed)
    val torchRandom=Option.when(request.torchCpuNoise)(new TorchCpuNormal(request.seed))
    def randomNoise(shape: Shape=latentShape): Tensor[Float]=Tensor.fromFloats(torchRandom match
      case Some(generator) => generator.normal(shape.numel.toInt)
      case None => Array.fill(shape.numel.toInt)(random.nextGaussian().toFloat),shape)
    def independentNoise(seed: Long,shape: Shape=latentShape): Tensor[Float] =
      val data=if request.torchCpuNoise then new TorchCpuNormal(seed).normal(shape.numel.toInt)
        else
          val rng=new java.util.Random(seed)
          Array.fill(shape.numel.toInt)(rng.nextGaussian().toFloat)
      Tensor.fromFloats(data,shape)
    def oldIndependentNoise(seed: Long): Tensor[Float] =
      val data=if request.torchCpuNoise then new TorchCpuNormal(seed).normal(latentShape.numel.toInt)
        else
          val rng=new java.util.Random(seed)
          Array.fill(latentShape.numel.toInt)(rng.nextGaussian().toFloat)
      Tensor.fromFloats(data,latentShape)
    // Diffusers starts a fully denoised inpaint pass from pure noise. A
    // four-channel U-Net still needs the original latent for mask blending;
    // a nine-channel inpaint U-Net does not encode that image at strength 1.
    val fullInpaint=source.exists(value => value.strength==1.0 &&
      (value.mask.isDefined || (nineChannelInput && value.latentTransform.isDefined)))
    val clean=source.filterNot(value => nineChannelInput && value.mask.isDefined && value.strength==1.0 && value.latentTransform.isEmpty).map { value =>
      checkCancelled()
      val moments=encodeVae(value.image,cancelled)
      val posterior=VaePosterior(Variable(moments,requiresGrad=false))
      checkCancelled()
      val posteriorShape=Shape.from(Seq(1L,latentChannels.toLong,value.image.shape.dims(2)/factor,value.image.shape.dims(3)/factor))
      val encoded=vae.toDiffusionLatents(posterior.sample(value.posteriorNoise.getOrElse(randomNoise(posteriorShape))))
      val latent=value.latentTransform.fold(encoded)(transform => Variable(transform(encoded.data),requiresGrad=false))
      require(latent.data.shape==latentShape && latent.data.device==Device.Cpu && latent.data.dtype==DType.Float32,
        "transformed source latent shape/device mismatch")
      require(latent.data.toFloatArray.forall(_.isFinite),"VAE encoder produced nonfinite latents")
      latent
    }
    val cleanLatents=latentSource.map(value => Variable(value.latents,requiresGrad=false))
    val latentMask=source.flatMap(_.mask).map { mask =>
      val sourceMask=mask.toFloatArray
      val width=request.width/factor;val height=request.height/factor
      val values=Array.tabulate(width*height) { i =>
        val x=i%width;val y=i/width
        // Diffusers binarizes its model mask before nearest-neighbor latent
        // resizing; keep the original soft mask for the final pixel overlay.
        if sourceMask((y*factor)*request.width+x*factor)>=0.5f then 1f else 0f
      }
      Tensor.fromFloats(values,Shape.from(Seq(1L,1L,height.toLong,width.toLong)))
    }
    val (schedBaseSteps,schedHiresSteps)=request.promptScheduleSteps.getOrElse((schedule.timesteps.size,None))
    request.kohyaHrFix.foreach(_.validate(unet.downBlocks.map(_.layers.size)))
    request.tokenDownsampling.foreach { td =>
      if td.enabledFor(schedHiresSteps.getOrElse(schedBaseSteps)) && (request.pag.isDefined || request.hyperTile.isDefined) then
        throw IllegalStateException("Token Downsampling cannot combine with PAG or HyperTile: their self-attention layouts differ")
    }
    val (emPipe,effReqPrompt,effReqNegPrompt)=EmbeddingMerge.preparePipelineAndPrompts(this,request.prompt,request.negativePrompt)
    def scheduleText(txt: String): Vector[(Int,String)] =
      val baseTxt=if request.promptFusion then PromptFusion.applyDeclarations(txt) else txt
      if request.weightGradient then WeightGradient.schedule(baseTxt,schedBaseSteps,schedHiresSteps)
      else PromptSchedule.schedule(baseTxt,schedBaseSteps,schedHiresSteps)
    val promptSched=scheduleText(effReqPrompt)
    val negSched=scheduleText(effReqNegPrompt)
    val initialPrompt=PromptSchedule.textAt(promptSched,1)
    val initialNegPrompt=PromptSchedule.textAt(negSched,1)
    case class EncodedPair(pos: Tensor[Float],neg: Option[Tensor[Float]],posAdded: Option[DiffusionAddedConditioning],negAdded: Option[DiffusionAddedConditioning],posSigns: Option[Array[Float]],negSigns: Option[Array[Float]])
    def encodeRawPair(posTxt: String,negTxt: String,withNeg: Boolean): (Tensor[Float],Option[Tensor[Float]],Option[DiffusionAddedConditioning],Option[DiffusionAddedConditioning]) =
      val cPos=if request.composableLora.exists(_.enabled) then ComposableLora.cleanPrompt(posTxt) else posTxt
      val cNeg=if request.composableLora.exists(_.enabled) then ComposableLora.cleanPrompt(negTxt) else negTxt
      val (rawP,rawN,rawPa,rawNa)=emPipe.sdxlConditioning match
        case Some(sc) =>
          val p=sc.encode(cPos,cNeg,request.clipSkip,cancelled)
          val size=request.sdxlSize.getOrElse(SdxlImageSize(request.height,request.width))
          val pa=Some(DiffusionAddedConditioning(p.pooled,size.timeIds(request.height,request.width)))
          val na=if withNeg then Some(DiffusionAddedConditioning(p.negativePooled,pa.get.timeIds)) else None
          (p.positive,if withNeg then Some(p.negative) else None,pa,na)
        case None =>
          val s=emPipe.singleClipConditioning.encode(cPos,cNeg,request.clipSkip,request.truncatePrompt,encodeNegative=withNeg,cancelled=cancelled)
          (s.positive,s.negative,None,None)
      request.aestheticGradients match
        case None => (rawP,rawN,rawPa,rawNa)
        case Some(agCfg) =>
          val slerpEnc=Some((t: String) => emPipe.sdxlConditioning.fold(emPipe.singleClipConditioning.encode(t,"",request.clipSkip,request.truncatePrompt,encodeNegative=false,cancelled=cancelled).positive)(sc => sc.encode(t,"",request.clipSkip,cancelled).positive))
          if !agCfg.negative then
            (AestheticGradients.personalizeConditioning(rawP,agCfg,slerpEnc),rawN,rawPa,rawNa)
          else if rawN.isDefined && cNeg.trim.nonEmpty then
            (rawP,rawN.map(n => AestheticGradients.personalizeConditioning(n,agCfg,slerpEnc)),rawPa,rawNa)
          else (rawP,rawN,rawPa,rawNa)
    def encodePair(posTxt: String,negTxt: String): EncodedPair =
      val withNeg=request.guidanceScale>1
      if !request.negPip then
        val (p,n,pa,na)=encodeRawPair(posTxt,negTxt,withNeg)
        EncodedPair(p,n,pa,na,None,None)
      else
        val (posClean,posTargets)=NegPip.extract(posTxt)
        val (negClean,negTargets)=if withNeg then NegPip.extract(negTxt) else (negTxt,Vector.empty)
        if posTargets.isEmpty && negTargets.isEmpty then
          val (p,n,pa,na)=encodeRawPair(posTxt,negTxt,withNeg)
          EncodedPair(p,n,pa,na,None,None)
        else
          val (basePos,baseNeg,pa,na)=encodeRawPair(posClean,negClean,withNeg)
          val toks=Seq(tokenizer)++sdxlConditioning.toSeq.map(_.tokenizer2)
          def buildSigned(base: Tensor[Float],targets: Vector[(String,Double)]): (Tensor[Float],Option[Array[Float]]) =
            if targets.isEmpty then (base,None)
            else
              val pieces=scala.collection.mutable.ArrayBuffer[Tensor[Float]](base)
              val signs=scala.collection.mutable.ArrayBuffer.fill[Float](base.shape.dims(1).toInt)(1f)
              targets.foreach { case (t,w) =>
                val absW=BigDecimal(math.abs(w)).bigDecimal.stripTrailingZeros().toPlainString
                val (word,_,_,_)=encodeRawPair(s"(${NegPip.escapeText(t)}:$absW)","",false)
                val posIdx=NegPip.tokenPositions(toks,t,word.shape.dims(1).toInt)
                val sliced=NegPip.sliceTokens(word,posIdx)
                pieces+=sliced
                signs++=Seq.fill(sliced.shape.dims(1).toInt)(-1f)
              }
              (NegPip.concatTokens(pieces.toSeq),Some(signs.toArray))
          val (finalPos,posSigns)=buildSigned(basePos,posTargets)
          val (finalNeg,negSigns)=baseNeg match
            case Some(bn) =>
              val (fn,ns)=buildSigned(bn,negTargets)
              (Some(fn),ns)
            case None => (None,None)
          EncodedPair(finalPos,finalNeg,pa,na,posSigns,negSigns)
    val rawInitialEncoded=encodePair(initialPrompt,initialNegPrompt)
    val initialEncoded=request.clipIo.fold(rawInitialEncoded) { cio =>
      val dir=ClipIo.resolveStorageDir(cio)
      val inPosMode=if cio.inputEnabled && cio.inputName.trim.nonEmpty then cio.inputMode else "off"
      val inNegMode=if cio.inputEnabled && cio.negativeInputName.trim.nonEmpty then cio.negativeInputMode else "off"
      if cio.outputEnabled && !request.isHiresPass then
        if cio.outputName.trim.nonEmpty && inPosMode=="off" then
          val tgt=ClipIo.safeTarget(dir,cio.outputName,cio.outputFormat)
          if !cio.overwrite && java.nio.file.Files.exists(tgt) then request.onWarning(s"Clip Output: '${cio.outputName}' already exists and Overwrite is off; not written")
          else ClipIo.writeConditioning(tgt,cio.outputFormat,ClipIo.CondData(rawInitialEncoded.pos,rawInitialEncoded.posAdded.map(_.pooledText)))
        if cio.negativeOutputName.trim.nonEmpty && inNegMode=="off" then
          val nCond=rawInitialEncoded.neg.getOrElse(encodeRawPair("",initialNegPrompt,true)._2.get)
          val tgt=ClipIo.safeTarget(dir,cio.negativeOutputName,cio.outputFormat)
          if !cio.overwrite && java.nio.file.Files.exists(tgt) then request.onWarning(s"Clip Output: '${cio.negativeOutputName}' already exists and Overwrite is off; not written")
          else ClipIo.writeConditioning(tgt,cio.outputFormat,ClipIo.CondData(nCond,rawInitialEncoded.negAdded.map(_.pooledText)))
      if inPosMode=="off" && inNegMode=="off" then rawInitialEncoded
      else
        val emptyChunk=encodeRawPair("","",false)._1.slice(1,0,77)
        val (nextPos,nextPosPool)=ClipIo.applySide(inPosMode,cio.inputName,rawInitialEncoded.pos,rawInitialEncoded.posAdded.map(_.pooledText),emptyChunk,dir,cio.inputFormat)
        val (nextNegOpt,nextNegPool)=rawInitialEncoded.neg match
          case Some(bn) =>
            val (nn,np)=ClipIo.applySide(inNegMode,cio.negativeInputName,bn,rawInitialEncoded.negAdded.map(_.pooledText),emptyChunk,dir,cio.inputFormat)
            (Some(nn),np)
          case None => (None,None)
        val maxChunks=math.max(nextPos.shape.dims(1)/77,nextNegOpt.map(_.shape.dims(1)/77).getOrElse(1L)).toInt
        val padPos=ClipPromptEncoding.padTo(nextPos,maxChunks,emptyChunk,67108864L)
        val padNeg=nextNegOpt.map(n => ClipPromptEncoding.padTo(n,maxChunks,emptyChunk,67108864L))
        val nextPa=rawInitialEncoded.posAdded.map(pa => pa.copy(pooledText=nextPosPool.getOrElse(pa.pooledText)))
        val nextNa=rawInitialEncoded.negAdded.map(na => na.copy(pooledText=nextNegPool.getOrElse(na.pooledText)))
        rawInitialEncoded.copy(pos=padPos,neg=padNeg,posAdded=nextPa,negAdded=nextNa)
    }
    val revInitialEncoded=request.revision.filter(_.enabled).fold(initialEncoded) { revCfg =>
      if sdxlConditioning.isEmpty || initialEncoded.posAdded.isEmpty then
        request.onWarning("Revision is an SDXL feature; ignored")
        initialEncoded
      else
        val (revPos,revPool)=Revision.applyToSdxlConditioning(initialEncoded.pos,initialEncoded.posAdded.get.pooledText,revCfg)
        initialEncoded.copy(pos=revPos,posAdded=initialEncoded.posAdded.map(_.copy(pooledText=revPool)))
    }
    val conditional=Variable(revInitialEncoded.pos,requiresGrad=false)
    val unconditional=revInitialEncoded.neg.map(n => Variable(n,requiresGrad=false))
    val positiveAdded=revInitialEncoded.posAdded
    val negativeAdded=revInitialEncoded.negAdded
    /*
      // old singleClipConditioning
      request.clipSkip,request.truncatePrompt,encodeNegative=request.guidanceScale>1,cancelled=cancelled)) else None
    val conditional=sdxl.fold(Variable(single.get.positive,requiresGrad=false))(p => Variable(p.positive,requiresGrad=false))
    val unconditional=if request.guidanceScale>1 then Some(sdxl.fold(
      Variable(single.get.negative.get,requiresGrad=false))(p => Variable(p.negative,requiresGrad=false))) else None
    val positiveAdded=sdxl.map { pair =>
      val size=request.sdxlSize.getOrElse(SdxlImageSize(request.height,request.width))
      DiffusionAddedConditioning(pair.pooled,size.timeIds(request.height,request.width))
    }
    val negativeAdded=if request.guidanceScale>1 then sdxl.map(pair =>
      */
    checkCancelled()
    // Latent Upscale encodes masked conditioning at source resolution before
    // the inpaint pipeline draws target-sized diffusion noise, even for 4-channel U-Nets.
    val transformedMasked=source.filter(_.latentTransform.isDefined).flatMap { value =>
      val masked=value.maskedImage.orElse(Option.when(nineChannelInput && value.mask.isEmpty) {
        Tensor.fromFloats(new Array[Float](value.image.shape.numel.toInt),value.image.shape)
      })
      masked.map { image =>
      val moments=encodeVae(image,cancelled)
      val shape=Shape.from(Seq(1L,latentChannels.toLong,image.shape.dims(2)/factor,image.shape.dims(3)/factor))
      val encoded=vae.toDiffusionLatents(VaePosterior(Variable(moments,requiresGrad=false)).sample(randomNoise(shape))).data
      val transformed=value.latentTransform.get(encoded)
      require(transformed.shape==latentShape && transformed.device==Device.Cpu && transformed.dtype==DType.Float32 &&
        transformed.toFloatArray.forall(_.isFinite),"transformed masked latent shape/device mismatch")
      transformed
      }
    }
    val initial=noise.orElse(request.initialNoiseOverride).getOrElse {
      def seedNoise(s: Long): Tensor[Float] = request.noiseResizeFrom.fold(independentNoise(s)) { case (rw,rh) => HiresFixTweaks.resizedNoise(latentShape,s,request.variationSeed,request.variationStrength,rw,rh,factor,(sd,sh)=>independentNoise(sd,sh)) }
      val base=if (request.torchCpuNoise && source.isEmpty && latentSource.isEmpty) || request.noiseResizeFrom.nonEmpty then seedNoise(request.seed) else randomNoise()
      if request.variationStrength>0 && source.isEmpty && latentSource.isEmpty && request.noiseResizeFrom.isEmpty then
        val variation=seedNoise(request.variationSeed.get)
        VariationNoise.slerp(base,variation,request.variationStrength)
      else base
    }
    val nineChannelMask=if nineChannelInput then Some(latentMask.getOrElse(
      Tensor.fromFloats(Array.fill((request.width/factor)*(request.height/factor))(1f),
        Shape.from(Seq(1L,1L,(request.height/factor).toLong,(request.width/factor).toLong))))) else None
    val nineChannelMaskedLatents=if !nineChannelInput then None else Some(transformedMasked.orElse(source.flatMap(_.maskedImage).map { image =>
      checkCancelled()
      val moments=encodeVae(image,cancelled)
      checkCancelled()
      val encoded=vae.toDiffusionLatents(VaePosterior(Variable(moments,requiresGrad=false)).sample(randomNoise()))
      require(encoded.data.toFloatArray.forall(_.isFinite),"masked-image VAE encoding produced nonfinite latents")
      encoded.data
    }).getOrElse(Tensor.fromFloats(new Array[Float](latentShape.numel.toInt),latentShape)))
    // The four-channel Diffusers inpaint pipeline also samples a masked-image
    // posterior, although its U-Net never consumes those latents. Advance the
    // same RNG stream so subsequent stochastic sampler steps stay aligned.
    if !nineChannelInput && source.exists(value => value.mask.isDefined && value.latentTransform.isEmpty) then
      randomNoise()
      ()
    var latent=if fullInpaint then Variable(initial,requiresGrad=false)*schedule.initialNoiseSigma.toFloat
    else clean.orElse(cleanLatents).fold(Variable(initial,requiresGrad=false)*schedule.initialNoiseSigma.toFloat) { encoded =>
      val sigma=schedule.sigmas.head
      val alpha=if dpm || ddim.nonEmpty then 1/math.hypot(1,sigma) else 1.0
      encoded*alpha.toFloat+Variable(initial,requiresGrad=false)*(sigma*alpha).toFloat
    }
    val scheduler=EulerScheduler(schedule,prediction,maxElements=maxLatentElements)
    val heunSolver=if heun then Some(HeunScheduler(schedule,prediction,maxLatentElements)) else None
    val startStepIdx=fullSchedule.timesteps.size-schedule.timesteps.size
    val dpmSolver=if dpm && samplerKind=="Euler" then Some(DpmPlusPlus2M(schedule.sigmas,prediction,maxLatentElements)) else None
    val lmsSolver=if samplerKind=="LMS" then Some(LmsScheduler(schedule,prediction,maxElements=maxLatentElements,startStepIndex=startStepIdx,fullSigmasOverride=Some(fullSchedule.sigmas))) else None
    val kdpm2Solver=if samplerKind=="DPM2" || samplerKind=="DPM2 a" then Some(Kdpm2Scheduler(schedule,prediction,ancestral=(samplerKind=="DPM2 a"),maxElements=maxLatentElements)) else None
    val dpm2SaSolver=if samplerKind=="DPM++ 2S a" then Some(DpmPlusPlus2Sa(schedule,prediction,maxElements=maxLatentElements,totalStepsOverride=Some(fullSchedule.timesteps.size),startStepIndex=startStepIdx)) else None
    val dpmSdeSolver=if samplerKind=="DPM++ SDE" then Some(DpmPlusPlusSde(schedule,prediction,maxElements=maxLatentElements)) else None
    val dpmMultistepSdeSolver=if samplerKind=="DPM++ 2M SDE" || samplerKind=="DPM++ 2M SDE Heun" || samplerKind=="DPM++ 3M SDE" then
      Some(DpmPlusPlusMultistepSde(schedule.sigmas,prediction,order=if samplerKind=="DPM++ 3M SDE" then 3 else 2,heun=(samplerKind=="DPM++ 2M SDE Heun"),maxElements=maxLatentElements,totalStepsOverride=Some(fullSchedule.timesteps.size),startStepIndex=startStepIdx)) else None
    val uniPcSolver=if samplerKind=="UniPC" then Some(UniPcScheduler(schedule.sigmas,prediction,maxElements=maxLatentElements,totalStepsOverride=Some(fullSchedule.timesteps.size),startStepIndex=startStepIdx)) else None
    val lcmSolver=lcmSchedule.map(s => LcmScheduler(s,prediction))
    val ancestralSolver=if ancestral then Some(EulerAncestralScheduler(schedule,prediction,maxLatentElements)) else None
    val sonarStepper=request.sonar.map { sCfg =>
      val refLatent=request.sonarReferenceImage.filter(_ => sCfg.sampler=="Naive").map { refImg =>
        checkCancelled()
        val moments=encodeVae(refImg,cancelled)
        val mean=(VaePosterior(Variable(moments,requiresGrad=false)).mean*vae.config.scalingFactor.toFloat).data
        Sonar.normalizeReference(mean)
      }
      new SonarStepper(sCfg,schedule.timesteps.size,request.seed,ancestral,refLatent)
    }
    val sgBundle=request.semanticGuidance.map { sgCfg =>
      val emptyChunk=encodeRawPair("","",false)._1.slice(1,0,77)
      val concepts=SemanticGuidance.parseConcepts(sgCfg.prompt,sgCfg.negativePrompt).map { case (cText,cStr) =>
        val (emb,_,pa,_)=encodeRawPair(cText,"",false)
        EncodedSemanticConcept(emb,pa.map(_.pooledText),cStr)
      }
      val (_,uOpt,_,uaOpt)=if initialEncoded.neg.isDefined then (initialEncoded.pos,initialEncoded.neg,initialEncoded.posAdded,initialEncoded.negAdded) else encodeRawPair("",initialNegPrompt,true)
      (sgCfg,emptyChunk,concepts,uOpt.get,uaOpt)
    }
    var history: Option[Variable]=None
    var sharpenPrevious: Option[Tensor[Float]]=None
    val antiBurn=request.antiBurn.map(_.newState())
    val vectorscope=request.vectorscope.map(_.state(request.latentHookSeed.getOrElse(request.seed),sdxlConditioning.nonEmpty,
      request.latentHookBatchSize,request.latentHookBatchIndex))
    val detailMultipliers=request.detailDaemon.map(_.multipliers(fullSchedule.timesteps.size))
    val cadsState=request.cads.map(_.newState(request.latentHookSeed.getOrElse(request.seed)))
    val hyperTileState=request.hyperTile.map(_.newState(request.height/factor,request.width/factor,request.latentHookSeed.getOrElse(request.seed)))
    val tilingConfig=request.tiling.map(_.copy())
    val guidanceHookState=if request.apg.nonEmpty || request.mahiro || request.latentModifier.nonEmpty then Some(new GuidanceHookState(request.apg,request.mahiro,request.trainingSigmas,request.latentModifier,request.latentHookSeed.getOrElse(request.seed))) else None
    val chgRuntime=request.characteristicGuidance.map { chgCfg =>
      val ac=if request.trainingSigmas.nonEmpty then request.trainingSigmas.map(s => 1.0/(1.0+s.toDouble*s.toDouble)) else DynamicThresholding.defaultTrainingSigmas.map(s => 1.0/(1.0+s.toDouble*s.toDouble))
      new CharacteristicGuidance.State(chgCfg,schedule.timesteps.size,ac)
    }
    val pairCache=scala.collection.mutable.HashMap[(String,String),EncodedPair]((initialPrompt,initialNegPrompt) -> revInitialEncoded)
    if promptSched.size>1 || negSched.size>1 then
      for stepIdx <- 1 to schedule.timesteps.size do
        val key=(PromptSchedule.textAt(promptSched,stepIdx),PromptSchedule.textAt(negSched,stepIdx))
        if !pairCache.contains(key) then pairCache(key)=encodePair(key._1,key._2)
    val loadedIpAdapter=request.ipAdapter.filter(_.enabled).map { ipCfg =>
      val ctxDim=unet.midBlock.attention.blocks.head.attn2.toK.weight.data.shape.dims(1).toInt
      IpAdapter.load(ipCfg,ctxDim,request.controlNetPaths)
    }
    val loadedLLLite=request.controlLLLite.filter(_.enabled).flatMap { llCfg =>
      if sdxlConditioning.isEmpty then
        request.onWarning("ControlNet-LLLite is an SDXL feature; ignored")
        None
      else
        val (cImg,gain)=ControlLLLite.load(llCfg,request.width,request.height,None,request.controlNetPaths)
        Some((llCfg,cImg,gain))
    }
    val loadedFooocus=request.fooocusInpaint.filter(_.enabled).flatMap { fiCfg =>
      if sdxlConditioning.isEmpty || latentMask.isEmpty || nineChannelInput then
        request.onWarning("Fooocus Inpaint needs SDXL and an inpaint mask; ignored")
        None
      else
        val head=FooocusInpaint.loadHead(fiCfg,request.controlNetPaths)
        val maskedLat=source.flatMap(_.maskedImage).map { img =>
          val m=encodeVae(img,cancelled)
          vae.toDiffusionLatents(VaePosterior(Variable(m,requiresGrad=false)).mean).data
        }.orElse(clean.map { cl =>
          val clArr=cl.data.toFloatArray
          val mArr=latentMask.get.toFloatArray
          val hw=mArr.length
          val out=new Array[Float](clArr.length)
          for c <- 0 until latentChannels; p <- 0 until hw do
            out(c*hw+p)=clArr(c*hw+p)*(1.0f-mArr(p))
          Tensor.fromFloats(out,cl.data.shape)
        }).getOrElse(Tensor.fromFloats(new Array[Float](latentShape.numel.toInt),latentShape))
        val feed5=FooocusInpaint.buildFeed5Channel(latentMask.get,maskedLat)
        Some((fiCfg,head,feed5))
    }
    val refBankState=request.reference.filter(_.enabled).flatMap { refCfg =>
      if nineChannelInput then
        request.onWarning("Reference-only is not supported for 9-channel inpainting models; ignored")
        None
      else
        val refImg=ReferenceOnly.loadNormalizedImageTensor(refCfg.image,request.width,request.height,"Reference-only")
        val moments=encodeVae(refImg,cancelled)
        val refLat=vae.toDiffusionLatents(VaePosterior(Variable(moments,requiresGrad=false)).mean).data
        Some(new ReferenceOnly.BankState(refCfg,refLat))
    }
    val activeSag=request.sag.filter(_.enabled).flatMap { sagCfg =>
      if sdxlConditioning.nonEmpty || source.isDefined || latentSource.isDefined || nineChannelInput then
        request.onWarning("SAG is only supported for SD1.5 txt2img; ignored")
        None
      else if !Sag.SupportedSamplers.contains(samplerKind) then
        request.onWarning(s"SAG is not supported for sampler '$samplerKind'; ignored")
        None
      else Some(sagCfg)
    }
    val fusionPosSpec=if request.promptFusion then PromptFusion.parseSpec(effReqPrompt,schedule.timesteps.size) else None
    val fusionNegSpec=if request.promptFusion then PromptFusion.parseSpec(effReqNegPrompt,schedule.timesteps.size) else None
    val fusionControls: Option[(Option[(PromptFusion.FusionSpec,Vector[EncodedPair])],Option[(PromptFusion.FusionSpec,Vector[EncodedPair])])] =
      if fusionPosSpec.isEmpty && fusionNegSpec.isEmpty then None
      else
        val emptyChunk=encodeRawPair("","",false)._1.slice(1,0,77)
        val pCtrls=fusionPosSpec.map { sp =>
          val encs=sp.controlPrompts.map(cp => encodePair(cp,initialNegPrompt))
          val maxC=encs.map(_.pos.shape.dims(1)/77).max.toInt
          (sp,encs.map(e => e.copy(pos=ClipPromptEncoding.padTo(e.pos,maxC,emptyChunk,67108864L))))
        }
        val nCtrls=fusionNegSpec.map { sp =>
          val encs=sp.controlPrompts.map(cn => encodePair(initialPrompt,cn))
          val maxC=encs.flatMap(_.neg.map(_.shape.dims(1)/77)).foldLeft(1L)(math.max).toInt
          (sp,encs.map(e => e.copy(neg=e.neg.map(n => ClipPromptEncoding.padTo(n,maxC,emptyChunk,67108864L)))))
        }
        Some((pCtrls,nCtrls))
    val neutralStaticPlan: Option[(NeutralPrompt.Group,Vector[(EncodedPair,Vector[ComposableLora.Controller])],Double)] =
      val npPhi=request.neutralPrompt.map(_.cfgRescale).getOrElse(0.0)
      val tree=NeutralPrompt.parse(effReqPrompt)
      val lvs=NeutralPrompt.leaves(tree)
      val activeNp=(request.neutralPrompt.isDefined && NeutralPrompt.isActive(effReqPrompt,npPhi)) || (request.composableLora.exists(_.enabled) && lvs.size>1)
      if !activeNp then None
      else
        val emptyChunk=encodeRawPair("","",false)._1.slice(1,0,77)
        val rawBranches=lvs.map { lf =>
          val ctrls=if request.composableLora.exists(_.enabled) then ComposableLora.parseBranchControllers(lf.text) else Vector.empty
          val teScope=if ctrls.isEmpty then Map.empty[scalatorch.nn.LowRankAdapter,Float] else
            val mults=ComposableLora.evaluateBranchMultipliers(ctrls,-1,schedule.timesteps.size)
            request.namedLoraAdapters.flatMap { case (k,adps) => mults.get(k).map(w => adps.map(_ -> w)).getOrElse(Vector.empty) }
          val ep=scalatorch.nn.LinearAdapterScope.withMultipliers(teScope)(encodePair(lf.text,initialNegPrompt))
          (ep,ctrls)
        }
        val maxC=math.max(initialEncoded.pos.shape.dims(1)/77,rawBranches.map(_._1.pos.shape.dims(1)/77).max).toInt
        val aligned=rawBranches.map { case (ep,ctrls) => (ep.copy(pos=ClipPromptEncoding.padTo(ep.pos,maxC,emptyChunk,67108864L)),ctrls) }
        Some((tree,aligned,npPhi))
    val singleBranchClora=if neutralStaticPlan.isEmpty && request.composableLora.exists(_.enabled) then
      val c=ComposableLora.parseBranchControllers(effReqPrompt)
      if c.nonEmpty then Some(c) else None
    else None
    scalatorch.nn.CircularConvPadding.withConfig(tilingConfig) { HyperTile.withState(hyperTileState) {
    unetExecution.withPredictor(unet,conditional.data,unconditional.map(_.data),cancelled,positiveAdded,negativeAdded,
      request.controlNet.toVector++request.controlNets) { predict =>
      var i=0;var stopEarly=false
      while i<schedule.timesteps.size && !stopEarly do
        tilingConfig.foreach(_.updateForStep(i,schedule.timesteps.size))
        checkCancelled()
        def predictAt(sample: Variable,evalIndex: Int,controlStart: Double,controlEnd: Double,sigmaOverride: Option[Double]=None,timestepOverride: Option[Float]=None): Variable =
          val modelInputLatent=if sigmaOverride.isDefined then SamplerPrediction.scaleEulerInput(sigmaOverride.get,sample)
            else if dpm || ddim.nonEmpty then sample

            else if heunSolver.nonEmpty then heunSolver.get.scaleModelInput(evalIndex,sample)
            else scheduler.scaleModelInput(evalIndex,sample)
          val modelInput=if nineChannelInput then EulerImagePipeline.concatenateInpaintChannels(
            modelInputLatent.data,nineChannelMask.get,nineChannelMaskedLatents.get) else modelInputLatent.data
          val timestep=timestepOverride.getOrElse(schedule.timesteps(evalIndex))
          val time=Tensor.fromFloats(Array(timestep),Shape.from(Seq(1)))
          val scheduled=request.cfgSchedule.filter(_=>request.guidanceScale>1).map { cfg =>
            val index=fullSchedule.timesteps.indices.minBy(j=>math.abs(fullSchedule.timesteps(j)-timestep))
            cfg.at(index.toDouble/math.max(1,fullSchedule.timesteps.size-1))
          }
          val detail=request.detailDaemon.flatMap { settings =>
            val index=fullSchedule.timesteps.indices.minBy(j=>math.abs(fullSchedule.timesteps(j)-timestep))
            settings.adjustment(timestep,detailMultipliers.get(index)).map { value =>
              if dpm || ddim.nonEmpty then value.copy(timestep=math.rint(value.timestep.toDouble).toFloat) else value
            }
          }
          val cadsStep=cadsState.map { st =>
            val index=fullSchedule.timesteps.indices.minBy(j=>math.abs(fullSchedule.timesteps(j)-timestep))
            (st,index.toDouble/math.max(1,fullSchedule.timesteps.size-1))
          }
          val stepKey=(PromptSchedule.textAt(promptSched,i+1),PromptSchedule.textAt(negSched,i+1))
          val baseStepEncoded=pairCache(stepKey)
          val stepEncoded=fusionControls.fold(baseStepEncoded) { case (pCtrlsOpt,nCtrlsOpt) =>
            val pUpdated=pCtrlsOpt.fold(baseStepEncoded) { case (sp,encs) =>
              val interpPos=PromptFusion.interpolateTensor(encs.map(_.pos),i,schedule.timesteps.size,sp)
              val interpPa=baseStepEncoded.posAdded.map { pa =>
                val pPools=encs.flatMap(_.posAdded.map(_.pooledText))
                if pPools.size==encs.size then pa.copy(pooledText=PromptFusion.interpolateTensor(pPools,i,schedule.timesteps.size,sp)) else pa
              }
              baseStepEncoded.copy(pos=interpPos,posAdded=interpPa)
            }
            nCtrlsOpt.fold(pUpdated) { case (sp,encs) =>
              val nTensors=encs.flatMap(_.neg)
              if nTensors.size!=encs.size then pUpdated
              else
                val interpNeg=PromptFusion.interpolateTensor(nTensors,i,schedule.timesteps.size,sp)
                val interpNa=pUpdated.negAdded.map { na =>
                  val nPools=encs.flatMap(_.negAdded.map(_.pooledText))
                  if nPools.size==encs.size then na.copy(pooledText=PromptFusion.interpolateTensor(nPools,i,schedule.timesteps.size,sp)) else na
                }
                pUpdated.copy(neg=Some(interpNeg),negAdded=interpNa)
            }
          }
          val (rawCond,rawPosAdded)=(stepEncoded.pos,stepEncoded.posAdded)
          val (sgStepCond,effStepPosAdded)=sgBundle.filter(b => i>=b._1.warmup && b._3.nonEmpty).fold((rawCond,rawPosAdded)) {
            case (sgCfg,emptyChunk,concepts,uFall,uaFall) =>
              SemanticGuidance.applyToConditioning(rawCond,stepEncoded.neg.getOrElse(uFall),emptyChunk,concepts,sgCfg,rawPosAdded,stepEncoded.negAdded.orElse(uaFall),unet.additionTimeWidth)
          }
          val effStepCond=request.incantations.filter(ic => ic.hasT2i0 && ic.t2i0CorrectionStrength!=0.0 && i>=ic.t2i0StepStart && i<=ic.t2i0StepEnd).fold(sgStepCond) { ic =>
            val toks=ic.t2i0Tokens.split(",").map(_.trim).filter(_.nonEmpty).flatMap(_.toIntOption).toVector
            Incantations.cbs(sgStepCond,ic.t2i0CorrectionThreshold,ic.t2i0WindowSize,ic.t2i0CorrectionStrength,toks)
          }
          val (condOvr,posAddedOvr,uncondOvr,negAddedOvr)=if stepKey==(initialPrompt,initialNegPrompt) && (effStepCond eq rawCond) && fusionControls.isEmpty then (None,None,None,None) else
            (Some(effStepCond),effStepPosAdded,stepEncoded.neg,stepEncoded.negAdded)

          val tdStep=request.tokenDownsampling.map(td => (td,sdxlConditioning.nonEmpty,i+1,schedHiresSteps.getOrElse(schedBaseSteps)))
          val effScheduled=scheduled.orElse(request.incantations.filter(ic => ic.hasCfgInterval && request.guidanceScale>1f).map(ic => Incantations.cfgIntervalValue(ic,i,schedule.timesteps.size,request.guidanceScale.toDouble)))
          val segUncond=request.incantations.filter(ic => ic.hasSeg && i>=ic.segStartStep && i<=ic.segEndStep).map(ic => (ic.segBlurSigma,request.height,request.width))
          val effPag=request.pag.orElse(request.incantations.filter(ic => ic.hasPag && i>=ic.pagStartStep && i<=ic.pagEndStep).map(ic => PagConfig(scale=ic.pagScale,adaptiveScale=0.0,layers=Vector("mid"))))
          val chgStep=chgRuntime.map(st => (st,i))
          val stepNeutralPlan=neutralStaticPlan.map { case (tree,branches,phi) =>
            val bConds=branches.map { case (ep,ctrls) =>
              val scope=if ctrls.isEmpty then Map.empty[scalatorch.nn.LowRankAdapter,Float] else
                val mults=ComposableLora.evaluateBranchMultipliers(ctrls,i,schedule.timesteps.size)
                request.namedLoraAdapters.flatMap { case (k,adps) => mults.get(k).map(w => adps.map(_ -> w)).getOrElse(Vector.empty) }
              NeutralPrompt.BranchConditioning(ep.pos,ep.posAdded,ep.posSigns,scope)
            }
            NeutralPrompt.StepPlan(tree,bConds,phi)
          }
          val singleCloraScope=singleBranchClora.map { ctrls =>
            val mults=ComposableLora.evaluateBranchMultipliers(ctrls,i,schedule.timesteps.size)
            request.namedLoraAdapters.flatMap { case (k,adps) => mults.get(k).map(w => adps.map(_ -> w)).getOrElse(Vector.empty) }
          }.getOrElse(Map.empty)
          val curSigma=sigmaOverride.getOrElse(schedule.sigmas(evalIndex))
          val stepPct=i.toDouble/math.max(1,schedule.timesteps.size-1)
          val sagStepOpt=activeSag.map { sagCfg =>
            val effSig=if dpm || ddim.nonEmpty then curSigma else curSigma/math.hypot(1.0,curSigma)
            (sagCfg,effSig)
          }
          val refStepOpt=refBankState.map { bank =>
            val noisedRef=ReferenceOnly.noiseReferenceLatents(bank.refLatents,curSigma,request.seed+evalIndex)
            val scaledRef=if sigmaOverride.isDefined then SamplerPrediction.scaleEulerInput(sigmaOverride.get,Variable(noisedRef)).data
              else if dpm || ddim.nonEmpty then noisedRef
              else if heunSolver.nonEmpty then heunSolver.get.scaleModelInput(evalIndex,Variable(noisedRef)).data
              else scheduler.scaleModelInput(evalIndex,Variable(noisedRef)).data
            (bank,scaledRef)
          }
          val activeLLLiteOpt=loadedLLLite.map { case (llCfg,cImg,gain) => ControlLLLite.ActiveLLLite(llCfg,cImg,stepPct,gain) }
          val activeFooocusOpt=loadedFooocus.map { case (fiCfg,head,feed5) => FooocusInpaint.ActiveFooocusInpaint(fiCfg,head,feed5,stepPct) }
          def runPrediction()=scalatorch.nn.LinearAdapterScope.withMultipliers(singleCloraScope) {
            IpAdapter.withAdapter(loadedIpAdapter) {
              ControlLLLite.withLLLite(activeLLLiteOpt) {
                FooocusInpaint.withFooocus(activeFooocusOpt) {
                  Llul.withStep(request.llul,i+1,schedule.timesteps.size,request.width,request.height) {
                    AgentAttention.withStep(request.agentAttention,i,request.isHiresPass,request.height/factor,request.width/factor,request.seed) {
                      predict(modelInput,time,request.guidanceScale,controlStart,request.guidanceRescale,effScheduled,detail,controlEnd,request.dynamicThresholding,prediction,cadsStep,guidanceHookState,effPag,condOvr,posAddedOvr,uncondOvr,negAddedOvr,stepEncoded.posSigns,stepEncoded.negSigns,request.kohyaHrFix,tdStep,request.multiDiffusion,segUncond,chgStep,stepNeutralPlan,sagStepOpt,refStepOpt)
                    }
                  }
                }
              }
            }
          }
          val output=request.loraStepFilter.fold(runPrediction()) { filter =>
            scalatorch.nn.LinearAdapterScope.withFilter(item=>filter(i+1,schedule.timesteps.size,item._1))(runPrediction())
          }
          Variable(output,requiresGrad=false)
        val isTwoStage=heunSolver.nonEmpty || kdpm2Solver.nonEmpty || dpmSdeSolver.nonEmpty
        val heunStart=(2*i).toDouble/(2*schedule.timesteps.size)
        val midpoint=(2*i+1).toDouble/(2*schedule.timesteps.size)
        val heunEnd=(2*i+2).toDouble/(2*schedule.timesteps.size)
        val stepNoisy=latent.data
        val rawPredicted=predictAt(latent,i,if isTwoStage then heunStart else i.toDouble/schedule.timesteps.size,
          if isTwoStage then midpoint else (i+1).toDouble/schedule.timesteps.size)
        val rawDenoised=if dpm || ddim.nonEmpty then SamplerPrediction.toOriginalVp(schedule.sigmas(i),prediction,rawPredicted,latent)
          else SamplerPrediction.toOriginalEuler(schedule.sigmas(i),prediction,rawPredicted,latent)
        val (predicted,stepDenoised)=request.cdTuner match
          case None => (rawPredicted,rawDenoised.data)
          case Some(cdCfg) =>
            val shifted=CdTuner.applyStepShift(rawDenoised.data,cdCfg,sdxlConditioning.nonEmpty,i,schedule.timesteps.size,request.isHiresPass)
            val shiftedVar=Variable(shifted,requiresGrad=false)
            val newPred=if dpm || ddim.nonEmpty then SamplerPrediction.fromOriginalVp(schedule.sigmas(i),prediction,shiftedVar,latent,rawPredicted)
              else SamplerPrediction.fromOriginalEuler(schedule.sigmas(i),prediction,shiftedVar,latent,rawPredicted)
            (newPred,shifted)

        checkCancelled()
        val activeTrainingSigmas=if request.trainingSigmas.nonEmpty then request.trainingSigmas else DynamicThresholding.defaultTrainingSigmas
        def midTimestep(midSigma: Double): Float =
          SamplerPrediction.sigmaToTimestep(activeTrainingSigmas, midSigma)
        if kdpm2Solver.nonEmpty then
          val solver=kdpm2Solver.get
          val first=solver.predict(i,predicted,latent)
          latent=if first.requiresSecondEvaluation then
            checkCancelled()
            val next=predictAt(first.midpointSample,i,midpoint,heunEnd,Some(first.midpointSigma),Some(midTimestep(first.midpointSigma)))
            val z=if solver.ancestral && first.sigmaUp>0 then Some(Variable(randomNoise(),requiresGrad=false)) else None
            solver.correct(i,first,next,z).previous
          else first.midpointSample
        else if dpm2SaSolver.nonEmpty then
          latent=dpm2SaSolver.get.stepVp(i,predicted,latent).previous
        else if dpmSdeSolver.nonEmpty then
          val solver=dpmSdeSolver.get
          val z1=if schedule.sigmas(i+1)>0 then Variable(randomNoise(),requiresGrad=false) else Variable(Tensor.fromFloats(new Array[Float](latentShape.numel.toInt),latentShape),requiresGrad=false)
          val first=solver.predict(i,predicted,latent,z1)
          latent=if first.requiresSecondEvaluation then
            checkCancelled()
            val next=predictAt(first.midpointSample,i,midpoint,heunEnd,Some(first.midpointSigma),Some(midTimestep(first.midpointSigma)))
            val z2=Variable(randomNoise(),requiresGrad=false)
            solver.correct(i,first,next,z2).previous
          else first.midpointSample
        else if lmsSolver.nonEmpty then
          latent=lmsSolver.get.step(i,predicted,latent).previous
        else if dpmMultistepSdeSolver.nonEmpty then
          val z=Variable(randomNoise(),requiresGrad=false)
          latent=dpmMultistepSdeSolver.get.step(i,predicted,latent,z).previous
        else if uniPcSolver.nonEmpty then
          latent=uniPcSolver.get.step(i,predicted,latent).previous
        else if lcmSolver.nonEmpty then
          val z=if i+1<schedule.timesteps.size then Some(Variable(randomNoise(),requiresGrad=false)) else None
          latent=lcmSolver.get.step(startStepIdx+i,predicted,latent,z).previous
        else heunSolver match
          case Some(solver) =>
            val first=solver.predict(i,predicted,latent)
            latent=if first.requiresSecondEvaluation then
              checkCancelled()
              val next=predictAt(first.predictor,i+1,midpoint,heunEnd)
              solver.correct(i,first,next).previous
            else first.predictor
          case None =>
            dpmSolver match
              case Some(solver) =>
                val result=solver.step(i,predicted,latent,history)
                latent=result.previous
                history=Some(result.predictedOriginal)
              case None => ancestralSolver match
                case Some(solver) =>
                  val next=schedule.sigmas(i+1)
                  val noise=if next>0 then randomNoise()
                    else Tensor.fromFloats(new Array[Float](latentShape.numel.toInt),latentShape)
                  latent=sonarStepper.fold(solver.step(i,predicted,latent,Variable(noise,requiresGrad=false)).previous)(st => Variable(st.step(latent.data,predicted.data,schedule.sigmas(i),next,prediction,Some(noise)),requiresGrad=false))
                case None => latent=ddim match
                  case Some(solver) =>
                    val noise=Option.when(ddimEta>0)(Variable(randomNoise(),requiresGrad=false))
                    solver.step(i+fullSchedule.timesteps.size-schedule.timesteps.size,predicted,latent,ddimEta,noise)
                  case None => sonarStepper.fold(scheduler.step(i,predicted,latent).previous)(st => Variable(st.step(latent.data,predicted.data,schedule.sigmas(i),schedule.sigmas(i+1),prediction,None),requiresGrad=false))
        if !nineChannelInput then latentMask.foreach { mask =>
          val sigma=schedule.sigmas(i+1)
          val alpha=if dpm || ddim.nonEmpty then 1/math.hypot(1,sigma) else 1.0
          val preserved=clean.get*alpha.toFloat+Variable(initial,requiresGrad=false)*(sigma*alpha).toFloat
          val original=preserved.data.toFloatArray
          val preBlendLatent=latent.data; val generated=preBlendLatent.toFloatArray
          val repaint=mask.toFloatArray
          val plane=mask.shape.dims(2).toInt*mask.shape.dims(3).toInt
          latent=Variable(Tensor.fromFloats(Array.tabulate(generated.length) { j =>
            val m=repaint(j%plane)
            original(j)*(1f-m)+generated(j)*m
          },latent.data.shape),requiresGrad=false)
          request.softInpainting.foreach { siCfg =>
            latent=Variable(SoftInpainting.softInpaintBlend(preBlendLatent,preserved.data,source.get.mask.get,i,schedule.timesteps.size,siCfg),requiresGrad=false)
          }
        }
        vectorscope.foreach { settings =>
          latent=Variable(settings.step(latent.data,i+1,schedule.timesteps.size),requiresGrad=false)
        }
        request.latentMirroring.foreach { settings =>
          latent=Variable(settings.step(latent.data,i+1,schedule.timesteps.size),requiresGrad=false)
        }
        request.resharpen.foreach { settings =>
          val adjusted=settings.step(latent.data,sharpenPrevious,i+1,schedule.timesteps.size)
          sharpenPrevious=Some(adjusted)
          latent=Variable(adjusted,requiresGrad=false)
        }
        request.colorGrading.foreach { settings =>
          latent=Variable(settings.step(latent.data,i+1,schedule.timesteps.size,sdxlConditioning.nonEmpty),requiresGrad=false)
        }
        antiBurn.foreach { state =>
          val (result,stop)=state.step(latent.data,i+1,schedule.timesteps.size)
          latent=Variable(result,requiresGrad=false);stopEarly=stop
        }
        require(latent.data.toFloatArray.forall(_.isFinite),"denoising produced nonfinite latents")
        checkCancelled()
        request.cdTuner.filter(c => (c.sat!=0.0 || c.sat2!=0.0) && i==schedule.timesteps.size-1).foreach { cdCfg =>
          latent=Variable(CdTuner.applyStepShift(latent.data,cdCfg.copy(d1=0,d2=0,hd1=0,hd2=0,con1=0,con2=0,bri=0,col1=0,col2=0,col3=0,colors=""),sdxlConditioning.nonEmpty,i,schedule.timesteps.size,request.isHiresPass),requiresGrad=false)
        }
        request.onStepObservation.foreach(cb => cb(i+1,schedule.timesteps.size,stepNoisy,stepDenoised,latent.data))
        onProgress(DiffusionImageProgress(i+1,schedule.timesteps.size,schedule.timesteps(i),latent.data))
        i+=1
    }
    checkCancelled()
    if returnLatents then latent.data
    else
      tilingConfig.foreach(_.enableForVaeDecode())
      val image=decodeLatents(latent.data,cancelled)
      checkCancelled()
      source.flatMap(s => if !s.finalOverlay then None else s.mask.map(mask =>
        val generated=image.toFloatArray
        val original=s.originalImage.get.toFloatArray
        val repaint=mask.toFloatArray
        Tensor.fromFloats(Array.tabulate(generated.length) { i =>
          val pixel=i%(request.width*request.height)
          // Diffusers clamps decoded pixels before NUI composites the image.
          val decoded=math.max(-1f,math.min(1f,generated(i)))
          original(i)*(1f-repaint(pixel))+decoded*repaint(pixel)
        },image.shape)
      ).map(hardComp => request.softInpainting.fold(hardComp)(siCfg => SoftInpainting.compositePixels(s.originalImage.get,Tensor.fromFloats(image.toFloatArray.map(v => math.max(-1f,math.min(1f,v))),image.shape),s.mask.get,siCfg)))).getOrElse(image)
    }}
  }

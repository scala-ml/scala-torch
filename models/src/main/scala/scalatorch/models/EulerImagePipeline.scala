package scalatorch.models

import scalatorch.*
import scalatorch.autograd.{Autograd,Variable}
import java.util.concurrent.CancellationException

final case class DiffusionImageRequest(prompt: String,negativePrompt: String,width: Int,height: Int,seed: Long,
    guidanceScale: Float=7.5f,truncatePrompt: Boolean=false)
final case class DiffusionImageProgress(completedSteps: Int,totalSteps: Int,timestep: Float,latents: Tensor[Float])

/** Single-image CPU inference composition; explicit fast tokenizer and Euler schedule, no hidden checkpoint defaults. */
final class EulerImagePipeline(val tokenizer: ClipFastTokenizer,val textEncoder: ClipTextModel,
    val unet: ConditionalUNet2d,val vae: VaeModel,maxImagePixels: Long=16777216L,maxLatentElements: Long=16777216L):
  require(maxImagePixels>=0 && maxLatentElements>=0,"negative image pipeline allocation limit")
  require(tokenizer.vocabSize<=textEncoder.tokenEmbedding.weight.data.shape.dims.head &&
    tokenizer.maxLength<=textEncoder.positionEmbedding.weight.data.shape.dims.head,"pipeline tokenizer/text encoder mismatch")
  require(textEncoder.finalNorm.width==unet.midBlock.attention.blocks.head.attn2.toK.weight.data.shape.dims(1),
    "pipeline text context width mismatch")
  require(unet.convIn.weight.data.shape.dims(1)==vae.config.latentChannels &&
    unet.convOut.weight.data.shape.dims.head==vae.config.latentChannels,"pipeline latent channel mismatch")
  private val factor=1 << (vae.decoder.upBlocks.size-1)

  /** Noise is unscaled standard-normal data; supplied noise overrides seed. Guidance is enabled above one. */
  def generate(request: DiffusionImageRequest,schedule: EulerSchedule,
      noise: Option[Tensor[Float]]=None,onProgress: DiffusionImageProgress => Unit=_ => (),
      cancelled: () => Boolean=() => false): Tensor[Float] = Autograd.noGrad {
    def checkCancelled(): Unit =
      if cancelled() || Thread.currentThread().isInterrupted then throw new CancellationException("image generation cancelled")
    checkCancelled()
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
    def encode(text: String): Variable =
      checkCancelled()
      val ids=tokenizer.batch(Vector(text),truncate=request.truncatePrompt).ids
      checkCancelled()
      textEncoder(ids)
    val conditional=encode(request.prompt)
    val unconditional=if request.guidanceScale>1 then Some(encode(request.negativePrompt)) else None
    checkCancelled()
    val initial=noise.getOrElse {
      val random=new java.util.Random(request.seed)
      Tensor.fromFloats(Array.fill(latentShape.numel.toInt)(random.nextGaussian().toFloat),latentShape)
    }
    var latent=Variable(initial,requiresGrad=false)*schedule.initialNoiseSigma.toFloat
    val scheduler=EulerScheduler(schedule,maxElements=maxLatentElements)
    for i <- schedule.timesteps.indices do
      checkCancelled()
      val input=scheduler.scaleModelInput(i,latent)
      val time=Tensor.fromFloats(Array(schedule.timesteps(i)),Shape.from(Seq(1)))
      val positive=unet(input,time,conditional)
      checkCancelled()
      val predicted=unconditional match
        case None => positive
        case Some(context) =>
          val negative=unet(input,time,context)
          checkCancelled()
          ClassifierFreeGuidance.combine(negative,positive,request.guidanceScale)
      latent=scheduler.step(i,predicted,latent).previous
      require(latent.data.toFloatArray.forall(_.isFinite),"denoising produced nonfinite latents")
      checkCancelled()
      onProgress(DiffusionImageProgress(i+1,schedule.timesteps.size,schedule.timesteps(i),latent.data))
    checkCancelled()
    val image=vae.decodeDiffusionLatents(latent).data
    require(image.toFloatArray.forall(_.isFinite),"VAE produced nonfinite image values")
    checkCancelled()
    image
  }

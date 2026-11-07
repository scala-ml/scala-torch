package scalatorch.models

import scalatorch.Shape
import scalatorch.io.{SafeDType,SafeTensorSet,TensorInfo}

/** Structural signature and fixed component configs for SD 2.x OpenCLIP single-file checkpoints.
  *
  * The released SD2 single-file checkpoints carry the OpenCLIP ViT-H/14 text tower, which has 24
  * transformer layers, while the ported Hugging Face `stable-diffusion-2` text encoder keeps the
  * first 23 layers and ignores the extra `resblock.23` weights. Diffusers (NUI's reference
  * behavior) loads 23 layers, so this loader does the same and treats `resblock.23.*` as
  * documented dead weight; see https://github.com/Stability-AI/stablediffusion/issues/262. */
object SingleFileSd2Config:
  val clipConfig: ClipTextConfig = ClipTextConfig(49408,1024,4096,23,16,77,activation=ClipActivation.Gelu)
  val unetConfig: UNetConfig = UNetConfig(4,4,Vector(320,640,1280,1280),Vector(true,true,true,false),
    Vector(false,true,true,true),2,5,1024,32,1e-5,1,1,1280,true,0,1,stageHeads=Vector(5,10,20,20),linearProjection=true)
  private val unetInput = "model.diffusion_model.input_blocks.0.0.weight"
  private val clipToken = "cond_stage_model.model.token_embedding.weight"
  private val clipPosition = "cond_stage_model.model.positional_embedding"
  private val clipFinalNorm = "cond_stage_model.model.ln_final.weight"
  private val clipFirstQkv = "cond_stage_model.model.transformer.resblocks.0.attn.in_proj_weight"
  private val vaeInput = "first_stage_model.encoder.conv_in.weight"
  private val crossAttention = "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight"

  private val supportedRoots = Vector("model.diffusion_model.","cond_stage_model.model.","first_stage_model.")
  private val incompatibleRoots = Vector("model_ema.","conditioner.","model.diffusion_model_ema.",
    "cond_stage_model.transformer.")
  private val schedulerTensors=Set("alphas_cumprod","alphas_cumprod_prev","betas","log_one_minus_alphas_cumprod",
    "posterior_log_variance_clipped","posterior_mean_coef1","posterior_mean_coef2","posterior_variance",
    "sqrt_alphas_cumprod","sqrt_one_minus_alphas_cumprod","sqrt_recip_alphas_cumprod","sqrt_recipm1_alphas_cumprod")
  private val auxiliary=schedulerTensors++Set("model_ema.decay","model_ema.num_updates",
    "cond_stage_model.model.text_projection","cond_stage_model.model.logit_scale")

  /** True when the tensor inventory uses the SD2 OpenCLIP `cond_stage_model.model.*` layout. */
  private[models] def isSd2(tensors: Map[String,TensorInfo]): Boolean =
    tensors.keySet.exists(_.startsWith("cond_stage_model.model."))

  /** Rejects non-SD2, partial and SD1 checkpoints before component allocation begins. */
  def inspect(archive: SafeTensorSet): Unit =
    inspectTensors(archive.tensors)

  private[models] def inspectTensors(tensors: Map[String,TensorInfo]): Unit =
    val names = tensors.keySet
    incompatibleRoots.find(root => names.exists(name => name.startsWith(root) && !auxiliary(name))).foreach { root =>
      throw IllegalArgumentException(s"unsupported SD2 single-file checkpoint variant: $root")
    }
    val unknown = names.filterNot(name => auxiliary(name) || supportedRoots.exists(name.startsWith))
    require(unknown.isEmpty,
      s"unsupported tensors in SD2 single-file checkpoint: ${unknown.toVector.sorted.take(8).mkString(", ")}")

    def requireShape(name: String, expected: Seq[Long]): Unit =
      val info = tensors.getOrElse(name,throw IllegalArgumentException(s"missing SD2 checkpoint tensor: $name"))
      require(info.shape == Shape.from(expected),s"unsupported SD2 checkpoint shape for $name: ${info.shape}")

    val input = tensors.getOrElse(unetInput,
      throw IllegalArgumentException(s"missing SD2 checkpoint tensor: $unetInput"))
    require(Set(Shape.from(Seq(320L,4L,3L,3L)),Shape.from(Seq(320L,9L,3L,3L)))(input.shape),
      s"unsupported SD2 checkpoint shape for $unetInput: ${input.shape}")
    requireShape(clipToken,Seq(49408L,1024L))
    requireShape(clipPosition,Seq(77L,1024L))
    requireShape(clipFinalNorm,Seq(1024L))
    requireShape(clipFirstQkv,Seq(3072L,1024L))
    requireShape(vaeInput,Seq(128L,3L,3L,3L))
    requireShape(crossAttention,Seq(320L,1024L))
    schedulerTensors.intersect(names).foreach(name => requireShape(name,Seq(1000L)))
    Set("model_ema.decay","model_ema.num_updates").intersect(names).foreach(name => requireShape(name,Seq.empty))

    val layer="cond_stage_model.model.transformer.resblocks.(\\d+)\\.(.*)".r
    val depths=names.iterator.collect { case layer(index,_) => index.toInt }.toSet
    require(depths.nonEmpty,"missing SD2 OpenCLIP transformer tensors")
    require(depths.min==0 && depths.max<=23,
      s"unsupported SD2 OpenCLIP transformer depth: ${depths.min}..${depths.max}")
    (0 to 22).foreach(i => require(depths(i),s"missing SD2 OpenCLIP transformer layer: $i"))

    val unsupportedClip=names.filter(name => name.startsWith("cond_stage_model.") &&
      !name.startsWith("cond_stage_model.model."))
    require(unsupportedClip.isEmpty,
      s"unsupported SD2 text encoder tensors: ${unsupportedClip.toVector.sorted.take(8).mkString(", ")}")
    val unsupportedTypes=tensors.iterator.collect {
      case (name,info) if !(Set(SafeDType.F16,SafeDType.BF16,SafeDType.F32)(info.dtype) ||
          (name=="model_ema.num_updates" && info.dtype==SafeDType.I32)) => name
    }.toVector.sorted
    require(unsupportedTypes.isEmpty,
      s"unsupported SD2 tensor dtypes: ${unsupportedTypes.take(8).mkString(", ")}")

    val unet = names.count(_.startsWith("model.diffusion_model."))
    val clip = names.count(_.startsWith("cond_stage_model.model."))
    val vae = names.count(_.startsWith("first_stage_model."))
    require(unet > 1 && clip > 1 && vae > 1,"checkpoint must contain U-Net, CLIP and VAE weights")

  private[models] def unetConfigFor(archive: SafeTensorSet): UNetConfig =
    unetConfigForTensors(archive.tensors)

  private[models] def unetConfigForTensors(tensors: Map[String,TensorInfo]): UNetConfig =
    inspectTensors(tensors)
    unetConfig.copy(inChannels = tensors(unetInput).shape.dims(1).toInt)

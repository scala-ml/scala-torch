package scalatorch.models

import scalatorch.Shape
import scalatorch.io.{SafeDType,SafeTensorSet,TensorInfo}

/** Structural signature and fixed component configs for conventional SD 1.x LDM checkpoints. */
object SingleFileSd1Config:
  val vaeConfig: VaeConfig = VaeConfig(3,3,4,Vector(128,256,512,512),2,32,sampleHeight=512,sampleWidth=512)
  val unetConfig: UNetConfig = UNetConfig(4,4,Vector(320,640,1280,1280),Vector(true,true,true,false),
    Vector(false,true,true,true),2,8,768,32,1e-5,1,1,1280,true,0,1)
  private val unetInput = "model.diffusion_model.input_blocks.0.0.weight"
  private val clipToken = "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"
  private val clipPosition = "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight"
  private val vaeInput = "first_stage_model.encoder.conv_in.weight"
  private val crossAttention = "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight"

  private val supportedRoots = Vector("model.diffusion_model.", "cond_stage_model.", "first_stage_model.")
  private val incompatibleRoots = Vector("model_ema.", "conditioner.", "model.diffusion_model_ema.")
  private val schedulerTensors=Set("alphas_cumprod","alphas_cumprod_prev","betas","log_one_minus_alphas_cumprod",
    "posterior_log_variance_clipped","posterior_mean_coef1","posterior_mean_coef2","posterior_variance",
    "sqrt_alphas_cumprod","sqrt_one_minus_alphas_cumprod","sqrt_recip_alphas_cumprod","sqrt_recipm1_alphas_cumprod")
  private val auxiliary=schedulerTensors++Set("model_ema.decay","model_ema.num_updates")

  /** Rejects non-SD1, partial and mixed-family checkpoints before component allocation begins. */
  def inspect(archive: SafeTensorSet): Unit =
    inspectTensors(archive.tensors)

  private[models] def inspectTensors(tensors: Map[String,TensorInfo]): Unit =
    val names = tensors.keySet
    incompatibleRoots.find(root => names.exists(name => name.startsWith(root) && !auxiliary(name))).foreach { root =>
      throw IllegalArgumentException(s"unsupported SD1 single-file checkpoint variant: $root")
    }
    val unknown = names.filterNot(name => auxiliary(name) || supportedRoots.exists(name.startsWith))
    require(unknown.isEmpty,
      s"unsupported tensors in SD1 single-file checkpoint: ${unknown.toVector.sorted.take(8).mkString(", ")}")

    def requireShape(name: String, expected: Seq[Long]): Unit =
      val info = tensors.getOrElse(name,throw IllegalArgumentException(s"missing SD1 checkpoint tensor: $name"))
      require(info.shape == Shape.from(expected),s"unsupported SD1 checkpoint shape for $name: ${info.shape}")

    val input = tensors.getOrElse(unetInput,
      throw IllegalArgumentException(s"missing SD1 checkpoint tensor: $unetInput"))
    require(Set(Shape.from(Seq(320L,4L,3L,3L)),Shape.from(Seq(320L,9L,3L,3L)))(input.shape),
      s"unsupported SD1 checkpoint shape for $unetInput: ${input.shape}")
    requireShape(clipToken,Seq(49408L,768L))
    requireShape(clipPosition,Seq(77L,768L))
    requireShape(vaeInput,Seq(128L,3L,3L,3L))
    requireShape(crossAttention,Seq(320L,768L))
    schedulerTensors.intersect(names).foreach(name => requireShape(name,Seq(1000L)))
    Set("model_ema.decay","model_ema.num_updates").intersect(names).foreach(name => requireShape(name,Seq.empty))

    val unsupportedClip=names.filter(name => name.startsWith("cond_stage_model.") &&
      !name.startsWith("cond_stage_model.transformer.text_model."))
    require(unsupportedClip.isEmpty,
      s"unsupported SD1 text encoder tensors: ${unsupportedClip.toVector.sorted.take(8).mkString(", ")}")
    val positionIds="cond_stage_model.transformer.text_model.embeddings.position_ids"
    val unsupportedTypes=tensors.iterator.collect {
      case (name,info) if !(Set(SafeDType.F16,SafeDType.BF16,SafeDType.F32)(info.dtype) ||
          (name==positionIds && info.dtype==SafeDType.I64) ||
          (name=="model_ema.num_updates" && info.dtype==SafeDType.I32)) => name
    }.toVector.sorted
    require(unsupportedTypes.isEmpty,
      s"unsupported SD1 tensor dtypes: ${unsupportedTypes.take(8).mkString(", ")}")

    val unet = names.count(_.startsWith("model.diffusion_model."))
    val clip = names.count(_.startsWith("cond_stage_model."))
    val vae = names.count(_.startsWith("first_stage_model."))
    require(unet > 1 && clip > 1 && vae > 1,"checkpoint must contain U-Net, CLIP and VAE weights")

  private[models] def unetConfigFor(archive: SafeTensorSet): UNetConfig =
    unetConfigForTensors(archive.tensors)

  private[models] def unetConfigForTensors(tensors: Map[String,TensorInfo]): UNetConfig =
    inspectTensors(tensors)
    unetConfig.copy(inChannels = tensors(unetInput).shape.dims(1).toInt)


package scalatorch.models

import scalatorch.Shape
import scalatorch.io.{SafeDType,SafeTensorSet,TensorInfo}

/** Structural signature and fixed component configs for SDXL 1.0 base single-file checkpoints.
  *
  * Released SDXL base checkpoints store:
  *   - CLIP-L (`conditioner.embedders.0.transformer.text_model.*`, 12 layers, width 768)
  *   - OpenCLIP ViT-bigG (`conditioner.embedders.1.model.*`, 32 layers, width 1280, transposed
  *     `text_projection` 1280x1280)
  *   - 3-stage U-Net (`model.diffusion_model.*`, channels 320/640/1280, transformer depths 1/2/10,
  *     heads 5/10/20, context width 2048, `label_emb` 2816 -> 1280)
  *   - SDXL VAE (`first_stage_model.*`, scaling factor 0.13025, 1024px sample size)
  */
object SingleFileSdxlConfig:
  val clipConfig: ClipTextConfig = ClipTextConfig(49408,768,3072,12,12,77)
  val clip2Config: ClipTextConfig = ClipTextConfig(49408,1280,5120,32,20,77,
    activation=ClipActivation.Gelu,projectionWidth=Some(1280))
  val vaeConfig: VaeConfig = VaeConfig(3,3,4,Vector(128,256,512,512),2,32,
    scalingFactor=0.13025,sampleHeight=1024,sampleWidth=1024)
  val unetConfig: UNetConfig = UNetConfig(4,4,Vector(320,640,1280),Vector(false,true,true),
    Vector(true,true,false),2,5,2048,32,1e-5,1,1,1280,true,0,1,
    stageHeads=Vector(5,10,20),stageTransformerLayers=Vector(1,2,10),
    linearProjection=true,additionTimeWidth=Some(256),additionInputWidth=Some(2816))

  private val unetInput = "model.diffusion_model.input_blocks.0.0.weight"
  private val labelEmb0 = "model.diffusion_model.label_emb.0.0.weight"
  private val labelEmb2 = "model.diffusion_model.label_emb.0.2.weight"
  private val crossAttention = "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight"
  private val clipLToken = "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight"
  private val clipLPosition = "conditioner.embedders.0.transformer.text_model.embeddings.position_embedding.weight"
  private val clipGToken = "conditioner.embedders.1.model.token_embedding.weight"
  private val clipGPosition = "conditioner.embedders.1.model.positional_embedding"
  private val clipGFinalNorm = "conditioner.embedders.1.model.ln_final.weight"
  private val clipGProjection = "conditioner.embedders.1.model.text_projection"
  private val clipGFirstQkv = "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight"
  private val vaeInput = "first_stage_model.encoder.conv_in.weight"

  private val supportedRoots = Vector(
    "model.diffusion_model.",
    "conditioner.embedders.0.transformer.",
    "conditioner.embedders.1.model.",
    "first_stage_model.")
  private val incompatibleRoots = Vector(
    "model_ema.",
    "model.diffusion_model_ema.",
    "cond_stage_model.",
    "conditioner.embedders.0.model.")
  private val schedulerTensors = Set("alphas_cumprod","alphas_cumprod_prev","betas","log_one_minus_alphas_cumprod",
    "posterior_log_variance_clipped","posterior_mean_coef1","posterior_mean_coef2","posterior_variance",
    "sqrt_alphas_cumprod","sqrt_one_minus_alphas_cumprod","sqrt_recip_alphas_cumprod","sqrt_recipm1_alphas_cumprod")
  private val auxiliary = schedulerTensors ++ Set("model_ema.decay","model_ema.num_updates",
    "conditioner.embedders.1.model.logit_scale")

  /** True when the tensor inventory uses the SDXL `conditioner.embedders.*` layout. */
  private[models] def isSdxl(tensors: Map[String,TensorInfo]): Boolean =
    tensors.keySet.exists(_.startsWith("conditioner.embedders."))

  /** Rejects non-SDXL, refiner, partial and mixed-family checkpoints before component allocation begins. */
  def inspect(archive: SafeTensorSet): Unit =
    inspectTensors(archive.tensors)

  private[models] def inspectTensors(tensors: Map[String,TensorInfo]): Unit =
    val names = tensors.keySet
    incompatibleRoots.find(root => names.exists(name => name.startsWith(root) && !auxiliary(name))).foreach { root =>
      throw IllegalArgumentException(s"unsupported SDXL single-file checkpoint variant: $root")
    }
    val unknown = names.filterNot(name => auxiliary(name) || supportedRoots.exists(name.startsWith))
    require(unknown.isEmpty,
      s"unsupported tensors in SDXL single-file checkpoint: ${unknown.toVector.sorted.take(8).mkString(", ")}")

    def requireShape(name: String, expected: Seq[Long]): Unit =
      val info = tensors.getOrElse(name,throw IllegalArgumentException(s"missing SDXL checkpoint tensor: $name"))
      require(info.shape == Shape.from(expected),s"unsupported SDXL checkpoint shape for $name: ${info.shape}")

    val input = tensors.getOrElse(unetInput,
      throw IllegalArgumentException(s"missing SDXL checkpoint tensor: $unetInput"))
    require(Set(Shape.from(Seq(320L,4L,3L,3L)),Shape.from(Seq(320L,9L,3L,3L)))(input.shape),
      s"unsupported SDXL checkpoint shape for $unetInput: ${input.shape}")
    requireShape(labelEmb0,Seq(1280L,2816L))
    requireShape(labelEmb2,Seq(1280L,1280L))
    requireShape(crossAttention,Seq(640L,2048L))
    requireShape(clipLToken,Seq(49408L,768L))
    requireShape(clipLPosition,Seq(77L,768L))
    requireShape(clipGToken,Seq(49408L,1280L))
    requireShape(clipGPosition,Seq(77L,1280L))
    requireShape(clipGFinalNorm,Seq(1280L))
    requireShape(clipGProjection,Seq(1280L,1280L))
    requireShape(clipGFirstQkv,Seq(3840L,1280L))
    requireShape(vaeInput,Seq(128L,3L,3L,3L))
    schedulerTensors.intersect(names).foreach(name => requireShape(name,Seq(1000L)))
    Set("model_ema.decay","model_ema.num_updates").intersect(names).foreach(name => requireShape(name,Seq.empty))

    val unetInputBlock = "model\\.diffusion_model\\.input_blocks\\.(\\d+)\\..+".r
    val maxInputBlock = names.iterator.collect { case unetInputBlock(index) => index.toInt }.maxOption.getOrElse(0)
    require(maxInputBlock <= 8,s"unsupported SDXL U-Net input block index: $maxInputBlock")
    val unetOutputBlock = "model\\.diffusion_model\\.output_blocks\\.(\\d+)\\..+".r
    val maxOutputBlock = names.iterator.collect { case unetOutputBlock(index) => index.toInt }.maxOption.getOrElse(0)
    require(maxOutputBlock <= 8,s"unsupported SDXL U-Net output block index: $maxOutputBlock")

    val clipLLayer = "conditioner\\.embedders\\.0\\.transformer\\.text_model\\.encoder\\.layers\\.(\\d+)\\..+".r
    val clipLDepths = names.iterator.collect { case clipLLayer(index) => index.toInt }.toSet
    require(clipLDepths.nonEmpty,"missing SDXL CLIP-L transformer tensors")
    require(clipLDepths.min == 0 && clipLDepths.max == 11,
      s"unsupported SDXL CLIP-L transformer depth: ${clipLDepths.min}..${clipLDepths.max}")
    (0 to 11).foreach(i => require(clipLDepths(i),s"missing SDXL CLIP-L transformer layer: $i"))

    val clipGLayer = "conditioner\\.embedders\\.1\\.model\\.transformer\\.resblocks\\.(\\d+)\\..+".r
    val clipGDepths = names.iterator.collect { case clipGLayer(index) => index.toInt }.toSet
    require(clipGDepths.nonEmpty,"missing SDXL CLIP-G transformer tensors")
    require(clipGDepths.min == 0 && clipGDepths.max == 31,
      s"unsupported SDXL CLIP-G transformer depth: ${clipGDepths.min}..${clipGDepths.max}")
    (0 to 31).foreach(i => require(clipGDepths(i),s"missing SDXL CLIP-G transformer layer: $i"))

    val unsupportedClipL = names.filter(name => name.startsWith("conditioner.embedders.0.") &&
      !name.startsWith("conditioner.embedders.0.transformer.text_model."))
    require(unsupportedClipL.isEmpty,
      s"unsupported SDXL CLIP-L tensors: ${unsupportedClipL.toVector.sorted.take(8).mkString(", ")}")
    val positionIds = "conditioner.embedders.0.transformer.text_model.embeddings.position_ids"
    val unsupportedTypes = tensors.iterator.collect {
      case (name,info) if !(Set(SafeDType.F16,SafeDType.BF16,SafeDType.F32)(info.dtype) ||
          (name == positionIds && info.dtype == SafeDType.I64) ||
          (name == "model_ema.num_updates" && info.dtype == SafeDType.I32)) => name
    }.toVector.sorted
    require(unsupportedTypes.isEmpty,
      s"unsupported SDXL tensor dtypes: ${unsupportedTypes.take(8).mkString(", ")}")

    val unet = names.count(_.startsWith("model.diffusion_model."))
    val clipL = names.count(_.startsWith("conditioner.embedders.0.transformer."))
    val clipG = names.count(_.startsWith("conditioner.embedders.1.model."))
    val vae = names.count(_.startsWith("first_stage_model."))
    require(unet > 1 && clipL > 1 && clipG > 1 && vae > 1,
      "SDXL checkpoint must contain U-Net, CLIP-L, CLIP-G and VAE weights")

  private[models] def unetConfigFor(archive: SafeTensorSet): UNetConfig =
    unetConfigForTensors(archive.tensors)

  private[models] def unetConfigForTensors(tensors: Map[String,TensorInfo]): UNetConfig =
    inspectTensors(tensors)
    unetConfig.copy(inChannels = tensors(unetInput).shape.dims(1).toInt)

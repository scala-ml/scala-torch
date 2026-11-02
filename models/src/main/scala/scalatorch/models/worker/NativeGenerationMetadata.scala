package scalatorch.models.worker

import scalatorch.models.{DiffusionPrediction,SdxlImageSize}

/** NUI retains string infotext locally; execution overrides must still be interpreted. */
private[worker] object NativeGenerationMetadata:
  import NativeJson.*
  def prediction(value: Option[NativeJson],configured: DiffusionPrediction): DiffusionPrediction =
    val fields=value match
      case None => Map.empty[String,NativeJson]
      case Some(Obj(items)) => items
      case _ => throw IllegalArgumentException("extra must be an object of strings")
    require(fields.values.forall(_.isInstanceOf[Str]),"extra values must be strings")
    // Java Gaussian noise is not PyTorch CPU/CUDA or NV Philox noise. Do not claim
    // seed compatibility when the caller explicitly selects one of those algorithms.
    if fields.contains("rng") then
      throw new NativeBackendError("unsupported","explicit rng selection is not implemented")
    fields.get("prediction_type").map(_.asInstanceOf[Str].value.trim).getOrElse("") match
      case "" => configured
      case "epsilon" => DiffusionPrediction.Epsilon
      case "v_prediction" => DiffusionPrediction.VPrediction
      case "sample" => DiffusionPrediction.Sample
      case name => throw IllegalArgumentException(s"unsupported prediction_type: $name")

  private val sizeKeys=Set("sdfx_sdxl_original_height","sdfx_sdxl_original_width","sdfx_sdxl_crop_top","sdfx_sdxl_crop_left")
  def sdxlSize(value: Option[NativeJson],isSdxl: Boolean): Option[SdxlImageSize] =
    val fields=value match
      case None => Map.empty[String,NativeJson]
      case Some(Obj(items)) => items
      case _ => throw IllegalArgumentException("extra must be an object of strings")
    val selected=fields.keySet intersect sizeKeys
    if selected.isEmpty then None
    else
      if !isSdxl then throw new NativeBackendError("unsupported","SDXL size metadata requires an SDXL checkpoint")
      val height=fields.get("sdfx_sdxl_original_height").map(_.asInstanceOf[Str].value.toInt)
      val width=fields.get("sdfx_sdxl_original_width").map(_.asInstanceOf[Str].value.toInt)
      require(height.isDefined && width.isDefined,"SDXL original height and width must be supplied together")
      Some(SdxlImageSize(height.get,width.get,
        fields.get("sdfx_sdxl_crop_top").fold(0)(_.asInstanceOf[Str].value.toInt),
        fields.get("sdfx_sdxl_crop_left").fold(0)(_.asInstanceOf[Str].value.toInt)))

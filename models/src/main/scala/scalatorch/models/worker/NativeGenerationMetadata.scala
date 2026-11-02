package scalatorch.models.worker

import scalatorch.models.DiffusionPrediction

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

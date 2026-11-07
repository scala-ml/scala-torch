package scalatorch.models.worker

import scalatorch.models.DiffusionPrediction

class NativeGenerationMetadataSuite extends munit.FunSuite:
  import NativeJson.*
  test("NUI string metadata does not change prediction, but explicit overrides do") {
    val metadata=Obj(Map("Model keyword settings"->Str("{\"placement\":\"keyword prompt\"}")))
    assertEquals(NativeGenerationMetadata.prediction(Some(metadata),DiffusionPrediction.VPrediction),DiffusionPrediction.VPrediction)
    for (name,expected) <- Vector("epsilon"->DiffusionPrediction.Epsilon,
        "v_prediction"->DiffusionPrediction.VPrediction,"sample"->DiffusionPrediction.Sample) do
      assertEquals(NativeGenerationMetadata.prediction(Some(Obj(Map("prediction_type"->Str(name)))),DiffusionPrediction.Epsilon),expected)
    assertEquals(NativeGenerationMetadata.prediction(Some(Obj(Map("prediction_type"->Str(" ")))),DiffusionPrediction.Sample),DiffusionPrediction.Sample)
  }
  test("metadata validates types and never silently ignores execution settings") {
    intercept[IllegalArgumentException](NativeGenerationMetadata.prediction(Some(Arr(Vector.empty)),DiffusionPrediction.Epsilon))
    intercept[IllegalArgumentException](NativeGenerationMetadata.prediction(Some(Obj(Map("key"->Number(1)))),DiffusionPrediction.Epsilon))
    intercept[IllegalArgumentException](NativeGenerationMetadata.prediction(Some(Obj(Map("prediction_type"->Str("typo")))),DiffusionPrediction.Epsilon))
    assertEquals(NativeGenerationMetadata.prediction(Some(Obj(Map("rng"->Str("cpu")))),DiffusionPrediction.Epsilon),DiffusionPrediction.Epsilon)
    for name <- Vector("gpu","nv") do
      intercept[NativeBackendError](NativeGenerationMetadata.prediction(Some(Obj(Map("rng"->Str(name)))),DiffusionPrediction.Epsilon))
  }
  test("GPU RNG request falls back to CPU only when CUDA is unavailable") {
    val gpu=Some(Obj(Map("rng"->Str("gpu"))))
    assert(NativeGenerationMetadata.torchCpuNoise(gpu,cudaAvailable=false))
    assertEquals(NativeGenerationMetadata.prediction(gpu,DiffusionPrediction.Epsilon,cudaAvailable=false),DiffusionPrediction.Epsilon)
    intercept[NativeBackendError](NativeGenerationMetadata.torchCpuNoise(gpu,cudaAvailable=true))
  }

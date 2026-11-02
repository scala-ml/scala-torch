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
    for name <- Vector("cpu","gpu","nv") do
      intercept[NativeBackendError](NativeGenerationMetadata.prediction(Some(Obj(Map("rng"->Str(name)))),DiffusionPrediction.Epsilon))
  }

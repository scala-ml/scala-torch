package scalatorch.models.worker

class NativeGenerationDefaultsSuite extends munit.FunSuite:
  import NativeJson.*
  test("NUI inactive options accept defaults but reject active or unknown nested options") {
    val defaults=NativeGenerationDefaults.generation
    assertEquals(NativeGenerationDefaults.strip(defaults.updated("prompt",Str("cat")),defaults),Map("prompt"->Str("cat")))
    for (key,value) <- Vector("batch_size"->Number(2),"hires"->Obj(Map("enabled"->Bool(true))),
        "hires"->Obj(Map("typo"->Bool(false))),"loras"->Arr(Vector(Str("lora"))),"eta"->Number(0)) do
      intercept[NativeBackendError](NativeGenerationDefaults.strip(Map(key->value),defaults))
    assertEquals(NativeGenerationDefaults.strip(Map("hires"->Obj(Map("enabled"->Bool(false)))),defaults),Map.empty)
    assertEquals(NativeGenerationDefaults.strip(NativeGenerationDefaults.envelope,NativeGenerationDefaults.envelope),Map.empty)
  }

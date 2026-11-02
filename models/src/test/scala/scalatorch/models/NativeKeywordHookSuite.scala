package scalatorch.models.worker

import java.nio.file.Files
import scalatorch.models.ModelKeyword

class NativeKeywordHookSuite extends munit.FunSuite:
  import NativeJson.*
  test("Diffusers hook looks up full checkpoint hash and plans per-image keywords") {
    val directory=Files.createTempDirectory("native-keyword")
    val custom=directory.resolve("custom-mappings.txt")
    try
      val hash="a"*64
      Files.writeString(custom,s"$hash,red|blue,tiny\n")
      val hooks=Some(Obj(Map("model_keyword"->Obj(Map("_mapping_dir"->Str(directory.toString),
        "_base_prompt"->Str("cat"),"model_choice"->Str("iterate"))))))
      val plan=NativeKeywordHook.plan(hooks,"different",hash,"tiny",2,42,ModelKeyword.State(),() => ()).get
      assertEquals(plan.prompts,Vector("red cat","blue cat"))
      assertEquals(plan.nextState,ModelKeyword.State(2,0))
      val missing=NativeKeywordHook.plan(hooks,"different","b"*64,"tiny",1,42,ModelKeyword.State(),() => ()).get
      assertEquals(missing.prompts,Vector("cat"))
    finally
      Files.delete(custom);Files.delete(directory)
  }
  test("unknown active hooks and selected unsupported networks fail instead of being dropped") {
    def run(config: NativeJson)=NativeKeywordHook.plan(Some(config),"cat","a"*64,"tiny",1,42,ModelKeyword.State(),() => ())
    assertEquals(run(Obj(Map.empty)),None)
    assertEquals(run(Obj(Map("unknown"->Bool(false)))),None)
    intercept[NativeBackendError](run(Obj(Map("unknown"->Obj(Map.empty)))))
    for field <- Vector("embedding","lora","_lora_path") do
      intercept[NativeBackendError](run(Obj(Map("model_keyword"->Obj(Map(field->Str("selected")))))))
    intercept[IllegalArgumentException](run(Obj(Map("model_keyword"->Obj(Map("typo"->Bool(false)))))))
  }

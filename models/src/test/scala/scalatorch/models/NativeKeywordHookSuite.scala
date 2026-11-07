package scalatorch.models.worker

import java.nio.file.Files
import scalatorch.models.{KeywordMappings,ModelKeyword}

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
    val hiresTweaks=Obj(Map("hires_fix_tweaks"->Obj(Map("enabled"->Bool(true)))))
    intercept[NativeBackendError](run(hiresTweaks))
    intercept[IllegalArgumentException](run(Obj(Map("model_keyword"->Obj(Map("typo"->Bool(false)))))))
  }
  test("selected embedding and LoRA use catalog keywords and retain independent iteration") {
    val dir=Files.createTempDirectory("native-keyword-selected")
    val lora=dir.resolve("actual.safetensors")
    val catalog=dir.resolve("lora-keyword.txt")
    val custom=dir.resolve("lora-keyword-user.txt")
    try
      Files.write(lora,Array[Byte](1,2,3))
      val hash=KeywordMappings.legacyHash(lora)
      Files.writeString(catalog,s"$hash,wrong,alias.safetensors\n")
      Files.writeString(custom,s"$hash,first|second,alias.safetensors\n")
      val hooks=Some(Obj(Map("model_keyword"->Obj(Map("_mapping_dir"->Str(dir.toString),
        "_lora_path"->Str(lora.toString),"lora"->Str("alias"),"lora_name"->Str("sub/alias.safetensors"),
        "lora_choice"->Str("< iterate >"),"embedding"->Str("myembedding"),"multiplier"->Number(0.5))))))
      val plan=NativeKeywordHook.plan(hooks,"cat","a"*64,"tiny",2,42,ModelKeyword.State(),() => ()).get
      assertEquals(plan.prompts,Vector("<lora:alias:0.5> myembedding first cat","<lora:alias:0.5> myembedding second cat"))
      assertEquals(plan.nextState,ModelKeyword.State(0,2))
    finally
      Files.deleteIfExists(custom);Files.deleteIfExists(catalog);Files.deleteIfExists(lora);Files.delete(dir)
  }

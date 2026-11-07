package scalatorch.models.worker

import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files,Path}
import NativeJson.*
import scalatorch.models.ClipFastTokenizer

class NativeTokenizerInspectionSuite extends munit.FunSuite:
  private def obj(v: NativeJson)=v.asInstanceOf[Obj].fields
  private def arr(v: NativeJson)=v.asInstanceOf[Arr].values
  test("added embedding token IDs decode with NUI spacing") {
    val directory=sys.props.get("scalatorch.test.models")
    assume(directory.nonEmpty,"tiny SD fixture required")
    val tokenizer=ClipFastTokenizer.loadDirectory(Path.of(directory.get).resolve("tiny-sd-pipe/tokenizer"))
      .withAddedTokens(Vector("sks"->2))
    val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    val result=NativeTokenizerInspection.execute(Map("input_mode"->Str("ids"),"text"->Str("68,1000,1001,434")),
      Vector(("tokenizer","",tokenizer)),context)
    assertEquals(obj(arr(result("streams")).head)("decoded"),Str("c sks sks_1 at"))
  }
  test("native tokenizer inspection follows NUI reference and loaded model lifecycle") {
    val directory=sys.props.get("scalatorch.test.models")
    assume(directory.nonEmpty,"tiny SD1 and SDXL fixtures required")
    val stream=getClass.getResourceAsStream("/nui-tokenize-reference.json")
    val bytes=try stream.readAllBytes() finally stream.close()
    assertEquals(java.util.HexFormat.of().formatHex(java.security.MessageDigest.getInstance("SHA-256").digest(bytes)),"e3da5ffde756b0ee23a3c873bb33a42bef9f2376a9d5f7f303ede24e486e9d98")
    val cases=arr(obj(NativeJson.parse(new String(bytes,UTF_8)))("cases"))
    val temp=Files.createTempDirectory("nui-tokenize")
    try
      for profile <- Vector("tiny-sd-pipe","tiny-sdxl-pipe") do
        val model=Path.of(directory.get).resolve(profile)
        val config=temp.resolve("scheduler.json");Files.writeString(config,"{}")
        val backend=new NativeDiffusionBackend(model,config)
        val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
        assert(backend.features.contains("tokenizer"))
        val empty=NativeRequest("inspect","tokenize",Map.empty,None)
        assertEquals(intercept[NativeBackendError](backend.execute(empty,context)).code,"no_model")
        backend.execute(NativeRequest("load","load_model",Map.empty,None),context)
        for item <- cases if obj(item)("profile")==Str(profile) do
          val f=obj(item)
          val actual=backend.execute(empty.copy(params=Map("text"->f("text"),"input_mode"->f("mode"))),context)
          val expected=obj(f("result"))
          val normalized=actual.updated("streams",Arr(arr(actual("streams")).map { v => Obj(obj(v).updated("source",Str(""))) }))
          assertEquals(normalized,expected,s"profile=$profile input=${f("text")}")
        intercept[IllegalArgumentException](backend.execute(empty.copy(params=Map("text"->Str("1,-2"),"input_mode"->Str("ids"))),context))
        intercept[IllegalArgumentException](backend.execute(empty.copy(params=Map("text"->Str("999999"),"input_mode"->Str("ids"))),context))
        backend.execute(NativeRequest("unload","unload",Map.empty,None),context)
        assertEquals(intercept[NativeBackendError](backend.execute(empty,context)).code,"no_model")
    finally
      val paths=Files.walk(temp)
      try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }

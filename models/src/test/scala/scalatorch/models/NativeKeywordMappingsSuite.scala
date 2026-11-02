package scalatorch.models.worker

import java.nio.file.Files

class NativeKeywordMappingsSuite extends munit.FunSuite:
  import NativeJson.*
  test("NUI mapping request schema checks saves and deletes a native directory mapping") {
    val root=Files.createTempDirectory("mapping-request")
    val context=new NativeWorkContext(() => false,() => true,(_,_) => ())
    def run(action: String)=NativeKeywordMappings.execute(Map("kind"->Str("checkpoint"),"action"->Str(action),
      "model_path"->Str(root.toString),"mapping_dir"->Str(root.toString),"keywords"->Str("red|blue")),root,() => "a"*64,context)
    try
      assertEquals(run("check")("source"),Str("no match"))
      val saved=run("save")
      assertEquals(saved("hash"),Str("a"*64))
      assertEquals(saved("keywords"),Arr(Vector(Str("red"),Str("blue"))))
      assertEquals(saved("source"),Str("custom"))
      assertEquals(saved("choices"),Arr(Vector("keyword1, keyword2","random","iterate","red","blue").map(Str(_))))
      assertEquals(run("delete")("deleted"),Bool(true))
      assertEquals(run("delete")("deleted"),Bool(false))
      assertEquals(run("check")("keywords"),Arr(Vector.empty))
    finally
      val files=Files.list(root)
      try files.forEach(p => {Files.delete(p);()}) finally files.close()
      Files.delete(root)
  }
  test("LoRA mappings use legacy file identity and preserve the requested alias") {
    val root=Files.createTempDirectory("mapping-lora")
    val model=root.resolve("actual.safetensors")
    Files.write(model,Array[Byte](1,2,3))
    val context=new NativeWorkContext(() => false,() => true,(_,_) => ())
    try
      val result=NativeKeywordMappings.execute(Map("kind"->Str("lora"),"action"->Str("save"),
        "model_path"->Str(model.toString),"mapping_dir"->Str(root.toString),"match_name"->Str("nested/alias.safetensors"),
        "keywords"->Str("style|texture")),root,() => throw new AssertionError("directory hash used for file"),context)
      assertEquals(result("hash"),Str("e3b0c442"))
      assertEquals(result("filename"),Str("nested/alias.safetensors"))
      assertEquals(result("choices"),Arr(Vector("None","style, texture","style","texture","< iterate >","< random >").map(Str(_))))
      assert(Files.exists(root.resolve("lora-keyword-user.txt")))
    finally
      val files=Files.list(root)
      try files.forEach(p => {Files.delete(p);()}) finally files.close()
      Files.delete(root)
  }

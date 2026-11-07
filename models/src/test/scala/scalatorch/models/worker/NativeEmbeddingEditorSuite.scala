package scalatorch.models.worker

import scalatorch.{Shape,Tensor}
import scalatorch.models.{ClipFastTokenizer,ClipTextualInversion}
import NativeJson.*
import java.nio.file.Files

class NativeEmbeddingEditorSuite extends munit.FunSuite:
  test("open returns loaded vectors and rejects a changed embedding file") {
    val tokenizer=ClipFastTokenizer.fromText("""{"<|startoftext|>":0,"<|endoftext|>":1,"a</w>":2}""","#version: 0.2\n")
      .withAddedTokens(Vector("example"->1))
    val weights=Tensor.fromFloats(Array(0f,0f,0f,1f,1f,0f,0.25f,0.75f),Shape.from(Seq(4,2)))
    val dir=Files.createTempDirectory("native-editor-open");val file=dir.resolve("example.safetensors")
    try
      ClipTextualInversion.writeSafetensors(file,ClipTextualInversion(Vector(Array(0.25f,0.75f)),None))
      val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
      val streams=Vector(("clip_l",tokenizer,weights))
      val result=NativeEmbeddingEditor.open("example",file,"model-1","a:#ff0000",streams,context)
      assertEquals(result("model_token"),Str("model-1"))
      assertEquals(result("vectors").asInstanceOf[Obj].fields("clip_l"),
        Arr(Vector(Arr(Vector(Number(BigDecimal(0.25)),Number(BigDecimal(0.75)))))))
      assertEquals(result("ranges").asInstanceOf[Obj].fields("clip_l"),
        Arr(Vector(Arr(Vector(Number(0),Number(1))),Arr(Vector(Number(0),Number(1))))))
      assertEquals(result("guidance").asInstanceOf[Arr].values.size,1)
      ClipTextualInversion.writeSafetensors(file,ClipTextualInversion(Vector(Array(0.5f,0.75f)),None),overwrite=true)
      intercept[IllegalArgumentException](NativeEmbeddingEditor.open("example",file,"model-1","",streams,context))
    finally
      Files.deleteIfExists(file);Files.deleteIfExists(dir)
  }

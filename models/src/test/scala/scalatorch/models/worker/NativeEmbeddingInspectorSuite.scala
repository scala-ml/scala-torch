package scalatorch.models.worker

import scalatorch.models.ClipFastTokenizer
import scalatorch.{Shape,Tensor}
import NativeJson.*

class NativeEmbeddingInspectorSuite extends munit.FunSuite:
  test("mixer resolves NUI token rows and returns weighted vectors") {
    val tokenizer=ClipFastTokenizer.fromText("""{"<|startoftext|>":0,"<|endoftext|>":1,"a</w>":2}""","#version: 0.2\n")
    val weights=Tensor.fromFloats(Array(0f,0f,0f,1f,1f,0.5f),Shape.from(Seq(3,2)))
    val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    val rows=Arr(Vector(Obj(Map("name"->Str("a"),"weight"->Number(BigDecimal(0.5)))),
      Obj(Map("ids"->Obj(Map("clip_l"->Number(1))),"weight"->Number(2)))))
    val result=NativeEmbeddingInspector.prepareMix(Map("rows"->rows),Vector(("clip_l",tokenizer,weights)),context)
    assertEquals(result("clip_l"),Vector(Vector(0.5f,2.25f)))
  }
  test("export publishes selected vectors and respects collisions and cancellation") {
    val tokenizer=ClipFastTokenizer.fromText("""{"<|startoftext|>":0,"<|endoftext|>":1,"a</w>":2}""","#version: 0.2\n")
    val weights=Tensor.fromFloats(Array(0f,0f,0f,1f,1f,0.5f),Shape.from(Seq(3,2)))
    val streams=Vector(("clip_l",tokenizer,weights))
    val dir=java.nio.file.Files.createTempDirectory("inspector-export")
    val target=dir.resolve("vectors.txt")
    try
      val params=Map("text"->Str("a"),"output"->Str(target.toString))
      val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
      NativeEmbeddingInspector.exportVectors(params,streams,context)
      val json=NativeJson.parse(java.nio.file.Files.readString(target)).asInstanceOf[Obj].fields
      assertEquals(json("clip_l"),Arr(Vector(Arr(Vector(Number(1),Number(BigDecimal(0.5)))))))
      val original=java.nio.file.Files.readAllBytes(target).toVector
      intercept[java.nio.file.FileAlreadyExistsException](NativeEmbeddingInspector.exportVectors(params,streams,context))
      assertEquals(java.nio.file.Files.readAllBytes(target).toVector,original)
      val denied=new NativeWorkContext(()=>false,()=>false,(_,_)=>())
      intercept[java.util.concurrent.CancellationException](NativeEmbeddingInspector.exportVectors(params.updated("overwrite",Bool(true)),streams,denied))
      assertEquals(java.nio.file.Files.readAllBytes(target).toVector,original)
      NativeEmbeddingInspector.exportVectors(params.updated("text",Str("#1")).updated("overwrite",Bool(true)),streams,context)
      val updated=NativeJson.parse(java.nio.file.Files.readString(target)).asInstanceOf[Obj].fields
      assertEquals(updated("clip_l"),Arr(Vector(Arr(Vector(Number(0),Number(1))))))
    finally
      val paths=java.nio.file.Files.walk(dir)
      try paths.sorted(java.util.Comparator.reverseOrder[java.nio.file.Path]()).forEach(java.nio.file.Files.deleteIfExists(_)) finally paths.close()
  }
  test("list separates active embeddings from catalog names and preserves encoder shapes") {
    val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    val loaded=Vector(NativeEmbeddingInspector.LoadedEmbedding("Cat",java.nio.file.Path.of("cat.safetensors"),2,768,Some(1280)))
    val params=Map("embedding_paths"->Arr(Vector(Str("C:/embeddings/CAT.safetensors"),Str("C:/embeddings/dog.pt"),
      Str("C:/other/Dog.bin"),Str("C:/other/ignored.txt"))))
    val result=NativeEmbeddingInspector.list(params,loaded,context)
    assertEquals(result("available"),Arr(Vector(Str("dog"))))
    val entry=result("loaded").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields
    assertEquals(entry("name"),Str("Cat"))
    val encoders=entry("encoders").asInstanceOf[Obj].fields
    assertEquals(encoders("clip_l"),Arr(Vector(Number(2),Number(768))))
    assertEquals(encoders("clip_g"),Arr(Vector(Number(2),Number(1280))))
    assertEquals(entry("file_metadata_status"),Str("file changed or could not be verified"))
  }
  test("SDXL file inspection preserves paired encoder vectors") {
    val tokenizer=ClipFastTokenizer.fromText("""{"<|startoftext|>":0,"<|endoftext|>":1,"a</w>":2}""","#version: 0.2\n")
    val left=Tensor.fromFloats(Array(0f,0f,0f,1f,1f,0f),Shape.from(Seq(3,2)))
    val right=Tensor.fromFloats(Array(0f,0f,0f,0f,1f,0f,1f,0f,0f),Shape.from(Seq(3,3)))
    val dir=java.nio.file.Files.createTempDirectory("inspector-sdxl")
    val path=dir.resolve("pair.safetensors")
    try
      scalatorch.models.ClipTextualInversion.writeSafetensors(path,
        scalatorch.models.ClipTextualInversion(Vector(Array(1f,0f)),Some(Vector(Array(0f,1f,0f)))))
      val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
      val params=Map("text"->Str("pair"),"embedding_paths"->Arr(Vector(Str(path.toString))))
      val result=NativeEmbeddingInspector.inspect(params,Vector(("clip_l",tokenizer,left),("clip_g",tokenizer,right)),context)
      val encoders=result("encoders").asInstanceOf[Obj].fields
      assertEquals(encoders("clip_l").asInstanceOf[Obj].fields("shape"),Arr(Vector(Number(1),Number(2))))
      assertEquals(encoders("clip_g").asInstanceOf[Obj].fields("shape"),Arr(Vector(Number(1),Number(3))))
      val output=dir.resolve("pair.txt")
      NativeEmbeddingInspector.exportVectors(params.updated("output",Str(output.toString)),
        Vector(("clip_l",tokenizer,left),("clip_g",tokenizer,right)),context)
      val exported=NativeJson.parse(java.nio.file.Files.readString(output)).asInstanceOf[Obj].fields
      assertEquals(exported("clip_l"),Arr(Vector(Arr(Vector(Number(1),Number(0))))))
      assertEquals(exported("clip_g"),Arr(Vector(Arr(Vector(Number(0),Number(1),Number(0))))))
      java.nio.file.Files.delete(output)
      intercept[IllegalArgumentException](NativeEmbeddingInspector.inspect(params.updated("embedding_paths",
        Arr(Vector(Str(path.toString),Str(dir.resolve("pair.pt").toString)))),Vector(("clip_l",tokenizer,left)),context))
    finally
      java.nio.file.Files.deleteIfExists(path);java.nio.file.Files.deleteIfExists(dir)
  }
  test("inspect reads an available embedding file without activating it") {
    val tokenizer=ClipFastTokenizer.fromText("""{"<|startoftext|>":0,"<|endoftext|>":1,"a</w>":2}""","#version: 0.2\n")
    val weights=Tensor.fromFloats(Array(0f,0f,0f,1f,1f,0f),Shape.from(Seq(3,2)))
    val dir=java.nio.file.Files.createTempDirectory("inspector-file")
    val path=dir.resolve("example.safetensors")
    try
      scalatorch.models.ClipTextualInversion.writeSafetensors(path,
        scalatorch.models.ClipTextualInversion(Vector(Array(0.25f,0.75f)),None),step=Some(12))
      val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
      val result=NativeEmbeddingInspector.inspect(Map("text"->Str("example"),
        "embedding_paths"->Arr(Vector(Str(path.toString)))),Vector(("clip_l",tokenizer,weights)),context)
      val encoder=result("encoders").asInstanceOf[Obj].fields("clip_l").asInstanceOf[Obj].fields
      assertEquals(encoder("ids"),Arr(Vector.empty))
      assertEquals(encoder("vectors"),Arr(Vector(Arr(Vector(Number(BigDecimal(0.25)),Number(BigDecimal(0.75)))))))
      assertEquals(result("metadata").asInstanceOf[Obj].fields("step"),Str("12"))
    finally
      java.nio.file.Files.deleteIfExists(path);java.nio.file.Files.deleteIfExists(dir)
  }
  test("inspect returns token vectors and stable cosine-ranked neighbours") {
    val tokenizer=ClipFastTokenizer.fromText("""{"<|startoftext|>":0,"<|endoftext|>":1,"a</w>":2,"b</w>":3}""","#version: 0.2\n")
    val weights=Tensor.fromFloats(Array(0f,0f,0f,1f,1f,0f,1f,0f),Shape.from(Seq(4,2)))
    val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    val result=NativeEmbeddingInspector.inspect(Map("text"->Str("a"),"limit"->Number(3)),
      Vector(("clip_l",tokenizer,weights)),context)
    val encoder=result("encoders").asInstanceOf[Obj].fields("clip_l").asInstanceOf[Obj].fields
    assertEquals(encoder("ids"),Arr(Vector(Number(2))))
    assertEquals(encoder("vectors"),Arr(Vector(Arr(Vector(Number(1),Number(0))))))
    val nearest=encoder("nearest").asInstanceOf[Arr].values.map(_.asInstanceOf[Obj].fields("id"))
    assertEquals(nearest,Vector(Number(2),Number(3),Number(0)))
    intercept[IllegalArgumentException](NativeEmbeddingInspector.inspect(Map("text"->Str("#4")),
      Vector(("clip_l",tokenizer,weights)),context))
  }
  test("embedding inspector tokenizes both encoders and limits output like NUI") {
    val tokenizer=ClipFastTokenizer.fromText("""{"<|startoftext|>":0,"<|endoftext|>":1,"a</w>":2}""","#version: 0.2\n")
    val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    val result=NativeEmbeddingInspector.execute(Map("operation"->Str("tokenize"),"text"->Str("A "*129)),
      Vector("clip_l"->tokenizer,"clip_g"->tokenizer),context)
    val encoders=result("encoders").asInstanceOf[Obj].fields
    for key <- Vector("clip_l","clip_g") do
      val tokens=encoders(key).asInstanceOf[Arr].values
      assertEquals(tokens.size,128)
      assertEquals(tokens.head,Obj(Map("id"->Number(2),"token"->Str("a</w>"))))
    assertEquals(result("warnings").asInstanceOf[Arr].values.size,2)
  }

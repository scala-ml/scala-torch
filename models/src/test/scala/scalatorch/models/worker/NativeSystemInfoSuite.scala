package scalatorch.models.worker

import java.nio.file.Files
import scalatorch.{Shape,Tensor}
import scalatorch.io.{SafeTensorWrite,SafeTensorsWriter}
import NativeJson.*

class NativeSystemInfoSuite extends munit.FunSuite:
  test("system information includes model inventory and optional hashes without a loaded model") {
    val dir=Files.createTempDirectory("native-system-info")
    val config=dir.resolve("scheduler.json")
    Files.writeString(config,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
    val model=dir.resolve("model.safetensors")
    SafeTensorsWriter.write(model,Map("weight"->SafeTensorWrite.F32(Tensor.fromFloats(Array(1f),Shape.from(Seq(1L))))))
    val embedding=Files.createDirectory(dir.resolve("embeddings"))
    SafeTensorsWriter.write(embedding.resolve("cat.safetensors"),Map("embedding"->SafeTensorWrite.F32(
      Tensor.fromFloats(Array.fill(12)(1f),Shape.from(Seq(3L,4L))))))
    val backend=new NativeDiffusionBackend(dir,config)
    val ctx=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    def request(hashes: Boolean)=
      val dirs=Obj(Map("checkpoints"->Arr(Vector(Str(dir.toString))),
        "embeddings"->Arr(Vector(Str(embedding.toString)))))
      backend.execute(NativeRequest("info","system_info",Map("dirs"->dirs,"hashes"->Bool(hashes)),None),ctx)
    assert(backend.supported.contains("system_info"))
    val raw=request(false)
    assert(raw.contains("memory") && raw.contains("platform") && raw.contains("features"))
    val networks=raw("networks").asInstanceOf[Obj].fields
    val entry=networks("embeddings").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields
    assertEquals(entry("vectors"),Number(3));assertEquals(entry("hash"),Null)
    val hashed=request(true)("networks").asInstanceOf[Obj].fields("embeddings").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields
    assert(hashed("hash").asInstanceOf[Str].value.matches("[0-9a-f]{10}"))
    intercept[IllegalArgumentException](backend.execute(NativeRequest("bad","system_info",Map("dirs"->Str("bad")),None),ctx))
  }

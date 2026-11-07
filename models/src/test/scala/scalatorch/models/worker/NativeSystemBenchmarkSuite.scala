package scalatorch.models.worker

import java.nio.file.{Files,Path}
import NativeJson.*

class NativeSystemBenchmarkSuite extends munit.FunSuite:
  override val munitTimeout: scala.concurrent.duration.Duration=scala.concurrent.duration.Duration(300,"seconds")
  test("all benchmark presets use supported sequential generation counts") {
    for count <- Vector(1,2,4,8,16,32,64) do
      val p=NativeSystemBenchmark.generationParams(count,256,256,10)
      assertEquals(p("batch_size"),Number(1))
      assertEquals(p("n_iter"),Number(count))
  }
  test("benchmark request can generate more than sixteen images on CUDA") {
    val model=sys.env.get("SCALATORCH_TINY_MODEL")
    assume(model.nonEmpty && sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"tiny model and CUDA required")
    val root=Path.of(model.get)
    val dir=Files.createTempDirectory("native-benchmark-sequential")
    val scheduler=dir.resolve("scheduler.json")
    Files.writeString(scheduler,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
    val backend=new NativeDiffusionBackend(root,scheduler,
      vaeExecution=scalatorch.models.VaeExecution.Cuda(offloadWeights=true),
      unetExecution=scalatorch.models.UNetExecution.Cuda(offloadWeights=true))
    val ctx=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    try
      backend.execute(NativeRequest("load","load_model",Map.empty,None),ctx)
      val generation=NativeSystemBenchmark.generationParams(32,64,64,1)
      val result=backend.execute(NativeRequest("generate","txt2img",Map("params"->Obj(generation),"output_dir"->Str(dir.toString)),None),ctx)
      assertEquals(result("images").asInstanceOf[Arr].values.size,32)
    finally
      val paths=Files.walk(dir)
      try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(p=>{Files.deleteIfExists(p);()}) finally paths.close()
  }
  test("worker rejects benchmark without a loaded checkpoint") {
    val root=Files.createTempDirectory("system-benchmark-empty")
    val config=root.resolve("scheduler.json")
    Files.writeString(config,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
    val backend=new NativeDiffusionBackend(root,config)
    val ctx=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    val error=intercept[NativeBackendError](backend.execute(NativeRequest("b","system_benchmark",Map.empty,None),ctx))
    assertEquals(error.code,"no_model")
  }
  test("benchmark measures batches, stops after an error, and removes temporary images") {
    val ctx=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    val seen=scala.collection.mutable.ArrayBuffer.empty[Int]
    var directory: Path=null
    val result=NativeSystemBenchmark.run(Map("level"->Str("normal"),"steps"->Str("turbo"),
      "width"->Number(256),"height"->Number(256),"warmup"->Bool(false)),Map("timestamp"->Str("now")),ctx) {
      (batch,_,_,_,output,_) =>
        directory=output;seen+=batch
        if batch==2 then throw new IllegalStateException("out of memory")
        Files.writeString(output.resolve("temporary.png"),"test")
        batch
    }
    assertEquals(seen.toVector,Vector(1,2))
    assertEquals(result("status"),Str("stopped_after_error"))
    assertEquals(result("config").asInstanceOf[Obj].fields("batch_execution"),Str("sequential"))
    val rows=result("results").asInstanceOf[Arr].values
    assertEquals(rows.map(_.asInstanceOf[Obj].fields("status")),Vector(Str("complete"),Str("oom")))
    assert(rows.forall(_.asInstanceOf[Obj].fields("batch_execution")==Str("sequential")))
    assert(!Files.exists(directory))
  }
  test("benchmark validates dimensions, steps and cancellation") {
    val ctx=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    def run(p: Map[String,NativeJson],c: NativeWorkContext)=NativeSystemBenchmark.run(p,Map("timestamp"->Str("now")),c) {
      (_,_,_,_,_,_)=>1
    }
    intercept[IllegalArgumentException](run(Map("width"->Number(64),"height"->Number(64)),ctx))
    intercept[IllegalArgumentException](run(Map("steps"->Str("bad")),ctx))
    intercept[java.util.concurrent.CancellationException](run(Map.empty,new NativeWorkContext(()=>true,()=>true,(_,_)=>())))
  }
  test("benchmark fingerprint is stable across model reloads and changes with model identity") {
    val ctx=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    def fingerprint(hash: String,token: String)=
      val system=Map[String,NativeJson]("timestamp"->Str("now"),"model"->Obj(Map(
        "name"->Str("model"),"hash"->Str(hash),"model_token"->Str(token),"path"->Str("absolute/local/path"))))
      NativeSystemBenchmark.run(Map("level"->Str("quick"),"steps"->Str("turbo"),
        "width"->Number(256),"height"->Number(256),"warmup"->Bool(false)),system,ctx) {
        (_,_,_,_,_,_)=>1
      }("fingerprint")
    assertEquals(fingerprint("abc","first"),fingerprint("abc","second"))
    assertNotEquals(fingerprint("abc","first"),fingerprint("def","first"))
  }
  test("wrapped CUDA allocation failure is reported as out of memory") {
    val ctx=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    val result=NativeSystemBenchmark.run(Map("level"->Str("quick"),"steps"->Str("turbo"),
      "width"->Number(256),"height"->Number(256),"warmup"->Bool(false)),Map("timestamp"->Str("now")),ctx) {
      (_,_,_,_,_,_) => throw new IllegalStateException("decode failed",new IllegalStateException("cuMemAlloc_v2 failed with CUDA status 2"))
    }
    assertEquals(result("results").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("status"),Str("oom"))
  }

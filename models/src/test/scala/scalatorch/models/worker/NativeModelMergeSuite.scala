package scalatorch.models.worker

import java.nio.file.{Files,Path}
import scalatorch.{Shape,Tensor}
import scalatorch.io.{SafeDType,SafeTensorWrite,SafeTensors,SafeTensorsWriter}
import scala.util.Using
import NativeJson.*

class NativeModelMergeSuite extends munit.FunSuite:
  test("merging a large float tensor avoids redundant full-size arrays") {
    val bean=java.lang.management.ManagementFactory.getThreadMXBean.asInstanceOf[com.sun.management.ThreadMXBean]
    assume(bean.isThreadAllocatedMemorySupported,"thread allocation accounting required")
    bean.setThreadAllocatedMemoryEnabled(true)
    val dir=Files.createTempDirectory("merge-allocation")
    val input=dir.resolve("input.safetensors");val output=dir.resolve("output.safetensors")
    val tensor=Tensor.fromFloats(Array.fill(4000000)(2.5f),Shape.from(Seq(4000000L)))
    val ctx=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    try
      SafeTensorsWriter.write(input,Map("weight"->SafeTensorWrite.F32(tensor)))
      val params=Map[String,NativeJson]("model_a"->Str(input.toString),"model_b"->Str(input.toString),
        "output"->Str(output.toString),"precision"->Str("fp16"))
      val before=bean.getThreadAllocatedBytes(Thread.currentThread().threadId())
      NativeModelMerge.execute(params,ctx)
      val allocated=bean.getThreadAllocatedBytes(Thread.currentThread().threadId())-before
      assert(allocated<48L*1024*1024,s"merge allocated $allocated bytes for two 16 MB inputs")
      Using.resource(SafeTensors.open(output))(file=>assertEquals(file.readFloat("weight").toFloatArray.last,2.5f))
    finally
      Files.deleteIfExists(output);Files.deleteIfExists(input);Files.deleteIfExists(dir)
  }
  test("checkpoint merge implements all recipes, block weights, VAE choice and raw passthrough") {
    val dir=Files.createTempDirectory("native-model-merge")
    def tensor(xs: Float*)=SafeTensorWrite.F32(Tensor.fromFloats(xs.toArray,Shape.from(Seq(xs.size.toLong))))
    def file(name: String,base: Float,includeMissing: Boolean): Path =
      val path=dir.resolve(name)
      val entries=Map[String,SafeTensorWrite]("model.diffusion_model.input_blocks.0.weight"->tensor(base,base+1),
        "first_stage_model.weight"->tensor(base+2),"counter"->SafeTensorWrite.Raw(SafeDType.I32,Shape.from(Seq(1L)),Array[Byte](1,2,3,4))) ++
        Option.when(includeMissing)("missing"->tensor(9f))
      SafeTensorsWriter.write(path,entries);path
    val a=file("a.safetensors",1,true);val b=file("b.safetensors",5,false);val c=file("c.safetensors",9,false)
    val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    val weights=(Vector("0.25")++Vector.fill(25)("0.5")).mkString(",")
    for (mode,expected)<-Vector(("Weight sum",3f),("Add difference",-1f),("Sum twice",4.5f),("Triple sum",5f)) do
      val output=dir.resolve(mode.replace(' ','-')+".safetensors")
      val params=Map[String,NativeJson]("model_a"->Str(a.toString),"model_b"->Str(b.toString),"model_c"->Str(c.toString),
        "output"->Str(output.toString),"mode"->Str(mode),"alpha"->Number(BigDecimal("0.5")),
        "beta"->Number(BigDecimal("0.25")),"block_weights"->Str(weights),"vae_source"->Str("B"),"precision"->Str("fp32"))
      val result=NativeModelMerge.execute(params,context)
      assertEquals(result("skipped"),Number(1))
      Using.resource(SafeTensors.open(output)) { archive =>
        val value=archive.readFloat("model.diffusion_model.input_blocks.0.weight").toFloatArray.head
        assertEqualsFloat(value,expected,1e-6f)
        assertEquals(archive.readFloat("first_stage_model.weight").toFloatArray.toVector,Vector(7f))
        assertEquals(archive.readFloat("missing").toFloatArray.toVector,Vector(9f))
        assertEquals(archive.readRaw("counter").toVector,Vector[Byte](1,2,3,4))
        assert(archive.metadata.contains("sd_merge_recipe"))
      }
    val flat=dir.resolve("flat.safetensors")
    NativeModelMerge.execute(Map("model_a"->Str(a.toString),"model_b"->Str(b.toString),"output"->Str(flat.toString),
      "block_weights"->Str("preset:FLAT_37"),"precision"->Str("fp32")),context)
    Using.resource(SafeTensors.open(flat)) { archive =>
      assertEqualsFloat(archive.readFloat("model.diffusion_model.input_blocks.0.weight").toFloatArray.head,2.48f,1e-6f)
    }
    intercept[IllegalArgumentException](NativeModelMerge.execute(Map("model_a"->Str(a.toString),"model_b"->Str(b.toString),
      "output"->Str(a.toString),"overwrite"->Bool(true)),context))
  }

  test("streamed checkpoint recipes match NUI Python across modes and precision") {
    val dir=Files.createTempDirectory("native-merge-oracle")
    def resource(name: String)=this.getClass.getResourceAsStream(s"/model-merge-reference/$name.safetensors")
    def copy(name: String): Path =
      val path=dir.resolve(name+".safetensors");val in=resource(name)
      try Files.copy(in,path) finally in.close()
      path
    val a=copy("a");val b=copy("b");val c=copy("c")
    val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    for mode<-Vector("Weight sum","Add difference","Sum twice","Triple sum");precision<-Vector("fp16","bf16","fp32") do
      val label=mode.toLowerCase(java.util.Locale.ROOT).replace(' ','-')+"-"+precision
      val expectedPath=copy(label)
      val output=dir.resolve(label+"-actual.safetensors")
      val result=NativeModelMerge.execute(Map("model_a"->Str(a.toString),"model_b"->Str(b.toString),"model_c"->Str(c.toString),
        "output"->Str(output.toString),"mode"->Str(mode),"alpha"->Number(BigDecimal("0.35")),"beta"->Number(BigDecimal("0.2")),
        "block_weights"->Str("preset:GRAD_V"),"vae_source"->Str("B"),"precision"->Str(precision)),context)
      assertEquals(result("sdxl"),Bool(true))
      Using.resource(SafeTensors.open(expectedPath)) { expected => Using.resource(SafeTensors.open(output)) { actual =>
        assertEquals(actual.tensors.keySet,expected.tensors.keySet)
        for key<-expected.tensors.keys do
          assertEquals((actual.tensors(key).shape,actual.tensors(key).dtype),(expected.tensors(key).shape,expected.tensors(key).dtype))
          expected.tensors(key).dtype match
            case SafeDType.F16 | SafeDType.BF16 | SafeDType.F32 =>
              actual.readFloat(key).toFloatArray.zip(expected.readFloat(key).toFloatArray).foreach((a,b)=>assertEqualsFloat(a,b,0.00001f))
            case SafeDType.F64 => assertEquals(actual.readDouble(key).toDoubleArray.toVector,expected.readDouble(key).toDoubleArray.toVector)
            case _ => assertEquals(actual.readRaw(key).toVector,expected.readRaw(key).toVector)
      }}
  }

  test("worker dispatch merges without a loaded model and rejects cancelled publication") {
    val dir=Files.createTempDirectory("native-merge-worker")
    val tensor=SafeTensorWrite.F32(Tensor.fromFloats(Array(2f),Shape.from(Seq(1L))))
    val a=dir.resolve("a.safetensors");val b=dir.resolve("b.safetensors")
    SafeTensorsWriter.write(a,Map("weight"->tensor));SafeTensorsWriter.write(b,Map("weight"->tensor))
    val config=dir.resolve("scheduler.json")
    Files.writeString(config,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
    val backend=new NativeDiffusionBackend(dir,config)
    assert(backend.supported.contains("merge_models"))
    val out=dir.resolve("merged.safetensors")
    val fields=Map[String,NativeJson]("model_a"->Str(a.toString),"model_b"->Str(b.toString),"output"->Str(out.toString))
    var published=false
    val denied=new NativeWorkContext(()=>false,()=>{published=true;false},(_,_)=>())
    intercept[java.util.concurrent.CancellationException](backend.execute(NativeRequest("merge","merge_models",fields,None),denied))
    assert(published);assert(!Files.exists(out))
    val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    val result=backend.execute(NativeRequest("merge","merge_models",fields,None),context)
    assertEquals(result("keys"),Number(1))
    Using.resource(SafeTensors.open(out))(archive=>assertEquals(archive.readFloat("weight").toFloatArray.toVector,Vector(2f)))
    intercept[IllegalArgumentException](backend.execute(NativeRequest("repeat","merge_models",fields,None),context))
  }

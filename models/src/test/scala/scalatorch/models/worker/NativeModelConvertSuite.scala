package scalatorch.models.worker

import java.nio.file.Files
import scala.util.Using
import scalatorch.{Shape,Tensor}
import scalatorch.io.{SafeTensors,SafeTensorsWriter,SafeTensorWrite,SafeDType}
import NativeJson.*

class NativeModelConvertSuite extends munit.FunSuite:
  override val munitTimeout: scala.concurrent.duration.Duration=scala.concurrent.duration.Duration(10,"minutes")
  test("real SD1 conversion preserves tensor values when cast to fp16") {
    val configured=sys.env.get("SCALATORCH_REAL_CONVERT_MODEL")
    assume(configured.nonEmpty,"real checkpoint required")
    val source=java.nio.file.Path.of(configured.get).toRealPath()
    val dir=Files.createTempDirectory("real-convert")
    val output=dir.resolve("converted.safetensors")
    val ctx=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    try
      val vae=sys.env.get("SCALATORCH_REAL_CONVERT_VAE")
      val result=NativeModelConvert.execute(Map("model"->Str(source.toString),"output"->Str(output.toString),
        "precision"->Str("fp16"),"ema"->Str("keep"),"fix_clip"->Bool(false),"drop_junk"->Bool(false),
        "bake_vae"->vae.map(Str.apply).getOrElse(Null)),ctx)
      if vae.nonEmpty then assertEquals(result("vae_baked"),Bool(true))
      assert(result("keys").asInstanceOf[Number].value.toInt>1000)
      Using.resource(SafeTensors.open(source)) { a => Using.resource(SafeTensors.open(output)) { b =>
        assertEquals(a.tensors.keySet,b.tensors.keySet)
        for (key,info)<-a.tensors do
          if info.dtype==SafeDType.F32 then
            val values=a.readFloat(key).toFloatArray;val bytes=b.readRaw(key)
            val encoded=java.nio.ByteBuffer.wrap(bytes).order(java.nio.ByteOrder.LITTLE_ENDIAN)
            assertEquals(b.tensors(key).dtype,SafeDType.F16,key)
            val changed=vae.nonEmpty && key=="first_stage_model.decoder.conv_out.bias"
            var i=0
            while i<values.length && encoded.getShort()==java.lang.Float.floatToFloat16(if changed then values(i)+1f else values(i)) do i+=1
            assertEquals(i,values.length,key)
          else assertEquals(b.readRaw(key).toVector,a.readRaw(key).toVector,key)
      }}
    finally
      Files.deleteIfExists(output)
      Files.deleteIfExists(dir)
  }
  test("conversion matches NUI Python for precision, EMA and optional edits") {
    val dir=Files.createTempDirectory("convert-reference")
    def copy(name: String): java.nio.file.Path =
      val out=dir.resolve(name);val in=getClass.getResourceAsStream(s"/model-convert-reference/$name")
      try Files.copy(in,out) finally in.close()
      out
    val source=copy("source.safetensors");val vae=copy("vae.safetensors")
    val ctx=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    for precision<-Vector("keep","fp16","bf16","fp32"); ema<-Vector("no-ema","ema-only","keep"); edit<-Vector(true,false) do
      val name=s"$precision-$ema-${if edit then 1 else 0}"
      val expected=copy(name+".safetensors");val output=dir.resolve(name+"-actual.safetensors")
      NativeModelConvert.execute(Map("model"->Str(source.toString),"output"->Str(output.toString),
        "precision"->Str(precision),"ema"->Str(ema),"fix_clip"->Bool(edit),"drop_junk"->Bool(edit),
        "bake_vae"->(if edit then Str(vae.toString) else Null)),ctx)
      Using.resource(SafeTensors.open(expected)) { a => Using.resource(SafeTensors.open(output)) { b =>
        assertEquals(b.tensors.keySet,a.tensors.keySet,name)
        for key<-a.tensors.keys do
          assertEquals((b.tensors(key).shape,b.tensors(key).dtype),(a.tensors(key).shape,a.tensors(key).dtype),s"$name $key")
          assertEquals(b.readRaw(key).toVector,a.readRaw(key).toVector,s"$name $key")
      }}
  }
  test("bakes a PyTorch VAE state_dict without Python") {
    val dir=Files.createTempDirectory("convert-torch-vae")
    def copy(name: String): java.nio.file.Path =
      val out=dir.resolve(name);val in=getClass.getResourceAsStream(s"/model-convert-reference/$name")
      try Files.copy(in,out) finally in.close()
      out
    val source=copy("source.safetensors");val vae=copy("vae.pt");val output=dir.resolve("actual.safetensors")
    val ctx=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    NativeModelConvert.execute(Map("model"->Str(source.toString),"output"->Str(output.toString),
      "precision"->Str("fp16"),"ema"->Str("ema-only"),"bake_vae"->Str(vae.toString)),ctx)
    Using.resource(SafeTensors.open(output)) { file =>
      assertEquals(file.readFloat("first_stage_model.decoder.weight").toFloatArray.toVector,Vector(7f,11f))
    }
  }
  test("conversion selects EMA, fixes positions and bakes VAE") {
    val dir=Files.createTempDirectory("native-convert")
    val source=dir.resolve("source.safetensors")
    val vae=dir.resolve("vae.safetensors")
    val output=dir.resolve("out.safetensors")
    def tensor(x: Float)=SafeTensorWrite.F32(Tensor.fromFloats(Array(x),Shape.from(Seq(1L))))
    val unet="model.diffusion_model.middle_block.1.weight"
    val positions="cond_stage_model.transformer.position_ids"
    SafeTensorsWriter.write(source,Map(unet->tensor(1),"model_ema.diffusion_modelmiddle_block1weight"->tensor(9),
      "optimizer.state"->tensor(5),"first_stage_model.decoder.weight"->tensor(1),
      positions->SafeTensorWrite.I64(Tensor.fromLongs(Array(10L,10L),Shape.from(Seq(1L,2L))))))
    SafeTensorsWriter.write(vae,Map("decoder.weight"->tensor(7)))
    val ctx=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    val config=dir.resolve("scheduler.json")
    Files.writeString(config,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
    val backend=new NativeDiffusionBackend(dir,config)
    assert(backend.supported.contains("convert_model"))
    val result=backend.execute(NativeRequest("convert","convert_model",Map("model"->Str(source.toString),"output"->Str(output.toString),
      "precision"->Str("fp16"),"ema"->Str("ema-only"),"bake_vae"->Str(vae.toString)),None),ctx)
    assertEquals(result("ema_weights_used"),Number(1));assertEquals(result("clip_fixed"),Number(1))
    Using.resource(SafeTensors.open(output)) { file =>
      assertEquals(file.readFloat(unet).toFloatArray.toVector,Vector(9f))
      assertEquals(file.readFloat("first_stage_model.decoder.weight").toFloatArray.toVector,Vector(7f))
      assertEquals(file.readLong(positions).toLongArray.toVector,Vector(0L,1L))
      assertEquals(file.tensors(unet).dtype,SafeDType.F16)
      assert(!file.tensors.contains("optimizer.state"))
      assert(!file.tensors.contains("model_ema.diffusion_modelmiddle_block1weight"))
    }
  }
  test("conversion keeps existing output when publication is denied and rejects input alias") {
    val dir=Files.createTempDirectory("native-convert-cancel")
    val source=dir.resolve("source.safetensors");val out=dir.resolve("out.safetensors")
    val tensor=SafeTensorWrite.F32(Tensor.fromFloats(Array(2f),Shape.from(Seq(1L))))
    SafeTensorsWriter.write(source,Map("weight"->tensor));Files.writeString(out,"existing")
    val params=Map[String,NativeJson]("model"->Str(source.toString),"output"->Str(out.toString),"overwrite"->Bool(true))
    val denied=new NativeWorkContext(()=>false,()=>false,(_,_)=>())
    intercept[java.util.concurrent.CancellationException](NativeModelConvert.execute(params,denied))
    assertEquals(Files.readString(out),"existing")
    intercept[IllegalArgumentException](NativeModelConvert.execute(params.updated("output",Str(source.toString)),denied))
    Using.resource(SafeTensors.open(source))(file=>assertEquals(file.readFloat("weight").toFloatArray.toVector,Vector(2f)))
  }

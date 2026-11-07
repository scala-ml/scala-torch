package scalatorch.models.worker

import java.nio.file.{Files,Path}
import scalatorch.models.{UNetExecution,VaeExecution}
import NativeJson.*

class NativeGenerationHardwareSuite extends munit.FunSuite:
  test("CUDA generation retains device access after hardware probing") {
    val directory=sys.props.get("scalatorch.sdxl.dir")
    assume(directory.nonEmpty && sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"tiny SDXL and CUDA required")
    val root=Path.of(directory.get)
    val output=Files.createTempDirectory("native-generation-hardware-cuda")
    try
      val backend=new NativeDiffusionBackend(root,root.resolve("scheduler/scheduler_config.json"),
        vaeExecution=VaeExecution.Cuda(offloadWeights=true),unetExecution=UNetExecution.Cuda(offloadWeights=true))
      val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
      backend.execute(NativeRequest("load","load_model",Map.empty,None),context)
      val params=Map[String,NativeJson]("params"->Obj(Map("prompt"->Str("apple"),"width"->Number(8),
        "height"->Number(8),"steps"->Number(1),"seed"->Number(42))),"output_dir"->Str(output.toString),
        "hardware_info"->Obj(Map("enabled"->Bool(true),"remark"->Str("GPU"))))
      val image=backend.execute(NativeRequest("generate","txt2img",params,None),context)("images")
        .asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields
      val info=image("hardware_info").asInstanceOf[Obj].fields
      assertEquals(info("remark"),Str("GPU"))
      assert(info("hardware_info").asInstanceOf[Str].value.nonEmpty)
      assert(Files.isRegularFile(Path.of(image("path").asInstanceOf[Str].value)))
    finally
      val paths=Files.walk(output)
      try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }
  test("generation includes per-image hardware metadata for batches and img2img") {
    val directory=sys.props.get("scalatorch.sdxl.dir")
    assume(directory.nonEmpty,"tiny SDXL fixture required")
    val root=Path.of(directory.get)
    val output=Files.createTempDirectory("native-generation-hardware")
    try
      val backend=new NativeDiffusionBackend(root,root.resolve("scheduler/scheduler_config.json"))
      val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
      def run(kind: String,params: Map[String,NativeJson]) =
        backend.execute(NativeRequest("r",kind,params,None),context)
      run("load_model",Map.empty)
      val generation=Map[String,NativeJson]("prompt"->Str("apple"),"width"->Number(8),"height"->Number(8),
        "steps"->Number(1),"seed"->Number(42),"batch_size"->Number(2),"n_iter"->Number(1))
      val options=Obj(Map("enabled"->Bool(true),"remark"->Str("hardware test")))
      val params=Map[String,NativeJson]("params"->Obj(generation),"output_dir"->Str(output.toString),"hardware_info"->options)
      val generated=run("txt2img",params)("images").asInstanceOf[Arr].values
      assertEquals(generated.size,2)
      generated.foreach { image =>
        val info=image.asInstanceOf[Obj].fields("hardware_info").asInstanceOf[Obj].fields
        assert(info("hardware_info").asInstanceOf[Str].value.nonEmpty)
        assert(info("time_taken").asInstanceOf[Str].value.endsWith(" sec."))
        assertEquals(info("remark"),Str("hardware test"))
      }
      val source=generated.head.asInstanceOf[Obj].fields("path")
      val img=generation.updated("batch_size",Number(1)).updated("img2img",
        Obj(Map("init_image"->source,"denoising_strength"->Number(BigDecimal("0.5")),"resize_mode"->Number(0))))
      val edited=run("img2img",params.updated("params",Obj(img)))("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj]
      assert(edited.fields.contains("hardware_info"))
      val plain=run("txt2img",params.removed("hardware_info").updated("params",Obj(generation.updated("batch_size",Number(1)))))
      assert(!plain("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields.contains("hardware_info"))
      val disabled=NativeHardwareInfo.options(Some(Obj(Map("enabled"->Bool(false),"remark"->Str("ignored")))))
      assertEquals(disabled,None)
      intercept[IllegalArgumentException](NativeHardwareInfo.options(Some(Obj(Map("enabled"->Str("yes"))))))
    finally
      val paths=Files.walk(output)
      try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }

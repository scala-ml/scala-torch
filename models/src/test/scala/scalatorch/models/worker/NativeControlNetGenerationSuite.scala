package scalatorch.models.worker

import java.awt.image.BufferedImage
import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import scala.concurrent.duration.*
import scalatorch.models.{DiffusionImageRequest,PretrainedControlNet,UNetExecution,VaeExecution}
import NativeJson.*

class NativeControlNetGenerationSuite extends munit.FunSuite:
  override val munitTimeout: Duration=300.seconds
  test("changed ControlNet config or weights invalidate the loaded model cache") {
    val pipeline=sys.env.get("SCALATORCH_CONTROLNET_PIPELINE")
    val control=sys.env.get("SCALATORCH_TINY_CONTROLNET")
    assume(pipeline.nonEmpty && control.nonEmpty,"tiny ControlNet and pipeline fixtures required")
    val root=Path.of(pipeline.get).toRealPath()
    val original=Path.of(control.get).toRealPath()
    val temp=Files.createTempDirectory("controlnet-cache-change")
    try
      val copy=Files.createDirectory(temp.resolve("control"))
      for name <- Vector("config.json","diffusion_pytorch_model.safetensors") do Files.copy(original.resolve(name),copy.resolve(name))
      val scheduler=temp.resolve("scheduler.json")
      Files.writeString(scheduler,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
      val image=temp.resolve("input.png")
      ImageIO.write(new BufferedImage(64,64,BufferedImage.TYPE_INT_RGB),"png",image.toFile)
      val cuda=sys.env.get("SCALATORCH_CUDA_TESTS").contains("1")
      val backend=new NativeDiffusionBackend(root,scheduler,
        vaeExecution=if cuda then VaeExecution.Cuda(offloadWeights=true) else VaeExecution.Cpu,
        unetExecution=if cuda then UNetExecution.Cuda(offloadWeights=true) else UNetExecution.Cpu)
      val ctx=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
      backend.execute(NativeRequest("load","load_model",Map.empty,None),ctx)
      val unit=Obj(Map("enabled"->Bool(true),"model"->Str(copy.toString),"image"->Str(image.toString)))
      val params=Map[String,NativeJson]("output_dir"->Str(temp.toString),"params"->Obj(Map("prompt"->Str("cat"),
        "width"->Number(64),"height"->Number(64),"steps"->Number(1),"seed"->Number(42),"controlnet"->Arr(Vector(unit)))))
      backend.execute(NativeRequest("first","txt2img",params,None),ctx)
      val previousDirectoryTime=Files.getLastModifiedTime(copy)
      Files.writeString(copy.resolve("config.json"),"not a valid config")
      Files.setLastModifiedTime(copy,previousDirectoryTime)
      intercept[Exception](backend.execute(NativeRequest("second","txt2img",params,None),ctx))
      Files.copy(original.resolve("config.json"),copy.resolve("config.json"),java.nio.file.StandardCopyOption.REPLACE_EXISTING)
      backend.execute(NativeRequest("third","txt2img",params,None),ctx)
      Files.write(copy.resolve("diffusion_pytorch_model.safetensors"),Array[Byte](1,2,3))
      Files.setLastModifiedTime(copy,previousDirectoryTime)
      intercept[Exception](backend.execute(NativeRequest("fourth","txt2img",params,None),ctx))
      Files.delete(copy.resolve("diffusion_pytorch_model.safetensors"))
      val shard=copy.resolve("controlnet-00001-of-00001.safetensors")
      Files.copy(original.resolve("diffusion_pytorch_model.safetensors"),shard)
      val names=scala.util.Using.resource(scalatorch.io.SafeTensors.open(shard))(_.tensors.keys.toVector)
      val index=Obj(Map("weight_map"->Obj(names.map(name=>name->Str(shard.getFileName.toString)).toMap)))
      Files.writeString(copy.resolve("diffusion_pytorch_model.safetensors.index.json"),NativeJson.render(index))
      backend.execute(NativeRequest("fifth","txt2img",params,None),ctx)
      Files.write(shard,Array[Byte](1,2,3))
      Files.setLastModifiedTime(copy,previousDirectoryTime)
      intercept[Exception](backend.execute(NativeRequest("sixth","txt2img",params,None),ctx))
    finally
      val paths=Files.walk(temp)
      try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }
  test("native worker generates images with ControlNet and Detail Daemon") {
    val model=sys.props.get("scalatorch.controlnet.pipeline.dir")
    val control=sys.props.get("scalatorch.controlnet.dir")
    assume(model.nonEmpty && control.nonEmpty,"factor-8 tiny pipeline and ControlNet required")
    val root=Path.of(model.get)
    val output=Files.createTempDirectory("native-controlnet-generation")
    try
      val source=new BufferedImage(64,64,BufferedImage.TYPE_INT_RGB)
      for y <- 0 until 64; x <- 0 until 64 do
        source.setRGB(x,y,((x*4)<<16)|((y*4)<<8)|((x+y)*2))
      val image=output.resolve("control.png")
      ImageIO.write(source,"png",image.toFile)
      val loaded=PretrainedControlNet.load(Path.of(control.get))
      def prepared(preprocessor: String,sourceResize: Boolean) =
        val unit=Obj(Map("enabled"->Bool(true),"model"->Str(control.get),"image"->Str(image.toString),
          "preprocessor"->Str(preprocessor),"source_compatible_resize"->Bool(sourceResize)))
        NativeControlNet.prepare(Some(Arr(Vector(unit))),Map.empty,None,64,64,()=>false,_=>loaded).head
      for (preprocessor,sourceResize,expected) <- Vector(("canny",true,true),("canny",false,false),("none",true,false)) do
        val baseControl=prepared(preprocessor,sourceResize)
        assert(!baseControl.hiresSoftInjection)
        val request=DiffusionImageRequest("","",64,64,42,controlNets=Vector(baseControl))
        val hires=NativeControlNet.resizeRequest(request,128,128,()=>false).controlNets.head
        assertEquals(hires.hiresSoftInjection,expected)
      val placements=Vector(false)++Option.when(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"))(true)
      for cuda <- placements do
        val scheduler=output.resolve("scheduler.json")
        Files.writeString(scheduler,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
        val backend=new NativeDiffusionBackend(root,scheduler,
          vaeExecution=if cuda then VaeExecution.Cuda(offloadWeights=true) else VaeExecution.Cpu,
          unetExecution=if cuda then UNetExecution.Cuda(offloadWeights=true) else UNetExecution.Cpu)
        val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
        backend.execute(NativeRequest("load","load_model",Map.empty,None),context)
        val unit=Obj(Map("enabled"->Bool(true),"model"->Str(control.get),"image"->Str(image.toString),
          "preprocessor"->Str("none"),"weight"->Number(BigDecimal("0.7"))))
        val base=Map[String,NativeJson]("prompt"->Str("a red cat"),"width"->Number(64),"height"->Number(64),
          "steps"->Number(3),"seed"->Number(42),"sampler_name"->Str("Euler"),"scheduler"->Str("Uniform"))
        def run(fields: Map[String,NativeJson],kind: String = "txt2img"): Vector[Int] =
          val params=Map[String,NativeJson]("output_dir"->Str(output.toString),"params"->Obj(fields))
          val result=backend.execute(NativeRequest("generate",kind,params,None),context)
          val path=result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
          val generated=ImageIO.read(Path.of(path).toFile)
          assertEquals((generated.getWidth,generated.getHeight),(64,64))
          generated.getRGB(0,0,64,64,null,0,64).toVector
        val plain=run(base)
        val conditioned=run(base.updated("controlnet",Arr(Vector(unit))))
        val hooked=run(base.updated("controlnet",Arr(Vector(unit))).updated("hooks",Obj(Map(
          "detail_daemon"->Obj(Map("mode"->Str("both"),"amount"->Number(BigDecimal("0.35"))))))))
        assertNotEquals(conditioned,plain,s"ControlNet must affect ${if cuda then "CUDA" else "CPU"} output")
        assertNotEquals(hooked,conditioned,s"Detail Daemon must affect ${if cuda then "CUDA" else "CPU"} output")
        def window(start: String,end: String): NativeJson = Obj(unit.asInstanceOf[Obj].fields ++ Map(
          "guidance_start"->Number(BigDecimal(start)),"guidance_end"->Number(BigDecimal(end))))
        def withWindow(fields: Map[String,NativeJson],start: String,end: String): Map[String,NativeJson] =
          fields.updated("controlnet",Arr(Vector(window(start,end))))
        // With three steps, neither [0,1/3] nor [1/3,2/3] fits inside [0.1,0.5].
        assertEquals(run(withWindow(base,"0.1","0.5")),plain)
        // Both intervals admit exactly the first step.
        val oneStep=run(withWindow(base,"0","0.5"))
        assertNotEquals(oneStep,plain)
        assertEquals(oneStep,run(withWindow(base,"0","0.34")))
        // img2img strength .5 leaves two of four steps; the window is relative
        // to those two remaining steps, as in Diffusers.
        val img=base.updated("steps",Number(4)).updated("img2img",Obj(Map(
          "init_image"->Str(image.toString),"denoising_strength"->Number(BigDecimal("0.5")),"resize_mode"->Number(0))))
        val imgPlain=run(img,"img2img")
        val imgNoStep=run(withWindow(img,"0.1","0.6"),"img2img")
        val imgOneStep=run(withWindow(img,"0","0.5"),"img2img")
        assertEquals(imgNoStep,imgPlain)
        assertNotEquals(imgOneStep,imgPlain)
    finally
      val paths=Files.walk(output)
      try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }

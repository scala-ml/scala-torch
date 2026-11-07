package scalatorch.models.worker

import java.awt.image.BufferedImage
import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import scalatorch.models.{SdxlSingleFileConverter,UNetExecution,VaeExecution}
import NativeJson.*

/** Opt-in worker verification for the SDXL 1.0 single-file checkpoint without Python. */
class NativeSingleFileSdxlSuite extends munit.FunSuite:
  override val munitTimeout: scala.concurrent.duration.Duration = scala.concurrent.duration.Duration(20,"minutes")

  private def setting(name: String): Option[String] =
    val properties=new java.util.Properties()
    val config=Vector(Path.of(".superpowers/sdxl-single-file-test.properties"),
      Path.of("../.superpowers/sdxl-single-file-test.properties")).find(Files.isRegularFile(_))
    config.foreach(path => scala.util.Using.resource(Files.newInputStream(path))(properties.load))
    sys.env.get(name).orElse(Option(properties.getProperty(name)))

  private def generate(backend: NativeDiffusionBackend,context: NativeWorkContext,output: Path): BufferedImage =
    val params=Obj(Map("prompt"->Str("a photograph of a red apple on a wooden table"),
      "width"->Number(64),"height"->Number(64),"steps"->Number(2),"seed"->Number(4321),
      "cfg_scale"->Number(BigDecimal("7")),"sampler_name"->Str("Euler"),"scheduler"->Str("Uniform")))
    val result=backend.execute(NativeRequest("gen","txt2img",
      Map("output_dir"->Str(output.toString),"params"->params),None),context)
    val image=result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields
    assertEquals(image("width"),Number(64))
    assertEquals(image("height"),Number(64))
    ImageIO.read(Path.of(image("path").asInstanceOf[Str].value).toFile)

  test("SDXL 1.0 single-file checkpoint loads and matches the diffusers directory generation") {
    val single=setting("SCALATORCH_SDXL_SINGLE_FILE")
    val directory=setting("SCALATORCH_SDXL_DIFFUSERS_DIR")
    assume(single.nonEmpty && directory.nonEmpty,
      "set SCALATORCH_SDXL_SINGLE_FILE and SCALATORCH_SDXL_DIFFUSERS_DIR to the SDXL 1.0 artifacts")
    val root=Path.of(directory.get).toRealPath()
    val singlePath=SdxlSingleFileConverter.ensureSingleFile(root,Path.of(single.get)).toRealPath()
    val siblingTokenizer=singlePath.getParent.resolve("tokenizer")
    if !Files.exists(siblingTokenizer) then
      Files.createDirectories(siblingTokenizer)
      for n <- Seq("vocab.json","merges.txt","tokenizer_config.json","special_tokens_map.json") do
        val src=root.resolve("tokenizer").resolve(n)
        if Files.exists(src) then Files.copy(src,siblingTokenizer.resolve(n))

    val useCuda=sys.env.get("SCALATORCH_CUDA_TESTS").contains("1")
    val vaeExec=if useCuda then VaeExecution.Cuda(offloadWeights=true) else VaeExecution.Cpu
    val unetExec=if useCuda then UNetExecution.Cuda(offloadWeights=true) else UNetExecution.Cpu
    val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    val output=Files.createTempDirectory("sdxl-single-file-")
    var matched=false
    try
      val singleImage = locally {
        val singleBackend=new NativeDiffusionBackend(singlePath,Path.of("-"),vaeExecution=vaeExec,unetExecution=unetExec)
        val loaded=singleBackend.execute(NativeRequest("load","load_model",Map.empty,None),context)
        assertEquals(loaded("family"),Str("sdxl"))
        assertEquals(loaded("kind"),Str("safetensors"))
        val img=generate(singleBackend,context,output)
        singleBackend.execute(NativeRequest("unload","unload",Map.empty,None),context)
        img
      }
      System.gc()
      val scheduler=Files.createTempFile("sdxl-euler-",".json")
      Files.writeString(scheduler,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
      val directoryImage = locally {
        val directoryBackend=new NativeDiffusionBackend(root,scheduler,vaeExecution=vaeExec,unetExecution=unetExec)
        directoryBackend.execute(NativeRequest("load","load_model",Map.empty,None),context)
        val img=generate(directoryBackend,context,output)
        directoryBackend.execute(NativeRequest("unload","unload",Map.empty,None),context)
        img
      }
      System.gc()
      assertEquals((directoryImage.getWidth,directoryImage.getHeight),(64,64))
      var difference=0
      for y <- 0 until 64; x <- 0 until 64 do
        val a=singleImage.getRGB(x,y)
        val b=directoryImage.getRGB(x,y)
        difference=math.max(difference,math.max(
          math.abs((a&0xff)- (b&0xff)),
          math.max(math.abs(((a>>8)&0xff)-((b>>8)&0xff)),math.abs(((a>>16)&0xff)-((b>>16)&0xff)))))
      assert(difference<=1,
        s"SDXL single-file image diverges from the diffusers directory reference by $difference levels; outputs=$output")
      matched=true
    finally
      if matched then
        val files=Files.walk(output)
        try files.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(p => {
          require(p.toAbsolutePath.normalize().startsWith(output.toAbsolutePath.normalize()))
          Files.delete(p);()
        }) finally files.close()
  }

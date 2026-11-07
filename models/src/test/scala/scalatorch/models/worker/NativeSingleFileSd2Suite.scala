package scalatorch.models.worker

import java.awt.image.BufferedImage
import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import NativeJson.*

/** Opt-in worker verification for the released SD2.1 single-file checkpoint without Python. */
class NativeSingleFileSd2Suite extends munit.FunSuite:
  override val munitTimeout: scala.concurrent.duration.Duration = scala.concurrent.duration.Duration(20,"minutes")

  private def setting(name: String): Option[String] =
    val properties=new java.util.Properties()
    val config=Vector(Path.of(".superpowers/sd2-single-file-test.properties"),
      Path.of("../.superpowers/sd2-single-file-test.properties")).find(Files.isRegularFile(_))
    config.foreach(path => scala.util.Using.resource(Files.newInputStream(path))(properties.load))
    sys.env.get(name).orElse(Option(properties.getProperty(name)))

  private def generate(backend: NativeDiffusionBackend,context: NativeWorkContext,output: Path): BufferedImage =
    val params=Obj(Map("prompt"->Str("a red barn beside a green field, clear morning light"),
      "width"->Number(64),"height"->Number(64),"steps"->Number(2),"seed"->Number(4321),
      "cfg_scale"->Number(BigDecimal("7")),"sampler_name"->Str("Euler"),"scheduler"->Str("Uniform")))
    val result=backend.execute(NativeRequest("gen","txt2img",
      Map("output_dir"->Str(output.toString),"params"->params),None),context)
    val image=result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields
    assertEquals(image("width"),Number(64))
    assertEquals(image("height"),Number(64))
    ImageIO.read(Path.of(image("path").asInstanceOf[Str].value).toFile)

  test("released SD2.1 single-file checkpoint loads and matches the diffusers directory generation") {
    val single=setting("SCALATORCH_SD2_SINGLE_FILE")
    val directory=setting("SCALATORCH_SD2_DIFFUSERS_DIR")
    assume(single.nonEmpty && directory.nonEmpty,
      "set SCALATORCH_SD2_SINGLE_FILE and SCALATORCH_SD2_DIFFUSERS_DIR to the released SD2.1 artifacts")
    val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    val singleBackend=new NativeDiffusionBackend(Path.of(single.get).toRealPath(),Path.of("-"))
    val loaded=singleBackend.execute(NativeRequest("load","load_model",Map.empty,None),context)
    assertEquals(loaded("family"),Str("sd2"))
    assertEquals(loaded("kind"),Str("safetensors"))
    assertEquals(loaded("hash"),
      Str("df955bdf6b682338ea9b55dfc0d8f3475aadf4836e204893d28b82355e0956d2"))
    val output=Files.createTempDirectory("sd2-single-file-")
    var matched=false
    try
      val singleImage=generate(singleBackend,context,output)
      singleBackend.execute(NativeRequest("unload","unload",Map.empty,None),context)
      System.gc()
      val scheduler=Files.createTempFile("sd2-euler-",".json")
      Files.writeString(scheduler,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
      val directoryBackend=new NativeDiffusionBackend(Path.of(directory.get).toRealPath(),scheduler)
      directoryBackend.execute(NativeRequest("load","load_model",Map.empty,None),context)
      val directoryImage=generate(directoryBackend,context,output)
      directoryBackend.execute(NativeRequest("unload","unload",Map.empty,None),context)
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
        s"single-file image diverges from the diffusers directory reference by $difference levels; outputs=$output")
      matched=true
    finally
      if matched then
        val files=Files.walk(output)
        try files.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(p => {
          require(p.toAbsolutePath.normalize().startsWith(output.toAbsolutePath.normalize()))
          Files.delete(p);()
        }) finally files.close()
  }

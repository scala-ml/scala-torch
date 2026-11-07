package scalatorch.models.worker

import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import NativeJson.*

class NativeEulerAncestralWorkerSuite extends munit.FunSuite:
  test("Euler a runs through generation, img2img and both Hires paths") {
    val configured=sys.props.get("scalatorch.sdxl.dir")
    assume(configured.nonEmpty,"tiny SDXL fixture required")
    val root=Path.of(configured.get)
    val out=Files.createTempDirectory("native-euler-a")
    try
      val scheduler=out.resolve("scheduler.json")
      Files.writeString(scheduler,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,
        "beta_schedule":"scaled_linear","timestep_spacing":"leading","steps_offset":1}""")
      val backend=new NativeDiffusionBackend(root,scheduler)
      val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
      def run(kind: String,p: Map[String,NativeJson]=Map.empty)=backend.execute(NativeRequest("r",kind,p,None),context)
      val samplers=run("list_samplers")("samplers").asInstanceOf[Arr].values
      assert(samplers.exists(_.asInstanceOf[Obj].fields("name")==Str("Euler a")))
      run("load_model")
      val base=Map[String,NativeJson]("prompt"->Str("apple"),"width"->Number(8),"height"->Number(8),
        "steps"->Number(2),"seed"->Number(42),"sampler_name"->Str("Euler a"),"scheduler"->Str("Automatic"))
      def params(g: Map[String,NativeJson])=Map[String,NativeJson]("output_dir"->Str(out.toString),"params"->Obj(g))
      def image(result: Map[String,NativeJson])=result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields
      def pixels(result: Map[String,NativeJson]) =
        val picture=ImageIO.read(Path.of(image(result)("path").asInstanceOf[Str].value).toFile)
        picture.getRGB(0,0,picture.getWidth,picture.getHeight,null,0,picture.getWidth).toVector
      val first=run("txt2img",params(base))
      assertEquals(pixels(first),pixels(run("txt2img",params(base))))
      assertNotEquals(pixels(first),pixels(run("txt2img",params(base.updated("seed",Number(43))))) )
      val fallback=run("txt2img",params(base.updated("scheduler",Str("Karras"))))
      assert(fallback("warnings").asInstanceOf[Arr].values.exists(_.asInstanceOf[Str].value.contains("Euler a")))
      val expFallback=run("txt2img",params(base.updated("scheduler",Str("Exponential"))))
      assert(expFallback("warnings").asInstanceOf[Arr].values.exists(_.asInstanceOf[Str].value.contains("Exponential")))
      assertEquals(pixels(expFallback),pixels(fallback))
      val img=base.updated("img2img",Obj(Map("init_image"->image(first)("path"),
        "denoising_strength"->Number(BigDecimal("0.5")),"resize_mode"->Number(0))))
      assertEquals(image(run("img2img",params(img)))("width"),Number(8))
      val hires=Obj(Map("enabled"->Bool(true),"scale"->Number(2),"upscaler"->Str("Latent"),
        "second_pass_steps"->Number(2),"denoising_strength"->Number(BigDecimal("0.5"))))
      assertEquals(image(run("txt2img",params(base.updated("hires",hires))))("width"),Number(64))
      val fix=Obj(Map("img2img_hires_fix"->Obj(Map("upscaler"->Str("Nearest"),"ratio"->Number(2),
        "steps"->Number(1),"denoise"->Number(BigDecimal("0.5")),"sampler"->Str("Euler a")))))
      assertEquals(image(run("txt2img",params(base.updated("hooks",fix))))("width"),Number(64))
      val bad=base.updated("extra",Obj(Map("prediction_type"->Str("sample"))))
      intercept[NativeBackendError](run("txt2img",params(bad)))
    finally
      val paths=Files.walk(out)
      try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }

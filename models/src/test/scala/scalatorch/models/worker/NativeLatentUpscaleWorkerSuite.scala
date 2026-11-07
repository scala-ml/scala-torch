package scalatorch.models.worker

import java.nio.file.{Files,Path}
import java.awt.image.BufferedImage
import javax.imageio.ImageIO
import NativeJson.*

class NativeLatentUpscaleWorkerSuite extends munit.FunSuite:
  test("img2img latent upscale preserves original image size and performs the requested steps") {
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty,"public tiny model required")
    val root=Path.of(configured.get).getParent
    val output=Files.createTempDirectory("latent-upscale-worker")
    try
      val config=output.resolve("scheduler.json")
      Files.writeString(config,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
      val image=new BufferedImage(16,16,BufferedImage.TYPE_INT_RGB)
      for y <- 0 until 16; x <- 0 until 16 do image.setRGB(x,y,(x*16<<16)|(y*16<<8)|64)
      val file=output.resolve("input.png");ImageIO.write(image,"png",file.toFile)
      val backend=new NativeDiffusionBackend(root,config)
      val progress=scala.collection.mutable.ArrayBuffer.empty[Map[String,NativeJson]]
      val context=new NativeWorkContext(()=>false,()=>true,(event,fields)=>if event=="progress" then progress+=fields)
      backend.execute(NativeRequest("load","load_model",Map.empty,None),context)
      val fields=NativeGenerationDefaults.generation++Map("width"->Number(24),"height"->Number(24),"steps"->Number(2),
        "seed"->Number(42),"sampler_name"->Str("Euler"),"img2img"->Obj(Map("init_image"->Str(file.toString),
          "denoising_strength"->Number(BigDecimal("0.5")))),
        "hooks"->Obj(Map("latent_upscale"->Obj(Map("method"->Str("nearest-exact"),"scheduler"->Str("simple"))))))
      val result=backend.execute(NativeRequest("generate","img2img",Map("params"->Obj(fields),"output_dir"->Str(output.toString)),None),context)
      val path=result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
      val actual=ImageIO.read(Path.of(path).toFile)
      assertEquals((actual.getWidth,actual.getHeight),(24,24))
      assertEquals(progress.map(_("step")).toVector,Vector(Number(1),Number(2)))
      assert(progress.forall(_("total")==Number(2)))
      val lowStrength=fields("img2img").asInstanceOf[Obj].fields.updated("denoising_strength",Number(BigDecimal("0.0005")))
      val longFields=fields.updated("steps",Number(1)).updated("img2img",Obj(lowStrength))
      progress.clear()
      val longResult=backend.execute(NativeRequest("long","img2img",Map("params"->Obj(longFields),"output_dir"->Str(output.toString)),None),context)
      val longPath=longResult("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
      assertEquals(ImageIO.read(Path.of(longPath).toFile).getWidth,24)
      assertEquals(progress.map(_("step")).toVector,Vector(Number(1)))
      val ancestralFields=longFields.updated("sampler_name",Str("Euler a")).updated("steps",Number(2))
        .updated("img2img",Obj(lowStrength.updated("denoising_strength",Number(BigDecimal("0.001")))))
      progress.clear()
      val ancestralResult=backend.execute(NativeRequest("long-ancestral","img2img",
        Map("params"->Obj(ancestralFields),"output_dir"->Str(output.toString)),None),context)
      val ancestralPath=ancestralResult("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
      assertEquals(ImageIO.read(Path.of(ancestralPath).toFile).getWidth,24)
      assertEquals(progress.map(_("step")).toVector,Vector(Number(1),Number(2)))
      for chosen <- Vector("simple","karras") do
        val dpmHook=Obj(Map("latent_upscale"->Obj(Map("method"->Str("nearest-exact"),"scheduler"->Str(chosen)))))
        val dpmFields=ancestralFields.updated("sampler_name",Str("DPM++ 2M")).updated("hooks",dpmHook)
        progress.clear()
        val dpm=backend.execute(NativeRequest("long-dpm","img2img",
          Map("params"->Obj(dpmFields),"output_dir"->Str(output.toString)),None),context)
        assertEquals(progress.map(_("step")).toVector,Vector(Number(1),Number(2)))
        val warnings=dpm("warnings").asInstanceOf[Arr].values
        assertEquals(warnings.contains(Str("repeated DPM sigma steps use a stable first-order fallback")),chosen=="simple")
      val mask=new BufferedImage(16,16,BufferedImage.TYPE_BYTE_GRAY)
      for y <- 0 until 16; x <- 0 until 16 do mask.setRGB(x,y,if x<8 then 0xff000000 else 0xffffffff)
      val maskFile=output.resolve("mask.png");ImageIO.write(mask,"png",maskFile.toFile)
      val masked=fields("img2img").asInstanceOf[Obj].fields++Map("mask"->Str(maskFile.toString),"mask_blur"->Number(0))
      progress.clear()
      val maskedResult=backend.execute(NativeRequest("masked","img2img",Map("params"->Obj(fields.updated("img2img",Obj(masked))),
        "output_dir"->Str(output.toString)),None),context)
      val maskedPath=maskedResult("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
      val maskedImage=ImageIO.read(Path.of(maskedPath).toFile)
      assertEquals((maskedImage.getWidth,maskedImage.getHeight),(24,24))
      assertEquals(progress.map(_("step")).toVector,Vector(Number(1),Number(2)))
      val odd=new BufferedImage(9,101,BufferedImage.TYPE_INT_RGB)
      for y <- 0 until 101; x <- 0 until 9 do odd.setRGB(x,y,(x*28<<16)|(y*2<<8)|64)
      ImageIO.write(odd,"png",file.toFile)
      val oddMasked=backend.execute(NativeRequest("odd-mask","img2img",Map("params"->Obj(fields.updated("img2img",Obj(masked))),
        "output_dir"->Str(output.toString)),None),context)
      val oddPath=oddMasked("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
      assertEquals(ImageIO.read(Path.of(oddPath).toFile).getWidth,24)
      def generate(mode: Int): Vector[Int] =
        val img=fields("img2img").asInstanceOf[Obj].fields.updated("resize_mode",Number(mode))
        val p=fields.updated("img2img",Obj(img))
        val out=backend.execute(NativeRequest("resize","img2img",Map("params"->Obj(p),"output_dir"->Str(output.toString)),None),context)
        val path=out("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
        ImageIO.read(Path.of(path).toFile).getRGB(0,0,24,24,null,0,24).toVector
      assertEquals(generate(2),generate(0),"latent upscale must ignore pixel resize mode")
    finally
      val paths=Files.walk(output)
      try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }

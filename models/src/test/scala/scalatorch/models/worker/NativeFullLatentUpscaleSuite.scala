package scalatorch.models.worker

import java.nio.file.{Files,Path}
import java.awt.image.BufferedImage
import javax.imageio.ImageIO
import NativeJson.*

class NativeFullLatentUpscaleSuite extends munit.FunSuite:
  test("full-resolution masked latent upscale matches NUI reference pixels") {
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty,"public tiny model required")
    val root=Path.of(configured.get).getParent
    val output=Files.createTempDirectory("full-latent-upscale")
    try
      val config=output.resolve("scheduler.json")
      Files.writeString(config,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear","timestep_spacing":"linspace"}""")
      val source=new BufferedImage(20,16,BufferedImage.TYPE_INT_RGB)
      val mask=new BufferedImage(20,16,BufferedImage.TYPE_BYTE_GRAY)
      for y <- 0 until 16; x <- 0 until 20 do
        val gray=(x*11+y*7)%256
        source.setRGB(x,y,(gray<<16)|(gray<<8)|gray)
        mask.setRGB(x,y,if x>=7 && x<13 && y>=5 && y<11 then 0xffffffff else 0xff000000)
      val sourcePath=output.resolve("source.png");val maskPath=output.resolve("mask.png")
      ImageIO.write(source,"png",sourcePath.toFile);ImageIO.write(mask,"png",maskPath.toFile)
      val backend=new NativeDiffusionBackend(root,config)
      val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
      backend.execute(NativeRequest("load","load_model",Map.empty,None),context)
      val fields=Map[String,NativeJson]("prompt"->Str("a red cat"),"width"->Number(16),"height"->Number(16),
        "steps"->Number(2),"seed"->Number(42),"cfg_scale"->Number(1),"sampler_name"->Str("Euler"),
        "extra"->Obj(Map("rng"->Str("cpu"))),"img2img"->Obj(Map("init_image"->Str(sourcePath.toString),
          "mask"->Str(maskPath.toString),"mask_blur"->Number(1),"inpainting_fill"->Number(1),
          "inpaint_full_res"->Bool(true),"inpaint_full_res_padding"->Number(2),"denoising_strength"->Number(BigDecimal("0.5")))),
        "hooks"->Obj(Map("latent_upscale"->Obj(Map("method"->Str("nearest-exact"),"scheduler"->Str("normal"))))))
      val result=backend.execute(NativeRequest("full","img2img",Map("params"->Obj(fields),"output_dir"->Str(output.toString)),None),context)
      val file=result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
      val actual=ImageIO.read(Path.of(file).toFile)
      assertEquals((actual.getWidth,actual.getHeight),(20,16))
      val stream=this.getClass.getResourceAsStream("/full-latent-upscale-reference.json")
      val expected=try NativeJson.parse(new String(stream.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8)).asInstanceOf[Obj].fields("rgb")
        .asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toInt) finally stream.close()
      val error=(for y <- 0 until 16; x <- 0 until 20; c <- 0 until 3 yield
        math.abs(((actual.getRGB(x,y) >>> (16-8*c))&255)-expected((y*20+x)*3+c))).max
      assert(error<=2,s"full-resolution output max channel error $error")
    finally
      val paths=Files.walk(output)
      try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }

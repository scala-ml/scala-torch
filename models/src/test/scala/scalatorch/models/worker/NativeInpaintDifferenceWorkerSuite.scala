package scalatorch.models.worker

import java.nio.file.{Files,Path}
import java.awt.image.BufferedImage
import javax.imageio.ImageIO
import NativeJson.*

class NativeInpaintDifferenceWorkerSuite extends munit.FunSuite:
  test("inpaint difference generation matches an explicitly supplied mask") {
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty,"public tiny model required")
    val root=Path.of(configured.get).getParent
    val output=Files.createTempDirectory("native-inpaint-difference")
    try
      val config=output.resolve("scheduler.json")
      Files.writeString(config,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
      val base=new BufferedImage(16,16,BufferedImage.TYPE_INT_RGB)
      val altered=new BufferedImage(16,16,BufferedImage.TYPE_INT_RGB)
      val mask=new BufferedImage(16,16,BufferedImage.TYPE_BYTE_GRAY)
      for y <- 0 until 16; x <- 0 until 16 do
        base.setRGB(x,y,0xff203040)
        altered.setRGB(x,y,if x>=8 then 0xffe0a040 else 0xff203040)
        mask.setRGB(x,y,if x>=8 then 0xffffffff else 0xff000000)
      val basePath=output.resolve("base.png");val imagePath=output.resolve("altered.png");val maskPath=output.resolve("mask.png")
      ImageIO.write(base,"png",basePath.toFile);ImageIO.write(altered,"png",imagePath.toFile);ImageIO.write(mask,"png",maskPath.toFile)
      val backend=new NativeDiffusionBackend(root,config)
      val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
      backend.execute(NativeRequest("load","load_model",Map.empty,None),context)
      val source=Map[String,NativeJson]("init_image"->Str(imagePath.toString),"denoising_strength"->Number(BigDecimal("0.75")),"mask_blur"->Number(0))
      val fields=Map[String,NativeJson]("prompt"->Str("a red cat"),"width"->Number(16),"height"->Number(16),
        "steps"->Number(2),"seed"->Number(42),"cfg_scale"->Number(1),"sampler_name"->Str("Euler"))
      def generate(extra: Map[String,NativeJson]): Vector[Int] =
        val result=backend.execute(NativeRequest("generate","img2img",Map("params"->Obj(fields++extra),"output_dir"->Str(output.toString)),None),context)
        val file=result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
        ImageIO.read(Path.of(file).toFile).getRGB(0,0,16,16,null,0,16).toVector
      val expected=generate(Map("img2img"->Obj(source.updated("mask",Str(maskPath.toString)))))
      val hook=Obj(Map("inpaint_difference"->Obj(Map("base_path"->Str(basePath.toString),"threshold"->Number(1)))))
      val actual=generate(Map("img2img"->Obj(source),"hooks"->hook))
      assertEquals(actual,expected)
      for y <- 0 until 16; x <- 0 until 16 do
        mask.setRGB(x,y,if x==8 && y>0 && y<15 then 0xffffffff else 0xff000000)
      ImageIO.write(mask,"png",maskPath.toFile)
      val contourExpected=generate(Map("img2img"->Obj(source.updated("mask",Str(maskPath.toString)))))
      val contourHook=Obj(Map("inpaint_difference"->Obj(Map("base_path"->Str(basePath.toString),"threshold"->Number(1),"contours"->Bool(true)))))
      assertEquals(generate(Map("img2img"->Obj(source),"hooks"->contourHook)),contourExpected)
    finally
      val paths=Files.walk(output)
      try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }

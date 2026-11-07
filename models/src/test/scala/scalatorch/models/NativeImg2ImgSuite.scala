package scalatorch.models.worker

import java.nio.file.{Files,Path}
import java.awt.image.BufferedImage
import javax.imageio.ImageIO

class NativeImg2ImgSuite extends munit.FunSuite:
  test("worker img2img decodes resizes generates replays and preserves unmasked inpaint pixels") {
    import NativeJson.*
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty,"public tiny model required")
    val root=Path.of(configured.get).getParent
    val output=Files.createTempDirectory("native-img2img")
    val config=output.resolve("scheduler.json")
    Files.writeString(config,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
    val input=output.resolve("input.png")
    val inpaintSource=output.resolve("inpaint-source.png")
    val maskFile=output.resolve("mask.png")
    val stream=this.getClass.getResourceAsStream("/image-input/orientation-6.png")
    try Files.write(input,stream.readAllBytes()) finally stream.close()
    val original=new BufferedImage(10,6,BufferedImage.TYPE_INT_RGB)
    val maskImage=new BufferedImage(10,6,BufferedImage.TYPE_BYTE_GRAY)
    for y <- 0 until 6; x <- 0 until 10 do
      original.setRGB(x,y,if x<5 then 0xff204060 else 0xffc08040)
      maskImage.setRGB(x,y,if x<5 then 0xff000000 else 0xffffffff)
    ImageIO.write(original,"png",inpaintSource.toFile);ImageIO.write(maskImage,"png",maskFile.toFile)
    try
      val backend=new NativeDiffusionBackend(root,config)
      val progress=scala.collection.mutable.ArrayBuffer.empty[Map[String,NativeJson]]
      val context=new NativeWorkContext(() => false,() => true,(event,fields) => {
        if event=="progress" then progress+=fields
      })
      def run(kind: String,p: Map[String,NativeJson])=backend.execute(NativeRequest("test",kind,p,None),context)
      run("load_model",Map.empty)
      assert(backend.supported.contains("img2img"))
      val imageParams=Map("init_image"->Str(input.toString),"denoising_strength"->Number(BigDecimal("0.5")),
        "resize_mode"->Number(2),"mask"->Null,"mask_blur"->Number(4),"inpainting_fill"->Number(1),
        "inpaint_full_res"->Bool(false),"inpaint_full_res_padding"->Number(32),"inpainting_mask_invert"->Bool(false))
      def request(sampler: String,source: Map[String,NativeJson])=Map("output_dir"->Str(output.toString),
        "params"->Obj(NativeGenerationDefaults.generation++Map("width"->Number(10),"height"->Number(6),
          "steps"->Number(4),"seed"->Number(42),"sampler_name"->Str(sampler),"img2img"->Obj(source))))
      def pixels(p: Map[String,NativeJson])=
        progress.clear()
        val result=run("img2img",p)
        val file=result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
        val image=ImageIO.read(Path.of(file).toFile)
        assertEquals(image.getWidth,10);assertEquals(image.getHeight,6)
        assertEquals(progress.map(_("step")).toVector,Vector(Number(1),Number(2)))
        assert(progress.forall(_("total")==Number(2)))
        image.getRGB(0,0,10,6,null,0,10).toVector
      for sampler <- Seq("Euler","DPM++ 2M") do
        val p=request(sampler,imageParams)
        val first=pixels(p)
        assertEquals(pixels(p),first)
        assert(pixels(request(sampler,imageParams.updated("resize_mode",Number(0))))!=first)
      val baseImg2Img=request("Euler",imageParams)
      val variationParams=baseImg2Img("params").asInstanceOf[Obj].fields.updated("subseed",Number(1337))
        .updated("subseed_strength",Number(0.7))
      val varied=baseImg2Img.updated("params",Obj(variationParams))
      assertEquals(pixels(varied),pixels(baseImg2Img),"img2img must ignore variation noise")
      val variationResponse=run("img2img",varied)
      assert(variationResponse("warnings").asInstanceOf[Arr].values.contains(Str("variation seed is only applied in txt2img")))
      def names()=
        val files=Files.list(output)
        try files.toArray.toVector finally files.close()
      intercept[NativeBackendError](run("txt2img",request("Euler",imageParams)))
      val inpaintParams=imageParams.updated("init_image",Str(inpaintSource.toString)).updated("resize_mode",Number(0))
        .updated("mask",Str(maskFile.toString)).updated("mask_blur",Number(0))
      val inpaintResult=run("img2img",request("Euler",inpaintParams))
      val inpaintPath=Path.of(inpaintResult("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
      val inpainted=ImageIO.read(inpaintPath.toFile)
      assertEquals(inpainted.getRGB(0,0)&0xffffff,original.getRGB(0,0)&0xffffff)
      assertEquals(inpainted.getRGB(4,5)&0xffffff,original.getRGB(4,5)&0xffffff)
      assert((5 until 10).exists(x => (0 until 6).exists(y => inpainted.getRGB(x,y)!=original.getRGB(x,y))))
      val alphaMask=output.resolve("alpha-mask.png")
      val alphaImage=new BufferedImage(10,6,BufferedImage.TYPE_INT_ARGB)
      for y <- 0 until 6; x <- 0 until 10 do
        alphaImage.setRGB(x,y,if x<5 then 0x00ffffff else 0xff000000)
      ImageIO.write(alphaImage,"png",alphaMask.toFile)
      val alphaResult=run("img2img",request("Euler",inpaintParams.updated("mask",Str(alphaMask.toString))))
      val alphaPath=Path.of(alphaResult("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
      val alphaPainted=ImageIO.read(alphaPath.toFile)
      assertEquals(alphaPainted.getRGB(0,0,10,6,null,0,10).toVector,inpainted.getRGB(0,0,10,6,null,0,10).toVector,
        "a partly transparent RGBA mask uses alpha, independent of its RGB colors")
      val beforeSupportedJobs=names().toSet
      val fullRes=run("img2img",request("Euler",inpaintParams.updated("inpaint_full_res",Bool(true))))
      assertEquals(fullRes("images").asInstanceOf[Arr].values.size,1)
      val blurFill=run("img2img",request("Euler",inpaintParams.updated("inpainting_fill",Number(0))))
      assertEquals(blurFill("images").asInstanceOf[Arr].values.size,1)
      intercept[IllegalArgumentException](run("img2img",request("Euler",inpaintParams.updated("mask",Str(output.resolve("missing-mask.png").toString)))))
      intercept[IllegalArgumentException](run("img2img",request("Euler",inpaintParams.updated("mask_blur",Number(65)))))
      assert(names().toSet.size>=beforeSupportedJobs.size+2,"supported full-resolution and blur-fill jobs should publish images")
      intercept[IllegalArgumentException](run("img2img",request("Euler",imageParams.updated("init_image",Str("")))))
      intercept[IllegalArgumentException](run("img2img",request("Euler",imageParams.updated("denoising_strength",Number(2)))))
      val denied=new NativeWorkContext(() => false,() => false,(_,_) => ())
      intercept[java.util.concurrent.CancellationException](backend.execute(
        NativeRequest("denied","img2img",request("Euler",imageParams),None),denied))
      val beforeCancelled=names().toSet
      intercept[java.util.concurrent.CancellationException](backend.execute(
        NativeRequest("denied-mask","img2img",request("Euler",inpaintParams),None),denied))
      assertEquals(names().toSet,beforeCancelled)
    finally
      val files=Files.list(output)
      try files.forEach(p => {Files.delete(p);()}) finally files.close()
      Files.delete(output)
  }

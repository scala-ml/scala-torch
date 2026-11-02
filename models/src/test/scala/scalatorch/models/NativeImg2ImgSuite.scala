package scalatorch.models.worker

import java.nio.file.{Files,Path}
import javax.imageio.ImageIO

class NativeImg2ImgSuite extends munit.FunSuite:
  test("worker img2img decodes resizes generates replays and rejects active inpainting") {
    import NativeJson.*
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty,"public tiny model required")
    val root=Path.of(configured.get).getParent
    val output=Files.createTempDirectory("native-img2img")
    val config=output.resolve("scheduler.json")
    Files.writeString(config,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
    val input=output.resolve("input.png")
    val stream=this.getClass.getResourceAsStream("/image-input/orientation-6.png")
    try Files.write(input,stream.readAllBytes()) finally stream.close()
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
      def names()=
        val files=Files.list(output)
        try files.toArray.toVector finally files.close()
      val before=names().toSet
      intercept[NativeBackendError](run("txt2img",request("Euler",imageParams)))
      intercept[NativeBackendError](run("img2img",request("Euler",imageParams.updated("mask",Str(input.toString)))))
      intercept[IllegalArgumentException](run("img2img",request("Euler",imageParams.updated("init_image",Str("")))))
      intercept[IllegalArgumentException](run("img2img",request("Euler",imageParams.updated("denoising_strength",Number(2)))))
      val denied=new NativeWorkContext(() => false,() => false,(_,_) => ())
      intercept[java.util.concurrent.CancellationException](backend.execute(
        NativeRequest("denied","img2img",request("Euler",imageParams),None),denied))
      assertEquals(names().toSet,before)
    finally
      val files=Files.list(output)
      try files.forEach(p => {Files.delete(p);()}) finally files.close()
      Files.delete(output)
  }

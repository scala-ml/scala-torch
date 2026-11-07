package scalatorch.models.worker

import scalatorch.models.PyTorchEmbeddingFixture
import java.nio.file.{Files,Path}
import java.nio.{ByteBuffer,ByteOrder}
import scala.concurrent.duration.*

class NativeDiffusionBackendSuite extends munit.FunSuite:
  override val munitTimeout: Duration=180.seconds
  test("pixel Hires Fix refines a txt2img result") {
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty,"public tiny model required")
    val root=Path.of(configured.get).getParent
    val output=Files.createTempDirectory("native-txt-hires")
    import NativeJson.*
    val backend=new NativeDiffusionBackend(root,root.resolve("euler-demo.json"))
    assert(backend.features.contains("img2img_hires_fix"))
    val passes=scala.collection.mutable.ArrayBuffer.empty[String]
    val context=new NativeWorkContext(() => false,() => true,(name,fields) =>
      if name=="progress" then passes+=fields("pass").asInstanceOf[Str].value)
    val upscalers=backend.execute(NativeRequest("samplers","list_samplers",Map.empty,None),context)("upscalers")
      .asInstanceOf[Arr].values.collect { case Str(name) => name }.toSet
    assert(upscalers.contains("Nearest") && upscalers.contains("Lanczos") && upscalers.contains("Latent (bicubic)"))
    backend.execute(NativeRequest("load","load_model",Map("path"->Str(root.toString)),None),context)
    val hook=Obj(Map("img2img_hires_fix"->Obj(Map("ratio"->Number(2),"upscaler"->Str("Nearest"),
      "denoise"->Number(0.5),"steps"->Number(1)))))
    val params=Obj(Map("prompt"->Str("a cat"),"width"->Number(32),"height"->Number(32),
      "steps"->Number(1),"seed"->Number(42),"hooks"->hook))
    val result=backend.execute(NativeRequest("generate","txt2img",Map("output_dir"->Str(output.toString),"params"->params),None),context)
    val image=result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields
    assertEquals((image("width"),image("height")),(Number(64),Number(64)))
    assertEquals(passes.toVector,Vector("base","img2img_hires"))
    passes.clear()
    val hires=Obj(Map("enabled"->Bool(true),"scale"->Number(2),"second_pass_steps"->Number(1),
      "denoising_strength"->Number(BigDecimal("0.5"))))
    val combined=Obj(params.asInstanceOf[Obj].fields.updated("hires",hires))
    val result2=backend.execute(NativeRequest("combined","txt2img",Map("output_dir"->Str(output.toString),"params"->combined),None),context)
    val image2=result2("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields
    assertEquals((image2("width"),image2("height")),(Number(128),Number(128)))
    assertEquals(passes.toVector,Vector("base","hires","img2img_hires"))
    val target=scalatorch.models.PretrainedUNet.loadWithLinearTargets(root.resolve("unet")).linearTargets("unet.time_embedding.linear_1")
    val adapter=output.resolve("tiny.safetensors")
    writeAdapter(adapter,"unet.time_embedding.linear_1",target.data.shape.dims(1).toInt,target.data.shape.dims(0).toInt)
    val ref=Obj(Map("name"->Str("tiny"),"weight"->Number(1)))
    val switchParams=Obj(params.asInstanceOf[Obj].fields.updated("loras",Arr(Vector(ref))))
    def switchImage(selection: Option[Arr]): Vector[Int] =
      val envelope=Map[String,NativeJson]("output_dir"->Str(output.toString),"params"->switchParams,
        "lora_paths"->Obj(Map("tiny"->Str(adapter.toString)))) ++ selection.map("img2img_hires_loras"->_)
      val response=backend.execute(NativeRequest("switch","txt2img",envelope,None),context)
      val path=response("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
      val decoded=javax.imageio.ImageIO.read(Path.of(path).toFile)
      decoded.getRGB(0,0,decoded.getWidth,decoded.getHeight,null,0,decoded.getWidth).toVector
    val inherited=switchImage(None)
    assertEquals(switchImage(Some(Arr(Vector(ref)))),inherited,"explicit refinement adapter must match inherited adapter")
    assertNotEquals(inherited,switchImage(Some(Arr(Vector.empty))),
      "explicitly empty refinement adapters must remove base-pass LoRA")
    def refinementTag(tag: Boolean,selection: Option[Arr]): Vector[Int] =
      val overrideHook=Obj(Map("img2img_hires_fix"->Obj(Map("ratio"->Number(2),"upscaler"->Str("Nearest"),
        "denoise"->Number(0.5),"steps"->Number(1),"prompt"->Str("a cat"+(if tag then "<lora:tiny:1>" else ""))))))
      val generation=Obj(params.asInstanceOf[Obj].fields.updated("hooks",overrideHook))
      val envelope=Map[String,NativeJson]("output_dir"->Str(output.toString),"params"->generation,
        "lora_paths"->Obj(Map("tiny"->Str(adapter.toString)))) ++ selection.map("img2img_hires_loras"->_)
      val result=backend.execute(NativeRequest("refinement-tag","txt2img",envelope,None),context)
      val path=result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
      val decoded=javax.imageio.ImageIO.read(Path.of(path).toFile)
      decoded.getRGB(0,0,decoded.getWidth,decoded.getHeight,null,0,decoded.getWidth).toVector
    assertEquals(refinementTag(true,None),refinementTag(false,Some(Arr(Vector(ref)))),
      "refinement prompt tag must match a resolved refinement LoRA")
    assertEquals(refinementTag(true,Some(Arr(Vector.empty))),refinementTag(false,Some(Arr(Vector.empty))),
      "explicit refinement selection takes precedence over prompt tags")
  }
  test("worker runs masked img2img with a synthetic nine-channel U-Net") {
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty,"public tiny model required")
    val root=Path.of(configured.get).getParent
    val model=Files.createTempDirectory("native-nine-channel")
    val output=Files.createTempDirectory("native-nine-output")
    val scheduler=output.resolve("scheduler.json")
    Files.writeString(scheduler,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
    import NativeJson.*
    try
      makeNineChannelModel(root,model)
      val backend=new NativeDiffusionBackend(model,scheduler)
      val passes=scala.collection.mutable.ArrayBuffer.empty[String]
      val context=new NativeWorkContext(() => false,() => true,(name,fields) => if name=="progress" then
        passes+=fields("pass").asInstanceOf[Str].value)
      backend.execute(NativeRequest("load","load_model",Map("path"->Str(model.toString)),None),context)
      val source=new java.awt.image.BufferedImage(16,16,java.awt.image.BufferedImage.TYPE_INT_RGB)
      val mask=new java.awt.image.BufferedImage(16,16,java.awt.image.BufferedImage.TYPE_INT_RGB)
      for y <- 0 until 16; x <- 0 until 16 do
        source.setRGB(x,y,0xff204060)
        mask.setRGB(x,y,if x>=8 then 0xffffffff else 0xff000000)
      val sourcePath=output.resolve("source.png");val maskPath=output.resolve("mask.png")
      javax.imageio.ImageIO.write(source,"png",sourcePath.toFile)
      javax.imageio.ImageIO.write(mask,"png",maskPath.toFile)
      val img2img=Obj(Map("init_image"->Str(sourcePath.toString),"mask"->Str(maskPath.toString),
        "mask_blur"->Number(0),"inpainting_fill"->Number(1),"denoising_strength"->Number(1)))
      val generation=Obj(Map("prompt"->Str("edit right half"),"width"->Number(16),"height"->Number(16),
        "steps"->Number(1),"seed"->Number(42),"img2img"->img2img))
      val params=Map("output_dir"->Str(output.toString),"params"->generation)
      val result=backend.execute(NativeRequest("inpaint","img2img",params,None),context)
      val entry=result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields
      val image=javax.imageio.ImageIO.read(Path.of(entry("path").asInstanceOf[Str].value).toFile)
      assertEquals((image.getWidth,image.getHeight),(16,16))
      assertEquals(image.getRGB(2,2)&0xffffff,source.getRGB(2,2)&0xffffff)
      assert((8 until 16).exists(x => (0 until 16).exists(y => image.getRGB(x,y)!=source.getRGB(x,y))))
      val refineHook=Obj(Map("img2img_hires_fix"->Obj(Map("ratio"->Number(2),"upscaler"->Str("Nearest"),
        "steps"->Number(1),"denoise"->Number(1),"prompt"->Str("a green apple"),"cfg"->Number(4),
        "sampler"->Str("Euler"),"scheduler"->Str("Automatic")))))
      passes.clear()
      val refined=backend.execute(NativeRequest("inpaint-refine","img2img",
        Map("output_dir"->Str(output.toString),"params"->Obj(generation.asInstanceOf[Obj].fields.updated("hooks",refineHook))),None),context)
      val refinedEntry=refined("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields
      val refinedImage=javax.imageio.ImageIO.read(Path.of(refinedEntry("path").asInstanceOf[Str].value).toFile)
      assertEquals((refinedImage.getWidth,refinedImage.getHeight),(64,64),
        "nine-channel inpaint result must support whole-image pixel Hires refinement")
      assertEquals(refinedEntry("width"),Number(64))
      assertEquals(passes.toVector,Vector("base","img2img_hires"),"refinement must report its own pass")
      val refinedAgain=backend.execute(NativeRequest("inpaint-refine-again","img2img",
        Map("output_dir"->Str(output.toString),"params"->Obj(generation.asInstanceOf[Obj].fields.updated("hooks",refineHook))),None),context)
      val againEntry=refinedAgain("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields
      val againImage=javax.imageio.ImageIO.read(Path.of(againEntry("path").asInstanceOf[Str].value).toFile)
      assertEquals(againImage.getRGB(0,0,64,64,null,0,64).toVector,
        refinedImage.getRGB(0,0,64,64,null,0,64).toVector,"same seed and refinement settings must replay deterministically")
      val zeroDenoise=Obj(Map("img2img_hires_fix"->Obj(Map("ratio"->Number(2),"upscaler"->Str("Lanczos"),"denoise"->Number(0)))))
      passes.clear()
      val resized=backend.execute(NativeRequest("inpaint-resize","img2img",
        Map("output_dir"->Str(output.toString),"params"->Obj(generation.asInstanceOf[Obj].fields.updated("hooks",zeroDenoise))),None),context)
      val resizedEntry=resized("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields
      assertEquals((resizedEntry("width"),resizedEntry("height")),(Number(64),Number(64)))
      assertEquals(passes.toVector,Vector("base"),"zero denoise should only resize the completed first pass")
      def publishedImages(): Set[String]=
        val listed=Files.list(output)
        try
          import scala.jdk.CollectionConverters.*
          listed.iterator().asScala.map(_.getFileName.toString).filter(name=>name.startsWith("native-") && name.endsWith(".png")).toSet
        finally listed.close()
      val beforeCancel=publishedImages()
      var cancelled=false
      val cancelContext=new NativeWorkContext(() => cancelled,() => !cancelled,(name,fields) =>
        if name=="progress" && fields.get("pass").contains(Str("base")) then cancelled=true)
      intercept[java.util.concurrent.CancellationException] {
        backend.execute(NativeRequest("cancel-refine","img2img",
          Map("output_dir"->Str(output.toString),"params"->Obj(generation.asInstanceOf[Obj].fields.updated("hooks",refineHook))),None),cancelContext)
      }
      assertEquals(publishedImages(),beforeCancel,"cancellation after the first pass must not publish an image")
      val activeControlNetValue=Arr(Vector(Obj(Map("enabled"->Bool(true)))))
      val activeControlNet=generation.asInstanceOf[Obj].fields.updated("controlnet",activeControlNetValue)
        .updated("hooks",refineHook)
      val controlFailure=intercept[IllegalArgumentException](backend.execute(NativeRequest("refine-controlnet","img2img",
        Map("output_dir"->Str(output.toString),"params"->Obj(activeControlNet)),None),context))
      assert(controlFailure.getMessage.contains("model is required"))
      val loraFailure=intercept[IllegalArgumentException](backend.execute(NativeRequest("refine-lora-switch","img2img",
        Map("output_dir"->Str(output.toString),"img2img_hires_loras"->Arr(Vector(Obj(Map("name"->Str("other"))))),
          "params"->Obj(generation.asInstanceOf[Obj].fields.updated("hooks",refineHook))),None),context))
      assert(loraFailure.getMessage.contains("missing resolved path for LoRA"))
      assert(!Files.exists(model.resolve("python")))
      val hires=Obj(Map("enabled"->Bool(true)))
      val hiresParams=Map("output_dir"->Str(output.toString),"params"->Obj(Map("width"->Number(16),
        "height"->Number(16),"steps"->Number(1),"hires"->hires)))
      val inpaintHires=backend.execute(NativeRequest("hires-inpaint","txt2img",hiresParams,None),context)
      assert(inpaintHires("warnings").asInstanceOf[Arr].values.exists {
        case Str(message) => message.contains("inpainting models; decoding first")
        case _ => false
      })
      passes.clear()
      val ignored=backend.execute(NativeRequest("hires-img2img","img2img",Map("output_dir"->Str(output.toString),
        "params"->Obj(generation.asInstanceOf[Obj].fields.updated("hires",hires))),None),context)
      val ignoredEntry=ignored("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields
      val ignoredImage=javax.imageio.ImageIO.read(Path.of(ignoredEntry("path").asInstanceOf[Str].value).toFile)
      assertEquals((ignoredImage.getWidth,ignoredImage.getHeight),(16,16))
      assertEquals(ignoredImage.getRGB(0,0,16,16,null,0,16).toVector,image.getRGB(0,0,16,16,null,0,16).toVector)
      assertEquals(passes.toVector,Vector("base"),"ordinary hires is ignored for img2img by NUI")
    finally
      for directory <- Vector(model,output) do
        val paths=Files.walk(directory)
        try
          import scala.jdk.CollectionConverters.*
          paths.iterator().asScala.toVector.sortBy(_.getNameCount).reverse.foreach(Files.delete(_))
        finally paths.close()
  }

  private[models] def makeNineChannelModel(source: Path,target: Path): Unit =
    import scala.jdk.CollectionConverters.*
    val paths=Files.walk(source)
    try paths.iterator().asScala.foreach { path =>
      val destination=target.resolve(source.relativize(path))
      if Files.isDirectory(path) then Files.createDirectories(destination)
      else Files.copy(path,destination)
    } finally paths.close()
    val unet=target.resolve("unet")
    val weights=unet.resolve("diffusion_pytorch_model.safetensors")
    val entries=scala.util.Using.resource(scalatorch.io.SafeTensors.open(weights)) { archive =>
      archive.tensors.toVector.sortBy(_._1).map { (name,info) =>
        val tensor=archive.readFloat(name)
        if name=="conv_in.weight" then
          val dims=info.shape.dims
          require(dims.size==4 && dims(1)==4,"fixture U-Net conv_in must have four latent channels")
          val out=dims(0).toInt;val height=dims(2).toInt;val width=dims(3).toInt
          val input=tensor.toFloatArray
          val widened=Array.fill(out*9*height*width)(0f)
          for o <- 0 until out; c <- 0 until 9; y <- 0 until height; x <- 0 until width do
            val destination=((o*9+c)*height+y)*width+x
            if c<4 then widened(destination)=input(((o*4+c)*height+y)*width+x)
            else widened(destination)=.025f*(c-3)
          (name,Vector(out.toLong,9L,height.toLong,width.toLong),widened)
        else (name,info.shape.dims,tensor.toFloatArray)
      }
    }
    var offset=0L
    val header=entries.map { (name,dims,values) =>
      val begin=offset;offset+=values.length.toLong*4
      s"\"$name\":{\"dtype\":\"F32\",\"shape\":[${dims.mkString(",")}],\"data_offsets\":[$begin,$offset]}"
    }.mkString("{",",","}").getBytes(java.nio.charset.StandardCharsets.UTF_8)
    val data=ByteBuffer.allocate(Math.toIntExact(offset)).order(ByteOrder.LITTLE_ENDIAN)
    entries.foreach((_,_,values)=>values.foreach(data.putFloat))
    Files.write(weights,ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(header.length.toLong).array()++header++data.array())
    val config=unet.resolve("config.json")
    val original=Files.readString(config)
    val widened=original.replaceFirst("\"in_channels\"\\s*:\\s*4","\"in_channels\": 9")
    require(widened!=original,"fixture U-Net config has no four-channel input declaration")
    Files.writeString(config,widened)

  private def writeAdapter(path: Path,name: String,input: Int,output: Int): Unit =
    val down=Array.fill(input)(1f/input)
    val up=Array.fill(output)(8f)
    val entries=Vector((s"$name.lora_down.weight",Vector(1,input),down),
      (s"$name.lora_up.weight",Vector(output,1),up))
    var offset=0
    val header=entries.map { (key,dims,values) =>
      val begin=offset;offset+=values.length*4
      s"\"$key\":{\"dtype\":\"F32\",\"shape\":[${dims.mkString(",")}],\"data_offsets\":[$begin,$offset]}"
    }.mkString("{",",","}").getBytes(java.nio.charset.StandardCharsets.UTF_8)
    val data=ByteBuffer.allocate(offset).order(ByteOrder.LITTLE_ENDIAN)
    entries.foreach((_,_,values)=>values.foreach(data.putFloat))
    Files.write(path,ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(header.length.toLong).array()++header++data.array())

  private def writeEmbedding(path: Path,width: Int): Unit =
    val values=Array.fill(width)(0.75f)
    val header=s"\"emb_params\":{\"dtype\":\"F32\",\"shape\":[${width}],\"data_offsets\":[0,${width*4}]}"
      .prepended('{').appended('}').getBytes(java.nio.charset.StandardCharsets.UTF_8)
    val data=ByteBuffer.allocate(width*4).order(ByteOrder.LITTLE_ENDIAN)
    values.foreach(data.putFloat)
    Files.write(path,ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(header.length.toLong).array()++header++data.array())

  private def writeSdxlEmbedding(path: Path,leftWidth: Int,rightWidth: Int): Unit =
    val entries=Vector("clip_l"->Array.fill(leftWidth)(0.75f),"clip_g"->Array.fill(rightWidth)(-0.5f))
    var offset=0L
    val header=entries.map { (name,values) =>
      val begin=offset;offset+=values.length.toLong*4
      s"\"$name\":{\"dtype\":\"F32\",\"shape\":[1,${values.length}],\"data_offsets\":[$begin,$offset]}"
    }.mkString("{",",","}").getBytes(java.nio.charset.StandardCharsets.UTF_8)
    val data=ByteBuffer.allocate(Math.toIntExact(offset)).order(ByteOrder.LITTLE_ENDIAN)
    entries.foreach((_,values)=>values.foreach(data.putFloat))
    Files.write(path,ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(header.length.toLong).array()++header++data.array())

  test("native SDXL worker loads paired textual inversion into both CLIP encoders and resets it on unload") {
    val configured=sys.props.get("scalatorch.sdxl.dir")
    assume(configured.nonEmpty,"public tiny SDXL model required")
    val root=Path.of(configured.get)
    val output=Files.createTempDirectory("native-sdxl-embedding")
    val scheduler=output.resolve("scheduler.json")
    Files.writeString(scheduler,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
    import NativeJson.*
    val backend=new NativeDiffusionBackend(root,scheduler)
    val context=new NativeWorkContext(() => false,() => true,(_,_) => ())
    def load(): Unit=backend.execute(NativeRequest("load","load_model",Map("path"->Str(root.toString)),None),context)
    def run(paths: Vector[String])=backend.execute(NativeRequest("image","txt2img",
      Map("output_dir"->Str(output.toString),"embedding_paths"->Arr(paths.map(Str.apply)),"params"->Obj(Map(
        "prompt"->Str("a red cat sks"),"width"->Number(16),"height"->Number(16),"steps"->Number(1),"seed"->Number(42)))),None),context)
    def pixels(result: Map[String,NativeJson])=
      val image=result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
      val decoded=javax.imageio.ImageIO.read(Path.of(image).toFile)
      decoded.getRGB(0,0,16,16,null,0,16).toVector
    try
      load()
      val baseline=pixels(run(Vector.empty))
      val clipLWidth=scalatorch.models.PretrainedClip.load(root.resolve("text_encoder"))
        .tokenEmbedding.weight.data.shape.dims(1).toInt
      val clipGWidth=scalatorch.models.PretrainedClip.load(root.resolve("text_encoder_2"))
        .tokenEmbedding.weight.data.shape.dims(1).toInt
      val embedding=output.resolve("sks.safetensors")
      writeSdxlEmbedding(embedding,clipLWidth,clipGWidth)
      def listing()=backend.execute(NativeRequest("list","embedding_inspector",Map("operation"->Str("list"),
        "embedding_paths"->Arr(Vector(Str(embedding.toString)))),None),context)
      assertEquals(listing()("available"),Arr(Vector(Str("sks"))))
      assertEquals(listing()("loaded"),Arr(Vector.empty))
      val active=pixels(run(Vector(embedding.toString)))
      val entry=listing()("loaded").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields
      assertEquals(entry("name"),Str("sks"))
      val encoders=entry("encoders").asInstanceOf[Obj].fields
      assertEquals(encoders("clip_l"),Arr(Vector(Number(1),Number(clipLWidth))))
      assertEquals(encoders("clip_g"),Arr(Vector(Number(1),Number(clipGWidth))))
      assertEquals(listing()("available"),Arr(Vector.empty))
      assertNotEquals(active,baseline,"SDXL textual inversion must reach both CLIP encoders")
      assertEquals(pixels(run(Vector.empty)),active,"loaded SDXL textual inversion must persist across requests")
      backend.execute(NativeRequest("unload","unload",Map.empty,None),context)
      load()
      assertEquals(listing()("loaded"),Arr(Vector.empty))
      assertEquals(listing()("available"),Arr(Vector(Str("sks"))))
      val ptBaseline=pixels(run(Vector.empty))
      val ptEmbedding=output.resolve("sks.pt")
      PyTorchEmbeddingFixture.write(ptEmbedding,"sdxl",("clip_l",Vector.fill(clipLWidth)(0.75f),Vector(1,clipLWidth)),
        ("clip_g",Vector.fill(clipGWidth)(-0.5f),Vector(1,clipGWidth)))
      val activePt=pixels(run(Vector(ptEmbedding.toString)))
      assertNotEquals(activePt,ptBaseline,"SDXL .pt textual inversion must reach both CLIP encoders")
      assertEquals(activePt,active,
        "restricted .pt SDXL embedding must match the equivalent safetensors tensors")
      backend.execute(NativeRequest("unload","unload",Map.empty,None),context)
      load()
      assertEquals(pixels(run(Vector.empty)),baseline,"unload must clear both SDXL tokenizers and embedding rows")
      val mixed=output.resolve("native_mixed.safetensors")
      val save=backend.execute(NativeRequest("mix","embedding_inspector",Map("operation"->Str("save"),
        "rows"->Arr(Vector(Obj(Map("ids"->Obj(Map("clip_l"->Number(1),"clip_g"->Number(1))),"weight"->Number(1))))),
        "output"->Str(mixed.toString),"embedding_paths"->Arr(Vector.empty)),None),context)
      assertEquals(save("path"),Str(mixed.toString))
      assert(Files.isRegularFile(mixed))
      val savedList=listing()("loaded").asInstanceOf[Arr].values.map(_.asInstanceOf[Obj].fields("name"))
      assert(savedList.contains(Str("native_mixed")))
      val inspected=backend.execute(NativeRequest("inspect","embedding_inspector",Map("operation"->Str("inspect"),
        "text"->Str("native_mixed"),"limit"->Number(1)),None),context)
      assertEquals(inspected("metadata").asInstanceOf[Obj].fields("source"),Str("loaded model"))
      val savedBytes=Files.readAllBytes(mixed).toVector
      val second=output.resolve("other.safetensors")
      val savedParams=Map("operation"->Str("save"),"rows"->Arr(Vector(Obj(Map("ids"->Obj(Map("clip_l"->Number(1),"clip_g"->Number(1))))))),
        "output"->Str(mixed.toString),"overwrite"->Bool(true))
      backend.execute(NativeRequest("other","embedding_inspector",savedParams.updated("output",Str(second.toString)),None),context)
      def inspectOther()=backend.execute(NativeRequest("other-inspect","embedding_inspector",Map("operation"->Str("inspect"),
        "text"->Str("other"),"limit"->Number(1)),None),context)("encoders").asInstanceOf[Obj].fields.map((k,v)=>
          k->v.asInstanceOf[Obj].fields("vectors"))
      val otherBefore=inspectOther()
      val interrupted=output.resolve("interrupted.safetensors")
      val denied=new NativeWorkContext(()=>false,()=>false,(_,_)=>())
      intercept[java.util.concurrent.CancellationException](backend.execute(NativeRequest("denied","embedding_inspector",
        savedParams,None),denied))
      assertEquals(Files.readAllBytes(mixed).toVector,savedBytes)
      assertEquals(inspectOther(),otherBefore)
      val newRows=Arr(Vector(Obj(Map("ids"->Obj(Map("clip_l"->Number(1),"clip_g"->Number(1))))),
        Obj(Map("ids"->Obj(Map("clip_l"->Number(2),"clip_g"->Number(2)))))))
      backend.execute(NativeRequest("replace","embedding_inspector",savedParams.updated("rows",newRows).updated("concat",Bool(true)),None),context)
      assertNotEquals(Files.readAllBytes(mixed).toVector,savedBytes)
      assertEquals(inspectOther(),otherBefore)
      val entries=listing()("loaded").asInstanceOf[Arr].values.map(_.asInstanceOf[Obj].fields)
      assertEquals(entries.count(_("name")==Str("native_mixed")),1)
      assertEquals(entries.find(_("name")==Str("native_mixed")).get("encoders").asInstanceOf[Obj].fields("clip_l"),
        Arr(Vector(Number(2),Number(clipLWidth))))
      val opened=backend.execute(NativeRequest("open","embedding_editor",Map("operation"->Str("open"),
        "name"->Str("native_mixed"),"path"->Str(mixed.toString),"guidance"->Str("a:#ff0000")),None),context)
      assertEquals(opened("name"),Str("native_mixed"))
      assertEquals(opened("guidance").asInstanceOf[Arr].values.size,1)
      val update=backend.execute(NativeRequest("guide","embedding_editor",Map("operation"->Str("guidance"),
        "name"->Str("native_mixed"),"draft"->Obj(opened),"guidance"->Str("")),None),context)
      assertEquals(update("guidance"),Arr(Vector.empty))
      intercept[IllegalArgumentException](backend.execute(NativeRequest("stale","embedding_editor",Map("operation"->Str("guidance"),
        "name"->Str("native_mixed"),"draft"->Obj(opened.updated("model_token",Str("stale"))),"guidance"->Str("")),None),context))
      val beforeEdit=Files.readAllBytes(mixed).toVector
      val originalVectors=opened("vectors").asInstanceOf[Obj].fields
      val lRows=originalVectors("clip_l").asInstanceOf[Arr].values
      val first=lRows.head.asInstanceOf[Arr].values.updated(0,Number(BigDecimal("0.25")))
      val edited=Obj(opened.updated("vectors",Obj(originalVectors.updated("clip_l",Arr(lRows.updated(0,Arr(first)))))))
      intercept[IllegalArgumentException](backend.execute(NativeRequest("no-consent","embedding_editor",Map("operation"->Str("save"),
        "name"->Str("native_mixed"),"draft"->edited),None),context))
      assertEquals(Files.readAllBytes(mixed).toVector,beforeEdit)
      val saved=backend.execute(NativeRequest("save-edit","embedding_editor",Map("operation"->Str("save"),
        "name"->Str("native_mixed"),"draft"->edited,"overwrite"->Bool(true)),None),context)
      assertEquals(saved("reloaded"),Bool(true))
      val backup=Path.of(saved("backup").asInstanceOf[Str].value)
      assertEquals(Files.readAllBytes(backup).toVector,beforeEdit)
      val reopened=backend.execute(NativeRequest("reopen","embedding_editor",Map("operation"->Str("open"),
        "name"->Str("native_mixed")),None),context)
      assertEquals(reopened("vectors").asInstanceOf[Obj].fields("clip_l").asInstanceOf[Arr].values.head.asInstanceOf[Arr].values.head,
        Number(BigDecimal("0.25")))
      intercept[IllegalArgumentException](backend.execute(NativeRequest("stale-save","embedding_editor",Map("operation"->Str("save"),
        "name"->Str("native_mixed"),"draft"->edited,"overwrite"->Bool(true)),None),context))
      val stable=Files.readAllBytes(mixed).toVector
      val previewVectors=reopened("vectors").asInstanceOf[Obj].fields
      val previewLeft=previewVectors("clip_l").asInstanceOf[Arr].values.map { value =>
        val row=value.asInstanceOf[Arr].values
        Arr(row.map(_=>Number(10)))
      }
      val draft=Obj(reopened.updated("vectors",Obj(previewVectors.updated("clip_l",Arr(previewLeft)))))
      val previewParams=Map("output_dir"->Str(output.toString),"params"->Obj(Map("prompt"->Str("native_mixed"),
        "width"->Number(16),"height"->Number(16),"steps"->Number(1),"seed"->Number(42))))
      def previewImage(params: Map[String,NativeJson])=pixels(backend.execute(NativeRequest("preview","txt2img",params,None),context))
      val normal=previewImage(previewParams)
      val preview=previewImage(previewParams.updated("embedding_editor_preview",draft))
      assertNotEquals(preview,normal,"unsaved embedding edits must affect preview generation")
      assertEquals(previewImage(previewParams),normal,"preview must restore the active model")
      assertEquals(Files.readAllBytes(mixed).toVector,stable)
      intercept[IllegalArgumentException](previewImage(previewParams.updated("embedding_editor_preview",
        Obj(draft.asInstanceOf[Obj].fields.updated("model_token",Str("stale"))))))
      assertEquals(previewImage(previewParams),normal)
      val invalidGeneration=previewParams.updated("params",Obj(Map("prompt"->Str("native_mixed"),
        "width"->Number(16),"height"->Number(16),"steps"->Number(0),"seed"->Number(42))))
      intercept[IllegalArgumentException](previewImage(invalidGeneration.updated("embedding_editor_preview",draft)))
      assertEquals(previewImage(previewParams),normal,"failed preview must restore the active model")
      assertEquals(Files.readAllBytes(mixed).toVector,stable)
      val addedDuringPreview=output.resolve("preview_extra.safetensors")
      writeSdxlEmbedding(addedDuringPreview,clipLWidth,clipGWidth)
      val addedParams=previewParams.updated("embedding_editor_preview",draft)
        .updated("embedding_paths",Arr(Vector(Str(addedDuringPreview.toString))))
        .updated("params",Obj(Map("prompt"->Str("native_mixed preview_extra"),"width"->Number(16),
          "height"->Number(16),"steps"->Number(1),"seed"->Number(42))))
      previewImage(addedParams)
      val afterList=listing()("loaded").asInstanceOf[Arr].values.map(_.asInstanceOf[Obj].fields("name"))
      assert(afterList.contains(Str("preview_extra")),"embedding loaded during preview must persist")
      val afterPreview=backend.execute(NativeRequest("editor-after","embedding_editor",Map("operation"->Str("open"),
        "name"->Str("native_mixed")),None),context)
      assertEquals(afterPreview("vectors"),reopened("vectors"),"unsaved preview rows must not persist")
      val legacyOutput=output.resolve("native_created.pt")
      val ptSaved=backend.execute(NativeRequest("save-pt","embedding_inspector",Map("operation"->Str("save"),
        "rows"->Arr(Vector(Obj(Map("ids"->Obj(Map("clip_l"->Number(1),"clip_g"->Number(1))))))),
        "output"->Str(legacyOutput.toString),"step"->Number(42)),None),context)
      assertEquals(ptSaved("path"),Str(legacyOutput.toString))
      val decoded=scalatorch.models.ClipTextualInversion.readPyTorch(legacyOutput,true,clipLWidth,clipGWidth)
      assertEquals(decoded.clipL.size,1)
      assertEquals(decoded.clipG.get.size,1)
      assert(listing()("loaded").asInstanceOf[Arr].values.exists(_.asInstanceOf[Obj].fields("name")==Str("native_created")))
      val ptBefore=Files.readAllBytes(legacyOutput).toVector
      val ptDraft=backend.execute(NativeRequest("pt-open","embedding_editor",Map("operation"->Str("open"),
        "name"->Str("native_created")),None),context)
      val ptVectors=ptDraft("vectors").asInstanceOf[Obj].fields
      val ptRows=ptVectors("clip_l").asInstanceOf[Arr].values
      val ptFirst=ptRows.head.asInstanceOf[Arr].values.updated(0,Number(BigDecimal("0.25")))
      val ptEdited=Obj(ptDraft.updated("vectors",Obj(ptVectors.updated("clip_l",Arr(ptRows.updated(0,Arr(ptFirst)))))))
      val ptResult=backend.execute(NativeRequest("pt-save","embedding_editor",Map("operation"->Str("save"),
        "name"->Str("native_created"),"draft"->ptEdited,"overwrite"->Bool(true)),None),context)
      assertEquals(Files.readAllBytes(Path.of(ptResult("backup").asInstanceOf[Str].value)).toVector,ptBefore)
      val ptAfter=scalatorch.models.ClipTextualInversion.readPyTorch(legacyOutput,true,clipLWidth,clipGWidth)
      assertEquals(ptAfter.clipL.head.head,0.25f)
      val batch=output.resolve("native_batch.safetensors")
      val conflict=output.resolve("native_batch_Binary.safetensors")
      Files.writeString(conflict,"existing")
      val batchParams=Map("operation"->Str("save"),"rows"->Arr(Vector(Obj(Map("ids"->Obj(Map(
        "clip_l"->Number(1),"clip_g"->Number(1))))))),"output"->Str(batch.toString),"batch_presets"->Bool(true))
      intercept[java.nio.file.FileAlreadyExistsException](backend.execute(NativeRequest("batch-conflict","embedding_inspector",batchParams,None),context))
      assertEquals(Files.readString(conflict),"existing")
      assert(!Files.exists(batch))
      Files.delete(conflict)
      val batchResult=backend.execute(NativeRequest("batch","embedding_inspector",batchParams,None),context)
      val batchPaths=batchResult("paths").asInstanceOf[Arr].values.map(_.asInstanceOf[Str].value)
      assertEquals(batchPaths.size,9)
      assert(batchPaths.forall(path=>Files.isRegularFile(Path.of(path))))
      val batchLoaded=listing()("loaded").asInstanceOf[Arr].values.map(_.asInstanceOf[Obj].fields("name"))
      assert(batchLoaded.contains(Str("native_batch_Boost")))
      assert(batchLoaded.contains(Str("native_batch_Crop_low")))
      assert(!Files.exists(interrupted))
    finally
      try backend.execute(NativeRequest("unload","unload",Map.empty,None),context) catch case _: Throwable => ()
      val files=Files.walk(output)
      try
        import scala.jdk.CollectionConverters.*
        files.iterator().asScala.toVector.sortBy(_.getNameCount).reverse.foreach(Files.delete(_))
      finally files.close()
  }

  test("native SDXL model keyword applies selected embedding and LoRA") {
    val configured=sys.props.get("scalatorch.sdxl.dir")
    assume(configured.nonEmpty,"public tiny SDXL model required")
    val root=Path.of(configured.get)
    val output=Files.createTempDirectory("native-sdxl-keyword")
    val scheduler=output.resolve("scheduler.json")
    Files.writeString(scheduler,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
    import NativeJson.*
    val backend=new NativeDiffusionBackend(root,scheduler)
    val context=new NativeWorkContext(() => false,() => true,(_,_) => ())
    try
      backend.execute(NativeRequest("load","load_model",Map("path"->Str(root.toString)),None),context)
      val clipL=scalatorch.models.PretrainedClip.load(root.resolve("text_encoder")).finalNorm.width
      val clipG=scalatorch.models.PretrainedClip.load(root.resolve("text_encoder_2")).finalNorm.width
      val embedding=output.resolve("myembedding.safetensors")
      writeSdxlEmbedding(embedding,clipL,clipG)
      val target=scalatorch.models.PretrainedUNet.loadWithLinearTargets(root.resolve("unet")).linearTargets("unet.time_embedding.linear_1")
      val lora=output.resolve("selected.safetensors")
      writeAdapter(lora,"unet.time_embedding.linear_1",target.data.shape.dims(1).toInt,target.data.shape.dims(0).toInt)
      val hook=Obj(Map("model_keyword"->Obj(Map("_mapping_dir"->Str(output.toString),"lora"->Str("selected"),
        "_lora_path"->Str(lora.toString),"lora_name"->Str("selected.safetensors"),
        "embedding"->Str("myembedding"),"multiplier"->Number(0.5)))))
      val base=Map[String,NativeJson]("prompt"->Str("cat"),"width"->Number(16),"height"->Number(16),
        "steps"->Number(1),"seed"->Number(42))
      val catalog=(0 until 64).map(i => s"catalog-$i"->Str(lora.toString)).toMap.updated("selected",Str(lora.toString))
      val envelope=Map[String,NativeJson]("output_dir"->Str(output.toString),
        "embedding_paths"->Arr(Vector(Str(embedding.toString))),"lora_paths"->Obj(catalog))
      def generate(fields: Map[String,NativeJson]): Map[String,NativeJson]=backend.execute(
        NativeRequest("sdxl-keyword","txt2img",envelope.updated("params",Obj(fields)),None),context)
        ("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields
      def pixels(fields: Map[String,NativeJson]): Vector[Int]=
        val image=javax.imageio.ImageIO.read(Path.of(fields("path").asInstanceOf[Str].value).toFile)
        image.getRGB(0,0,16,16,null,0,16).toVector
      val selected=generate(base.updated("hooks",hook))
      assertEquals(selected("effective_prompt"),Str("<lora:selected:0.5> myembedding cat"))
      val explicit=generate(base.updated("prompt",Str("myembedding cat")).updated("loras",
        Arr(Vector(Obj(Map("name"->Str("selected"),"weight"->Number(0.5)))))))
      assertEquals(pixels(selected),pixels(explicit),"selected keyword networks must match direct SDXL generation")
      val varied=base.updated("subseed",Number(1337)).updated("subseed_strength",Number(0.5))
      val firstVariation=generate(varied)
      assertEquals(pixels(firstVariation),pixels(generate(varied)),"SDXL variation must replay deterministically")
      assertNotEquals(pixels(firstVariation),pixels(generate(base)),"SDXL variation must affect the image")
      // Exercise explicit CPU noise through the public generation request.
      val cpu=base.updated("extra",Obj(Map("rng"->Str("cpu"))))
      val cpuImage=generate(cpu)
      assertEquals(pixels(cpuImage),pixels(generate(cpu)),"CPU RNG must replay deterministically")
      assertNotEquals(pixels(cpuImage),pixels(generate(base)),"explicit CPU RNG must not use the native default stream")
      val fullVariation=cpu.updated("subseed",Number(1337)).updated("subseed_strength",Number(1))
      assertEquals(pixels(generate(fullVariation)),pixels(generate(cpu.updated("seed",Number(1337)))),
        "CPU variation endpoint must use the independently seeded PyTorch CPU stream")
    finally
      try backend.execute(NativeRequest("unload","unload",Map.empty,None),context) catch case _: Throwable => ()
      val files=Files.walk(output)
      try
        import scala.jdk.CollectionConverters.*
        files.iterator().asScala.toVector.sortBy(_.getNameCount).reverse.foreach(Files.delete(_))
      finally files.close()
  }

  test("native SDXL Hires retains original-size and crop conditioning through the second pass") {
    val configured=sys.props.get("scalatorch.sdxl.dir")
    assume(configured.nonEmpty,"public tiny SDXL model required")
    val root=Path.of(configured.get)
    val output=Files.createTempDirectory("native-sdxl-hires")
    val scheduler=output.resolve("scheduler.json")
    Files.writeString(scheduler,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
    import NativeJson.*
    val backend=new NativeDiffusionBackend(root,scheduler)
    val context=new NativeWorkContext(() => false,() => true,(_,_) => ())
    try
      backend.execute(NativeRequest("load","load_model",Map("path"->Str(root.toString)),None),context)
      val hires=Obj(Map("enabled"->Bool(true),"scale"->Number(2),"upscaler"->Str("Latent (antialiased)"),
        "second_pass_steps"->Number(1),"denoising_strength"->Number(BigDecimal("0.55"))))
      val extra=Obj(Map("sdfx_sdxl_original_height"->Str("24"),"sdfx_sdxl_original_width"->Str("32"),
        "sdfx_sdxl_crop_top"->Str("2"),"sdfx_sdxl_crop_left"->Str("4")))
      val params=Map("output_dir"->Str(output.toString),"params"->Obj(Map("prompt"->Str("a red apple"),
        "negative_prompt"->Str("blurry"),"width"->Number(16),"height"->Number(16),"steps"->Number(1),
        "seed"->Number(42),"extra"->extra,"hires"->hires)))
      val result=backend.execute(NativeRequest("sdxl-hires","txt2img",params,None),context)
      val entry=result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields
      val image=javax.imageio.ImageIO.read(Path.of(entry("path").asInstanceOf[Str].value).toFile)
      assertEquals((image.getWidth,image.getHeight),(64,64))
      assertEquals(entry("width"),Number(64))
      assertEquals(entry("height"),Number(64))
      val overrides=hires.asInstanceOf[Obj].fields++Map("prompt"->Str("a blue ocean"),"negative_prompt"->Str("grainy"),
        "cfg_scale"->Number(0),"sampler_name"->Str("Euler"),"scheduler"->Str("Uniform"))
      val fields=params("params").asInstanceOf[Obj].fields.updated("hires",Obj(overrides))
      val changed=backend.execute(NativeRequest("sdxl-hires-override","txt2img",params.updated("params",Obj(fields)),None),context)
      val changedEntry=changed("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields
      val changedImage=javax.imageio.ImageIO.read(Path.of(changedEntry("path").asInstanceOf[Str].value).toFile)
      assertNotEquals(changedImage.getRGB(0,0,64,64,null,0,64).toVector,image.getRGB(0,0,64,64,null,0,64).toVector,
        "Hires overrides must change the second pass")
      val invalid=Obj(overrides.updated("sampler_name",Str("unknown sampler")))
      intercept[NativeBackendError](backend.execute(NativeRequest("sdxl-hires-invalid","txt2img",
        params.updated("params",Obj(fields.updated("hires",invalid))),None),context))
      val clipLWidth=scalatorch.models.PretrainedClip.load(root.resolve("text_encoder")).finalNorm.width
      val clipGWidth=scalatorch.models.PretrainedClip.load(root.resolve("text_encoder_2")).finalNorm.width
      val onlyHires=output.resolve("hiresonly.safetensors")
      writeSdxlEmbedding(onlyHires,clipLWidth,clipGWidth)
      val specific=Obj(hires.asInstanceOf[Obj].fields.updated("prompt",Str("hiresonly")))
      backend.execute(NativeRequest("sdxl-hires-embedding","txt2img",params
        .updated("embedding_paths",Arr(Vector(Str(onlyHires.toString))))
        .updated("params",Obj(fields.updated("hires",specific))),None),context)
      val loadedEmbeddings=backend.execute(NativeRequest("list","embedding_inspector",Map("operation"->Str("list")),None),context)
      assert(loadedEmbeddings("loaded").asInstanceOf[Arr].values.exists(_.asInstanceOf[Obj].fields("name")==Str("hiresonly")))
      val target=scalatorch.models.PretrainedUNet.loadWithLinearTargets(root.resolve("unet")).linearTargets("unet.time_embedding.linear_1")
      val adapter=output.resolve("xl-hires-lora.safetensors")
      writeAdapter(adapter,"unet.time_embedding.linear_1",target.data.shape.dims(1).toInt,target.data.shape.dims(0).toInt)
      val ref=Obj(Map("name"->Str("xl-hires-lora"),"weight"->Number(1)))
      val loraParams=params.updated("lora_paths",Obj(Map("xl-hires-lora"->Str(adapter.toString))))
        .updated("params",Obj(params("params").asInstanceOf[Obj].fields.updated("loras",Arr(Vector(ref)))))
      def loraImage(hiresLoras: Option[Arr]): Vector[Int] =
        val result=backend.execute(NativeRequest("xl-hires-lora","txt2img",loraParams++hiresLoras.map("hires_loras"->_),None),context)
        val entry=result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields
        javax.imageio.ImageIO.read(Path.of(entry("path").asInstanceOf[Str].value).toFile).getRGB(0,0,64,64,null,0,64).toVector
      val inherited=loraImage(None)
      assertEquals(loraImage(Some(Arr(Vector(ref)))),inherited)
      assert(loraImage(Some(Arr(Vector.empty)))!=inherited,"SDXL second-pass LoRA must be switchable")
    finally
      try backend.execute(NativeRequest("unload","unload",Map.empty,None),context) catch case _: Throwable => ()
      val files=Files.walk(output)
      try
        import scala.jdk.CollectionConverters.*
        files.iterator().asScala.toVector.sortBy(_.getNameCount).reverse.foreach(Files.delete(_))
      finally files.close()
  }

  test("public worker loads indexed CLIP UNet and VAE archives and hashes their source files") {
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty,"public tiny model required")
    val source=Path.of(configured.get).getParent
    val temp=Files.createTempDirectory("native-indexed")
    import NativeJson.*
    try
      for (part,filename) <- Vector("text_encoder"->"model.safetensors","unet"->"diffusion_pytorch_model.safetensors",
          "vae"->"diffusion_pytorch_model.safetensors") do
        val target=Files.createDirectory(temp.resolve(part))
        Files.copy(source.resolve(part).resolve("config.json"),target.resolve("config.json"))
        val shard="weights-00001-of-00001.safetensors"
        Files.copy(source.resolve(part).resolve(filename),target.resolve(shard))
        val names=scala.util.Using.resource(scalatorch.io.SafeTensors.open(target.resolve(shard)))(_.tensors.keys.toVector)
        Files.writeString(target.resolve(filename+".index.json"),NativeJson.render(Obj(Map(
          "weight_map"->Obj(names.map(_->Str(shard)).toMap)))))
      val tokenizer=Files.createDirectory(temp.resolve("tokenizer"))
      for name <- Vector("vocab.json","merges.txt") do Files.copy(source.resolve("tokenizer").resolve(name),tokenizer.resolve(name))
      val config=temp.resolve("scheduler.json")
      Files.writeString(config,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
      val backend=new NativeDiffusionBackend(temp,config)
      val context=new NativeWorkContext(() => false,() => true,(_,_) => ())
      val result=backend.execute(NativeRequest("load","load_model",Map("path"->Str(temp.toString)),None),context)
      assert(result("hash").asInstanceOf[Str].value.matches("[0-9a-f]{64}"))
      val image=backend.execute(NativeRequest("image","txt2img",Map("output_dir"->Str(temp.toString),
        "params"->Obj(Map("width"->Number(10),"height"->Number(6),"steps"->Number(1)))),None),context)
      assertEquals(image("images").asInstanceOf[Arr].values.size,1)
    finally
      val paths=Files.walk(temp)
      try
        import scala.jdk.CollectionConverters.*
        paths.iterator().asScala.toVector.sortBy(_.getNameCount).reverse.foreach(Files.delete(_))
      finally paths.close()
  }
  test("public native backend loads generates PNG and unloads without Python") {
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty,"public tiny model required")
    val root=Path.of(configured.get).getParent
    val config=Files.createTempFile("native-euler", ".json")
    val output=Files.createTempDirectory("native-output")
    Files.writeString(config,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
    try
      val backend=new NativeDiffusionBackend(root,config)
      val events=scala.collection.mutable.ArrayBuffer.empty[String]
      val context=new NativeWorkContext(() => false,() => true,(name,_) => { events+=name;() })
      def run(kind: String,params: Map[String,NativeJson])=
        backend.execute(NativeRequest("r",kind,params,None),context)
      import NativeJson.*
      val catalog=run("list_models",Map("dirs"->Arr(Vector(Str(root.getParent.toString)))))
      assertEquals(catalog("models").asInstanceOf[Arr].values.size,1)
      assertEquals(run("list_models",Map("dirs"->Arr(Vector(Str(output.toString)))))("models"),Arr(Vector.empty))
      val samplers=run("list_samplers",Map.empty)("samplers").asInstanceOf[Arr].values
      assertEquals(samplers.map(_.asInstanceOf[Obj].fields("name")),Vector("Euler","Euler a","Heun","LMS","DPM2","DPM2 a","DPM++ 2S a","DPM++ SDE","DPM++ 2M","DPM++ 2M SDE","DPM++ 2M SDE Heun","DPM++ 3M SDE","DDIM","UniPC","LCM").map(Str(_)))
      val metadata=run("load_model",Map("path"->Str(root.toString),"dtype"->Str("fp32"),"vram_mode"->Str("cpu")))
      assert(metadata("hash").asInstanceOf[Str].value.matches("[0-9a-f]{64}"))
      assertEquals(backend.systemStatus("model").asInstanceOf[Obj].fields("hash"),metadata("hash"))
      val externalVae=output.resolve("external VAE.fp16.safetensors")
      Files.copy(root.resolve("vae/diffusion_pytorch_model.safetensors"),externalVae)
      val swapped=run("set_vae",Map("path"->Str(externalVae.toString)))
      assert(swapped("vae_hash").asInstanceOf[Str].value.matches("[0-9a-f]{64}"))
      val retained=backend.systemStatus("model")
      intercept[Exception](run("set_vae",Map("path"->Str(output.resolve("missing.safetensors").toString))))
      assertEquals(backend.systemStatus("model"),retained)
      val deniedVae=new NativeWorkContext(() => false,() => false,(_,_) => ())
      intercept[java.util.concurrent.CancellationException](backend.execute(NativeRequest("vae-denied","set_vae",Map("path"->Null),None),deniedVae))
      assertEquals(backend.systemStatus("model"),retained)
      val restored=run("set_vae",Map("path"->Null))
      assertEquals(restored("vae_name"),Null)
      intercept[NativeBackendError](run("load_model",Map("dtype"->Str("fp16"))))
      val result=run("txt2img",NativeGenerationDefaults.envelope++Map("output_dir"->Str(output.toString),"vae_recovery_dir"->Str(output.toString),"params"->Obj(NativeGenerationDefaults.generation++Map(
        "checkpoint"->Str(root.toString),"vae"->Str("Automatic"),
        "prompt"->Str("a red cat"),"width"->Number(10),"height"->Number(6),"steps"->Number(2),
        "seed"->Number(42),"sampler_name"->Str("DPM++ 2M"),"scheduler"->Str("Automatic")))))
      assertEquals(result("scheduler"),Str("Karras"))
      val images=result("images").asInstanceOf[Arr].values
      val path=Path.of(images.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
      val image=javax.imageio.ImageIO.read(path.toFile)
      assertEquals(image.getWidth,10);assertEquals(image.getHeight,6)
      assertEquals(events.toVector,Vector("progress","progress"))
      def pixelsFor(prompt: String,negative: String="",clipSkip: Int=1): Vector[Int]=
        val generated=run("txt2img",Map("output_dir"->Str(output.toString),"params"->Obj(Map(
          "prompt"->Str(prompt),"negative_prompt"->Str(negative),"width"->Number(10),"height"->Number(6),
          "steps"->Number(2),"seed"->Number(42),"clip_skip"->Number(clipSkip)))))
        val file=generated("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
        val decoded=javax.imageio.ImageIO.read(Path.of(file).toFile)
        decoded.getRGB(0,0,10,6,null,0,10).toVector
      val plainPrompt=pixelsFor("a red cat")
      // Exercise the public request rather than only the interpolation helper.
      def variation(seed: Long,subseed: Long,strength: Double)=
        val response=run("txt2img",Map("output_dir"->Str(output.toString),"params"->Obj(Map(
          "prompt"->Str("a red cat"),"width"->Number(10),"height"->Number(6),"steps"->Number(2),
          "seed"->Number(seed),"subseed"->Number(subseed),"subseed_strength"->Number(strength)))))
        val entry=response("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields
        val image=javax.imageio.ImageIO.read(Path.of(entry("path").asInstanceOf[Str].value).toFile)
        (image.getRGB(0,0,10,6,null,0,10).toVector,entry)
      val variationImage=variation(42,1337,0.5)
      assertNotEquals(variationImage._1,plainPrompt,"variation must change the generated image")
      assertEquals(variation(42,1337,0.5)._1,variationImage._1,"variation must be deterministic")
      assertEquals(variation(42,1337,1.0)._1,variation(1337,42,0.0)._1,"full variation uses the subseed noise")
      assertEquals(variationImage._2("subseed"),Number(1337))
      assert(variation(42,-1,0.5)._2("subseed").asInstanceOf[Number].value>=0,"random subseed must be recorded")
      val textWidth=scalatorch.models.PretrainedClip.load(root.resolve("text_encoder")).tokenEmbedding.weight.data.shape.dims(1).toInt
      val idlePath=output.resolve("idle.safetensors")
      writeEmbedding(idlePath,textWidth)
      val inactive=run("txt2img",Map("output_dir"->Str(output.toString),"embedding_paths"->Arr(Vector(Str(idlePath.toString))),
        "params"->Obj(Map("prompt"->Str("a red cat"),"width"->Number(10),"height"->Number(6),"steps"->Number(2),"seed"->Number(42)))))
      assertEquals(inactive("warnings"),Arr(Vector.empty))
      assertEquals(javax.imageio.ImageIO.read(Path.of(inactive("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj]
        .fields("path").asInstanceOf[Str].value).toFile).getRGB(0,0,10,6,null,0,10).toVector,plainPrompt)
      val idleBaseline=pixelsFor("a red cat idle")
      assertEquals(pixelsFor("a red cat idle"),idleBaseline,"inactive embedding paths must not be cached or applied")
      val invalidPath=output.resolve("broken.safetensors");Files.write(invalidPath,Array[Byte](1,2,3))
      val invalidEmbedding=run("txt2img",Map("output_dir"->Str(output.toString),"embedding_paths"->Arr(Vector(Str(invalidPath.toString))),
        "params"->Obj(Map("prompt"->Str("a red cat broken"),"width"->Number(10),"height"->Number(6),"steps"->Number(2),"seed"->Number(42)))))
      assert(invalidEmbedding("warnings").asInstanceOf[Arr].values.nonEmpty)
      val rejectedList=run("embedding_inspector",Map("operation"->Str("list"),"embedding_paths"->Arr(Vector(Str(invalidPath.toString)))))
      assertEquals(rejectedList("loaded"),Arr(Vector.empty))
      assertEquals(rejectedList("available"),Arr(Vector(Str("broken"))))
      val invalidImage=invalidEmbedding("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
      val invalidPixels=javax.imageio.ImageIO.read(Path.of(invalidImage).toFile).getRGB(0,0,10,6,null,0,10).toVector
      val invalidCached=run("txt2img",Map("output_dir"->Str(output.toString),"params"->Obj(Map(
        "prompt"->Str("a red cat broken"),"width"->Number(10),"height"->Number(6),"steps"->Number(2),"seed"->Number(42)))))
      val invalidCachedPath=invalidCached("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
      assertEquals(javax.imageio.ImageIO.read(Path.of(invalidCachedPath).toFile).getRGB(0,0,10,6,null,0,10).toVector,invalidPixels,
        "malformed files must not partially alter the loaded tokenizer or encoder")
      val embeddingPath=output.resolve("sks.safetensors")
      writeEmbedding(embeddingPath,textWidth)
      val withEmbedding=run("txt2img",Map("output_dir"->Str(output.toString),"embedding_paths"->Arr(Vector(Str(embeddingPath.toString))),
        "params"->Obj(Map("prompt"->Str("a red cat sks"),"width"->Number(10),"height"->Number(6),"steps"->Number(2),"seed"->Number(42)))))
      val embeddedImage=withEmbedding("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
      val embeddedPixels=javax.imageio.ImageIO.read(Path.of(embeddedImage).toFile).getRGB(0,0,10,6,null,0,10).toVector
      assertNotEquals(embeddedPixels,plainPrompt)
      assertEquals(withEmbedding("warnings"),Arr(Vector.empty))
      val cached=run("txt2img",Map("output_dir"->Str(output.toString),"params"->Obj(Map(
        "prompt"->Str("a red cat sks"),"width"->Number(10),"height"->Number(6),"steps"->Number(2),"seed"->Number(42)))))
      val cachedPath=cached("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
      assertEquals(javax.imageio.ImageIO.read(Path.of(cachedPath).toFile).getRGB(0,0,10,6,null,0,10).toVector,embeddedPixels,
        "loaded textual inversion must persist without resending its path")
      val ptPath=output.resolve("legacy.pt")
      PyTorchEmbeddingFixture.write(ptPath,"string_to_param",("*",Vector.fill(textWidth)(-0.75f),Vector(1,textWidth)))
      val ptBaseline=pixelsFor("a red cat legacy")
      val pytorchEmbedding=run("txt2img",Map("output_dir"->Str(output.toString),"embedding_paths"->Arr(Vector(Str(ptPath.toString))),
        "params"->Obj(Map("prompt"->Str("a red cat legacy"),"width"->Number(10),"height"->Number(6),"steps"->Number(2),"seed"->Number(42)))))
      val pytorchImage=pytorchEmbedding("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
      val pytorchPixels=javax.imageio.ImageIO.read(Path.of(pytorchImage).toFile).getRGB(0,0,10,6,null,0,10).toVector
      assertNotEquals(pytorchPixels,ptBaseline,"legacy .pt textual inversion must affect native generation")
      assertEquals(pytorchEmbedding("warnings"),Arr(Vector.empty))
      val pytorchCached=run("txt2img",Map("output_dir"->Str(output.toString),"params"->Obj(Map(
        "prompt"->Str("a red cat legacy"),"width"->Number(10),"height"->Number(6),"steps"->Number(2),"seed"->Number(42)))))
      val pytorchCachedPath=pytorchCached("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
      assertEquals(javax.imageio.ImageIO.read(Path.of(pytorchCachedPath).toFile).getRGB(0,0,10,6,null,0,10).toVector,pytorchPixels,
        "loaded PyTorch textual inversion must persist without resending its path")
      val binPath=output.resolve("legacybin.bin")
      Files.copy(ptPath,binPath)
      val binBaseline=pixelsFor("a red cat legacybin")
      val binResult=run("txt2img",Map("output_dir"->Str(output.toString),"embedding_paths"->Arr(Vector(Str(binPath.toString))),
        "params"->Obj(Map("prompt"->Str("a red cat legacybin"),"width"->Number(10),"height"->Number(6),"steps"->Number(2),"seed"->Number(42)))))
      val binImage=binResult("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
      assertEquals(binResult("warnings"),Arr(Vector.empty))
      assertNotEquals(javax.imageio.ImageIO.read(Path.of(binImage).toFile).getRGB(0,0,10,6,null,0,10).toVector,binBaseline)
      val brokenPt=output.resolve("brokenpt.pt")
      Files.write(brokenPt,Array[Byte](1,2,3))
      val brokenPrompt="a red cat brokenpt"
      val brokenBase=pixelsFor(brokenPrompt)
      val brokenResult=run("txt2img",Map("output_dir"->Str(output.toString),"embedding_paths"->Arr(Vector(Str(brokenPt.toString))),
        "params"->Obj(Map("prompt"->Str(brokenPrompt),"width"->Number(10),"height"->Number(6),"steps"->Number(2),"seed"->Number(42)))))
      assert(brokenResult("warnings").asInstanceOf[Arr].values.nonEmpty,"malformed .pt embedding must produce a visible warning")
      val brokenPath=brokenResult("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
      assertEquals(javax.imageio.ImageIO.read(Path.of(brokenPath).toFile).getRGB(0,0,10,6,null,0,10).toVector,brokenBase,
        "rejected .pt data must leave CLIP and the loaded checkpoint unchanged")
      val target=scalatorch.models.PretrainedUNet.loadWithLinearTargets(root.resolve("unet"),maxParameters=1000000000L)
        .linearTargets("unet.time_embedding.linear_1")
      val adapterPath=output.resolve("tiny.safetensors")
      writeAdapter(adapterPath,"unet.time_embedding.linear_1",target.data.shape.dims(1).toInt,target.data.shape.dims(0).toInt)
      def loraPixels(weight: Double,dynamicRank: Option[Int]=None): Vector[Int]=
        val lora=Map[String,NativeJson]("name"->Str("tiny"),"weight"->Number(weight),"unet"->Number(weight)) ++
          dynamicRank.map(value=>"dyn"->Number(value))
        val response=run("txt2img",Map("output_dir"->Str(output.toString),"lora_paths"->Obj(Map("tiny"->Str(adapterPath.toString))),
          "params"->Obj(Map("prompt"->Str("a red cat"),"width"->Number(10),"height"->Number(6),"steps"->Number(2),
            "seed"->Number(42),"loras"->Arr(Vector(Obj(lora)))))))
        val imagePath=response("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
        val decoded=javax.imageio.ImageIO.read(Path.of(imagePath).toFile)
        decoded.getRGB(0,0,10,6,null,0,10).toVector
      assertEquals(loraPixels(0),plainPrompt,"zero-scale LoRA must preserve the unadapted image")
      val adaptedPrompt=loraPixels(1)
      assert(adaptedPrompt!=plainPrompt,"nonzero native LoRA must affect generated pixels")
      assertEquals(loraPixels(1,Some(0)),plainPrompt,"zero dynamic rank must disable the adapter")
      assertEquals(loraPixels(1,Some(1)),adaptedPrompt,"rank-one adapter must retain all factors")
      assertEquals(loraPixels(1),adaptedPrompt,"native LoRA replay must be deterministic")
      val tagged=run("txt2img",Map("output_dir"->Str(output.toString),"lora_paths"->Obj(Map("tiny"->Str(adapterPath.toString))),
        "params"->Obj(Map("prompt"->Str("a red cat<lora:TINY:1>"),"width"->Number(10),"height"->Number(6),
          "steps"->Number(2),"seed"->Number(42)))))
      val taggedPath=tagged("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
      assertEquals(javax.imageio.ImageIO.read(Path.of(taggedPath).toFile).getRGB(0,0,10,6,null,0,10).toVector,adaptedPrompt,
        "prompt LoRA must match an explicitly resolved adapter")
      assertEquals(pixelsFor("a red cat"),plainPrompt,"adapter factors must not mutate the base checkpoint")
      val weightedPrompt=pixelsFor("a (red:1.5) cat")
      assert(weightedPrompt!=plainPrompt,"single-CLIP worker must pass prompt emphasis into the U-Net")
      assertEquals(pixelsFor("a (red:1.5) cat"),weightedPrompt,"weighted prompt replay must be deterministic")
      val chunkedPrompt=pixelsFor("apple, "*40,"blur BREAK fog")
      assertEquals(pixelsFor("apple, "*40,"blur BREAK fog"),chunkedPrompt,
        "single-CLIP worker must handle unequal positive/negative chunk counts deterministically")
      // Exercise the public request boundary, not only metadata parsing: overrides
      // must reach both samplers and must not mutate the checkpoint configuration.
      for sampler <- Vector("Euler","DPM++ 2M") do
        def pixels(extra: Map[String,NativeJson]): Vector[Int]=
          val generated=run("txt2img",Map("output_dir"->Str(output.toString),"params"->Obj(Map(
            "prompt"->Str("cat"),"width"->Number(10),"height"->Number(6),"steps"->Number(2),
            "seed"->Number(42),"sampler_name"->Str(sampler),"extra"->Obj(extra)))))
          val file=generated("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
          val decoded=javax.imageio.ImageIO.read(Path.of(file).toFile)
          decoded.getRGB(0,0,10,6,null,0,10).toVector
        val baseline=pixels(Map.empty)
        assertEquals(pixels(Map("Model keyword settings"->Str("{}"),"prediction_type"->Str("epsilon"))),baseline)
        assert(pixels(Map("prediction_type"->Str("v_prediction")))!=baseline)
        assertEquals(pixels(Map.empty),baseline)
      val uniform=run("txt2img",Map("output_dir"->Str(output.toString),"params"->Obj(Map(
        "prompt"->Str("a red cat"),"width"->Number(10),"height"->Number(6),"steps"->Number(2),
        "seed"->Number(42),"clip_skip"->Number(2),"sampler_name"->Str("DPM++ 2M"),"scheduler"->Str("Uniform")))))
      assertEquals(uniform("scheduler"),Str("Uniform"))
      val uniformPath=Path.of(uniform("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
      assertEquals(javax.imageio.ImageIO.read(uniformPath.toFile).getWidth,10)
      val iterativeParams=Map("output_dir"->Str(output.toString),"params"->Obj(Map(
        "prompt"->Str("cat"),"width"->Number(10),"height"->Number(6),"steps"->Number(1),
        "seed"->Number(100),"subseed"->Number(7),"n_iter"->Number(2))))
      val iterative=run("txt2img",iterativeParams)("images").asInstanceOf[Arr].values
      assertEquals(iterative.map(_.asInstanceOf[Obj].fields("seed")),Vector(Number(100),Number(101)))
      assertEquals(iterative.map(_.asInstanceOf[Obj].fields("subseed")),Vector(Number(7),Number(8)))
      val batchEvents=scala.collection.mutable.ArrayBuffer.empty[Map[String,NativeJson]]
      val batchContext=new NativeWorkContext(() => false,() => true,(_,fields) => batchEvents+=fields)
      val batchedParams=iterativeParams.updated("params",Obj(Map("prompt"->Str("cat"),"width"->Number(10),
        "height"->Number(6),"steps"->Number(1),"seed"->Number(200),"subseed"->Number(9),
        "batch_size"->Number(2),"n_iter"->Number(2))))
      val batched=backend.execute(NativeRequest("batch-size","txt2img",batchedParams,None),batchContext)("images").asInstanceOf[Arr].values
      assertEquals(batched.map(_.asInstanceOf[Obj].fields("seed")),Vector(200,201,202,203).map(v=>Number(BigDecimal(v))))
      assertEquals(batched.map(_.asInstanceOf[Obj].fields("subseed")),Vector(9,10,11,12).map(v=>Number(BigDecimal(v))))
      val batchProgress=batchEvents.filter(_.get("batch_size").contains(Number(2)))
      assert(batchProgress.nonEmpty,"progress must report the requested logical batch size")
      assert(batchProgress.forall(_.get("images").contains(Number(4))))
      val initCanvas=new java.awt.image.BufferedImage(100,60,java.awt.image.BufferedImage.TYPE_INT_RGB)
      val inpaintMask=new java.awt.image.BufferedImage(100,60,java.awt.image.BufferedImage.TYPE_INT_RGB)
      for y <- 0 until 60; x <- 0 until 100 do
        initCanvas.setRGB(x,y,0xffff0000)
        inpaintMask.setRGB(x,y,if x>=40 && x<60 && y>=20 && y<40 then 0xffffffff else 0xff000000)
      val initPath=output.resolve("fullres-init.png");val maskPath=output.resolve("fullres-mask.png")
      javax.imageio.ImageIO.write(initCanvas,"png",initPath.toFile)
      javax.imageio.ImageIO.write(inpaintMask,"png",maskPath.toFile)
      val fullRes=run("img2img",Map("output_dir"->Str(output.toString),"params"->Obj(Map(
        "prompt"->Str("replace the center"),"width"->Number(64),"height"->Number(64),"steps"->Number(1),"seed"->Number(303),
        "img2img"->Obj(Map("init_image"->Str(initPath.toString),"mask"->Str(maskPath.toString),
          "denoising_strength"->Number(1),"resize_mode"->Number(0),"mask_blur"->Number(0),"inpainting_fill"->Number(0),
          "inpaint_full_res"->Bool(true),"inpaint_full_res_padding"->Number(8)))))))
      val fullResEntry=fullRes("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields
      assertEquals((fullResEntry("width"),fullResEntry("height")),(Number(100),Number(60)))
      val fullResImage=javax.imageio.ImageIO.read(Path.of(fullResEntry("path").asInstanceOf[Str].value).toFile)
      assertEquals((fullResImage.getWidth,fullResImage.getHeight),(100,60))
      assertEquals((fullResImage.getRGB(0,0)&0xffffff),0xff0000)
      assert((fullResImage.getRGB(50,30)&0xffffff)!=0xff0000)
      val custom=output.resolve("custom-mappings.txt")
      Files.writeString(custom,s"${metadata("hash").asInstanceOf[Str].value},a red|a blue,${root.getFileName}\n")
      val keywordParams=Map("prompt"->Str("cat"),"width"->Number(10),"height"->Number(6),"steps"->Number(2),
        "seed"->Number(42),"hooks"->Obj(Map("model_keyword"->Obj(Map("_mapping_dir"->Str(output.toString),"model_choice"->Str("iterate"))))))
      val keywordRequest=Map("output_dir"->Str(output.toString),"params"->Obj(keywordParams))
      def imageFields(result: Map[String,NativeJson])=result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields
      def pixelsOf(fields: Map[String,NativeJson])=
        val img=javax.imageio.ImageIO.read(Path.of(fields("path").asInstanceOf[Str].value).toFile)
        img.getRGB(0,0,10,6,null,0,10).toVector
      val firstKeyword=imageFields(run("txt2img",keywordRequest))
      assertEquals(firstKeyword("effective_prompt"),Str("a red cat"))
      assertEquals(pixelsOf(firstKeyword),image.getRGB(0,0,10,6,null,0,10).toVector)
      val rejectKeyword=new NativeWorkContext(() => false,() => false,(_,_) => ())
      intercept[java.util.concurrent.CancellationException](backend.execute(NativeRequest("keyword-reject","txt2img",keywordRequest,None),rejectKeyword))
      intercept[IllegalArgumentException](run("txt2img",keywordRequest.updated("params",Obj(keywordParams.updated("width",Number(0))))))
      val secondKeyword=imageFields(run("txt2img",keywordRequest))
      assertEquals(secondKeyword("effective_prompt"),Str("a blue cat"))
      assertEquals(secondKeyword("model_keyword_source_prompt"),Str("a blue cat"))
      val directBlue=imageFields(run("txt2img",keywordRequest.updated("params",Obj(keywordParams.removed("hooks").updated("prompt",Str("a blue cat"))))))
      assertEquals(pixelsOf(secondKeyword),pixelsOf(directBlue))
      Files.writeString(custom,s"${metadata("hash").asInstanceOf[Str].value},<lora:tiny:1>a red,${root.getFileName}\n")
      val resolved=Obj(Map("name"->Str("tiny"),"weight"->Number(1)))
      val keywordTagged=keywordRequest.updated("lora_paths",Obj(Map("tiny"->Str(adapterPath.toString))))
        .updated("manual_loras",Arr(Vector.empty))
        .updated("params",Obj(keywordParams.updated("loras",Arr(Vector(resolved)))))
      val taggedResult=run("txt2img",keywordTagged.updated("embedding_paths",Arr(Vector(Str(adapterPath.toString)))))
      assertEquals(taggedResult("warnings"),Arr(Vector.empty),"a LoRA name inside a tag is not an embedding reference")
      val taggedFields=imageFields(taggedResult)
      assertEquals(pixelsOf(taggedFields),adaptedPrompt,
        "keyword adapter must replace the pre-resolved list rather than applying it twice")
      assertEquals(taggedFields("effective_prompt"),Str("<lora:tiny:1>a red cat"))
      assertEquals(taggedFields("model_keyword_source_prompt"),Str("<lora:tiny:1>a red cat"))
      Files.writeString(custom,s"${metadata("hash").asInstanceOf[Str].value},<lora:tiny:1>a red|a blue,${root.getFileName}\n")
      val batchedKeyword=keywordTagged.updated("params",Obj(keywordParams.updated("batch_size",Number(2))))
      val entries=run("txt2img",batchedKeyword)("images").asInstanceOf[Arr].values.map(_.asInstanceOf[Obj].fields)
      assertEquals(entries.map(_("model_keyword_source_prompt")),Vector(Str("<lora:tiny:1>a red cat"),Str("a blue cat")))
      assertEquals(entries.map(_("effective_prompt")),Vector(Str("<lora:tiny:1>a red cat"),Str("<lora:tiny:1>a blue cat")))
      val keywordOnly=output.resolve("keywordonly.safetensors")
      writeEmbedding(keywordOnly,textWidth)
      val withoutKeywordEmbedding=pixelsFor("keywordonly cat")
      Files.writeString(custom,s"${metadata("hash").asInstanceOf[Str].value},keywordonly,${root.getFileName}\n")
      val keywordEmbedding=keywordRequest.updated("embedding_paths",Arr(Vector(Str(keywordOnly.toString))))
      val keywordEmbeddingFields=imageFields(run("txt2img",keywordEmbedding))
      val withKeywordEmbedding=pixelsOf(keywordEmbeddingFields)
      assertNotEquals(withKeywordEmbedding,withoutKeywordEmbedding,"keyword-only textual inversion must affect the output")
      assertEquals(withKeywordEmbedding,pixelsFor("keywordonly cat"),"keyword embedding must match a direct prompt")
      val refinementOnly=output.resolve("refinementonly.safetensors")
      writeEmbedding(refinementOnly,textWidth)
      val refinementHook=Obj(Map("img2img_hires_fix"->Obj(Map("ratio"->Number(1),"upscaler"->Str("Nearest"),
        "denoise"->Number(0.5),"steps"->Number(1),"prompt"->Str("refinementonly cat")))))
      run("txt2img",keywordRequest.updated("embedding_paths",Arr(Vector(Str(refinementOnly.toString))))
        .updated("params",Obj(keywordParams.updated("hooks",refinementHook).updated("steps",Number(1)))))
      val inspected=run("embedding_inspector",Map("operation"->Str("list"),
        "embedding_paths"->Arr(Vector(Str(refinementOnly.toString)))))
      assert(inspected("loaded").asInstanceOf[Arr].values.exists(_.asInstanceOf[Obj].fields("name")==Str("refinementonly")),
        "embedding referenced only by the refinement prompt must be loaded")
      Files.writeString(custom,s"${metadata("hash").asInstanceOf[Str].value},<lora:tiny:1>a red|a blue,${root.getFileName}\n")
      val keywordRefine=Obj(Map("img2img_hires_fix"->Obj(Map("ratio"->Number(1),"upscaler"->Str("Nearest"),
        "denoise"->Number(0.5),"steps"->Number(1)))))
      val keywordHook=keywordParams("hooks").asInstanceOf[Obj].fields
      val refineParams=keywordParams.updated("steps",Number(1)).updated("n_iter",Number(2))
        .updated("hooks",Obj(keywordHook++keywordRefine.asInstanceOf[Obj].fields))
      val keywordRefineRequest=keywordTagged.updated("params",Obj(refineParams))
      val keywordRefineImages=run("txt2img",keywordRefineRequest)("images").asInstanceOf[Arr].values
        .map(_.asInstanceOf[Obj].fields)
      assertEquals(keywordRefineImages(1)("model_keyword_refinement_prompt"),Str("<lora:tiny:1>a red cat"))
      val refHook=Obj(Map("img2img_hires_fix"->Obj(keywordRefine.asInstanceOf[Obj].fields("img2img_hires_fix")
        .asInstanceOf[Obj].fields.updated("prompt",Str("a red cat")))))
      val directRefine=keywordRefineRequest.updated("img2img_hires_loras",Arr(Vector(resolved)))
        .updated("params",Obj(keywordParams.removed("hooks").updated("prompt",Str("a blue cat"))
          .updated("seed",Number(43)).updated("steps",Number(1)).updated("hooks",refHook)))
      def fullPixels(fields: Map[String,NativeJson])=
        val img=javax.imageio.ImageIO.read(Path.of(fields("path").asInstanceOf[Str].value).toFile)
        img.getRGB(0,0,img.getWidth,img.getHeight,null,0,img.getWidth).toVector
      assertEquals(fullPixels(keywordRefineImages(1)),fullPixels(imageFields(run("txt2img",directRefine))),
        "keyword refinement must use the preview prompt and its LoRA for every batch")
      def filesNow()=
        val paths=Files.list(output)
        try
          import scala.jdk.CollectionConverters.*
          paths.iterator().asScala.toVector.map(_.getFileName.toString).toSet
        finally paths.close()
      val before=filesNow()
      var cancelBatch=false
      val cancelContext=new NativeWorkContext(() => cancelBatch,() => !cancelBatch,(_,fields) => {
        if fields.get("image").contains(Number(1)) then cancelBatch=true
      })
      intercept[java.util.concurrent.CancellationException](backend.execute(NativeRequest("batch","txt2img",iterativeParams,None),cancelContext))
      assertEquals(filesNow(),before)
      val rejectPublication=new NativeWorkContext(() => false,() => false,(_,_) => ())
      intercept[java.util.concurrent.CancellationException](backend.execute(NativeRequest("reject","txt2img",iterativeParams,None),rejectPublication))
      assertEquals(filesNow(),before)
      val oversized=iterativeParams.updated("params",Obj(Map("seed"->Number(BigDecimal(Long.MaxValue)),"n_iter"->Number(2))))
      intercept[IllegalArgumentException](run("txt2img",oversized))
      val cancelled=new NativeWorkContext(() => true,() => false,(_,_) => ())
      intercept[java.util.concurrent.CancellationException](backend.execute(NativeRequest("x","unload",Map.empty,None),cancelled))
      val invalid=intercept[NativeBackendError](run("txt2img",Map("unknown"->Bool(true))))
      assertEquals(invalid.code,"unsupported")
      run("unload",Map.empty)
      assertEquals(backend.systemStatus("model"),Obj(Map.empty))
      intercept[NativeBackendError](run("txt2img",Map.empty))
    finally
      val files=Files.list(output)
      try files.forEach(p => { Files.delete(p);() }) finally files.close()
      Files.delete(output);Files.delete(config)
  }

  test("worker runs deterministic native latent Hires as two passes") {
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty,"public tiny model required")
    val root=Path.of(configured.get).getParent
    val config=Files.createTempFile("native-hires-euler", ".json")
    val output=Files.createTempDirectory("native-hires-output")
    Files.writeString(config,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
    import NativeJson.*
    try
      val backend=new NativeDiffusionBackend(root,config)
      val events=scala.collection.mutable.ArrayBuffer.empty[(String,Int,Int)]
      val context=new NativeWorkContext(() => false,() => true,(name,fields) => if name=="progress" then
        events+=((fields("pass").asInstanceOf[Str].value,fields("step").asInstanceOf[Number].value.toInt,
          fields("total").asInstanceOf[Number].value.toInt)))
      backend.execute(NativeRequest("load","load_model",Map("path"->Str(root.toString)),None),context)
      val hires=Obj(Map("enabled"->Bool(true),"scale"->Number(2),"upscaler"->Str("Latent (bicubic)"),
        "second_pass_steps"->Number(1),"denoising_strength"->Number(BigDecimal("0.55"))))
      val params=Map("output_dir"->Str(output.toString),"params"->Obj(Map("prompt"->Str("a red cat"),
        "width"->Number(16),"height"->Number(16),"steps"->Number(1),"seed"->Number(42),"hires"->hires)))
      def generate()=backend.execute(NativeRequest("hires","txt2img",params,None),context)
      def image(result: Map[String,NativeJson]): (Vector[Int],Int,Int)=
        val entry=result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields
        val decoded=javax.imageio.ImageIO.read(Path.of(entry("path").asInstanceOf[Str].value).toFile)
        (decoded.getRGB(0,0,decoded.getWidth,decoded.getHeight,null,0,decoded.getWidth).toVector,decoded.getWidth,decoded.getHeight)
      val first=generate();val firstImage=image(first)
      events.clear()
      val second=generate();val secondImage=image(second)
      assertEquals(firstImage._2,64,"final width")
      assertEquals(firstImage._3,64,"final height")
      assertEquals(secondImage,firstImage,"same seed and request replay deterministically")
      assertEquals(events.map(_._1).toVector,Vector("base","hires"),"each pass reports progress independently")
      assertEquals(events.map(_._2).toVector,Vector(1,1),"active steps per pass")
      assertEquals(events.map(_._3).toVector,Vector(1,1),"pass-local progress totals")
      val target=scalatorch.models.PretrainedUNet.loadWithLinearTargets(root.resolve("unet")).linearTargets("unet.time_embedding.linear_1")
      val adapter=output.resolve("hires-lora.safetensors")
      writeAdapter(adapter,"unet.time_embedding.linear_1",target.data.shape.dims(1).toInt,target.data.shape.dims(0).toInt)
      val lora=Obj(Map("name"->Str("hires-lora"),"weight"->Number(1)))
      val withBase=params.updated("lora_paths",Obj(Map("hires-lora"->Str(adapter.toString))))
        .updated("params",Obj(params("params").asInstanceOf[Obj].fields.updated("loras",Arr(Vector(lora)))))
      def switched(value: Option[Arr])=image(backend.execute(NativeRequest("hires-lora","txt2img",
        withBase++value.map("hires_loras"->_),None),context))._1
      val inherited=switched(None)
      assertEquals(switched(Some(Arr(Vector(lora)))),inherited,"explicit same Hires LoRA must match inherited LoRA")
      assert(inherited!=switched(Some(Arr(Vector.empty))),"empty Hires LoRA must disable the base adapter in the second pass")
      assertEquals(switched(None),inherited,"Hires LoRA selection must not leak into later requests")
      def taggedHires(tag: Boolean,selection: Option[Arr]): Vector[Int] =
        val overrideHires=Obj(hires.asInstanceOf[Obj].fields.updated("prompt",Str("a red cat"+(if tag then "<lora:hires-lora:1>" else ""))))
        val generation=Obj(params("params").asInstanceOf[Obj].fields.updated("hires",overrideHires))
        val envelope=params.updated("params",generation).updated("lora_paths",Obj(Map("hires-lora"->Str(adapter.toString)))) ++
          selection.map("hires_loras"->_)
        image(backend.execute(NativeRequest("hires-tag","txt2img",envelope,None),context))._1
      assertEquals(taggedHires(true,None),taggedHires(false,Some(Arr(Vector(lora)))),
        "Hires prompt tag must match a resolved Hires LoRA")
      assertEquals(taggedHires(true,Some(Arr(Vector.empty))),taggedHires(false,Some(Arr(Vector.empty))),
        "explicit Hires selection takes precedence over prompt tags")
    finally
      Files.deleteIfExists(config)
      val files=Files.walk(output)
      try
        import scala.jdk.CollectionConverters.*
        files.iterator().asScala.toVector.sortBy(_.getNameCount).reverse.foreach(Files.delete(_))
      finally files.close()
  }

  test("worker cancels after Hires base pass without publishing a partial image") {
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty,"public tiny model required")
    val root=Path.of(configured.get).getParent
    val config=Files.createTempFile("native-hires-cancel-euler", ".json")
    val output=Files.createTempDirectory("native-hires-cancel-output")
    Files.writeString(config,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
    import NativeJson.*
    try
      val backend=new NativeDiffusionBackend(root,config)
      val context=new NativeWorkContext(() => false,() => true,(_,_) => ())
      backend.execute(NativeRequest("load","load_model",Map("path"->Str(root.toString)),None),context)
      val hires=Obj(Map("enabled"->Bool(true),"scale"->Number(2),"upscaler"->Str("Latent"),
        "second_pass_steps"->Number(1)))
      val params=Map("output_dir"->Str(output.toString),"params"->Obj(Map("prompt"->Str("a red cat"),
        "width"->Number(16),"height"->Number(16),"steps"->Number(1),"seed"->Number(42),"hires"->hires)))
      var cancelled=false
      val cancelContext=new NativeWorkContext(() => cancelled,() => !cancelled,(name,fields) =>
        if name=="progress" && fields.get("pass").contains(Str("base")) then cancelled=true)
      intercept[java.util.concurrent.CancellationException] {
        backend.execute(NativeRequest("cancel-hires","txt2img",params,None),cancelContext)
      }
      val after=Files.list(output)
      try assertEquals(after.count(),0L,"cancelled base pass must not publish a partial image") finally after.close()
    finally
      Files.deleteIfExists(config)
      val files=Files.walk(output)
      try
        import scala.jdk.CollectionConverters.*
        files.iterator().asScala.toVector.sortBy(_.getNameCount).reverse.foreach(Files.delete(_))
      finally files.close()
  }

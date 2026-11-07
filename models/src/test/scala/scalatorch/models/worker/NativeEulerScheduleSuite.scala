package scalatorch.models.worker

import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import scalatorch.models.*
import NativeJson.*
import scala.concurrent.duration.*

class NativeEulerScheduleSuite extends munit.FunSuite:
  override val munitTimeout: Duration=180.seconds
  test("Euler Uniform uses NUI timesteps rather than checkpoint leading spacing") {
    val directory=sys.props.get("scalatorch.sdxl.dir")
    assume(directory.nonEmpty,"tiny SDXL fixture required")
    val root=Path.of(directory.get)
    val output=Files.createTempDirectory("native-euler-schedule")
    try
      val config=output.resolve("scheduler.json")
      Files.writeString(config,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,
        "beta_schedule":"scaled_linear","timestep_spacing":"leading","steps_offset":1}""")
      val backend=new NativeDiffusionBackend(root,config)
      val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
      backend.execute(NativeRequest("load","load_model",Map.empty,None),context)
      val params=Map[String,NativeJson]("output_dir"->Str(output.toString),"params"->Obj(Map(
        "prompt"->Str("apple"),"width"->Number(8),"height"->Number(8),"steps"->Number(1),
        "seed"->Number(42),"sampler_name"->Str("Euler"),"scheduler"->Str("Uniform"))))
      val result=backend.execute(NativeRequest("generate","txt2img",params,None),context)
      val aliasFields=params("params").asInstanceOf[Obj].fields.updated("sampler_name",Str("k_euler"))
        .updated("scheduler",Str("uniform"))
      val aliasResult=backend.execute(NativeRequest("alias","txt2img",params.updated("params",Obj(aliasFields)),None),context)
      assertEquals(aliasResult("scheduler"),Str("Uniform"))
      val aliasPath=aliasResult("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
      val path=result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
      val actual=ImageIO.read(Path.of(path).toFile)
      val aliasImage=ImageIO.read(Path.of(aliasPath).toFile)
      assertEquals(aliasImage.getRGB(0,0,8,8,null,0,8).toVector,actual.getRGB(0,0,8,8,null,0,8).toVector)
      for (alias,canonical,requested,resolved) <- Vector(
          ("eULer A","Euler a","sgm_uniform","SGM Uniform"),
          ("k_dpmpp_2m","DPM++ 2M"," automatic ","Karras"),
          ("DPM++ 2M Karras","DPM++ 2M","Automatic","Karras")) do
        def generate(name: String,schedule: String) =
          val fields=aliasFields.updated("sampler_name",Str(name)).updated("scheduler",Str(schedule))
          backend.execute(NativeRequest("aliases","txt2img",params.updated("params",Obj(fields)),None),context)
        val fromAlias=generate(alias,requested)
        val fromCanonical=generate(canonical,resolved)
        assertEquals(fromAlias("scheduler"),Str(resolved))
        def pixels(result: Map[String,NativeJson]) =
          val file=result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
          ImageIO.read(Path.of(file).toFile).getRGB(0,0,8,8,null,0,8).toVector
        assertEquals(pixels(fromAlias),pixels(fromCanonical),alias)
      val first=ClipFastTokenizer.loadDirectory(root.resolve("tokenizer"));val clip=PretrainedClip.load(root.resolve("text_encoder"))
      val conditioning=new SdxlTextConditioning(first,clip,ClipFastTokenizer.loadDirectory(root.resolve("tokenizer_2")),
        PretrainedClip.load(root.resolve("text_encoder_2")))
      val pipeline=new EulerImagePipeline(first,clip,PretrainedUNet.load(root.resolve("unet")),PretrainedVae.load(root.resolve("vae")),
        sdxlConditioning=Some(conditioning))
      val schedule=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true).nuiUniformEuler(1)
      val expected=DiffusionImageIO.toBufferedImage(pipeline.generate(DiffusionImageRequest("apple","",8,8,42,7f),schedule))
      assertEquals(actual.getRGB(0,0,8,8,null,0,8).toVector,expected.getRGB(0,0,8,8,null,0,8).toVector)
      val exponentialFields=params("params").asInstanceOf[Obj].fields.updated("scheduler",Str("Exponential")).updated("steps",Number(3))
      val exponentialParams=params.updated("params",Obj(exponentialFields))
      val exponentialResult=backend.execute(NativeRequest("generate-exp","txt2img",exponentialParams,None),context)
      assertEquals(exponentialResult("scheduler"),Str("Exponential"))
      val polyFields=exponentialFields.updated("scheduler",Str("Polyexponential"))
      val polyResult=backend.execute(NativeRequest("poly","txt2img",params.updated("params",Obj(polyFields)),None),context)
      assertEquals(polyResult("scheduler"),Str("Polyexponential"))
      assertEquals(polyResult("warnings"),Arr(Vector.empty))
      def pixels(result: Map[String,NativeJson]) =
        val file=result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
        ImageIO.read(Path.of(file).toFile).getRGB(0,0,8,8,null,0,8).toVector
      assertEquals(pixels(polyResult),pixels(exponentialResult))
      for sampler <- Vector("Euler a","DPM++ 2M") do
        val fields=polyFields.updated("sampler_name",Str(sampler))
        val actualPoly=backend.execute(NativeRequest("poly-fallback","txt2img",params.updated("params",Obj(fields)),None),context)
        val expectedFallback=backend.execute(NativeRequest("karras-fallback","txt2img",params.updated("params",Obj(fields.updated("scheduler",Str("Karras")))),None),context)
        assertEquals(pixels(actualPoly),pixels(expectedFallback),sampler)
        val warnings=actualPoly("warnings").asInstanceOf[Arr].values
        assert(warnings.contains(Str(s"schedule 'Polyexponential' is not available for sampler '$sampler'; using Karras")))
        if sampler=="Euler a" then assert(warnings.contains(Str("schedule 'Polyexponential' is not supported by sampler 'Euler a'; using the sampler's default spacing")))
      val exponentialPath=exponentialResult("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
      val exponentialActual=ImageIO.read(Path.of(exponentialPath).toFile)
      val exponentialExpected=DiffusionImageIO.toBufferedImage(pipeline.generate(DiffusionImageRequest("apple","",8,8,42,7f),
        DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true).exponentialEuler(3)))
      assertEquals(exponentialActual.getRGB(0,0,8,8,null,0,8).toVector,exponentialExpected.getRGB(0,0,8,8,null,0,8).toVector)
      val supported=backend.execute(NativeRequest("schedules","list_samplers",Map.empty,None),context)("schedulers").asInstanceOf[Arr].values
      for name <- Vector("Normal","Simple","SGM Uniform") do assert(supported.contains(Str(name)))
      val noise=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
      for sampler <- Vector("Euler","Euler a","DPM++ 2M"); name <- Vector("Normal","Simple","SGM Uniform") do
        val fields=exponentialFields.updated("sampler_name",Str(sampler)).updated("scheduler",Str(name))
        val received=backend.execute(NativeRequest("extra","txt2img",params.updated("params",Obj(fields)),None),context)
        assertEquals(received("scheduler"),Str(name))
        assertEquals(received("warnings"),Arr(Vector.empty))
        val file=received("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
        val actualImage=ImageIO.read(Path.of(file).toFile)
        val request=DiffusionImageRequest("apple","",8,8,42,7f)
        val expectedTensor=if sampler=="Euler" then
          val layout=name match
            case "Normal" => noise.normalEuler(3)
            case "Simple" => noise.simpleEuler(3)
            case _ => noise.sgmUniformEuler(3)
          pipeline.generate(request,layout)
        else if sampler=="Euler a" then
          val spacing=if name=="Normal" then DiffusionTimestepSpacing.Linspace else DiffusionTimestepSpacing.Trailing
          pipeline.generateAncestral(request,noise.euler(3,spacing))
        else pipeline.generateDpm(request,if name=="Normal" then DpmSchedule.uniform(noise,3) else DpmSchedule.trailing(noise,3))
        val expectedImage=DiffusionImageIO.toBufferedImage(expectedTensor)
        assertEquals(actualImage.getRGB(0,0,8,8,null,0,8).toVector,expectedImage.getRGB(0,0,8,8,null,0,8).toVector,s"$sampler / $name")
      val dpmParams=params.updated("params",Obj(exponentialFields.updated("sampler_name",Str("DPM++ 2M"))))
      val dpmResult=backend.execute(NativeRequest("dpm-exp","txt2img",dpmParams,None),context)
      val dpmPath=dpmResult("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
      val dpmActual=ImageIO.read(Path.of(dpmPath).toFile)
      val dpmExpected=DiffusionImageIO.toBufferedImage(pipeline.generateDpm(DiffusionImageRequest("apple","",8,8,42,7f),
        DpmSchedule.exponential(DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true),3)))
      assertEquals(dpmActual.getRGB(0,0,8,8,null,0,8).toVector,dpmExpected.getRGB(0,0,8,8,null,0,8).toVector)
      val hires=Obj(Map("enabled"->Bool(true),"scale"->Number(2),"upscaler"->Str("Latent"),
        "second_pass_steps"->Number(2),"denoising_strength"->Number(BigDecimal("0.5"))))
      val hiresResult=backend.execute(NativeRequest("hires","txt2img",params.updated("params",Obj(exponentialFields.updated("hires",hires))),None),context)
      assertEquals(hiresResult("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("width"),Number(64))
      val dpmHires=exponentialFields.updated("sampler_name",Str("DPM++ 2M")).updated("hires",hires)
      val dpmHiresResult=backend.execute(NativeRequest("dpm-hires","txt2img",params.updated("params",Obj(dpmHires)),None),context)
      assertEquals(dpmHiresResult("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("width"),Number(64))
    finally
      val paths=Files.walk(output)
      try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }

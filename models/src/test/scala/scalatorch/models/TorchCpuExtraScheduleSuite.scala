package scalatorch.models

import java.nio.file.Path
import scalatorch.{Shape,Tensor}
import scalatorch.models.worker.NativeJson
import NativeJson.*
import scala.concurrent.duration.*

class TorchCpuExtraScheduleSuite extends munit.FunSuite:
  override val munitTimeout: Duration=180.seconds
  test("NUI extra schedules match PyTorch latent outputs for text and image inputs") {
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty,"public tiny SD model required")
    val root=Path.of(configured.get).getParent
    val input=this.getClass.getResourceAsStream("/extra-schedule-pipeline-reference.json")
    val cases=try NativeJson.parse(new String(input.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8)).asInstanceOf[Arr].values finally input.close()
    val pipe=new EulerImagePipeline(ClipFastTokenizer.load(root.resolve("tokenizer/vocab.json"),root.resolve("tokenizer/merges.txt")),
      PretrainedClip.load(root.resolve("text_encoder")),PretrainedUNet.load(root.resolve("unet")),PretrainedVae.load(root.resolve("vae")))
    val image=Tensor.fromFloats(Array.tabulate(3*16*16)(i=>(i%256).toFloat/127.5f-1f),Shape.from(Seq(1L,3L,16L,16L)))
    val source=Some(DiffusionImageSource(image,.75))
    val request=DiffusionImageRequest("a red cat","",16,16,42,guidanceScale=1f,torchCpuNoise=true)
    val noise=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
    cases.foreach { value =>
      val row=value.asInstanceOf[Obj].fields
      val sampler=row("sampler").asInstanceOf[Str].value
      val schedule=row("schedule").asInstanceOf[Str].value
      def generate(source: Option[DiffusionImageSource], request: DiffusionImageRequest): Tensor[Float] = sampler match
        case "Euler" =>
          val layout=schedule match
            case "Normal" => noise.normalEuler(4)
            case "Simple" => noise.simpleEuler(4)
            case "Polyexponential" => noise.exponentialEuler(4)
            case _ => noise.sgmUniformEuler(4)
          pipe.generateLatents(request,layout,source=source)
        case "Euler a" =>
          val spacing=if schedule=="Normal" || schedule=="Polyexponential" then DiffusionTimestepSpacing.Linspace else DiffusionTimestepSpacing.Trailing
          pipe.generateAncestralLatents(request,noise.euler(4,spacing),source=source)
        case _ =>
          val layout=if schedule=="Normal" then DpmSchedule.uniform(noise,4)
            else if schedule=="Polyexponential" then DpmSchedule.karras(noise,4) else DpmSchedule.trailing(noise,4)
          pipe.generateDpmLatents(request,layout,source=source)
      val guidedRequest=DiffusionImageRequest("a red cat","blurry",16,16,42,guidanceScale=7.5f,torchCpuNoise=true)
      for (key,src,params) <- Vector(("txt",None,request),("img",source,request),
                                     ("guided_txt",None,guidedRequest),("guided_img",source,guidedRequest)) do
        val actual=generate(src,params).toFloatArray
        val reference=row(key).asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat)
        assertEquals(actual.length,reference.length)
        val max=actual.zip(reference).map((a,b)=>math.abs(a.toDouble-b)).max
        println(s"$sampler / $schedule / $key max latent error $max")
        assert(max<.001,s"$sampler / $schedule / $key error $max")
    }
  }

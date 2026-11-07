package scalatorch.models

import java.nio.file.Path
import scalatorch.{Shape,Tensor}
import scalatorch.models.worker.NativeJson
import NativeJson.*
import scala.concurrent.duration.*

class LongDpmPipelineSuite extends munit.FunSuite:
  override val munitTimeout: Duration=180.seconds
  test("long Karras and exponential DPM++ schedules match PyTorch latents") {
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty,"public tiny model required")
    val root=Path.of(configured.get).getParent
    val stream=this.getClass.getResourceAsStream("/long-dpm-reference.json")
    val rows=try NativeJson.parse(new String(stream.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8)).asInstanceOf[Arr].values finally stream.close()
    val pipe=new EulerImagePipeline(ClipFastTokenizer.load(root.resolve("tokenizer/vocab.json"),root.resolve("tokenizer/merges.txt")),
      PretrainedClip.load(root.resolve("text_encoder")),PretrainedUNet.load(root.resolve("unet")),PretrainedVae.load(root.resolve("vae")))
    val image=Tensor.fromFloats(Array.tabulate(3*16*16)(i=>(i%256).toFloat/127.5f-1f),Shape.from(Seq(1L,3L,16L,16L)))
    val request=DiffusionImageRequest("a red cat","",16,16,42,guidanceScale=1f,torchCpuNoise=true)
    val noise=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
    for value <- rows do
      val row=value.asInstanceOf[Obj].fields
      val name=row("schedule").asInstanceOf[Str].value
      if Set("Karras","Exponential","Polyexponential")(name) then
        assertEquals(row("finite"),Bool(true))
        val count=row("count").asInstanceOf[Number].value.toInt
        val strength=row("strength").asInstanceOf[Number].value.toDouble
        val schedule=if name=="Exponential" then DpmSchedule.exponential(noise,count) else DpmSchedule.karras(noise,count)
        val actual=pipe.generateDpmLatents(request,schedule,source=Some(DiffusionImageSource(image,strength))).toFloatArray
        val expected=row("latent").asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat)
        val error=actual.zip(expected).map((a,b)=>math.abs(a.toDouble-b)).max
        assert(error<.001,s"$name $count / $strength max error $error")
  }
  test("long discrete DPM++ schedules remain finite when timesteps repeat") {
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty,"public tiny model required")
    val root=Path.of(configured.get).getParent
    val pipe=new EulerImagePipeline(ClipFastTokenizer.load(root.resolve("tokenizer/vocab.json"),root.resolve("tokenizer/merges.txt")),
      PretrainedClip.load(root.resolve("text_encoder")),PretrainedUNet.load(root.resolve("unet")),PretrainedVae.load(root.resolve("vae")))
    val image=Tensor.fromFloats(Array.tabulate(3*16*16)(i=>(i%256).toFloat/127.5f-1f),Shape.from(Seq(1L,3L,16L,16L)))
    val request=DiffusionImageRequest("a red cat","",16,16,42,guidanceScale=1f,torchCpuNoise=true)
    val noise=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
    val stream=this.getClass.getResourceAsStream("/long-dpm-reference.json")
    val rows=try NativeJson.parse(new String(stream.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8)).asInstanceOf[Arr].values finally stream.close()
    for count <- Vector(2000,10000); trailing <- Vector(false,true); active <- Vector(3,4) do
      val schedule=if trailing then DpmSchedule.trailing(noise,count) else DpmSchedule.uniform(noise,count)
      val result=pipe.generateDpmLatents(request,schedule,source=Some(DiffusionImageSource(image,(active+1e-6)/count)))
      assert(result.toFloatArray.forall(_.isFinite),s"$count trailing=$trailing active=$active")
      val name=if trailing then "Simple" else "Normal"
      rows.map(_.asInstanceOf[Obj].fields).find(row => row("schedule")==Str(name) &&
        row("count")==Number(count) && row("strength").asInstanceOf[Number].value.toDouble==active.toDouble/count &&
        row("finite")==Bool(true)).foreach { row =>
          val expected=row("latent").asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat)
          val error=result.toFloatArray.zip(expected).map((a,b)=>math.abs(a.toDouble-b)).max
          assert(error<.001,s"$name $count active=$active max error $error")
        }
  }

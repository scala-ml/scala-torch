package scalatorch.models

import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.Path
import scalatorch.{Shape,Tensor}
import scalatorch.models.worker.NativeJson
import NativeJson.*

class LatentColorGradingSuite extends munit.FunSuite:
  private def fixture(): Map[String,NativeJson] =
    val stream=getClass.getResourceAsStream("/latent-color-grading-reference.json")
    try NativeJson.parse(new String(stream.readAllBytes(),UTF_8)).asInstanceOf[Obj].fields finally stream.close()
  private val grading=LatentColorGrading(true,.8,.4,-.3,.7,true,.6)
  private def compare(actual: Tensor[Float],value: NativeJson,tolerance: Double,label: String): Unit =
    val expected=value.asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat)
    val error=actual.toFloatArray.zip(expected).map((a,b)=>math.abs(a.toDouble-b)).max
    assert(error<tolerance,s"$label error $error")
  test("SD1 and SDXL latent vectors match NUI before and on the final step") {
    val refs=fixture()("vectors").asInstanceOf[Obj].fields
    val data=Array.tabulate(64)(i=>math.sin(i.toFloat*.17f).toFloat*6f)
    val input=Tensor.fromFloats(data,Shape.from(Seq(1L,4L,4L,4L)))
    for xl <- Vector(false,true);last <- Vector(false,true) do
      compare(grading.step(input,if last then 4 else 2,4,xl),refs(s"${if xl then "True" else "False"}_${if last then "True" else "False"}"),.00001,s"xl=$xl last=$last")
  }
  test("color grading follows NUI with mirroring and resharpen through generation") {
    val configured=sys.props.get("scalatorch.clip.dir");assume(configured.nonEmpty,"public tiny model required")
    val root=Path.of(configured.get).getParent
    val refs=fixture()("cases").asInstanceOf[Obj].fields
    val pipe=new EulerImagePipeline(ClipFastTokenizer.load(root.resolve("tokenizer/vocab.json"),root.resolve("tokenizer/merges.txt")),
      PretrainedClip.load(root.resolve("text_encoder")),PretrainedUNet.load(root.resolve("unet")),PretrainedVae.load(root.resolve("vae")))
    val input=Tensor.fromFloats(Array.tabulate(3*16*16)(i=>(i%256).toFloat/127.5f-1f),Shape.from(Seq(1L,3L,16L,16L)))
    val layout=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
    val request=DiffusionImageRequest("a red cat","",16,16,42,guidanceScale=7.5f,torchCpuNoise=true,
      latentMirroring=Some(LatentMirroring("Blend Average","90 Degree Rotation",1,.6)),resharpen=Some(Resharpen(1.5,"Cos")),colorGrading=Some(grading))
    for mode <- Vector("txt2img","img2img");sampler <- Vector("euler","dpm") do
      val source=Option.when(mode=="img2img")(DiffusionImageSource(input,.75))
      val actual=if sampler=="euler" then pipe.generateLatents(request,layout.euler(4),source=source)
        else pipe.generateDpmLatents(request,DpmSchedule.uniform(layout,4),source=source)
      compare(actual,refs(s"${mode}_${sampler}"),.001,s"$mode $sampler")
  }

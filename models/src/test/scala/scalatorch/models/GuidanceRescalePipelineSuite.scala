package scalatorch.models

import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.Path
import scalatorch.{Shape,Tensor}
import scalatorch.models.worker.NativeJson
import NativeJson.*

class GuidanceRescalePipelineSuite extends munit.FunSuite:
  test("CFG rescale follows Diffusers over txt2img and img2img with three samplers") {
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty,"public tiny model required")
    val root=Path.of(configured.get).getParent
    val stream=getClass.getResourceAsStream("/rescale-cfg-pipeline-reference.json")
    val expected=try NativeJson.parse(new String(stream.readAllBytes(),UTF_8)).asInstanceOf[Obj].fields finally stream.close()
    val pipe=new EulerImagePipeline(ClipFastTokenizer.load(root.resolve("tokenizer/vocab.json"),root.resolve("tokenizer/merges.txt")),
      PretrainedClip.load(root.resolve("text_encoder")),PretrainedUNet.load(root.resolve("unet")),PretrainedVae.load(root.resolve("vae")))
    val request=DiffusionImageRequest("a red cat","",16,16,42,guidanceScale=7.5f,torchCpuNoise=true,guidanceRescale=.7f)
    val input=Tensor.fromFloats(Array.tabulate(3*16*16)(i=>(i%256).toFloat/127.5f-1f),Shape.from(Seq(1L,3L,16L,16L)))
    val layout=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
    for mode <- Vector("txt2img","img2img"); sampler <- Vector("euler","ancestral","dpm") do
      val source=Option.when(mode=="img2img")(DiffusionImageSource(input,.75))
      val effective=if mode=="img2img" then request.copy(guidanceRescale=0f) else request
      val actual=sampler match
        case "euler" => pipe.generateLatents(effective,layout.euler(4),source=source)
        case "ancestral" => pipe.generateAncestralLatents(effective,layout.euler(4),source=source)
        case _ => pipe.generateDpmLatents(effective,DpmSchedule.uniform(layout,4),source=source)
      val reference=expected(s"${mode}_${sampler}").asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat)
      val error=actual.toFloatArray.zip(reference).map((a,b)=>math.abs(a.toDouble-b)).max
      println(s"$mode $sampler rescale maximum latent error $error")
      assert(error<0.001,s"$mode $sampler maximum latent error $error")
  }

package scalatorch.models

import java.nio.file.Path
import scalatorch.{Shape,Tensor}
import scalatorch.models.worker.NativeJson
import NativeJson.*

class LongAncestralPipelineSuite extends munit.FunSuite:
  test("long trailing Euler ancestral consumes noise at repeated timesteps like PyTorch") {
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty,"public tiny model required")
    val root=Path.of(configured.get).getParent
    val stream=this.getClass.getResourceAsStream("/long-ancestral-reference.json")
    val reference=try NativeJson.parse(new String(stream.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8)).asInstanceOf[Obj].fields finally stream.close()
    val pipe=new EulerImagePipeline(ClipFastTokenizer.load(root.resolve("tokenizer/vocab.json"),root.resolve("tokenizer/merges.txt")),
      PretrainedClip.load(root.resolve("text_encoder")),PretrainedUNet.load(root.resolve("unet")),PretrainedVae.load(root.resolve("vae")))
    val image=Tensor.fromFloats(Array.tabulate(3*16*16)(i=>(i%256).toFloat/127.5f-1f),Shape.from(Seq(1L,3L,16L,16L)))
    val request=DiffusionImageRequest("a red cat","",16,16,42,guidanceScale=1f,torchCpuNoise=true)
    val noise=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
    for (count,strength) <- Vector((2000,.0015),(2000,.002),(10000,.0002),(10000,.0003)) do
      val schedule=noise.euler(count,DiffusionTimestepSpacing.Trailing)
      val actual=pipe.generateAncestralLatents(request,schedule,source=Some(DiffusionImageSource(image,strength))).toFloatArray
      val key=s"$count:${BigDecimal(strength).bigDecimal.stripTrailingZeros.toPlainString}"
      val expected=reference(key).asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat)
      val error=actual.zip(expected).map((a,b)=>math.abs(a.toDouble-b)).max
      assert(error<.001,s"strength $strength max error $error")
  }

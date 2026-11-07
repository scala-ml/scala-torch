package scalatorch.models

import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.Path
import scalatorch.{Shape,Tensor}
import scalatorch.models.worker.NativeJson
import NativeJson.*

class LatentMirroringSuite extends munit.FunSuite:
  override val munitTimeout: scala.concurrent.duration.Duration=scala.concurrent.duration.Duration(3,"minutes")
  test("mirroring follows NUI and composes with resharpen") {
    val configured=sys.props.get("scalatorch.clip.dir");assume(configured.nonEmpty,"public tiny model required")
    val root=Path.of(configured.get).getParent
    val stream=getClass.getResourceAsStream("/latent-mirroring-reference.json")
    val refs=try NativeJson.parse(new String(stream.readAllBytes(),UTF_8)).asInstanceOf[Obj].fields finally stream.close()
    val pipe=new EulerImagePipeline(ClipFastTokenizer.load(root.resolve("tokenizer/vocab.json"),root.resolve("tokenizer/merges.txt")),
      PretrainedClip.load(root.resolve("text_encoder")),PretrainedUNet.load(root.resolve("unet")),PretrainedVae.load(root.resolve("vae")))
    val input=Tensor.fromFloats(Array.tabulate(3*16*16)(i=>(i%256).toFloat/127.5f-1f),Shape.from(Seq(1L,3L,16L,16L)))
    val layout=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
    for mode <- Vector("txt2img","img2img");sampler <- Vector("euler","dpm");
        style <- Vector("Vertical Mirroring","Horizontal Mirroring","Horizontal+Vertical Mirroring","90 Degree Rotation","180 Degree Rotation");
        method <- Vector("Alternate Steps","Blend Average") do
      val request=DiffusionImageRequest("a red cat","",16,16,42,guidanceScale=7.5f,torchCpuNoise=true,
        resharpen=Some(Resharpen(1.5,"Cos")),latentMirroring=Some(LatentMirroring(method,style,1,.6)))
      val source=Option.when(mode=="img2img")(DiffusionImageSource(input,.75))
      val actual=if sampler=="euler" then pipe.generateLatents(request,layout.euler(4),source=source)
        else pipe.generateDpmLatents(request,DpmSchedule.uniform(layout,4),source=source)
      val key=s"$mode|$sampler|$style|$method"
      val expected=refs(key).asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat)
      val error=actual.toFloatArray.zip(expected).map((a,b)=>math.abs(a.toDouble-b)).max
      assert(error<.001,s"$key: $error")
  }
  test("rotation on rectangular latents follows NUI's 180-degree fallback") {
    val value=Tensor.fromFloats(Array(1f,2f,3f,4f,5f,6f),Shape.from(Seq(1L,1L,2L,3L)))
    val rotated=LatentMirroring(style="90 Degree Rotation",maxStep=1).step(value,1,2)
    assertEquals(rotated.toFloatArray.toVector,Vector(6f,5f,4f,3f,2f,1f))
    assertEquals(LatentMirroring(maxStep=0).step(value,1,2).toFloatArray.toVector,value.toFloatArray.toVector)
  }

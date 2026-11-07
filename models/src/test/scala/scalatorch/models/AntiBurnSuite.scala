package scalatorch.models

import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.Path
import scalatorch.{Shape,Tensor}
import scalatorch.models.worker.NativeJson
import NativeJson.*

class AntiBurnSuite extends munit.FunSuite:
  override val munitTimeout: scala.concurrent.duration.Duration=scala.concurrent.duration.Duration(3,"minutes")
  test("anti-burn follows NUI across samplers, image modes and latent callbacks") {
    val configured=sys.props.get("scalatorch.clip.dir");assume(configured.nonEmpty,"public tiny model required")
    val root=Path.of(configured.get).getParent
    val stream=getClass.getResourceAsStream("/anti-burn-reference.json")
    val refs=try NativeJson.parse(new String(stream.readAllBytes(),UTF_8)).asInstanceOf[Obj].fields finally stream.close()
    val pipe=new EulerImagePipeline(ClipFastTokenizer.load(root.resolve("tokenizer/vocab.json"),root.resolve("tokenizer/merges.txt")),
      PretrainedClip.load(root.resolve("text_encoder")),PretrainedUNet.load(root.resolve("unet")),PretrainedVae.load(root.resolve("vae")))
    val input=Tensor.fromFloats(Array.tabulate(3*16*16)(i=>(i%256).toFloat/127.5f-1f),Shape.from(Seq(1L,3L,16L,16L)))
    val layout=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
    for mode <- Vector("txt2img","img2img");sampler <- Vector("euler","ancestral","dpm");(skip,avg) <- Vector((0,1),(0,3),(2,3),(8,3)) do
      val request=DiffusionImageRequest("a red cat","",16,16,42,guidanceScale=7.5f,torchCpuNoise=true,
        latentMirroring=Some(LatentMirroring("Blend Average","90 Degree Rotation",1,.6)),resharpen=Some(Resharpen(1.5,"Cos")),
        colorGrading=Some(LatentColorGrading(true,.8,.4,-.3,.7,true,.6)),antiBurn=Some(AntiBurn(skip,avg)))
      val source=Option.when(mode=="img2img")(DiffusionImageSource(input,.75))
      var steps=0
      val actual=sampler match
        case "euler" => pipe.generateLatents(request,layout.euler(4),source=source,onProgress=_=>steps+=1)
        case "ancestral" => pipe.generateAncestralLatents(request,layout.euler(4),source=source,onProgress=_=>steps+=1)
        case _ => pipe.generateDpmLatents(request,DpmSchedule.uniform(layout,4),source=source,onProgress=_=>steps+=1)
      val key=s"${mode}_${sampler}_${skip}_${avg}"
      val expected=refs(key).asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat)
      val error=actual.toFloatArray.zip(expected).map((a,b)=>math.abs(a.toDouble-b)).max
      assert(error<.001,s"$key: $error")
      val total=if mode=="img2img" then 3 else 4
      assertEquals(steps,if skip==0 then total else math.max(1,total-skip),key)
  }
  test("invalid anti-burn options fail") {
    intercept[IllegalArgumentException](AntiBurn(-1,3))
    intercept[IllegalArgumentException](AntiBurn(0,0))
  }
  test("random vectorscope composes with all latent callbacks through generation") {
    val configured=sys.props.get("scalatorch.clip.dir");assume(configured.nonEmpty,"public tiny model required")
    val root=Path.of(configured.get).getParent
    val stream=getClass.getResourceAsStream("/vectorscope-pipeline-reference.json")
    val refs=try NativeJson.parse(new String(stream.readAllBytes(),UTF_8)).asInstanceOf[Obj].fields finally stream.close()
    val pipe=new EulerImagePipeline(ClipFastTokenizer.load(root.resolve("tokenizer/vocab.json"),root.resolve("tokenizer/merges.txt")),
      PretrainedClip.load(root.resolve("text_encoder")),PretrainedUNet.load(root.resolve("unet")),PretrainedVae.load(root.resolve("vae")))
    val input=Tensor.fromFloats(Array.tabulate(3*16*16)(i=>(i%256).toFloat/127.5f-1f),Shape.from(Seq(1L,3L,16L,16L)))
    val layout=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
    for mode <- Vector("txt2img","img2img");sampler <- Vector("euler","ancestral","dpm");(skip,avg) <- Vector((0,3),(2,3)) do
      val request=DiffusionImageRequest("a red cat","",16,16,42,guidanceScale=7.5f,torchCpuNoise=true,
        latentMirroring=Some(LatentMirroring("Blend Average","90 Degree Rotation",1,.6)),resharpen=Some(Resharpen(1.5,"Cos")),
        colorGrading=Some(LatentColorGrading(true,.8,.4,-.3,.7,true,.6)),antiBurn=Some(AntiBurn(skip,avg)),
        vectorscope=Some(LatentVectorscope(.4,-.7,.8,.2,-.3,.6,"N.Random","Cos",1.3)))
      val source=Option.when(mode=="img2img")(DiffusionImageSource(input,.75))
      val actual=sampler match
        case "euler" => pipe.generateLatents(request,layout.euler(4),source=source)
        case "ancestral" => pipe.generateAncestralLatents(request,layout.euler(4),source=source)
        case _ => pipe.generateDpmLatents(request,DpmSchedule.uniform(layout,4),source=source)
      val key=s"${mode}_${sampler}_${skip}_${avg}"
      val expected=refs(key).asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat)
      val error=actual.toFloatArray.zip(expected).map((a,b)=>math.abs(a.toDouble-b)).max
      assert(error<.001,s"$key: $error")
  }

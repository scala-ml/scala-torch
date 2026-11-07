package scalatorch.models

import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.Path
import scalatorch.{Shape,Tensor}
import scalatorch.models.worker.NativeJson
import NativeJson.*

class CfgScheduleSuite extends munit.FunSuite:
  test("CFG schedule matches NUI trajectories with rescale and image inputs") {
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty,"public tiny model required")
    val root=Path.of(configured.get).getParent
    val stream=getClass.getResourceAsStream("/cfg-schedule-reference.json")
    val refs=try NativeJson.parse(new String(stream.readAllBytes(),UTF_8)).asInstanceOf[Obj].fields finally stream.close()
    val pipe=new EulerImagePipeline(ClipFastTokenizer.load(root.resolve("tokenizer/vocab.json"),root.resolve("tokenizer/merges.txt")),
      PretrainedClip.load(root.resolve("text_encoder")),PretrainedUNet.load(root.resolve("unet")),PretrainedVae.load(root.resolve("vae")))
    val input=Tensor.fromFloats(Array.tabulate(3*16*16)(i=>(i%256).toFloat/127.5f-1f),Shape.from(Seq(1L,3L,16L,16L)))
    val layout=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
    for mode <- Vector("txt2img","img2img");sampler <- Vector("euler","ancestral","dpm");curve <- Vector("linear","cosine","square","sqrt") do
      val request=DiffusionImageRequest("a red cat","",16,16,42,guidanceScale=7.5f,torchCpuNoise=true,
        guidanceRescale=if mode=="img2img" then 0f else .7f,cfgSchedule=Some(CfgSchedule(10,2,curve,.2,.8)))
      val source=Option.when(mode=="img2img")(DiffusionImageSource(input,.75))
      val actual=sampler match
        case "euler" => pipe.generateLatents(request,layout.euler(4),source=source)
        case "ancestral" => pipe.generateAncestralLatents(request,layout.euler(4),source=source)
        case _ => pipe.generateDpmLatents(request,DpmSchedule.uniform(layout,4),source=source)
      val key=s"${mode}_${sampler}_${curve}"
      val expected=refs(key).asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat)
      val error=actual.toFloatArray.zip(expected).map((a,b)=>math.abs(a.toDouble-b)).max
      println(s"$key CFG schedule max error $error")
      assert(error<.001,s"$key: $error")
  }
  test("schedule endpoints and reversed intervals follow NUI") {
    assertEquals(CfgSchedule(7,3,"linear",.7,.2).at(.5),7f)
    assertEquals(CfgSchedule(7,3,"linear",.7,.2).at(.8),3f)
    assertEquals(CfgSchedule(7,3,"cosine").at(0),7f)
    assertEquals(CfgSchedule(7,3,"cosine").at(1),3f)
  }

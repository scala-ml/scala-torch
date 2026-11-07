package scalatorch.models

import scalatorch.models.worker.NativeJson
import scalatorch.{Shape,Tensor}
import NativeJson.*

class DetailDaemonSuite extends munit.FunSuite:
  override val munitTimeout: scala.concurrent.duration.Duration=scala.concurrent.duration.Duration(3,"minutes")
  test("detail daemon matches NUI across CFG branches and image modes") {
    val configured=sys.props.get("scalatorch.clip.dir");assume(configured.nonEmpty,"public tiny model required")
    val root=java.nio.file.Path.of(configured.get).getParent
    val stream=getClass.getResourceAsStream("/detail-daemon-pipeline-reference.json")
    val refs=try NativeJson.parse(new String(stream.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8)).asInstanceOf[Obj].fields finally stream.close()
    val pipe=new EulerImagePipeline(ClipFastTokenizer.load(root.resolve("tokenizer/vocab.json"),root.resolve("tokenizer/merges.txt")),
      PretrainedClip.load(root.resolve("text_encoder")),PretrainedUNet.load(root.resolve("unet")),PretrainedVae.load(root.resolve("vae")))
    val input=Tensor.fromFloats(Array.tabulate(3*16*16)(i=>(i%256).toFloat/127.5f-1f),Shape.from(Seq(1L,3L,16L,16L)))
    val layout=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
    for kind <- Vector("txt2img","img2img");sampler <- Vector("euler","dpm");mode <- Vector("both","cond","uncond");cfg <- Vector(1f,7.5f) do
      val options=DetailDaemon(mode=mode,amount=.35,start=.2,end=.8,bias=.5,exponent=1,startOffset=.1,endOffset=.15,
        trainingSigmas=layout.floatTrainingSigmas.map(_.toDouble))
      val request=DiffusionImageRequest("a red cat","",16,16,42,guidanceScale=cfg,torchCpuNoise=true,detailDaemon=Some(options))
      val source=Option.when(kind=="img2img")(DiffusionImageSource(input,.75))
      val actual=if sampler=="euler" then pipe.generateLatents(request,layout.euler(4),source=source)
        else pipe.generateDpmLatents(request,DpmSchedule.uniform(layout,4),source=source)
      val key=s"$kind|$sampler|$mode|${cfg.toDouble}"
      val expected=refs(key).asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat)
      val error=actual.toFloatArray.zip(expected).map((a,b)=>math.abs(a.toDouble-b)).max
      assert(error<.002,s"$key error=$error")
  }
  test("step envelopes and timestep transforms match NUI reference") {
    val stream=getClass.getResourceAsStream("/detail-daemon-reference.json")
    val data=try NativeJson.parse(new String(stream.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8)).asInstanceOf[Obj].fields finally stream.close()
    val sigmas=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true).floatTrainingSigmas.map(_.toDouble)
    def num(f: Map[String,NativeJson],k: String)=f(k).asInstanceOf[Number].value.toDouble
    for entry <- data("cases").asInstanceOf[Arr].values do
      val row=entry.asInstanceOf[Obj].fields;val f=row("options").asInstanceOf[Obj].fields
      val daemon=DetailDaemon(amount=num(f,"amount"),start=num(f,"start"),end=num(f,"end"),bias=num(f,"bias"),
        exponent=num(f,"exponent"),startOffset=num(f,"start_offset"),endOffset=num(f,"end_offset"),fade=num(f,"fade"),
        smooth=f("smooth").asInstanceOf[Bool].value,trainingSigmas=sigmas)
      val actual=daemon.multipliers(num(row,"steps").toInt)
      val expected=row("values").asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toDouble)
      assertEquals(actual.size,expected.size)
      assert(actual.zip(expected).forall((a,b)=>math.abs(a-b)<1e-12),s"$f $actual vs $expected")
    val daemon=DetailDaemon(trainingSigmas=sigmas)
    for entry <- data("transforms").asInstanceOf[Arr].values do
      val row=entry.asInstanceOf[Obj].fields
      val result=daemon.adjustment(num(row,"t").toFloat,num(row,"m")).get
      assert(math.abs(result.scale-num(row,"scale"))<1e-5,s"scale $row $result")
      assert(math.abs(result.timestep-num(row,"target"))<.002,s"timestep $row $result")
  }

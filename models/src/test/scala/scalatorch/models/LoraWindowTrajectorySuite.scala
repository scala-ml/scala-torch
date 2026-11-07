package scalatorch.models

import scalatorch.{Shape,Tensor}
import scalatorch.nn.{LinearAdapterScope,LowRankAdapter,Parameter}
import scalatorch.models.worker.NativeJson
import NativeJson.*

class LoraWindowTrajectorySuite extends munit.FunSuite:
  override val munitTimeout: scala.concurrent.duration.Duration=scala.concurrent.duration.Duration(3,"minutes")
  test("LoRA windows match each Python Diffusers denoising step") {
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty,"tiny SD fixture required")
    val root=java.nio.file.Path.of(configured.get).getParent
    val loaded=PretrainedUNet.loadWithLinearTargets(root.resolve("unet"))
    val weight=loaded.linearTargets("unet.time_embedding.linear_1")
    val in=weight.data.shape.dims(1).toInt;val out=weight.data.shape.dims(0).toInt
    val adapter=LowRankAdapter(Parameter(Tensor.fromFloats(Array.fill(in)(1f/in),Shape.from(Seq(1L,in.toLong))),false),
      Parameter(Tensor.fromFloats(Array.fill(out)(8f),Shape.from(Seq(out.toLong,1L))),false),1.0)
    val tokenizer=ClipFastTokenizer.load(root.resolve("tokenizer/vocab.json"),root.resolve("tokenizer/merges.txt"))
    val clip=PretrainedClip.load(root.resolve("text_encoder"));val vae=PretrainedVae.load(root.resolve("vae"))
    val sourceImage=Tensor.fromFloats(Array.tabulate(3*16*16)(i=>(i%256).toFloat/127.5f-1f),Shape.from(Seq(1L,3L,16L,16L)))
    val layout=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
    val stream=getClass.getResourceAsStream("/lora-window-trajectory-reference.json")
    val refs=try NativeJson.parse(new String(stream.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8)).asInstanceOf[Obj].fields finally stream.close()
    val windows=Vector((None,None,"None|None"),(Some(2.0),None,"2|None"),(None,Some(2.0),"None|2"),
      (Some(.5),Some(.9),"0.5|0.9"),(Some(5.0),None,"5|None"))
    val placements=Vector(UNetExecution.Cpu)++Option.when(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"))(UNetExecution.Cuda(offloadWeights=true))
    for placement <- placements; kind <- Vector("txt2img","img2img"); sampler <- Vector("euler","ancestral","dpm"); (start,stop,label) <- windows do
      val pipe=new EulerImagePipeline(tokenizer,clip,loaded.model,vae,unetExecution=placement)
      val window=LoraStepWindow(start,stop)
      val request=DiffusionImageRequest("a red cat","",16,16,42,guidanceScale=7.5f,torchCpuNoise=true,
        loraStepFilter=Some((step,total,item)=>(item ne adapter) || window.active(step,total)))
      val source=Option.when(kind=="img2img")(DiffusionImageSource(sourceImage,.75))
      val actual=scala.collection.mutable.ArrayBuffer.empty[Array[Float]]
      LinearAdapterScope.withAdapters(Map(weight->Vector(adapter->1f))) {
        val report: DiffusionImageProgress=>Unit=p=>actual+=p.latents.toFloatArray
        sampler match
          case "euler" => pipe.generateLatents(request,layout.euler(4),source=source,onProgress=report)
          case "ancestral" => pipe.generateAncestralLatents(request,layout.euler(4),source=source,onProgress=report)
          case _ => pipe.generateDpmLatents(request,DpmSchedule.uniform(layout,4),source=source,onProgress=report)
      }
      val expected=refs(s"$kind|$sampler|$label").asInstanceOf[Arr].values
      assertEquals(actual.size,expected.size)
      for i <- actual.indices do
        val wanted=expected(i).asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat)
        val error=actual(i).zip(wanted).map((a,b)=>math.abs(a.toDouble-b)).max
        assert(error<.002,s"$kind $sampler $label ${placement.device} step $i error=$error")
  }

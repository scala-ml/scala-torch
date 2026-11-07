package scalatorch.models

import java.nio.file.Path
import scalatorch.{Shape,Tensor}
import scalatorch.models.worker.NativeJson
import NativeJson.*

class ControlNetDetailSuite extends munit.FunSuite:
  test("ControlNet and Detail Daemon predictions match independent Diffusers") {
    val configured=sys.env.get("SCALATORCH_CONTROLNET_PIPELINE").map(p=>Path.of(p).resolve("text_encoder").toString)
      .orElse(sys.props.get("scalatorch.clip.dir"))
    val controlDir=sys.env.get("SCALATORCH_TINY_CONTROLNET").orElse(sys.props.get("scalatorch.controlnet.dir"))
    assume(configured.nonEmpty && controlDir.nonEmpty,"tiny SD and generated ControlNet required")
    val root=Path.of(configured.get).getParent
    val unet=PretrainedUNet.load(root.resolve("unet"))
    val control=PretrainedControlNet.load(Path.of(controlDir.get))
    control.config.requireCompatible(UNetConfig.read(root.resolve("unet/config.json")))
    val sample=Tensor.fromFloats(Array.tabulate(256)(i=>math.sin(i.toFloat*.07f).toFloat),Shape.from(Seq(1L,4L,8L,8L)))
    val image=Tensor.fromFloats(Array.tabulate(3*64*64)(i=>(i%251).toFloat/250f),Shape.from(Seq(1L,3L,64L,64L)))
    val pos=Tensor.fromFloats(Array.tabulate(77*32)(i=>math.sin(i.toFloat*.013f).toFloat),Shape.from(Seq(1L,77L,32L)))
    val neg=Tensor.fromFloats(Array.tabulate(77*32)(i=>math.cos(i.toFloat*.017f).toFloat),pos.shape)
    val time=Tensor.fromFloats(Array(666f),Shape.from(Seq(1L)))
    val stream=sys.env.get("SCALATORCH_CONTROLNET_REFERENCE").map(p=>java.nio.file.Files.newInputStream(Path.of(p)))
      .getOrElse(getClass.getResourceAsStream("/controlnet-detail-reference.json"))
    val refs=try NativeJson.parse(new String(stream.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8)).asInstanceOf[Obj].fields finally stream.close()
    val noise=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
    val request=DiffusionControlNetRequest(control,image,weight=.7)
    // Diffusers enables a step only if its entire [i/N,(i+1)/N] interval
    // lies inside the requested interval, including equality at the end.
    val partial=request.copy(guidanceStart=.2,guidanceEnd=.7)
    assertEquals((0 until 4).map(i=>partial.scales(i/4.0,(i+1)/4.0).isDefined).toVector,
      Vector(false,true,false,false))
    assert(partial.scales(.25,.7).isDefined)
    val soft=request.copy(controlMode="My prompt is more important").scales(.5).get
    for i <- soft.indices do assert(math.abs(soft(i)-(.7*math.pow(.825,12-i)))<1e-6)
    assertEquals(request.copy(hiresSoftInjection=true).scales(.5).get,soft)
    val unconditioned=UNetExecution.Cpu.withPredictor(unet,pos,Some(neg),()=>false) { predictor =>
      predictor(sample,time,7.5f)
    }
    val controlReference=refs("none").asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat)
    assert(unconditioned.toFloatArray.zip(controlReference).exists((a,b)=>math.abs(a-b)>1e-4),"ControlNet fixture must affect prediction")
    val placements=Vector(UNetExecution.Cpu)++Option.when(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"))(UNetExecution.Cuda(offloadWeights=true))
    for placement <- placements; mode <- Vector("none","both","cond","uncond") do
      val config=DetailDaemon(mode=if mode=="none" then "both" else mode,amount=.35,start=.2,end=.8,bias=.5,
        exponent=1,startOffset=.1,endOffset=.15,trainingSigmas=noise.floatTrainingSigmas.map(_.toDouble))
      val adjustment=Option.when(mode!="none")(config.adjustment(666f,config.multipliers(4)(1)).get)
      val result=placement.withPredictor(unet,pos,Some(neg),()=>false,controlNets=Vector(request)) {
        predictor => predictor(sample,time,7.5f,.5,0f,None,adjustment)
      }
      val expected=refs(mode).asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat)
      val error=result.toFloatArray.zip(expected).map((a,b)=>math.abs(a.toDouble-b)).max
      println(s"ControlNet detail $mode ${placement.device}: max error $error")
      assert(error<.001,s"$mode ${placement.device} error=$error")
  }

package scalatorch.models

import scalatorch.{Shape,Tensor}
import scalatorch.models.worker.NativeJson
import NativeJson.*

class LatentVectorscopeSuite extends munit.FunSuite:
  override val munitTimeout: scala.concurrent.duration.Duration=scala.concurrent.duration.Duration(3,"minutes")
  test("all Vectorscope methods and schedules follow NUI across repeated steps") {
    val stream=getClass.getResourceAsStream("/vectorscope-reference.json")
    val refs=try NativeJson.parse(new String(stream.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8)).asInstanceOf[Obj].fields finally stream.close()
    for xl <- Vector(false,true);method <- Vector("Straight","Cross","N.Random","U.Random");
        scaling <- Vector("Flat","Cos","Sin","1 - Cos","1 - Sin") do
      val config=LatentVectorscope(.4,-.7,.8,.2,-.3,.6,method,scaling,1.3)
      val state=config.state(42,xl)
      var x=Tensor.fromFloats(Array.tabulate(64)(i=>math.sin(i.toFloat*.17f).toFloat),Shape.from(Seq(1L,4L,4L,4L)))
      for i <- 1 to 4 do x=state.step(x,i,4)
      val key=s"${if xl then "True" else "False"}|$method|$scaling"
      val expected=refs(key).asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat)
      val error=x.toFloatArray.zip(expected).map((a,b)=>math.abs(a.toDouble-b)).max
      assert(error<.00001,s"$key error=$error")
  }
  test("batched random vectorscope preserves stream position and hires seed") {
    val stream=getClass.getResourceAsStream("/vectorscope-reference.json")
    val refs=try NativeJson.parse(new String(stream.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8)).asInstanceOf[Obj].fields finally stream.close()
    for seed <- Vector(42L,43L);batchIndex <- 0 until 2 do
      val state=LatentVectorscope(.4,-.7,.8,.2,-.3,.6,"N.Random","Cos",1.3).state(seed,false,2,batchIndex)
      var x=Tensor.fromFloats(Array.tabulate(64)(i=>math.sin((i+batchIndex*64).toFloat*.17f).toFloat),Shape.from(Seq(1L,4L,4L,4L)))
      for i <- 1 to 4 do x=state.step(x,i,4)
      val expected=refs(s"batch|$seed").asInstanceOf[Arr].values.slice(batchIndex*64,(batchIndex+1)*64).map(_.asInstanceOf[Number].value.toFloat)
      val error=x.toFloatArray.zip(expected).map((a,b)=>math.abs(a.toDouble-b)).max
      assert(error<.00001,s"seed=$seed batch=$batchIndex error=$error")
  }
  test("two-image batch sampling matches NUI seeded Vectorscope trajectories") {
    val configured=sys.props.get("scalatorch.clip.dir");assume(configured.nonEmpty,"public tiny model required")
    val root=java.nio.file.Path.of(configured.get).getParent
    val stream=getClass.getResourceAsStream("/vectorscope-batch-reference.json")
    val refs=try NativeJson.parse(new String(stream.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8)).asInstanceOf[Obj].fields finally stream.close()
    val pipe=new EulerImagePipeline(ClipFastTokenizer.load(root.resolve("tokenizer/vocab.json"),root.resolve("tokenizer/merges.txt")),
      PretrainedClip.load(root.resolve("text_encoder")),PretrainedUNet.load(root.resolve("unet")),PretrainedVae.load(root.resolve("vae")))
    val layout=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
    for method <- Vector("N.Random","U.Random");sampler <- Vector("euler","ancestral","dpm");index <- 0 until 2 do
      val request=DiffusionImageRequest("a red cat","",16,16,42+index,guidanceScale=7.5f,torchCpuNoise=true,
        vectorscope=Some(LatentVectorscope(.4,-.7,.8,.2,-.3,.6,method,"Cos",1.3)),
        latentHookSeed=Some(42),latentHookBatchSize=2,latentHookBatchIndex=index)
      val actual=sampler match
        case "euler" => pipe.generateLatents(request,layout.euler(4))
        case "ancestral" => pipe.generateAncestralLatents(request,layout.euler(4))
        case _ => pipe.generateDpmLatents(request,DpmSchedule.uniform(layout,4))
      val expected=refs(s"$sampler|$method").asInstanceOf[Arr].values.slice(index*256,(index+1)*256).map(_.asInstanceOf[Number].value.toFloat)
      val error=actual.toFloatArray.zip(expected).map((a,b)=>math.abs(a.toDouble-b)).max
      assert(error<.001,s"$method $sampler index=$index error=$error")
      if sampler=="euler" then
        val small=actual.toFloatArray
        val enlarged=Tensor.fromFloats(Array.tabulate(4*16*16) { i =>
          val c=i/256;val y=(i%256)/16;val x=i%16
          small(c*64+(y/2)*8+x/2)
        },Shape.from(Seq(1L,4L,16L,16L)))
        val second=request.copy(width=32,height=32,latentHookSeed=Some(43))
        var last: Tensor[Float]=null
        pipe.generate(second,layout.euler(4),latentSource=Some(DiffusionLatentSource(enlarged,.75)),onProgress=p=>last=p.latents)
        val hires=refs(s"hires|$method").asInstanceOf[Arr].values.slice(index*1024,(index+1)*1024).map(_.asInstanceOf[Number].value.toFloat)
        val diff=last.toFloatArray.zip(hires).map((a,b)=>math.abs(a.toDouble-b)).max
        assert(diff<.002,s"$method hires index=$index error=$diff")
  }

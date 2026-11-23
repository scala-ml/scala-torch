package scalatorch.models

import java.nio.file.{Files,Path}
import scalatorch.*

class PostDiffPipelineSuite extends munit.FunSuite:
  test("Euler ancestral switches size and replays with request-local noise") {
    val root=Path.of(sys.env.getOrElse("SCALATORCH_POSTDIFF_TINY", ".superpowers/models/tiny-sd-pipe")).toAbsolutePath
    assume(Files.exists(root.resolve("unet/config.json")),"tiny SD fixture required")
    val pipe=new EulerImagePipeline(ClipFastTokenizer.loadDirectory(root.resolve("tokenizer")),
      PretrainedClip.load(root.resolve("text_encoder")),PretrainedUNet.load(root.resolve("unet")),PretrainedVae.load(root.resolve("vae")))
    val schedule=EulerSchedule(Vector(750f,500f,250f,0f),Vector(2.0,1.0,0.5,0.1,0.0),2.0)
    val request=DiffusionImageRequest("a red cat","",16,16,42,guidanceScale=1f,
      postDiff=Some(PostDiffConfig(0.5,0.5)))
    var shapes=Vector.empty[Seq[Long]]
    val first=pipe.generateAncestralLatents(request,schedule,onProgress=p=>shapes :+= p.latents.shape.dims)
    assertEquals(shapes.map(_.takeRight(2)),Vector(Seq(4L,4L),Seq(8L,8L),Seq(8L,8L),Seq(8L,8L)))
    assertEquals(first.toFloatArray.toVector,pipe.generateAncestralLatents(request,schedule).toFloatArray.toVector)
    var firstStep=Option.empty[(Array[Float],Array[Float],Array[Float])]
    val torchRequest=request.copy(torchCpuNoise=true,onStepObservation=Some((step,_,before,clean,after) =>
      if step==1 then firstStep=Some((before.toFloatArray,clean.toFloatArray,after.toFloatArray))))
    pipe.generateAncestralLatents(torchRequest,schedule)
    val (before,clean,after)=firstStep.get
    val rng=new TorchCpuNormal(42)
    rng.normal(before.length) // initial low-resolution latent consumes the first draw
    val expectedNext=rng.normal(before.length)
    val sigma=2.0; val next=1.0
    val up=next*math.sqrt(1.0-(next/sigma)*(next/sigma))
    val down=math.sqrt(next*next-up*up)
    val inferred=before.indices.map(i => (after(i)-before(i)-(before(i)-clean(i))/sigma*(down-sigma))/up)
    val noiseError=inferred.zip(expectedNext).map((a,b)=>math.abs(a-b)).max
    assert(noiseError<1e-5,s"first ancestral noise must follow initial noise; max error=$noiseError")
    var transition=Option.empty[(Array[Float],Array[Float])]
    val capture=request.copy(torchCpuNoise=true,onStepObservation=Some((step,_,noisy,denoised,_) =>
      if step==2 then transition=Some((noisy.toFloatArray,denoised.toFloatArray))))
    pipe.generateAncestralLatents(capture,schedule)
    val (resized,estimated)=transition.get
    val transitionRng=new TorchCpuNormal(42)
    transitionRng.normal(before.length)
    transitionRng.normal(before.length)
    val fresh=transitionRng.normal(resized.length)
    val transitionError=resized.indices.map(i=>math.abs((resized(i)-estimated(i))/schedule.sigmas(1)-fresh(i))).max
    assert(transitionError<1e-5,s"transition must use fresh noise for same-step prediction; max error=$transitionError")
    val explicitNoise=Tensor.fromFloats(Array.fill(256)(0f),Shape.from(Seq(1,4,8,8)))
    intercept[IllegalArgumentException](pipe.generateAncestralLatents(request,schedule,noise=Some(explicitNoise)))
    val image=Tensor.fromFloats(Array.fill(3*16*16)(0f),Shape.from(Seq(1,3,16,16)))
    intercept[IllegalArgumentException](pipe.generateAncestralLatents(request,schedule,source=Some(DiffusionImageSource(image,0.5))))
    intercept[IllegalArgumentException](pipe.generateAncestralLatents(request.copy(variationSeed=Some(43),variationStrength=0.5),schedule))
    intercept[IllegalArgumentException](pipe.generateAncestralLatents(request.copy(postDiff=Some(PostDiffConfig(0.01,0.5))),schedule))
    intercept[IllegalArgumentException](pipe.generateAncestralLatents(request.copy(cfgSchedule=Some(CfgSchedule(7,3,"linear"))),schedule))
    DeepCache.install(Some(DeepCacheConfig(3,0,0)))
    try intercept[IllegalArgumentException](pipe.generateAncestralLatents(request,schedule))
    finally DeepCache.install(None)
    var cancelled=false
    intercept[java.util.concurrent.CancellationException] {
      pipe.generateAncestralLatents(request,schedule,onProgress=_=>cancelled=true,cancelled=()=>cancelled)
    }
    for torch <- Seq(false,true) do
      val base=request.copy(torchCpuNoise=torch)
      val one=pipe.generateAncestralLatents(base,schedule).toFloatArray
      assertEquals(one.toVector,pipe.generateAncestralLatents(base,schedule).toFloatArray.toVector)
      intercept[BatchedSamplingUnsupported](pipe.generateAncestralLatents(base.copy(batchSeeds=Vector(42,43)),schedule))
    if sys.env.get("SCALATORCH_CUDA_TESTS").contains("1") then
      val gpu=new EulerImagePipeline(pipe.tokenizer,pipe.textEncoder,pipe.unet,pipe.vae,
        vaeExecution=VaeExecution.Cuda(),unetExecution=UNetExecution.Cuda())
      for torch <- Seq(false,true) do
        val base=request.copy(torchCpuNoise=torch)
        val normal=base.copy(postDiff=None)
        val normalExpected=Seq(42L,43L).flatMap(seed=>gpu.generateAncestralLatents(normal.copy(seed=seed),schedule).toFloatArray)
        val normalBatch=gpu.generateAncestralLatents(normal.copy(batchSeeds=Vector(42,43)),schedule).toFloatArray
        val normalError=normalBatch.zip(normalExpected).map((a,b)=>math.abs(a.toDouble-b.toDouble)).max
        println(s"baseline CUDA batch parity torch=$torch max error=$normalError")
        val expected=Seq(42L,43L).flatMap(seed=>gpu.generateAncestralLatents(base.copy(seed=seed),schedule).toFloatArray)
        val batched=gpu.generateAncestralLatents(base.copy(batchSeeds=Vector(42,43)),schedule).toFloatArray
        assertEquals(batched.length,expected.size)
        val error=batched.zip(expected).map((a,b)=>math.abs(a.toDouble-b.toDouble)).max
        println(s"PostDiff CUDA batch parity torch=$torch max error=$error")
        val limit=if sys.env.get("SCALATORCH_CUDA_FP16").contains("0") then 5e-4 else 0.02
        assert(error<limit,s"CUDA batch parity torch=$torch max error=$error limit=$limit")
        val n=batched.length/2
        val seedDifference=batched.take(n).zip(batched.drop(n)).map((a,b)=>math.abs(a.toDouble-b.toDouble)).max
        assert(seedDifference>0.1,s"batch seeds unexpectedly similar: $seedDifference")
    intercept[IllegalArgumentException](pipe.generateLatents(request,schedule))
  }

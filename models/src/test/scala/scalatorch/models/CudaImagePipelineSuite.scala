package scalatorch.models

import scalatorch.*
import java.nio.file.Path
import java.util.concurrent.CancellationException

class CudaImagePipelineSuite extends munit.FunSuite:
  private def pipeline(execution: VaeExecution): EulerImagePipeline =
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"real GPU test disabled")
    val dir=sys.props.get("scalatorch.vae.dir")
    assume(dir.nonEmpty,"public tiny model required")
    val root=Path.of(dir.get).getParent
    new EulerImagePipeline(ClipFastTokenizer.load(root.resolve("tokenizer/vocab.json"),root.resolve("tokenizer/merges.txt")),
      PretrainedClip.load(root.resolve("text_encoder")),PretrainedUNet.load(root.resolve("unet")),
      PretrainedVae.load(root.resolve("vae")),vaeExecution=execution)
  test("Euler and DPM image pipelines use CUDA decode with one latent scaling") {
    val cpu=pipeline(VaeExecution.Cpu)
    val gpu=new EulerImagePipeline(cpu.tokenizer,cpu.textEncoder,cpu.unet,cpu.vae,vaeExecution=VaeExecution.Cuda())
    val noise=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
    val request=DiffusionImageRequest("a red cat","",10,6,42,guidanceScale=1f)
    for dpm <- Seq(false,true) do
      def run(p: EulerImagePipeline)=
        if dpm then p.generateDpm(request,DpmSchedule.karras(noise,2)) else p.generate(request,noise.euler(2))
      val expected=run(cpu);val actual=run(gpu)
      assertEquals(actual.shape,expected.shape)
      val max=actual.toFloatArray.zip(expected.toFloatArray).map((a,e) => math.abs(a.toDouble-e)).max
      println(s"CUDA VAE pipeline dpm=$dpm maxAbsError=$max")
      assert(max<2e-5)
    intercept[CancellationException](gpu.generate(request,noise.euler(1),cancelled=() => true))
  }
  test("CUDA decode failure retains exact input for recovery and cancellation is not wrapped") {
    val p=pipeline(VaeExecution.Cuda(maxBytes=1))
    val request=DiffusionImageRequest("a cat","",10,6,42,guidanceScale=1f)
    val schedule=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true).euler(1)
    var latents: Tensor[Float]=null
    val error=intercept[VaeDecodeFailure](p.generate(request,schedule,onProgress=x => latents=x.latents))
    assertEquals(error.decoderInput.toFloatArray.toVector,
      latents.toFloatArray.toVector.map(_*(1/p.vae.config.scalingFactor).toFloat))
    var checks=0
    intercept[CancellationException] {
      VaeExecution.Cuda().decode(p.vae.decoder,error.decoderInput,() => { checks+=1;checks>5 })
    }
    val result=VaeExecution.Cuda().decode(p.vae.decoder,error.decoderInput,() => false)
    assert(result.toFloatArray.forall(_.isFinite))
  }
  test("worker VAE device settings are explicit and reject invalid values") {
    assertEquals(VaeExecution.fromEnvironment(Map.empty),VaeExecution.Cpu)
    assertEquals(VaeExecution.fromEnvironment(Map("SDFX_SCALATORCH_VAE_DEVICE"->"cuda")),VaeExecution.Cuda())
    assertEquals(VaeExecution.fromEnvironment(Map("SDFX_SCALATORCH_VAE_DEVICE"->"cuda:2",
      "SDFX_SCALATORCH_VAE_MAX_BYTES"->"4096")),VaeExecution.Cuda(2,4096))
    for device <- Seq("gpu","cuda:-1","cuda:foo") do
      intercept[IllegalArgumentException](VaeExecution.fromEnvironment(Map("SDFX_SCALATORCH_VAE_DEVICE"->device)))
    intercept[IllegalArgumentException](VaeExecution.fromEnvironment(Map("SDFX_SCALATORCH_VAE_DEVICE"->"cuda",
      "SDFX_SCALATORCH_VAE_MAX_BYTES"->"0")))
  }
  test("resident CUDA U-Net stays available across Euler and DPM steps with classifier-free guidance") {
    val cpu=pipeline(VaeExecution.Cpu)
    val gpu=new EulerImagePipeline(cpu.tokenizer,cpu.textEncoder,cpu.unet,cpu.vae,
      vaeExecution=VaeExecution.Cuda(),unetExecution=UNetExecution.Cuda())
    val noise=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
    for dpm <- Seq(false,true); guidance <- Seq(1f,7.5f) do
      val request=DiffusionImageRequest("a red cat","green trees",10,6,42,guidanceScale=guidance)
      def run(p: EulerImagePipeline)=
        if dpm then p.generateDpm(request,DpmSchedule.karras(noise,3)) else p.generate(request,noise.euler(3))
      val expected=run(cpu);val actual=run(gpu)
      assertEquals(actual.shape,expected.shape)
      val error=actual.toFloatArray.zip(expected.toFloatArray).map((a,e) => math.abs(a.toDouble-e)).max
      println(s"CUDA U-Net pipeline dpm=$dpm cfg=$guidance maxAbsError=$error")
      assert(error<2e-5)
    val request=DiffusionImageRequest("a cat","",10,6,43)
    var cancelled=false
    intercept[CancellationException] {
      gpu.generate(request,noise.euler(3),onProgress=_ => cancelled=true,cancelled=() => cancelled)
    }
    assert(gpu.generate(request,noise.euler(1)).toFloatArray.forall(_.isFinite))
    val limited=new EulerImagePipeline(cpu.tokenizer,cpu.textEncoder,cpu.unet,cpu.vae,unetExecution=UNetExecution.Cuda(maxBytes=1))
    intercept[IllegalArgumentException](limited.generate(request,noise.euler(1)))
  }
  /** Float32 kernels for a comparison, whatever precision an earlier suite of this test JVM left selected. */
  private def float32[A](body: => A): A =
    val previous=CudaDiffusionRuntime.fp16Default
    CudaDiffusionRuntime.setHalfPrecision(Some(false))
    try body finally CudaDiffusionRuntime.setHalfPrecision(Some(previous))
  test("batched sampling gives every image the latents and the noise stream of its own seed") { float32 {
    val cpu=pipeline(VaeExecution.Cpu)
    val gpu=new EulerImagePipeline(cpu.tokenizer,cpu.textEncoder,cpu.unet,cpu.vae,
      vaeExecution=VaeExecution.Cuda(),unetExecution=UNetExecution.Cuda())
    val noise=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
    val seeds=Vector(173L,174L,175L)
    val samplers=Vector[(String,DiffusionImageRequest => Tensor[Float])](
      "Euler" -> (r => gpu.generateLatents(r,noise.euler(4))),
      "Euler a" -> (r => gpu.generateAncestralLatents(r,noise.euler(4))),
      "Heun" -> (r => gpu.generateHeun(r,noise.discreteSchedule(4,DiffusionTimestepSpacing.Linspace,0,"none",false),returnLatents=true)),
      "DPM++ 2M" -> (r => gpu.generateDpmLatents(r,DpmSchedule.karras(noise,4))),
      "DDIM" -> (r => gpu.generateDdim(r,DdimSchedule(noise,4,DiffusionTimestepSpacing.Leading,1),false,1f,returnLatents=true)),
      "LMS" -> (r => gpu.generateLms(r,noise.discreteSchedule(4,DiffusionTimestepSpacing.Linspace,0,"none",false),returnLatents=true)),
      "DPM2 a" -> (r => gpu.generateKdpm2(r,noise.discreteSchedule(4,DiffusionTimestepSpacing.Linspace,0,"none",true),ancestral=true,returnLatents=true)),
      "DPM++ 2S a" -> (r => gpu.generateDpm2Sa(r,DpmSchedule.karras(noise,4),returnLatents=true)),
      "DPM++ SDE" -> (r => gpu.generateDpmSde(r,noise.discreteSchedule(4,DiffusionTimestepSpacing.Linspace,0,"karras",false),returnLatents=true)),
      "DPM++ 2M SDE" -> (r => gpu.generateDpmMultistepSde(r,DpmSchedule.exponential(noise,4),returnLatents=true)),
      "DPM++ 3M SDE" -> (r => gpu.generateDpmMultistepSde(r,DpmSchedule.exponential(noise,4),order=3,returnLatents=true)),
      "UniPC" -> (r => gpu.generateUniPc(r,DpmSchedule.karras(noise,4),returnLatents=true)),
      "LCM" -> (r => gpu.generateLcm(r,LcmSchedule(noise,4),returnLatents=true)))
    // every sampler with guidance; the stochastic ones also with the torch noise stream and without guidance (one batch of three)
    val variants=Set("Euler a","DPM++ SDE")
    for (name,run) <- samplers; (guidance,torchNoise) <- Seq((7.5f,false),(7.5f,true),(1f,false)).take(if variants(name) then 3 else 1) do
      val request=DiffusionImageRequest("a red cat","green trees",10,6,seeds.head,guidanceScale=guidance,torchCpuNoise=torchNoise)
      val batched=run(request.copy(batchSeeds=seeds))
      val single=seeds.map(seed => run(request.copy(seed=seed)))
      assertEquals(batched.shape.dims.toSeq,3L+:single.head.shape.dims.toSeq.drop(1))
      val expected=single.flatMap(_.toFloatArray).toArray
      val error=batched.toFloatArray.zip(expected).map((a,e) => math.abs(a.toDouble-e)).max
      println(s"batched sampling $name cfg=$guidance torchNoise=$torchNoise maxAbsError=$error")
      // Float32 rounding of the batched U-Net pass, accumulated over the steps (latents reach the initial sigma of about 14);
      // a wrong seed or noise stream gives errors of order one.
      assert(error<5e-4,s"$name cfg=$guidance torchNoise=$torchNoise")
      assert(single(0).toFloatArray.zip(single(1).toFloatArray).exists((a,b) => math.abs(a-b)>1e-3),"seeds must give different latents")
    // an image pass, a hook or a prompt that changes between steps is not sampled as a batch
    val request=DiffusionImageRequest("a red cat","green trees",10,6,173,batchSeeds=seeds)
    intercept[BatchedSamplingUnsupported](gpu.generate(request,noise.euler(2)))
    intercept[BatchedSamplingUnsupported](gpu.generateLatents(request.copy(guidanceRescale=0.5f),noise.euler(2)))
    intercept[BatchedSamplingUnsupported](gpu.generateLatents(request.copy(prompt="a [red:blue:0.5] cat"),noise.euler(2)))
    assert(EulerImagePipeline.batchable(request) && !EulerImagePipeline.batchable(request.copy(mahiro=true)) &&
      !EulerImagePipeline.batchable(request.copy(variationStrength=0.3,variationSeed=Some(1L))))
  } }
  test("worker U-Net placement and budget settings reject invalid values") {
    assertEquals(UNetExecution.fromEnvironment(Map.empty),UNetExecution.Cpu)
    assertEquals(UNetExecution.fromEnvironment(Map("SDFX_SCALATORCH_UNET_DEVICE"->"cuda")),UNetExecution.Cuda())
    assertEquals(UNetExecution.fromEnvironment(Map("SDFX_SCALATORCH_UNET_DEVICE"->"cuda:2",
      "SDFX_SCALATORCH_UNET_MAX_BYTES"->"4096")),UNetExecution.Cuda(2,4096))
    for device <- Seq("gpu","cuda:-1","cuda:foo") do
      intercept[IllegalArgumentException](UNetExecution.fromEnvironment(Map("SDFX_SCALATORCH_UNET_DEVICE"->device)))
    intercept[IllegalArgumentException](UNetExecution.Cuda(maxBytes=0))
  }
  test("offload environment settings reach complete low-budget CUDA generation") {
    val env=Map("SDFX_SCALATORCH_UNET_DEVICE"->"cuda","SDFX_SCALATORCH_VAE_DEVICE"->"cuda",
      "SDFX_SCALATORCH_UNET_OFFLOAD_WEIGHTS"->"true","SDFX_SCALATORCH_VAE_OFFLOAD_WEIGHTS"->"true",
      "SDFX_SCALATORCH_UNET_MAX_BYTES"->"2097152","SDFX_SCALATORCH_VAE_MAX_BYTES"->"2097152")
    assertEquals(UNetExecution.fromEnvironment(env),UNetExecution.Cuda(maxBytes=2097152,offloadWeights=true))
    assertEquals(VaeExecution.fromEnvironment(env),VaeExecution.Cuda(maxBytes=2097152,offloadWeights=true))
    for value <- Seq("yes","1","") do
      intercept[IllegalArgumentException](UNetExecution.fromEnvironment(env.updated("SDFX_SCALATORCH_UNET_OFFLOAD_WEIGHTS",value)))
      intercept[IllegalArgumentException](VaeExecution.fromEnvironment(env.updated("SDFX_SCALATORCH_VAE_OFFLOAD_WEIGHTS",value)))
    val cpu=pipeline(VaeExecution.Cpu)
    val resident=new EulerImagePipeline(cpu.tokenizer,cpu.textEncoder,cpu.unet,cpu.vae,
      vaeExecution=VaeExecution.Cuda(),unetExecution=UNetExecution.Cuda())
    val offloaded=new EulerImagePipeline(cpu.tokenizer,cpu.textEncoder,cpu.unet,cpu.vae,
      vaeExecution=VaeExecution.fromEnvironment(env),unetExecution=UNetExecution.fromEnvironment(env))
    val request=DiffusionImageRequest("a cat","green trees",10,6,42,guidanceScale=7.5f)
    val noise=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
    for dpm <- Seq(false,true) do
      def run(p: EulerImagePipeline)=if dpm then p.generateDpm(request,DpmSchedule.karras(noise,2)) else p.generate(request,noise.euler(2))
      assertEquals(run(offloaded).toFloatArray.toVector,run(resident).toFloatArray.toVector)
  }
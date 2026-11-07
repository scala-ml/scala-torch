package scalatorch.models

import scalatorch.*
import java.nio.file.Path

class EulerAncestralPipelineSuite extends munit.FunSuite:
  test("Euler a replays seeds and works with img2img, latent refinement and CUDA") {
    val directory=sys.props.get("scalatorch.sdxl.dir")
    assume(directory.nonEmpty,"tiny SDXL fixture required")
    val root=Path.of(directory.get)
    def tokenizer(part: String)=ClipFastTokenizer.loadDirectory(root.resolve(part))
    val first=tokenizer("tokenizer");val clip=PretrainedClip.load(root.resolve("text_encoder"))
    val text=new SdxlTextConditioning(first,clip,tokenizer("tokenizer_2"),PretrainedClip.load(root.resolve("text_encoder_2")))
    val unet=PretrainedUNet.load(root.resolve("unet"));val vae=PretrainedVae.load(root.resolve("vae"))
    val cpu=new EulerImagePipeline(first,clip,unet,vae,sdxlConditioning=Some(text))
    val schedule=EulerSchedule(Vector(500f,250f,0f),Vector(1.0,.5,.1,0.0),1.0)
    val request=DiffusionImageRequest("a red apple","blur",8,8,42)
    val shape=Shape.from(Seq(1L,4L,4L,4L))
    val noise=Tensor.fromFloats(Array.tabulate(shape.numel.toInt)(i=>math.sin(i*.1).toFloat),shape)
    val firstImage=cpu.generateAncestral(request,schedule,Some(noise))
    assertEquals(cpu.generateAncestral(request,schedule,Some(noise)).toFloatArray.toVector,firstImage.toFloatArray.toVector)
    assertNotEquals(cpu.generateAncestral(request.copy(seed=43),schedule,Some(noise)).toFloatArray.toVector,
      firstImage.toFloatArray.toVector)
    val source=DiffusionImageSource(firstImage,.7,Some(noise))
    assertEquals(cpu.generateAncestral(request,schedule,Some(noise),source=Some(source)).shape,firstImage.shape)
    val latents=cpu.generateAncestralLatents(request,schedule,Some(noise))
    assertEquals(latents.shape,shape)
    assertEquals(cpu.generateAncestral(request,schedule,Some(noise),latentSource=Some(DiffusionLatentSource(latents,.5))).shape,firstImage.shape)
    if sys.env.get("SCALATORCH_CUDA_TESTS").contains("1") then
      val gpu=new EulerImagePipeline(first,clip,unet,vae,
        vaeExecution=VaeExecution.Cuda(maxBytes=16777216,offloadWeights=true),
        unetExecution=UNetExecution.Cuda(maxBytes=16777216,offloadWeights=true),sdxlConditioning=Some(text))
      val actual=gpu.generateAncestral(request,schedule,Some(noise)).toFloatArray
      val error=actual.zip(firstImage.toFloatArray).map((a,b)=>math.abs(a.toDouble-b)).max
      assert(error<1e-4,s"Euler a CUDA max error $error")
    var cancelled=false
    intercept[java.util.concurrent.CancellationException](cpu.generateAncestral(request,schedule,Some(noise),
      onProgress=_=>cancelled=true,cancelled=()=>cancelled))
  }

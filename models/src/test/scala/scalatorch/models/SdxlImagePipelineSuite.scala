package scalatorch.models

import scalatorch.*
import java.nio.{ByteBuffer,ByteOrder}
import java.nio.file.Path
import scalatorch.models.worker.NativeJson
import NativeJson.*

class SdxlImagePipelineSuite extends munit.FunSuite:
  private def reference(name: String,shape: Shape): Tensor[Float] =
    val stream=getClass.getResourceAsStream(s"/tiny-sdxl-pipeline/$name.bin")
    val bytes=try ByteBuffer.wrap(stream.readAllBytes()).order(ByteOrder.LITTLE_ENDIAN) finally stream.close()
    Tensor.fromFloats(Array.fill(bytes.remaining()/4)(bytes.getFloat()),shape)
  test("SDXL rescale CFG matches Diffusers latents for text and image input") {
    val configured=sys.props.get("scalatorch.sdxl.dir")
    assume(configured.nonEmpty,"tiny SDXL required")
    val root=Path.of(configured.get)
    def tok(part: String)=ClipFastTokenizer.load(root.resolve(s"$part/vocab.json"),root.resolve(s"$part/merges.txt"))
    val first=tok("tokenizer");val clip=PretrainedClip.load(root.resolve("text_encoder"))
    val text=new SdxlTextConditioning(first,clip,tok("tokenizer_2"),PretrainedClip.load(root.resolve("text_encoder_2")))
    val pipe=new EulerImagePipeline(first,clip,PretrainedUNet.load(root.resolve("unet")),PretrainedVae.load(root.resolve("vae")),sdxlConditioning=Some(text))
    val shape=Shape.from(Seq(1L,4L,4L,4L));val imageShape=Shape.from(Seq(1L,3L,8L,8L))
    val noise=reference("noise",shape)
    val source=DiffusionImageSource(reference("source",imageShape),.5,Some(reference("posterior-noise",shape)))
    val schedule=EulerSchedule(Vector(500f,250f),Vector(1.0,.5,0.0),1.0)
    val stream=getClass.getResourceAsStream("/sdxl-rescale-reference.json")
    val refs=try NativeJson.parse(new String(stream.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8)).asInstanceOf[Obj].fields finally stream.close()
    for img <- Seq(false,true);dpm <- Seq(false,true) do
      val request=DiffusionImageRequest("a red apple","blur",8,8,42,guidanceScale=7.5f,guidanceRescale=.7f,
        sdxlSize=Option.when(img)(SdxlImageSize(12,16,1,2)))
      var last: Tensor[Float]=null
      val sourceOpt=Option.when(img)(source)
      if dpm then pipe.generateDpm(request,DpmSchedule(schedule.timesteps,schedule.sigmas),Some(noise),onProgress=p=>last=p.latents,source=sourceOpt)
      else pipe.generate(request,schedule,Some(noise),onProgress=p=>last=p.latents,source=sourceOpt)
      val key=(if dpm then "dpm_" else "euler_")+(if img then "img2img" else "txt2img")
      val expected=refs(key).asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat)
      val error=last.toFloatArray.zip(expected).map((a,b)=>math.abs(a.toDouble-b)).max
      println(s"SDXL $key rescale max error $error")
      assert(error<0.0001,s"$key $error")
    if sys.env.get("SCALATORCH_CUDA_TESTS").contains("1") then
      val gpu=new EulerImagePipeline(first,clip,pipe.unet,pipe.vae,
        vaeExecution=VaeExecution.Cuda(maxBytes=16777216,offloadWeights=true),
        unetExecution=UNetExecution.Cuda(maxBytes=16777216,offloadWeights=true),sdxlConditioning=Some(text))
      val request=DiffusionImageRequest("a red apple","blur",8,8,42,guidanceScale=7.5f,resharpen=Some(Resharpen(1.5,"Cos")),
        colorGrading=Some(LatentColorGrading(true,.8,.4,-.3,.7,true,.6)))
      val longer=EulerSchedule(Vector(500f,375f,250f),Vector(1.0,.75,.5,0.0),1.0)
      val cpu=pipe.generate(request,longer,Some(noise)).toFloatArray
      val cuda=gpu.generate(request,longer,Some(noise)).toFloatArray
      val error=cpu.zip(cuda).map((a,b)=>math.abs(a.toDouble-b)).max
      assert(error<.0001,s"SDXL CUDA resharpen error $error")
  }
  test("native SDXL Euler and DPM images match Diffusers with CFG, size IDs and image input") {
    val directory=sys.props.get("scalatorch.sdxl.dir")
    assume(directory.nonEmpty,"tiny SDXL required")
    val root=Path.of(directory.get)
    def tok(part: String)=ClipFastTokenizer.load(root.resolve(s"$part/vocab.json"),root.resolve(s"$part/merges.txt"))
    val first=tok("tokenizer");val clip=PretrainedClip.load(root.resolve("text_encoder"))
    val text=new SdxlTextConditioning(first,clip,tok("tokenizer_2"),PretrainedClip.load(root.resolve("text_encoder_2")))
    val pipeline=new EulerImagePipeline(first,clip,PretrainedUNet.load(root.resolve("unet")),PretrainedVae.load(root.resolve("vae")),sdxlConditioning=Some(text))
    val shape=Shape.from(Seq(1L,4L,4L,4L));val imageShape=Shape.from(Seq(1L,3L,8L,8L))
    val noise=reference("noise",shape)
    val source=DiffusionImageSource(reference("source",imageShape),0.5,Some(reference("posterior-noise",shape)))
    val schedule=EulerSchedule(Vector(500f,250f),Vector(1.0,0.5,0.0),1.0)
    for dpm <- Seq(false,true); name <- Seq("plain","empty","unguided","img2img") do
      val request=DiffusionImageRequest("a red apple",if name=="empty" then "" else "blur",8,8,42,
        guidanceScale=if name=="unguided" then 1f else 7.5f,
        sdxlSize=if Set("unguided","img2img").contains(name) then Some(SdxlImageSize(12,16,1,2)) else None)
      var finalLatent: Tensor[Float]=null
      def run(p: EulerImagePipeline): Tensor[Float] =
        val input=if name=="img2img" then Some(source) else None
        if dpm then p.generateDpm(request,DpmSchedule(schedule.timesteps,schedule.sigmas),Some(noise),
          onProgress=p => finalLatent=p.latents,source=input)
        else p.generate(request,schedule,Some(noise),onProgress=p => finalLatent=p.latents,source=input)
      val output=run(pipeline)
      for (actual,label) <- Seq(output->"image",finalLatent->"latent") do
        val expected=reference(s"${if dpm then "dpm-" else ""}$name-$label",actual.shape)
        val error=actual.toFloatArray.zip(expected.toFloatArray).map((a,b) => math.abs(a.toDouble-b)).max
        println(s"SDXL dpm=$dpm $name $label maxAbsError=$error")
        assert(error<5e-5)
      if sys.env.get("SCALATORCH_CUDA_TESTS").contains("1") then
        val gpu=new EulerImagePipeline(first,clip,pipeline.unet,pipeline.vae,
          vaeExecution=VaeExecution.Cuda(maxBytes=16777216,offloadWeights=true),
          unetExecution=UNetExecution.Cuda(maxBytes=16777216,offloadWeights=true),sdxlConditioning=Some(text))
        val error=run(gpu).toFloatArray.zip(output.toFloatArray).map((a,b) => math.abs(a.toDouble-b)).max
        println(s"SDXL CUDA offload dpm=$dpm $name maxAbsError=$error")
        assert(error<5e-5)
    intercept[java.util.concurrent.CancellationException](pipeline.generate(
      DiffusionImageRequest("apple","",8,8,42),schedule,cancelled=() => true))
    intercept[IllegalArgumentException](SdxlImageSize(0,8))
    intercept[IllegalArgumentException](SdxlImageSize(8,8,-1,0))
    val replay=DiffusionImageRequest("apple","",8,8,123,guidanceScale=1f)
    val dpmSchedule=DpmSchedule(schedule.timesteps,schedule.sigmas)
    val expected=pipeline.generateDpm(replay,dpmSchedule).toFloatArray.toVector
    var cancel=false
    intercept[java.util.concurrent.CancellationException](pipeline.generateDpm(replay,dpmSchedule,
      onProgress=_ => cancel=true,cancelled=() => cancel))
    assertEquals(pipeline.generateDpm(replay,dpmSchedule).toFloatArray.toVector,expected)
    var textChecks=0
    intercept[java.util.concurrent.CancellationException](text.encode("apple BREAK tree","",cancelled=() => {
      textChecks+=1;textChecks>=4
    }))
  }

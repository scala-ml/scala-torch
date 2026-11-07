package scalatorch.models

import scalatorch.{Shape,Tensor}
import scalatorch.models.worker.NativeJson
import NativeJson.*
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.Path

class TorchCpuNoisePipelineSuite extends munit.FunSuite:
  test("tiny SD Euler CPU latent agrees with independent PyTorch and Diffusers") {
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty,"public tiny model required")
    val root=Path.of(configured.get).getParent
    val stream=getClass.getResourceAsStream("/torch-cpu-tiny-euler-reference.json")
    val fixture=try NativeJson.parse(new String(stream.readAllBytes(),UTF_8)).asInstanceOf[Obj].fields finally stream.close()
    val expected=fixture("latent").asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat).toArray
    val pipe=new EulerImagePipeline(ClipFastTokenizer.load(root.resolve("tokenizer/vocab.json"),root.resolve("tokenizer/merges.txt")),
      PretrainedClip.load(root.resolve("text_encoder")),PretrainedUNet.load(root.resolve("unet")),PretrainedVae.load(root.resolve("vae")))
    val schedule=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true).euler(3)
    val actual=pipe.generateLatents(DiffusionImageRequest("a red cat","",16,16,42,guidanceScale=1f,torchCpuNoise=true),schedule).toFloatArray
    assertEquals(actual.length,expected.length)
    val error=actual.zip(expected).map((a,b)=>math.abs(a.toDouble-b))
    val max=error.max
    println(s"tiny SD CPU Euler versus PyTorch: maximum latent error $max, mean ${error.sum/error.length}")
    assert(max<0.001,s"tiny SD Euler latent error $max")
    val expectedImage=fixture("image").asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat).toArray
    val decoded=pipe.generate(DiffusionImageRequest("a red cat","",16,16,42,guidanceScale=1f,torchCpuNoise=true),schedule)
      .toFloatArray.map(value => math.max(0f,math.min(1f,value/2f+0.5f)))
    assertEquals(decoded.length,expectedImage.length)
    val imageError=decoded.zip(expectedImage).map((a,b)=>math.abs(a.toDouble-b))
    println(s"tiny SD CPU Euler decoded image: maximum error ${imageError.max}, mean ${imageError.sum/imageError.length}")
    assert(imageError.max<0.001,s"tiny SD decoded image error ${imageError.max}")
    val expectedVariation=fixture("variation_latent").asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat).toArray
    val variation=DiffusionImageRequest("a red cat","",16,16,42,guidanceScale=7.5f,
      variationSeed=Some(1337),variationStrength=0.45,torchCpuNoise=true)
    val varied=pipe.generateLatents(variation,schedule).toFloatArray
    assertEquals(varied.length,expectedVariation.length)
    val variationError=varied.zip(expectedVariation).map((a,b)=>math.abs(a.toDouble-b))
    println(s"tiny SD CPU Euler variation: maximum latent error ${variationError.max}, mean ${variationError.sum/variationError.length}")
    assert(variationError.max<0.001,s"tiny SD variation error ${variationError.max}")
    def compareReference(key: String,actual: Tensor[Float]): Unit =
      val reference=fixture(key).asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat).toArray
      assertEquals(actual.toFloatArray.length,reference.length)
      val differences=actual.toFloatArray.zip(reference).map((a,b)=>math.abs(a.toDouble-b))
      println(s"tiny SD CPU $key: maximum latent error ${differences.max}, mean ${differences.sum/differences.length}")
      assert(differences.max<0.001,s"$key latent error ${differences.max}")
    val request=DiffusionImageRequest("a red cat","",16,16,42,guidanceScale=1f,torchCpuNoise=true)
    val layout=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
    compareReference("ancestral_latent",pipe.generateAncestralLatents(request,layout.euler(3)))
    compareReference("dpm_latent",pipe.generateDpmLatents(request,DpmSchedule.uniform(layout,3)))
    val input=Tensor.fromFloats(Array.tabulate(3*16*16)(i => (i%256).toFloat/127.5f-1f),
      Shape.from(Seq(1L,3L,16L,16L)))
    val source=Some(DiffusionImageSource(input,0.75))
    compareReference("img2img_euler",pipe.generateLatents(request,layout.euler(4),source=source))
    compareReference("img2img_ancestral",pipe.generateAncestralLatents(request,layout.euler(4),source=source))
    compareReference("img2img_dpm",pipe.generateDpmLatents(request,DpmSchedule.uniform(layout,4),source=source))
    val imgReference=fixture("img2img_image").asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat).toArray
    val imgResult=pipe.generate(request,layout.euler(4),source=source).toFloatArray.map(v => math.max(0f,math.min(1f,v/2f+0.5f)))
    assertEquals(imgResult.length,imgReference.length)
    val imgError=imgResult.zip(imgReference).map((a,b)=>math.abs(a.toDouble-b))
    println(s"tiny SD CPU img2img image: maximum error ${imgError.max}, mean ${imgError.sum/imgError.length}")
    assert(imgError.max<0.001,s"img2img decoded image error ${imgError.max}")
  }
  test("CPU seed drives the same initial latent as PyTorch in Euler, DPM and Euler ancestral") {
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty,"public tiny model required")
    val root=Path.of(configured.get).getParent
    val stream=getClass.getResourceAsStream("/torch-cpu-normal-reference.json")
    val data=try NativeJson.parse(new String(stream.readAllBytes(),UTF_8)).asInstanceOf[Obj].fields finally stream.close()
    val reference=data("cases").asInstanceOf[Arr].values.map(_.asInstanceOf[Obj].fields).find { row =>
      row("seed")==Str("42") && row("sizes")==Arr(Vector(Number(4096)))
    }.get("values").asInstanceOf[Arr].values.head.asInstanceOf[Arr].values.take(256)
      .map(_.asInstanceOf[Number].value.toFloat).toArray
    val noise=Tensor.fromFloats(reference,Shape.from(Seq(1L,4L,8L,8L)))
    val pipe=new EulerImagePipeline(ClipFastTokenizer.load(root.resolve("tokenizer/vocab.json"),root.resolve("tokenizer/merges.txt")),
      PretrainedClip.load(root.resolve("text_encoder")),PretrainedUNet.load(root.resolve("unet")),PretrainedVae.load(root.resolve("vae")))
    val request=DiffusionImageRequest("a red cat","",16,16,42,torchCpuNoise=true)
    val layout=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
    def compare(actual: Tensor[Float],expected: Tensor[Float],sampler: String): Unit =
      assertEquals(actual.shape,expected.shape)
      val max=actual.toFloatArray.zip(expected.toFloatArray).map((a,b)=>math.abs(a.toDouble-b)).max
      assert(max<0.0001,s"$sampler latent maximum error $max")
    val euler=layout.nuiUniformEuler(3)
    compare(pipe.generateLatents(request,euler),pipe.generateLatents(request,euler,noise=Some(noise)),"Euler")
    val dpm=DpmSchedule.uniform(layout,3)
    compare(pipe.generateDpmLatents(request,dpm),pipe.generateDpmLatents(request,dpm,noise=Some(noise)),"DPM")
    val ancestral=layout.euler(3,DiffusionTimestepSpacing.Linspace,0)
    compare(pipe.generateAncestralLatents(request,ancestral),
      pipe.generateAncestralLatents(request,ancestral,noise=Some(noise)),"Euler ancestral")
  }

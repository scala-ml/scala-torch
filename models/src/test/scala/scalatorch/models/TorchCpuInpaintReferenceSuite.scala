package scalatorch.models

import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files,Path}
import scalatorch.{Shape,Tensor}
import scalatorch.models.worker.{NativeDiffusionBackendSuite,NativeJson}
import NativeJson.*

class TorchCpuInpaintReferenceSuite extends munit.FunSuite:
  test("nine-channel latent upscale without a mask matches PyTorch") {
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty,"public tiny model required")
    val originalRoot=Path.of(configured.get).getParent
    val root=Files.createTempDirectory("unmasked-inpaint-upscale")
    try
      new NativeDiffusionBackendSuite().makeNineChannelModel(originalRoot,root)
      val stream=getClass.getResourceAsStream("/unmasked-inpaint-latent-upscale-reference.json")
      val reference=try NativeJson.parse(new String(stream.readAllBytes(),UTF_8)).asInstanceOf[Obj].fields finally stream.close()
      val pipe=new EulerImagePipeline(ClipFastTokenizer.load(root.resolve("tokenizer/vocab.json"),root.resolve("tokenizer/merges.txt")),
        PretrainedClip.load(root.resolve("text_encoder")),PretrainedUNet.load(root.resolve("unet")),PretrainedVae.load(root.resolve("vae")))
      val input=Tensor.fromFloats(Array.tabulate(3*16*16)(i => (i%256).toFloat/127.5f-1f),Shape.from(Seq(1L,3L,16L,16L)))
      def resize(input: Tensor[Float]): Tensor[Float] =
        val source=input.toFloatArray
        Tensor.fromFloats(Array.tabulate(4*12*12) { i =>
          val c=i/144;val y=i%144/12;val x=i%12
          source(c*64+((2*y+1)*8/24)*8+(2*x+1)*8/24)
        },Shape.from(Seq(1L,4L,12L,12L)))
      val request=DiffusionImageRequest("a red cat","",24,24,42,guidanceScale=1f,torchCpuNoise=true)
      val noise=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
      for strength <- Vector(.75,1.0); sampler <- Vector("Euler","Euler a","DPM++ 2M") do
        val src=Some(DiffusionImageSource(input,strength,latentTransform=Some(resize)))
        val actual=(sampler match
          case "Euler" => pipe.generateLatents(request,noise.normalEuler(4),source=src)
          case "Euler a" => pipe.generateAncestralLatents(request,noise.euler(4),source=src)
          case _ => pipe.generateDpmLatents(request,DpmSchedule.uniform(noise,4),source=src)).toFloatArray
        val expected=reference(s"$sampler:$strength").asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat)
        val max=actual.zip(expected).map((a,b)=>math.abs(a.toDouble-b)).max
        assert(max<.001,s"$sampler at $strength max latent error $max")
    finally
      val paths=Files.walk(root)
      try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }
  for channels <- Vector(4,9); strength <- Vector(.75,1.0) do test(s"$channels-channel masked source-size latent upscale at $strength matches PyTorch sampling sequence") {
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty,"public tiny model required")
    val originalRoot=Path.of(configured.get).getParent
    val temp=if channels==9 then Some(Files.createTempDirectory("masked-upscale-model")) else None
    try {
    temp.foreach(path => new NativeDiffusionBackendSuite().makeNineChannelModel(originalRoot,path))
    val root=temp.getOrElse(originalRoot)
    val fixture=s"/masked-latent-upscale${if channels==9 then "-9" else ""}${if strength==1.0 then "-full" else ""}-reference.json"
    val stream=getClass.getResourceAsStream(fixture)
    val reference=try NativeJson.parse(new String(stream.readAllBytes(),UTF_8)).asInstanceOf[Obj].fields finally stream.close()
    val pipe=new EulerImagePipeline(ClipFastTokenizer.load(root.resolve("tokenizer/vocab.json"),root.resolve("tokenizer/merges.txt")),
      PretrainedClip.load(root.resolve("text_encoder")),PretrainedUNet.load(root.resolve("unet")),PretrainedVae.load(root.resolve("vae")))
    val values=Array.tabulate(3*16*16)(i => (i%256).toFloat/127.5f-1f)
    val input=Tensor.fromFloats(values,Shape.from(Seq(1L,3L,16L,16L)))
    val masked=Tensor.fromFloats(values.indices.map(i => if i%16>=8 then 0f else values(i)).toArray,input.shape)
    val mask=Tensor.fromFloats(Array.tabulate(24*24)(i=>if i%24>=12 then 1f else 0f),Shape.from(Seq(1L,1L,24L,24L)))
    val original=Tensor.fromFloats(new Array[Float](3*24*24),Shape.from(Seq(1L,3L,24L,24L)))
    def resize(input: Tensor[Float]): Tensor[Float] =
      val source=input.toFloatArray
      Tensor.fromFloats(Array.tabulate(4*12*12) { i =>
        val c=i/144;val y=i%144/12;val x=i%12
        source(c*64+((2*y+1)*8/24)*8+(2*x+1)*8/24)
      },Shape.from(Seq(1L,4L,12L,12L)))
    val src=Some(DiffusionImageSource(input,strength,mask=Some(mask),originalImage=Some(original),finalOverlay=false,
      maskedImage=Some(masked),latentTransform=Some(resize)))
    val request=DiffusionImageRequest("a red cat","",24,24,42,guidanceScale=1f,torchCpuNoise=true)
    val noise=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
    for sampler <- Vector("Euler","Euler a","DPM++ 2M") do
      val actual=(sampler match
        case "Euler" => pipe.generateLatents(request,noise.normalEuler(4),source=src)
        case "Euler a" => pipe.generateAncestralLatents(request,noise.euler(4),source=src)
        case _ => pipe.generateDpmLatents(request,DpmSchedule.uniform(noise,4),source=src)).toFloatArray
      val expected=reference(sampler).asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat)
      val max=actual.zip(expected).map((a,b)=>math.abs(a.toDouble-b)).max
      assert(max<.001,s"$sampler max latent error $max")
    } finally temp.foreach { path =>
      val paths=Files.walk(path)
      try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
    }
  }
  test("source-sized VAE sampling and nearest-exact latent upscale match PyTorch") {
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty,"public tiny model required")
    val root=Path.of(configured.get).getParent
    val stream=getClass.getResourceAsStream("/source-latent-upscale-reference.json")
    val reference=try NativeJson.parse(new String(stream.readAllBytes(),UTF_8)).asInstanceOf[Obj].fields finally stream.close()
    val pipe=new EulerImagePipeline(ClipFastTokenizer.load(root.resolve("tokenizer/vocab.json"),root.resolve("tokenizer/merges.txt")),
      PretrainedClip.load(root.resolve("text_encoder")),PretrainedUNet.load(root.resolve("unet")),PretrainedVae.load(root.resolve("vae")))
    val input=Tensor.fromFloats(Array.tabulate(3*16*16)(i => (i%256).toFloat/127.5f-1f),Shape.from(Seq(1L,3L,16L,16L)))
    def resize(input: Tensor[Float]): Tensor[Float] =
      val source=input.toFloatArray
      val out=Array.tabulate(4*12*12) { i =>
        val c=i/144;val y=(i%144)/12;val x=i%12
        source(c*64+((2*y+1)*8/24)*8+(2*x+1)*8/24)
      }
      Tensor.fromFloats(out,Shape.from(Seq(1L,4L,12L,12L)))
    val src=Some(DiffusionImageSource(input,.75,latentTransform=Some(resize)))
    val request=DiffusionImageRequest("a red cat","",24,24,42,guidanceScale=1f,torchCpuNoise=true)
    val noise=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
    for sampler <- Vector("Euler","Euler a","DPM++ 2M") do
      val actual=(sampler match
        case "Euler" => pipe.generateLatents(request,noise.normalEuler(4),source=src)
        case "Euler a" => pipe.generateAncestralLatents(request,noise.euler(4),source=src)
        case _ => pipe.generateDpmLatents(request,DpmSchedule.uniform(noise,4),source=src)).toFloatArray
      val expected=reference(sampler).asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat)
      assertEquals(actual.length,expected.length)
      val max=actual.zip(expected).map((a,b)=>math.abs(a.toDouble-b)).max
      assert(max<.001,s"$sampler max latent error $max")
  }
  test("source latent transform preserves seeded img2img results when it is the identity") {
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty,"public tiny model required")
    val root=Path.of(configured.get).getParent
    val pipe=new EulerImagePipeline(ClipFastTokenizer.load(root.resolve("tokenizer/vocab.json"),root.resolve("tokenizer/merges.txt")),
      PretrainedClip.load(root.resolve("text_encoder")),PretrainedUNet.load(root.resolve("unet")),PretrainedVae.load(root.resolve("vae")))
    val input=Tensor.fromFloats(Array.tabulate(3*16*16)(i => (i%256).toFloat/127.5f-1f),Shape.from(Seq(1L,3L,16L,16L)))
    val request=DiffusionImageRequest("a red cat","",16,16,42,guidanceScale=1f,torchCpuNoise=true)
    val schedule=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true).euler(4)
    val original=pipe.generateLatents(request,schedule,source=Some(DiffusionImageSource(input,.75))).toFloatArray
    val transformed=pipe.generateLatents(request,schedule,source=Some(DiffusionImageSource(input,.75,latentTransform=Some(identity)))).toFloatArray
    assertEquals(transformed.toVector,original.toVector)
  }
  for channels <- Vector(4,9); sampler <- Vector("euler","ancestral","dpm"); percent <- Vector(75,100,-1)
      if percent>=0 || sampler=="euler" do
    test(s"$channels-channel $sampler inpainting at ${if percent<0 then "soft mask" else s"$percent%"} matches PyTorch CPU") {
      val configured=sys.props.get("scalatorch.clip.dir")
      assume(configured.nonEmpty,"public tiny model required")
      val original=Path.of(configured.get).getParent
      val temp=if channels==9 then Some(Files.createTempDirectory("inpaint-reference-model")) else None
      try
        temp.foreach(path => new NativeDiffusionBackendSuite().makeNineChannelModel(original,path))
        val root=temp.getOrElse(original)
        val stream=getClass.getResourceAsStream("/torch-cpu-tiny-euler-reference.json")
        val fixture=try NativeJson.parse(new String(stream.readAllBytes(),UTF_8)).asInstanceOf[Obj].fields finally stream.close()
        val key=if percent<0 then s"inpaint_${channels}_soft" else s"inpaint_${channels}_${sampler}_$percent"
        val reference=fixture(key).asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat).toArray
        val pipe=new EulerImagePipeline(ClipFastTokenizer.load(root.resolve("tokenizer/vocab.json"),root.resolve("tokenizer/merges.txt")),
          PretrainedClip.load(root.resolve("text_encoder")),PretrainedUNet.load(root.resolve("unet")),PretrainedVae.load(root.resolve("vae")))
        val inputValues=Array.tabulate(3*16*16)(i => (i%256).toFloat/127.5f-1f)
        val input=Tensor.fromFloats(inputValues,Shape.from(Seq(1L,3L,16L,16L)))
        val maskValues=if percent<0 then fixture("soft_mask").asInstanceOf[Arr].values
          .map(_.asInstanceOf[Number].value.toFloat/255f).toArray
        else Array.tabulate(16*16)(i => if i%16>=8 then 1f else 0f)
        val mask=Tensor.fromFloats(maskValues,Shape.from(Seq(1L,1L,16L,16L)))
        val masked=Option.when(channels==9)(Tensor.fromFloats(inputValues.indices.map { i =>
          if maskValues(i%256)>=0.5f then 0f else inputValues(i)
        }.toArray,input.shape))
        val strength=if percent<0 then 0.75 else percent/100.0
        val source=Some(DiffusionImageSource(input,strength,mask=Some(mask),originalImage=Some(input),
          finalOverlay=false,maskedImage=masked))
        val request=DiffusionImageRequest("a red cat","",16,16,42,guidanceScale=1f,torchCpuNoise=true)
        val noise=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
        val actual=sampler match
          case "euler" => pipe.generateLatents(request,noise.euler(4),source=source)
          case "ancestral" => pipe.generateAncestralLatents(request,noise.euler(4),source=source)
          case _ => pipe.generateDpmLatents(request,DpmSchedule.uniform(noise,4),source=source)
        val values=actual.toFloatArray
        assertEquals(values.length,reference.length)
        val error=values.zip(reference).map((a,b)=>math.abs(a.toDouble-b))
        println(s"$key max error ${error.max}, mean ${error.sum/error.length}")
        assert(error.max<0.001,s"$key maximum latent error ${error.max}")
      finally
        temp.foreach { root =>
          val paths=Files.walk(root)
          try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
        }
    }

package scalatorch.models

import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files,Path}
import scalatorch.{Shape,Tensor}
import scalatorch.models.worker.{NativeJson,NativeDiffusionBackendSuite}
import NativeJson.*

class ResharpenSuite extends munit.FunSuite:
  test("resharpen matches masked 4 and 9 channel inpainting") {
    val configured=sys.props.get("scalatorch.clip.dir");assume(configured.nonEmpty,"public tiny model required")
    val original=Path.of(configured.get).getParent
    val temp=Files.createTempDirectory("resharpen-nine-channel")
    try
      new NativeDiffusionBackendSuite().makeNineChannelModel(original,temp)
      val stream=getClass.getResourceAsStream("/resharpen-reference.json")
      val refs=try NativeJson.parse(new String(stream.readAllBytes(),UTF_8)).asInstanceOf[Obj].fields finally stream.close()
      val values=Array.tabulate(3*16*16)(i=>(i%256).toFloat/127.5f-1f)
      val input=Tensor.fromFloats(values,Shape.from(Seq(1L,3L,16L,16L)))
      val mask=Tensor.fromFloats(Array.tabulate(16*16)(i=>if i%16>=8 then 1f else 0f),Shape.from(Seq(1L,1L,16L,16L)))
      val layout=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
      for channels <- Vector(4,9) do
        val root=if channels==9 then temp else original
        val pipe=new EulerImagePipeline(ClipFastTokenizer.load(root.resolve("tokenizer/vocab.json"),root.resolve("tokenizer/merges.txt")),
          PretrainedClip.load(root.resolve("text_encoder")),PretrainedUNet.load(root.resolve("unet")),PretrainedVae.load(root.resolve("vae")))
        for strength <- Vector(.75,1.0);sampler <- Vector("euler","ancestral","dpm") do
          val masked=Option.when(channels==9)(Tensor.fromFloats(values.indices.map(i=>if i%16>=8 then 0f else values(i)).toArray,input.shape))
          val source=Some(DiffusionImageSource(input,strength,mask=Some(mask),originalImage=Some(input),finalOverlay=false,maskedImage=masked))
          val request=DiffusionImageRequest("a red cat","",16,16,42,guidanceScale=7.5f,torchCpuNoise=true,resharpen=Some(Resharpen(1.5,"Cos")))
          val actual=sampler match
            case "euler" => pipe.generateLatents(request,layout.euler(4),source=source)
            case "ancestral" => pipe.generateAncestralLatents(request,layout.euler(4),source=source)
            case _ => pipe.generateDpmLatents(request,DpmSchedule.uniform(layout,4),source=source)
          val key=s"inpaint_${channels}_${sampler}_${(strength*100).toInt}"
          val expected=refs(key).asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat)
          val error=actual.toFloatArray.zip(expected).map((a,b)=>math.abs(a.toDouble-b)).max
          println(s"$key resharpen max error $error")
          assert(error<.001,s"$key: $error")
    finally
      val paths=Files.walk(temp)
      try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }
  test("resharpen matches NUI's step callback across image modes and samplers") {
    val configured=sys.props.get("scalatorch.clip.dir");assume(configured.nonEmpty,"public tiny model required")
    val root=Path.of(configured.get).getParent
    val stream=getClass.getResourceAsStream("/resharpen-reference.json")
    val refs=try NativeJson.parse(new String(stream.readAllBytes(),UTF_8)).asInstanceOf[Obj].fields finally stream.close()
    val pipe=new EulerImagePipeline(ClipFastTokenizer.load(root.resolve("tokenizer/vocab.json"),root.resolve("tokenizer/merges.txt")),
      PretrainedClip.load(root.resolve("text_encoder")),PretrainedUNet.load(root.resolve("unet")),PretrainedVae.load(root.resolve("vae")))
    val input=Tensor.fromFloats(Array.tabulate(3*16*16)(i=>(i%256).toFloat/127.5f-1f),Shape.from(Seq(1L,3L,16L,16L)))
    val layout=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
    for mode <- Vector("txt2img","img2img");sampler <- Vector("euler","ancestral","dpm");scale <- Vector("Flat","Cos","Sin","1 - Cos","1 - Sin") do
      val request=DiffusionImageRequest("a red cat","",16,16,42,guidanceScale=7.5f,torchCpuNoise=true,resharpen=Some(Resharpen(1.5,scale)))
      val source=Option.when(mode=="img2img")(DiffusionImageSource(input,.75))
      val actual=sampler match
        case "euler" => pipe.generateLatents(request,layout.euler(4),source=source)
        case "ancestral" => pipe.generateAncestralLatents(request,layout.euler(4),source=source)
        case _ => pipe.generateDpmLatents(request,DpmSchedule.uniform(layout,4),source=source)
      val key=s"${mode}_${sampler}_${scale}"
      val expected=refs(key).asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat)
      val error=actual.toFloatArray.zip(expected).map((a,b)=>math.abs(a.toDouble-b)).max
      println(s"$key resharpen max error $error")
      assert(error<.001,s"$key: $error")
  }
  test("first and last steps preserve input and scaling is validated") {
    val shape=Shape.from(Seq(1L,1L,1L,2L));val first=Tensor.fromFloats(Array(1f,2f),shape)
    val second=Tensor.fromFloats(Array(2f,4f),shape);val config=Resharpen(1,"Flat")
    assertEquals(config.step(first,None,1,3).toFloatArray.toVector,Vector(1f,2f))
    assertEquals(config.step(second,Some(first),3,3).toFloatArray.toVector,Vector(2f,4f))
    intercept[IllegalArgumentException](Resharpen(1,"bad"))
  }

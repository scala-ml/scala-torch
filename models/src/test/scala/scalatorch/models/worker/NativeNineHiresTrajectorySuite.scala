package scalatorch.models.worker

import java.nio.{ByteBuffer,ByteOrder}
import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import scalatorch.models.*

class NativeNineHiresTrajectorySuite extends munit.FunSuite:
  override val munitTimeout: scala.concurrent.duration.Duration=scala.concurrent.duration.Duration(180,"seconds")
  test("nine-channel second-pass latents match contiguous Diffusers input") {
    val configured=sys.props.get("scalatorch.nine.dir")
    val reference=sys.props.get("scalatorch.nine.reference.dir")
    assume(configured.nonEmpty,"reference tiny model required")
    val root=Path.of(configured.get)
    val tokenizer=ClipFastTokenizer.load(root.resolve("tokenizer/vocab.json"),root.resolve("tokenizer/merges.txt"))
      .withNuiFallbackNormalization
    val clip=PretrainedClip.load(root.resolve("text_encoder"))
    val unet=PretrainedUNet.load(root.resolve("unet"))
    val vae=PretrainedVae.load(root.resolve("vae"))
    val cuda=sys.env.get("SCALATORCH_CUDA_TESTS").contains("1")
    val pipeline=if cuda then new EulerImagePipeline(tokenizer,clip,unet,vae,
      vaeExecution=VaeExecution.Cuda(offloadWeights=true),unetExecution=UNetExecution.Cuda(offloadWeights=true))
    else new EulerImagePipeline(tokenizer,clip,unet,vae)
    def referenceBytes(external: String,resource: String): Array[Byte]=reference match
      case Some(dir) => Files.readAllBytes(Path.of(dir).resolve(external))
      case None =>
        val input=getClass.getResourceAsStream(resource)
        try input.readAllBytes() finally input.close()
    val image=ImageIO.read(new java.io.ByteArrayInputStream(referenceBytes("nine-contiguous-source.png","/nine-contiguous-source.png")))
    val tensor=NativeImg2ImgHires.tensor(image,()=>false)
    val mask=scalatorch.Tensor.fromFloats(Array.fill(128*128)(1f),scalatorch.Shape.from(Seq(1L,1L,128L,128L)))
    val masked=scalatorch.Tensor.fromFloats(new Array[Float](3*128*128),tensor.shape)
    val source=DiffusionImageSource(tensor,.5,mask=Some(mask),originalImage=Some(tensor),maskedImage=Some(masked),finalOverlay=false)
    val request=DiffusionImageRequest("a red cat","",128,128,42,guidanceScale=7.5f,torchCpuNoise=true)
    val schedule=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true).euler(4)
    val actual=scala.collection.mutable.ArrayBuffer.empty[Array[Float]]
    val decoded=pipeline.generate(request,schedule,source=Some(source),onProgress=p=>actual+=p.latents.toFloatArray)
    val actualImage=DiffusionImageIO.toBufferedImage(decoded)
    val expectedImage=ImageIO.read(new java.io.ByteArrayInputStream(referenceBytes("nine-contiguous-output.png","/nine-contiguous-output.png")))
    val pixelError=for y<-0 until 128; x<-0 until 128; shift<-Vector(0,8,16) yield
      math.abs(((actualImage.getRGB(x,y)>>>shift)&255)-((expectedImage.getRGB(x,y)>>>shift)&255))
    println(s"nine-channel direct decoded ${if cuda then "CUDA" else "CPU"} max=${pixelError.max} mean=${pixelError.sum.toDouble/pixelError.size}")
    assert(pixelError.max<=1 && pixelError.sum.toDouble/pixelError.size<0.01)
    val bytes=referenceBytes("nine-contiguous-latents.f32","/nine-contiguous-latents.f32")
    val buffer=ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    assertEquals(actual.size,2)
    actual.zipWithIndex.foreach { (values,index) =>
      val expected=Array.fill(values.length)(buffer.getFloat())
      val error=values.indices.map(i=>math.abs(values(i).toDouble-expected(i))).max
      println(s"nine-channel second-pass ${if cuda then "CUDA" else "CPU"} step=${index+1} max-latent-error=$error")
      assert(error<0.00005,s"step=$index max-latent-error=$error")
    }
    assertEquals(buffer.remaining(),0)
  }

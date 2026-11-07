package scalatorch.models.worker

import java.awt.image.BufferedImage
import java.nio.file.{Files,Path}
import scalatorch.*
import scalatorch.models.*

class NativeTinyVaeToolsSuite extends munit.FunSuite:
  override val munitTimeout: scala.concurrent.duration.Duration=scala.concurrent.duration.Duration(5,"minutes")
  test("trained tiny VAE encodes RGB pixels and decodes their exact latent PNG") {
    val root=Vector(Path.of(".superpowers/models"),Path.of("../.superpowers/models"))
      .find(p=>Files.isRegularFile(p.resolve("taesd/diffusion_pytorch_model.safetensors")))
    assume(root.nonEmpty,"trained TAESD weights required")
    for name<-Vector("taesd","taesdxl") do
      val path=root.get.resolve(name)
      val encoder=TinyAutoencoderEncoder.load(path)
      val decoder=TinyAutoencoderDecoder.load(path)
      val tool=new NativeTinyVaeTools(encoder,decoder,VaeExecution.Cuda(0,1073741824L))
      val image=new BufferedImage(19,17,BufferedImage.TYPE_INT_ARGB)
      val pixels=new Array[Float](19*17*3)
      for y<-0 until 17; x<-0 until 19 do
        val rgb=((x*13%256)<<16)|((y*15%256)<<8)|((x*y%256))
        image.setRGB(x,y,rgb)
        for c<-0 until 3 do pixels(c*19*17+y*19+x)=((rgb>>(16-8*c))&255)/127.5f-1f
      val expected=encoder.encodeCuda(Tensor.fromFloats(pixels,Shape.from(Seq(1,3,17,19))))
      val latent=tool.encode(image,()=>())
      assertEquals(latent.toFloatArray.toVector,expected.toFloatArray.toVector)
      val file=Files.createTempFile("tiny-vae-latent-",".png")
      Files.delete(file)
      NativeVaeLatent.writeExactPng(latent,file)
      val restored=NativeVaeLatent.readExactPng(file,4)
      val decoded=tool.decode(restored,()=>())
      val reference=NativePreview.decodedRgb(decoder.decodeCuda(expected))
      assertEquals((decoded.getWidth,decoded.getHeight),(24,24))
      for y<-0 until 24; x<-0 until 24 do assertEquals(decoded.getRGB(x,y),reference.getRGB(x,y))
      intercept[java.util.concurrent.CancellationException](tool.encode(image,()=>throw new java.util.concurrent.CancellationException()))
      Files.delete(file)
  }

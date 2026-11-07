package scalatorch.models

import java.nio.{ByteBuffer,ByteOrder}
import java.nio.file.{Files,Path}
import scalatorch.*

class TinyAutoencoderEncoderSuite extends munit.FunSuite:
  override val munitTimeout: scala.concurrent.duration.Duration=scala.concurrent.duration.Duration(10,"minutes")
  test("TAESD CUDA encoder handles 512px and recovers after cancellation") {
    val root=Vector(Path.of(".superpowers/models/taesd"),Path.of("../.superpowers/models/taesd"))
      .find(p=>Files.isRegularFile(p.resolve("diffusion_pytorch_model.safetensors")))
    assume(root.nonEmpty,"trained TAESD weights required")
    val model=TinyAutoencoderEncoder.load(root.get)
    val image=Tensor.fromFloats(Array.tabulate(3*512*512)(i=>(math.sin(i*.007)*.5).toFloat),Shape.from(Seq(1,3,512,512)))
    var checks=0
    intercept[java.util.concurrent.CancellationException] {
      model.encodeCuda(image,cancelled=()=>{
        checks+=1
        if checks==12 then throw new java.util.concurrent.CancellationException("test interruption")
      })
    }
    assertEquals(checks,12)
    val start=System.nanoTime()
    val latent=model.encodeCuda(image)
    println(s"TAESD CUDA encode 512px seconds=${(System.nanoTime()-start)/1e9}")
    assertEquals(latent.shape,Shape.from(Seq(1,4,64,64)))
    assert(latent.toFloatArray.forall(_.isFinite))
  }
  test("trained SD1 and SDXL tiny encoders match Diffusers for even and odd images") {
    val root=Vector(Path.of(".superpowers/models"),Path.of("../.superpowers/models"))
      .find(p=>Files.isRegularFile(p.resolve("taesd/diffusion_pytorch_model.safetensors")))
    assume(root.nonEmpty,"trained TAESD weights required")
    for name<-Vector("taesd","taesdxl"); (height,width)<-Vector((16,24),(17,19)) do
      val model=TinyAutoencoderEncoder.load(root.get.resolve(name))
      val image=Tensor.fromFloats(Array.tabulate(3*height*width)(i=>math.sin(i*.037).toFloat),
        Shape.from(Seq(1,3,height,width)))
      val stream=this.getClass.getResourceAsStream(s"/tiny-autoencoder/$name-encode-${height}x$width.f32")
      val raw=try stream.readAllBytes() finally stream.close()
      val buffer=ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)
      val expected=Array.fill(raw.length/4)(buffer.getFloat())
      val cpu=model.encode(image)
      val gpu=model.encodeCuda(image)
      assertEquals(cpu.shape,Shape.from(Seq(1,4,(height+7)/8,(width+7)/8)))
      assertEquals(gpu.shape,cpu.shape)
      for (actual,label,tolerance)<-Vector((cpu,"CPU",1e-4f),(gpu,"CUDA",1e-3f)) do
        val max=actual.toFloatArray.zip(expected).map((a,b)=>math.abs(a-b)).max
        assert(max<tolerance,s"$name $height x $width $label max error=$max")
      val decoded=TinyAutoencoderDecoder.load(root.get.resolve(name)).decodeCuda(gpu)
      val roundtripStream=this.getClass.getResourceAsStream(s"/tiny-autoencoder/$name-roundtrip-${height}x$width.f32")
      val roundtripRaw=try roundtripStream.readAllBytes() finally roundtripStream.close()
      val roundtripBuffer=ByteBuffer.wrap(roundtripRaw).order(ByteOrder.LITTLE_ENDIAN)
      val roundtrip=Array.fill(roundtripRaw.length/4)(roundtripBuffer.getFloat())
      assertEquals(decoded.shape,Shape.from(Seq(1,3,((height+7)/8)*8,((width+7)/8)*8)))
      val error=decoded.toFloatArray.zip(roundtrip).map((a,b)=>math.abs(a-b)).max
      assert(error<2e-3,s"$name $height x $width roundtrip max error=$error")
  }

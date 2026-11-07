package scalatorch.models

import java.nio.{ByteBuffer,ByteOrder}
import java.nio.file.{Files,Path}
import scalatorch.*

class TinyAutoencoderDecoderSuite extends munit.FunSuite:
  override val munitTimeout: scala.concurrent.duration.Duration=scala.concurrent.duration.Duration(10,"minutes")
  test("trained TAESD CUDA decoder handles a 512px output") {
    val root=Vector(Path.of(".superpowers/models/taesd"),Path.of("../.superpowers/models/taesd"))
      .find(p=>Files.isRegularFile(p.resolve("diffusion_pytorch_model.safetensors")))
    assume(root.nonEmpty,"trained TAESD reference weights required")
    val model=TinyAutoencoderDecoder.load(root.get)
    val latent=Tensor.fromFloats(Array.tabulate(4*64*64)(i=>(math.sin(i*.07)*.5).toFloat),Shape.from(Seq(1,4,64,64)))
    val start=System.nanoTime()
    val output=model.decodeCuda(latent)
    println(s"TAESD CUDA 512px seconds=${(System.nanoTime()-start)/1e9}")
    assertEquals(output.shape,Shape.from(Seq(1,3,512,512)))
    assert(output.toFloatArray.forall(_.isFinite))
  }
  test("trained SD1 and SDXL tiny decoders match Diffusers outputs") {
    val root=Vector(Path.of(".superpowers/models"),Path.of("../.superpowers/models"))
      .find(p=>Files.isRegularFile(p.resolve("taesd/diffusion_pytorch_model.safetensors")))
    assume(root.nonEmpty,"trained TAESD reference weights required")
    val latent=Tensor.fromFloats(Array.tabulate(24)(i=>(math.sin(i*.37)*2).toFloat),Shape.from(Seq(1,4,2,3)))
    for name<-Vector("taesd","taesdxl") do
      val model=TinyAutoencoderDecoder.load(root.get.resolve(name))
      val output=model.decode(latent)
      assertEquals(output.shape,Shape.from(Seq(1,3,16,24)))
      val stream=this.getClass.getResourceAsStream(s"/tiny-autoencoder/$name-decode.f32")
      val raw=try stream.readAllBytes() finally stream.close()
      val buffer=ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)
      val expected=Array.fill(raw.length/4)(buffer.getFloat())
      val actual=output.toFloatArray
      val max=actual.zip(expected).map((a,b)=>math.abs(a-b)).max
      assert(max<1e-4,s"$name max error=$max")
      val gpu=model.decodeCuda(latent)
      val gpuMax=gpu.toFloatArray.zip(expected).map((a,b)=>math.abs(a-b)).max
      assert(gpuMax<1e-3,s"$name CUDA max error=$gpuMax")
      var checks=0
      intercept[java.util.concurrent.CancellationException] {
        model.decodeCuda(latent,cancelled=()=>{
          checks+=1
          if checks==12 then throw new java.util.concurrent.CancellationException("test interruption")
        })
      }
      assertEquals(checks,12)
      val recovered=model.decodeCuda(latent)
      assert(recovered.toFloatArray.zip(expected).forall((a,b)=>math.abs(a-b)<1e-3))
  }

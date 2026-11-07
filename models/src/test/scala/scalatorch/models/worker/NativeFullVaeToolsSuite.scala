package scalatorch.models.worker

import java.nio.file.{Files,Path}
import scalatorch.*
import scalatorch.models.*

class NativeFullVaeToolsSuite extends munit.FunSuite:
  override val munitTimeout: scala.concurrent.duration.Duration=scala.concurrent.duration.Duration(15,"minutes")
  test("real SD1.5 VAE processes an image on a bounded CUDA session") {
    assume(sys.env.get("SDFX_TEST_REAL_VAE").contains("1"),"set SDFX_TEST_REAL_VAE=1 to run the real-model check")
    val root=Vector(Path.of(".superpowers/models/sd15/vae"),Path.of("../.superpowers/models/sd15/vae"))
      .find(p=>Files.isRegularFile(p.resolve("diffusion_pytorch_model.safetensors")))
    assume(root.nonEmpty,"SD1.5 VAE weights required")
    val vae=PretrainedVae.load(root.get)
    val tool=new NativeFullVaeTools(vae,VaeExecution.Cuda(0,1610612736L,true),true)
    val width=sys.env.get("SDFX_TEST_REAL_VAE_WIDTH").map(_.toInt).getOrElse(512)
    require(width>=512 && width<=1024 && width%8==0)
    val image=new java.awt.image.BufferedImage(width,512,java.awt.image.BufferedImage.TYPE_INT_RGB)
    for y<-0 until 512; x<-0 until width do image.setRGB(x,y,((x*255/(width-1))<<16)|((y/2)<<8)|((x*255/(width-1)+y/2)/2))
    val start=System.nanoTime()
    val latent=tool.encodeSeeded(image,42,()=>())
    assertEquals(latent.shape,Shape.from(Seq(1,4,64,width/8)))
    assert(latent.toFloatArray.forall(_.isFinite))
    val decoded=tool.decode(latent,()=>())
    assertEquals((decoded.getWidth,decoded.getHeight),(width,512))
    var error=0L
    for y<-0 until 512; x<-0 until width; shift<-Vector(0,8,16) do
      error+=math.abs(((image.getRGB(x,y)>>>shift)&255)-((decoded.getRGB(x,y)>>>shift)&255))
    val meanError=error.toDouble/(width*512*3)
    println(s"real SD1.5 VAE pixel mean absolute error=$meanError")
    assert(meanError<40,s"VAE reconstruction error is unexpectedly high: $meanError")
    println(s"real SD1.5 VAE ${width}x512 encode/decode seconds=${(System.nanoTime()-start)/1e9}")
  }
  test("Full VAE encoding uses seed and produces diffusion-scaled latents") {
    val root=Vector(Path.of(".superpowers/models/tiny-sd-pipe/vae"),Path.of("../.superpowers/models/tiny-sd-pipe/vae"))
      .find(p=>Files.isRegularFile(p.resolve("diffusion_pytorch_model.safetensors")))
    assume(root.nonEmpty,"tiny VAE fixture required")
    val vae=PretrainedVae.load(root.get)
    val image=new java.awt.image.BufferedImage(16,16,java.awt.image.BufferedImage.TYPE_INT_RGB)
    val tool=new NativeFullVaeTools(vae,VaeExecution.Cuda(),false)
    val first=tool.encodeSeeded(image,42,()=>())
    val same=tool.encodeSeeded(image,42,()=>())
    val different=tool.encodeSeeded(image,43,()=>())
    assertEquals(first.toFloatArray.toVector,same.toFloatArray.toVector)
    assertNotEquals(first.toFloatArray.toVector,different.toFloatArray.toVector)
    val pixels=Tensor.fromFloats(Array.fill(3*16*16)(-1f),Shape.from(Seq(1,3,16,16)))
    val moments=VaeExecution.Cuda().encode(vae.encoder,pixels,()=>false)
    val posterior=VaePosterior(scalatorch.autograd.Variable(moments,false))
    val noise=Tensor.fromFloats(new TorchCpuNormal(42).normal(posterior.mean.data.shape.numel.toInt),posterior.mean.data.shape)
    val expected=vae.toDiffusionLatents(posterior.sample(noise)).data
    assertEquals(first.toFloatArray.toVector,expected.toFloatArray.toVector)
    val dir=Files.createTempDirectory("full-vae-encode")
    val source=dir.resolve("source.png")
    javax.imageio.ImageIO.write(image,"png",source.toFile)
    val params=Map[String,NativeJson]("input_paths"->NativeJson.Arr(Vector(NativeJson.Str(source.toString))),
      "output_dir"->NativeJson.Str(dir.toString),"operation"->NativeJson.Str("encode+decode"),
      "approximation"->NativeJson.Str("Full"),"seed"->NativeJson.Number(42))
    val report=NativeVaeTools.execute(params,tool,new NativeWorkContext(()=>false,()=>true,(_,_)=>()))
    val item=report("results").asInstanceOf[NativeJson.Arr].values.head.asInstanceOf[NativeJson.Obj].fields
    assertEquals(item("status"),NativeJson.Str("complete"))
    assert(Files.isRegularFile(Path.of(item("path").asInstanceOf[NativeJson.Str].value)))
    assert(report("warnings").asInstanceOf[NativeJson.Arr].values.nonEmpty)
  }
  test("Full VAE decoder applies diffusion scaling and supports tiled execution") {
    val root=Vector(Path.of(".superpowers/models/tiny-sd-pipe/vae"),Path.of("../.superpowers/models/tiny-sd-pipe/vae"))
      .find(p=>Files.isRegularFile(p.resolve("diffusion_pytorch_model.safetensors")))
    assume(root.nonEmpty,"tiny VAE fixture required")
    val vae=PretrainedVae.load(root.get)
    val latent=Tensor.fromFloats(Array.tabulate(4*4*4)(i=>math.sin(i*.1).toFloat),Shape.from(Seq(1,4,4,4)))
    val raw=Tensor.fromFloats(latent.toFloatArray.map(_/vae.config.scalingFactor.toFloat),latent.shape)
    val expected=NativePreview.decodedRgb(VaeExecution.Cuda().decode(vae.decoder,raw,()=>false))
    val tool=new NativeFullVaeTools(vae,VaeExecution.Cuda(),false)
    val actual=tool.decode(latent,()=>())
    assertEquals((actual.getWidth,actual.getHeight),(expected.getWidth,expected.getHeight))
    for y<-0 until actual.getHeight; x<-0 until actual.getWidth do
      assertEquals(actual.getRGB(x,y),expected.getRGB(x,y))
    val tiled=new NativeFullVaeTools(vae,VaeExecution.Cuda(),true).decode(latent,()=>())
    assertEquals((tiled.getWidth,tiled.getHeight),(actual.getWidth,actual.getHeight))
    val dir=Files.createTempDirectory("full-vae-tools")
    val input=dir.resolve("latent.png")
    NativeVaeLatent.writeExactPng(latent,input)
    val params=Map[String,NativeJson]("input_paths"->NativeJson.Arr(Vector(NativeJson.Str(input.toString))),
      "output_dir"->NativeJson.Str(dir.toString),"operation"->NativeJson.Str("decode"),
      "approximation"->NativeJson.Str("Full"),"tiled"->NativeJson.Bool(true))
    val report=NativeVaeTools.execute(params,new NativeFullVaeTools(vae,VaeExecution.Cuda(),true),
      new NativeWorkContext(()=>false,()=>true,(_,_)=>()))
    val item=report("results").asInstanceOf[NativeJson.Arr].values.head.asInstanceOf[NativeJson.Obj].fields
    assertEquals(item("status"),NativeJson.Str("complete"))
    assert(Files.isRegularFile(Path.of(item("path").asInstanceOf[NativeJson.Str].value)))
    intercept[java.util.concurrent.CancellationException](tool.decode(latent,()=>throw new java.util.concurrent.CancellationException()))
  }

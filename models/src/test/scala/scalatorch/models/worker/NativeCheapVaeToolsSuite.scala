package scalatorch.models.worker

import java.awt.image.BufferedImage
import scalatorch.*

class NativeCheapVaeToolsSuite extends munit.FunSuite:
  test("cheap latent decoding matches NUI SD1 and SDXL reference pixels") {
    val values=Array(0f,1f,-1f,2f,-2f,.5f,.3f,-.7f,1.2f,-1.5f,.9f,0f,
      .2f,.1f,-.3f,.7f,.4f,-.9f,1f,-1f,.5f,-.5f,1.5f,-1.5f)
    val latent=Tensor.fromFloats(values,Shape.from(Seq(1,4,2,3)))
    val encoder=new NativeVaeToolsCodec:
      def encode(image: BufferedImage,cancelled: () => Unit)=latent
      def decode(latent: Tensor[Float],cancelled: () => Unit)=throw AssertionError("wrong decoder")
    val cases=Vector(false->Vector(107,109,81,170,165,202,112,120,87,165,160,201,30,65,17,200,171,201),
      true->Vector(75,97,101,238,216,208,26,51,63,255,255,247,0,0,0,208,192,197))
    for (xl,expected)<-cases do
      val image=new NativeCheapVaeTools(encoder,xl).decode(latent,()=>())
      val actual=(for y<-0 until 2;x<-0 until 3;shift<-Vector(16,8,0) yield (image.getRGB(x,y)>>>shift)&255).toVector
      assertEquals(actual,expected)
    val dir=java.nio.file.Files.createTempDirectory("cheap-vae-tools")
    val path=dir.resolve("latent.png")
    NativeVaeLatent.writeExactPng(latent,path)
    val params=Map[String,NativeJson]("input_paths"->NativeJson.Arr(Vector(NativeJson.Str(path.toString))),
      "output_dir"->NativeJson.Str(dir.toString),"operation"->NativeJson.Str("decode"),
      "approximation"->NativeJson.Str("Approx cheap"))
    val report=NativeVaeTools.execute(params,new NativeCheapVaeTools(encoder,false),
      new NativeWorkContext(()=>false,()=>true,(_,_)=>()))
    val item=report("results").asInstanceOf[NativeJson.Arr].values.head.asInstanceOf[NativeJson.Obj].fields
    assertEquals(item("status"),NativeJson.Str("complete"))
    val output=javax.imageio.ImageIO.read(java.nio.file.Path.of(item("path").asInstanceOf[NativeJson.Str].value).toFile)
    assertEquals((output.getWidth,output.getHeight),(3,2))
    val source=dir.resolve("source.png")
    javax.imageio.ImageIO.write(new BufferedImage(16,16,BufferedImage.TYPE_INT_RGB),"png",source.toFile)
    val cmyk=NativeVaeTools.execute(params.updated("input_paths",NativeJson.Arr(Vector(NativeJson.Str(source.toString))))
      .updated("operation",NativeJson.Str("encode+decode")).updated("representation",NativeJson.Str("CMYK")),
      new NativeCheapVaeTools(encoder,false),new NativeWorkContext(()=>false,()=>true,(_,_)=>()))
    val cmykItem=cmyk("results").asInstanceOf[NativeJson.Arr].values.head.asInstanceOf[NativeJson.Obj].fields
    assertEquals(cmykItem("status"),NativeJson.Str("complete"))
  }

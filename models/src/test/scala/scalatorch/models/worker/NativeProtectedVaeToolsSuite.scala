package scalatorch.models.worker

import java.awt.image.BufferedImage
import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import scalatorch.*
import NativeJson.*

class NativeProtectedVaeToolsSuite extends munit.FunSuite:
  test("legacy protected grayscale latent is restored exactly and warns when unverified") {
    val dir=Files.createTempDirectory("protected-latent-l")
    def fixture(name: String): Path =
      val out=dir.resolve(name)
      val stream=getClass.getResourceAsStream(s"/vae-latent/$name")
      try Files.copy(stream,out) finally stream.close()
      out
    val input=fixture("l-4-legacy-protected.png")
    val expected=NativeVaeLatent.readExactPng(fixture("l-4.png"),4).toFloatArray
    var actual=Array.empty[Float]
    val codec=new NativeVaeToolsCodec:
      def encode(image: BufferedImage,cancelled: () => Unit)=throw UnsupportedOperationException()
      def decode(latent: Tensor[Float],cancelled: () => Unit): BufferedImage =
        actual=latent.toFloatArray
        new BufferedImage(1,1,BufferedImage.TYPE_INT_RGB)
    val params=Map[String,NativeJson]("input_paths"->Arr(Vector(Str(input.toString))),"output_dir"->Str(dir.toString),
      "operation"->Str("decode"),"representation"->Str("L"),"approximation"->Str("Full"))
    val ctx=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    val report=NativeVaeTools.execute(params,codec,ctx,Map.empty,Some("fixture-password"))
    assertEquals(report("results").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("status"),Str("complete"))
    assertEquals(actual.map(java.lang.Float.floatToRawIntBits).toVector,expected.map(java.lang.Float.floatToRawIntBits).toVector)
    assert(report("warnings").asInstanceOf[Arr].values.exists(_.asInstanceOf[Str].value.contains("unverified")))
  }
  test("protected VAE output requires consent and roundtrips without plaintext images") {
    val dir=Files.createTempDirectory("protected-vae-tools")
    val input=dir.resolve("input.png")
    val password="fixture-password"
    val image=new BufferedImage(16,16,BufferedImage.TYPE_INT_RGB)
    NativeProtectedPng.write(image,input,password,Vector.empty)
    val tensor=Tensor.fromFloats(Array.fill(4*2*2)(.5f),Shape.from(Seq(1,4,2,2)))
    val codec=new NativeVaeToolsCodec:
      def encode(image: BufferedImage,cancelled: () => Unit)=tensor
      def decode(latent: Tensor[Float],cancelled: () => Unit)=image
    val params=Map[String,NativeJson]("input_paths"->Arr(Vector(Str(input.toString))),"output_dir"->Str(dir.toString),
      "operation"->Str("encode"),"representation"->Str("RGBA"),"approximation"->Str("Full"))
    val ctx=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    intercept[IllegalArgumentException](NativeVaeTools.execute(params,codec,ctx,Map.empty,Some(password)))
    val report=NativeVaeTools.execute(params.updated("allow_plaintext_latent",Bool(true)),codec,ctx,Map.empty,Some(password))
    val item=report("results").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields
    assertEquals(item("status"),Str("complete"))
    val path=Path.of(item("path").asInstanceOf[Str].value)
    assert(NativeProtectedPng.read(path,Some(password)).scrambled)
    assert(NativeProtectedPng.read(Path.of(item("preview_path").asInstanceOf[Str].value),Some(password)).scrambled)
    val decode=params.updated("input_paths",Arr(Vector(Str(path.toString)))).updated("operation",Str("decode"))
    val restored=NativeVaeTools.execute(decode,codec,ctx,Map.empty,Some(password))
    assertEquals(restored("results").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("status"),Str("complete"))
    val denied=NativeVaeTools.execute(decode,codec,ctx,Map.empty,Some("wrong"))
    assertEquals(denied("results").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("status"),Str("error"))
    val roundtrip=NativeVaeTools.execute(params.updated("operation",Str("encode+decode"))
      .updated("representation",Str("L")),codec,ctx,Map.empty,Some(password))
    val roundtripItem=roundtrip("results").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields
    assertEquals(roundtripItem("status"),Str("complete"))
    val roundtripDir=Path.of(roundtripItem("path").asInstanceOf[Str].value).getParent
    assert(!Files.exists(roundtripDir.resolve("001-encoded.png")))
    assert(!Files.exists(roundtripDir.resolve("001-latent.safetensors")))
    assert(NativeProtectedPng.read(roundtripDir.resolve("001-decoded.png"),Some(password)).scrambled)
  }

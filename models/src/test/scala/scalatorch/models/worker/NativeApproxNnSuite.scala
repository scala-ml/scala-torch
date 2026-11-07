package scalatorch.models.worker

import java.nio.{ByteBuffer,ByteOrder}
import java.nio.file.{Files,Path}
import scalatorch.*

class NativeApproxNnSuite extends munit.FunSuite:
  test("official release download produces loadable pinned weights") {
    assume(sys.env.get("SDFX_TEST_APPROX_DOWNLOAD").contains("1"),"network download test is opt-in")
    val root=Files.createTempDirectory("approx-official")
    for name<-Vector("model.pt","vaeapprox-sdxl.pt") do
      val target=root.resolve(name)
      NativeApproxDownload.official(target,name,()=>())
      assert(Files.isRegularFile(target))
      NativeApproxNn.load(target)
  }
  test("optional official weight download is bounded, verified and never replaces a cached file") {
    val root=Files.createTempDirectory("approx-download")
    val target=root.resolve("model.pt")
    val bytes="verified fixture".getBytes(java.nio.charset.StandardCharsets.UTF_8)
    val hash=java.util.HexFormat.of().formatHex(java.security.MessageDigest.getInstance("SHA-256").digest(bytes))
    def source()=new java.io.ByteArrayInputStream(bytes)
    intercept[IllegalArgumentException](NativeApproxDownload.fetch(target,"0"*64,()=>source(),()=>()))
    assert(!Files.exists(target))
    NativeApproxDownload.fetch(target,hash,()=>source(),()=>())
    assertEquals(Files.readAllBytes(target).toVector,bytes.toVector)
    intercept[java.nio.file.FileAlreadyExistsException](NativeApproxDownload.fetch(target,hash,()=>source(),()=>()))
    assertEquals(Files.readAllBytes(target).toVector,bytes.toVector)
    val files=Files.list(root)
    try assertEquals(files.count(),1L) finally files.close()
  }
  test("official SDXL Approx NN matches Python reference") {
    val path=Vector(Path.of(".superpowers/models/vae-approx/vaeapprox-sdxl.pt"),Path.of("../.superpowers/models/vae-approx/vaeapprox-sdxl.pt"))
      .find(Files.isRegularFile(_))
    assume(path.nonEmpty,"official SDXL Approx NN checkpoint required")
    val input=Tensor.fromFloats(Array.tabulate(24)(i=>math.sin(i*.13).toFloat),Shape.from(Seq(1,4,2,3)))
    val actual=NativeApproxNn.load(path.get).decodeTensor(input,()=>()).toFloatArray
    val stream=getClass.getResourceAsStream("/vae-approx/sdxl-reference.f32")
    val bytes=try stream.readAllBytes() finally stream.close()
    val buffer=ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    val expected=Array.fill(bytes.length/4)(buffer.getFloat())
    val error=actual.zip(expected).map((a,b)=>math.abs(a-b)).max
    assert(error<1e-4f,s"max error $error")
  }
  test("official SD1 Approx NN matches Python reference with bounded local weights") {
    val path=Vector(Path.of(".superpowers/models/vae-approx/model.pt"),Path.of("../.superpowers/models/vae-approx/model.pt"))
      .find(Files.isRegularFile(_))
    assume(path.nonEmpty,"official Approx NN checkpoint required")
    val input=Tensor.fromFloats(Array.tabulate(24)(i=>math.sin(i*.13).toFloat),Shape.from(Seq(1,4,2,3)))
    val result=NativeApproxNn.load(path.get).decodeTensor(input,()=>())
    assertEquals(result.shape,Shape.from(Seq(1,3,4,6)))
    val stream=getClass.getResourceAsStream("/vae-approx/sd15-reference.f32")
    val bytes=try stream.readAllBytes() finally stream.close()
    val buffer=ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    val expected=Array.fill(bytes.length/4)(buffer.getFloat())
    val error=result.toFloatArray.zip(expected).map((a,b)=>math.abs(a-b)).max
    assert(error<1e-4f,s"max error $error")
    val root=Files.createTempDirectory("native-approx-nn")
    val latentPath=root.resolve("latent.png")
    NativeVaeLatent.writeExactPng(input,latentPath)
    val encoder=new NativeVaeToolsCodec:
      def encode(image: java.awt.image.BufferedImage,cancelled: () => Unit)=input
      def decode(value: Tensor[Float],cancelled: () => Unit)=throw UnsupportedOperationException()
    val codec=new NativeApproxNnVaeTools(encoder,NativeApproxNn.load(path.get))
    val params=Map[String,NativeJson]("input_paths"->NativeJson.Arr(Vector(NativeJson.Str(latentPath.toString))),
      "output_dir"->NativeJson.Str(root.toString),"operation"->NativeJson.Str("decode"),
      "approximation"->NativeJson.Str("Approx NN"))
    val report=NativeVaeTools.execute(params,codec,new NativeWorkContext(()=>false,()=>true,(_,_)=>()))
    val item=report("results").asInstanceOf[NativeJson.Arr].values.head.asInstanceOf[NativeJson.Obj].fields
    assertEquals(item("status"),NativeJson.Str("complete"))
    val image=javax.imageio.ImageIO.read(Path.of(item("path").asInstanceOf[NativeJson.Str].value).toFile)
    assertEquals((image.getWidth,image.getHeight),(6,4))
    intercept[java.util.concurrent.CancellationException](NativeApproxNn.load(path.get).decodeTensor(input,()=>throw new java.util.concurrent.CancellationException()))
  }

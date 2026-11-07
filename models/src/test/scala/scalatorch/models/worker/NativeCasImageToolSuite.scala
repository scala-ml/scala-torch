package scalatorch.models.worker

import java.awt.image.BufferedImage
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files,Path}
import java.util.concurrent.CancellationException
import javax.imageio.ImageIO
import NativeJson.*

class NativeCasImageToolSuite extends munit.FunSuite:
  private def obj(v: NativeJson)=v.asInstanceOf[Obj].fields
  private def arr(v: NativeJson)=v.asInstanceOf[Arr].values
  private def num(v: NativeJson)=v.asInstanceOf[Number].value.toDouble
  test("native CAS image tool matches NUI pixels and publishes only on success") {
    val stream=getClass.getResourceAsStream("/nui-cas-reference.json")
    val bytes=try stream.readAllBytes() finally stream.close()
    assertEquals(java.util.HexFormat.of().formatHex(java.security.MessageDigest.getInstance("SHA-256").digest(bytes)),"c6ff3b34fd476abb8a5ebcf5bd5ba00bc682a19e17a49214dcf82bbd3b08f325")
    val data=obj(NativeJson.parse(new String(bytes,UTF_8)))
    val root=Files.createTempDirectory("native-cas-test")
    try
      val source=new BufferedImage(4,3,BufferedImage.TYPE_INT_ARGB)
      arr(data("pixels")).zipWithIndex.foreach { (pixel,i) =>
        val c=arr(pixel).map(v=>num(v).toInt)
        source.setRGB(i%4,i/4,(c(3)<<24)|(c(0)<<16)|(c(1)<<8)|c(2))
      }
      val sourcePath=root.resolve("rgba.png");ImageIO.write(source,"png",sourcePath.toFile)
      val scheduler=root.resolve("scheduler.json");Files.writeString(scheduler,"{}")
      val backend=new NativeDiffusionBackend(root,scheduler)
      assert(backend.supported.contains("image_tool"))
      val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
      val output=Files.createDirectory(root.resolve("out"))
      for item <- arr(data("cases")) do
        val fields=obj(item)
        val params=Map[String,NativeJson]("tool"->Str("cas_upscale"),"image_path"->Str(sourcePath.toString),
          "output_dir"->Str(output.toString),"scale"->fields("scale"),"sharpness"->fields("sharpness"))
        val result=backend.execute(NativeRequest("cas","image_tool",params,None),context)
        assertEquals(result("outputs"),Obj(Map.empty));assertEquals(result("info"),Obj(Map.empty))
        val image=ImageIO.read(Path.of(result("path").asInstanceOf[Str].value).toFile)
        assertEquals((image.getWidth,image.getHeight),(num(fields("width")).toInt,num(fields("height")).toInt))
        arr(fields("pixels")).zipWithIndex.foreach { (pixel,i) =>
          val expected=arr(pixel).map(v=>num(v).toInt)
          val actual=image.getRGB(i%image.getWidth,i/image.getWidth)
          for channel <- 0 to 2 do assert(math.abs(((actual >>> (16-8*channel))&255)-expected(channel))<=1)
        }
      val denied=new NativeWorkContext(()=>false,()=>false,(_,_)=>())
      val params=Map[String,NativeJson]("tool"->Str("cas_upscale"),"image_path"->Str(sourcePath.toString),"output_dir"->Str(output.toString))
      intercept[CancellationException](backend.execute(NativeRequest("cancel","image_tool",params,None),denied))
      val files=Files.list(output);try assertEquals(files.count(),arr(data("cases")).size.toLong) finally files.close()
      intercept[NativeBackendError](backend.execute(NativeRequest("unsupported","image_tool",params.updated("tool",Str("waifu2x")),None),context))
    finally
      val paths=Files.walk(root)
      try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }

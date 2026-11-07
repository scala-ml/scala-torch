package scalatorch.models.worker

import java.awt.image.BufferedImage
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import NativeJson.*

class NativeLoopbackImageToolSuite extends munit.FunSuite:
  private def obj(v: NativeJson)=v.asInstanceOf[Obj].fields
  private def arr(v: NativeJson)=v.asInstanceOf[Arr].values
  private def num(v: NativeJson)=v.asInstanceOf[Number].value.toInt
  test("Loopback finishing matches NUI Pillow reference") {
    val stream=getClass.getResourceAsStream("/nui-loopback-reference.json")
    val bytes=try stream.readAllBytes() finally stream.close()
    assertEquals(java.util.HexFormat.of().formatHex(java.security.MessageDigest.getInstance("SHA-256").digest(bytes)),"8b8b48e10fb4a98e91f7c8f5aaa49a49cb51ead78c803d4edfd862626a4e2211")
    val root=Files.createTempDirectory("native-loopback")
    try
      val config=root.resolve("scheduler.json");Files.writeString(config,"{}")
      val backend=new NativeDiffusionBackend(root,config)
      val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
      var maximumDifference=0
      for (item,caseIndex) <- arr(obj(NativeJson.parse(new String(bytes,UTF_8)))("cases")).zipWithIndex do
        val f=obj(item);val width=num(f("width"));val height=num(f("height"))
        val src=new BufferedImage(width,height,BufferedImage.TYPE_INT_ARGB)
        arr(f("source")).zipWithIndex.foreach { (p,i) =>
          val c=arr(p).map(num);src.setRGB(i%width,i/width,(c(3)<<24)|(c(0)<<16)|(c(1)<<8)|c(2))
        }
        val path=root.resolve("source.png");ImageIO.write(src,"png",path.toFile)
        val params=obj(f("params"))++Map("tool"->Str("loopback_scaler"),"image_path"->Str(path.toString),"output_dir"->Str(root.toString))
        val result=backend.execute(NativeRequest("loop","image_tool",params,None),context)
        assertEquals(result("outputs"),Obj(Map.empty));assertEquals(result("info"),Obj(Map.empty))
        val output=Path.of(result("path").asInstanceOf[Str].value)
        val image=ImageIO.read(output.toFile);Files.delete(output)
        assertEquals((image.getWidth,image.getHeight),(width,height))
        arr(f("pixels")).zipWithIndex.foreach { (p,i) =>
          val expected=arr(p).map(num);val pixel=image.getRGB(i%width,i/width)
          for channel <- 0 to 2 do
            maximumDifference=math.max(maximumDifference,math.abs(((pixel >>> (16-8*channel))&255)-expected(channel)))
        }
      val invalid=Map[String,NativeJson]("tool"->Str("loopback_scaler"),"image_path"->Str(root.resolve("source.png").toString),
        "output_dir"->Str(root.toString),"sharpness"->Number(3))
      intercept[IllegalArgumentException](backend.execute(NativeRequest("bad","image_tool",invalid,None),context))
      val denied=new NativeWorkContext(()=>false,()=>false,(_,_)=>())
      intercept[java.util.concurrent.CancellationException](backend.execute(
        NativeRequest("cancel","image_tool",invalid.updated("sharpness",Number(1)),None),denied))
      val remaining=Files.list(root)
      try assertEquals(remaining.count(),2L) finally remaining.close()
      assert(maximumDifference<=3,s"maximum Pillow channel difference: $maximumDifference")
    finally
      val paths=Files.walk(root)
      try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }

package scalatorch.models.worker

import java.awt.image.BufferedImage
import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import NativeJson.*

class NativeColorEnhanceSuite extends munit.FunSuite:
  test("color enhancement on a generated image agrees with NUI") {
    val source=sys.env.get("SCALATORCH_COLOR_SOURCE");val fixture=sys.env.get("SCALATORCH_ENHANCE_FIXTURE")
    assume(source.nonEmpty && fixture.nonEmpty,"real generated image fixture required")
    val root=Files.createTempDirectory("native-enhance-real")
    try
      val scheduler=root.resolve("scheduler.json");Files.writeString(scheduler,"{}")
      val backend=new NativeDiffusionBackend(root,scheduler)
      val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
      val result=backend.execute(NativeRequest("enhance","image_tool",Map("tool"->Str("color_enhance"),
        "image_path"->Str(source.get),"output_dir"->Str(root.toString)),None),context)
      val actual=ImageIO.read(Path.of(result("path").asInstanceOf[Str].value).toFile)
      val expected=ImageIO.read(Path.of(fixture.get).resolve("result.png").toFile)
      assertEquals((actual.getWidth,actual.getHeight),(expected.getWidth,expected.getHeight))
      var maximum=0;var total=0L
      for y <- 0 until actual.getHeight; x <- 0 until actual.getWidth; c <- 0 until 3 do
        val shift=16-c*8
        val error=math.abs(((actual.getRGB(x,y)>>>shift)&255)-((expected.getRGB(x,y)>>>shift)&255))
        maximum=math.max(maximum,error);total+=error
      println(s"enhance ${actual.getWidth}x${actual.getHeight} max=$maximum mean=${total.toDouble/(actual.getWidth*actual.getHeight*3)}")
      assert(maximum<=2,s"maximum channel error $maximum")
      val info=NativeJson.parse(Files.readString(Path.of(fixture.get).resolve("info.json")))
      assertEquals(result("info"),info)
    finally
      val paths=Files.walk(root)
      try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }
  test("color enhancement follows NUI reference for color and grayscale") {
    val input=this.getClass.getResourceAsStream("/nui-color-enhance-reference.json")
    val cases=try NativeJson.parse(new String(input.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8)).asInstanceOf[Arr].values finally input.close()
    val root=Files.createTempDirectory("native-color-enhance")
    try
      val scheduler=root.resolve("scheduler.json");Files.writeString(scheduler,"{}")
      val backend=new NativeDiffusionBackend(root,scheduler)
      val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
      cases.foreach { entry =>
        val row=entry.asInstanceOf[Obj].fields
        val gray=row("kind")==Str("gray")
        val source=new BufferedImage(19,17,BufferedImage.TYPE_INT_RGB)
        for y <- 0 until 17; x <- 0 until 19 do
          val v=(x*31+y*17)%256
          val pixel=if gray then (v<<16)|(v<<8)|v else (((x*23+y*7)%256)<<16)|(((x*11+y*19)%256)<<8)|((x*3+y*29)%256)
          source.setRGB(x,y,pixel)
        val path=root.resolve("source.png");ImageIO.write(source,"png",path.toFile)
        val params=Map[String,NativeJson]("tool"->Str("color_enhance"),"image_path"->Str(path.toString),
          "output_dir"->Str(root.toString),"strength"->row("strength"),"percentile"->row("percentile"))
        val result=backend.execute(NativeRequest("enhance","image_tool",params,None),context)
        val actual=ImageIO.read(Path.of(result("path").asInstanceOf[Str].value).toFile)
        val expected=row("rgb").asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toInt)
        val errors=for y <- 0 until 17; x <- 0 until 19; c <- 0 until 3 yield
          math.abs(((actual.getRGB(x,y) >>> (16-8*c))&255)-expected((y*19+x)*3+c))
        assert(errors.max<=2,s"${row("kind")} strength=${row("strength")} percentile=${row("percentile")} max=${errors.max}")
        assertEquals(result("info"),row("info"),s"metadata ${row("kind")} strength=${row("strength")} percentile=${row("percentile")}")
      }
      val outDir=Files.createDirectory(root.resolve("cancelled"))
      val params=Map[String,NativeJson]("tool"->Str("color_enhance"),"image_path"->Str(root.resolve("source.png").toString),
        "output_dir"->Str(outDir.toString))
      val denied=new NativeWorkContext(()=>false,()=>false,(_,_)=>())
      intercept[java.util.concurrent.CancellationException](backend.execute(NativeRequest("denied","image_tool",params,None),denied))
      val paths=Files.list(outDir)
      try assertEquals(paths.count(),0L,"cancelled output must be removed") finally paths.close()
      intercept[IllegalArgumentException](backend.execute(NativeRequest("bad","image_tool",params.updated("percentile",Number(101)),None),context))
    finally
      val paths=Files.walk(root)
      try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }

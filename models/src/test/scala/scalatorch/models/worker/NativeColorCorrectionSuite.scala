package scalatorch.models.worker

import java.awt.image.BufferedImage
import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import NativeJson.*

class NativeColorCorrectionSuite extends munit.FunSuite:
  test("color correction matches NUI on generated images") {
    val source=sys.env.get("SCALATORCH_COLOR_SOURCE")
    val reference=sys.env.get("SCALATORCH_COLOR_REFERENCE")
    val fixture=sys.env.get("SCALATORCH_COLOR_FIXTURES")
    assume(source.nonEmpty && reference.nonEmpty && fixture.nonEmpty,"generated image fixtures required")
    val root=Files.createTempDirectory("native-color-real")
    try
      val scheduler=root.resolve("scheduler.json");Files.writeString(scheduler,"{}")
      val backend=new NativeDiffusionBackend(root,scheduler)
      val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
      for method <- Vector("A1111","AdaIN","Wavelet") do
        val params=Map[String,NativeJson]("tool"->Str("color_correction"),"image_path"->Str(source.get),
          "reference_path"->Str(reference.get),"output_dir"->Str(root.toString),"method"->Str(method),
          "strength"->Number(BigDecimal("0.4")))
        val started=System.nanoTime()
        val result=backend.execute(NativeRequest("color-real","image_tool",params,None),context)
        val output=ImageIO.read(Path.of(result("path").asInstanceOf[Str].value).toFile)
        val expected=ImageIO.read(Path.of(fixture.get).resolve(s"$method.png").toFile)
        assertEquals((output.getWidth,output.getHeight),(expected.getWidth,expected.getHeight))
        var max=0;var sum=0L
        for y <- 0 until output.getHeight; x <- 0 until output.getWidth; c <- 0 until 3 do
          val shift=16-8*c
          val diff=math.abs(((output.getRGB(x,y)>>>shift)&255)-((expected.getRGB(x,y)>>>shift)&255))
          max=math.max(max,diff);sum+=diff
        println(s"$method ${output.getWidth}x${output.getHeight} max=$max mean=${sum.toDouble/(output.getWidth*output.getHeight*3)} time=${(System.nanoTime()-started)/1e9}s")
        assert(max<=1,s"$method maximum error $max")
    finally
      val paths=Files.walk(root)
      try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }
  test("native color correction matches NUI reference with resized reference image") {
    val input=this.getClass.getResourceAsStream("/nui-color-correction-reference.json")
    val cases=try NativeJson.parse(new String(input.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8)).asInstanceOf[Arr].values finally input.close()
    val root=Files.createTempDirectory("native-color-correction")
    try
      def make(w: Int,h: Int,salt: Int): BufferedImage =
        val image=new BufferedImage(w,h,BufferedImage.TYPE_INT_RGB)
        for y <- 0 until h; x <- 0 until w do
          image.setRGB(x,y,(((x*53+y*29+salt)%256)<<16)|(((x*11+y*71+salt*3)%256)<<8)|((x*31+y*7+salt*5)%256))
        image
      val source=root.resolve("source.png");val reference=root.resolve("reference.png")
      ImageIO.write(make(11,7,2),"png",source.toFile)
      ImageIO.write(make(7,13,17),"png",reference.toFile)
      val scheduler=root.resolve("scheduler.json");Files.writeString(scheduler,"{}")
      val backend=new NativeDiffusionBackend(root,scheduler)
      val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
      cases.foreach { entry =>
        val row=entry.asInstanceOf[Obj].fields
        val params=Map[String,NativeJson]("tool"->Str("color_correction"),"image_path"->Str(source.toString),
          "reference_path"->Str(reference.toString),"output_dir"->Str(root.toString),
          "method"->row("method"),"strength"->row("strength"))
        val result=backend.execute(NativeRequest("color","image_tool",params,None),context)
        val output=ImageIO.read(Path.of(result("path").asInstanceOf[Str].value).toFile)
        assertEquals((output.getWidth,output.getHeight),(11,7))
        val expected=row("rgb").asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toInt)
        val differences=for y <- 0 until 7; x <- 0 until 11; c <- 0 until 3 yield
          math.abs(((output.getRGB(x,y) >>> (16-c*8))&255)-expected((y*11+x)*3+c))
        assert(differences.max<=1,s"${row("method")} strength=${row("strength")} max error ${differences.max}")
        assertEquals(result("info").asInstanceOf[Obj].fields("method"),row("method"))
      }
      val outputDir=Files.createDirectory(root.resolve("comparisons"))
      val methods=Vector("A1111","AdaIN","Wavelet")
      def parameters(method: String,all: Boolean)=Map[String,NativeJson]("tool"->Str("color_correction"),
        "image_path"->Str(source.toString),"reference_path"->Str(reference.toString),"output_dir"->Str(outputDir.toString),
        "method"->Str(method),"strength"->Number(BigDecimal("0.4")),"all_methods"->Bool(all))
      def pixels(path: NativeJson)=
        val img=ImageIO.read(Path.of(path.asInstanceOf[Str].value).toFile)
        img.getRGB(0,0,11,7,null,0,11).toVector
      val individual=methods.map(m=>m->pixels(backend.execute(NativeRequest("single","image_tool",parameters(m,false),None),context)("path"))).toMap
      methods.foreach { method =>
        var publicationCount=0
        val guarded=new NativeWorkContext(()=>false,()=>{publicationCount+=1;true},(_,_)=>())
        val result=backend.execute(NativeRequest("all","image_tool",parameters(method,true),None),guarded)
        assertEquals(publicationCount,1)
        assertEquals(pixels(result("path")),individual(method))
        val outputs=result("outputs").asInstanceOf[Obj].fields
        assertEquals(outputs.keySet,methods.toSet-method)
        outputs.foreach { (name,path) => assertEquals(pixels(path),individual(name),name) }
      }
      val before=Files.list(outputDir)
      val count=try before.count() finally before.close()
      val denied=new NativeWorkContext(()=>false,()=>false,(_,_)=>())
      intercept[java.util.concurrent.CancellationException](backend.execute(NativeRequest("denied","image_tool",parameters("A1111",true),None),denied))
      val after=Files.list(outputDir)
      try assertEquals(after.count(),count) finally after.close()
    finally
      val paths=Files.walk(root)
      try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }

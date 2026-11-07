package scalatorch.models.worker

import java.awt.image.BufferedImage
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files,Path}
import java.util.concurrent.CancellationException
import javax.imageio.ImageIO
import NativeJson.*

class NativePreprocessSuite extends munit.FunSuite:
  private def obj(v: NativeJson)=v.asInstanceOf[Obj].fields
  private def arr(v: NativeJson)=v.asInstanceOf[Arr].values
  private def num(v: NativeJson)=v.asInstanceOf[Number].value.toInt
  test("preprocess matches NUI reference without loading a model") {
    val stream=getClass.getResourceAsStream("/nui-preprocess-reference.json")
    val bytes=try stream.readAllBytes() finally stream.close()
    assertEquals(java.util.HexFormat.of().formatHex(java.security.MessageDigest.getInstance("SHA-256").digest(bytes)),"5303a71179a4fb09961b0d40335f358d19ed69004ca3e861c8d2dee8bb640fba")
    val data=obj(NativeJson.parse(new String(bytes,UTF_8)))
    val root=Files.createTempDirectory("native-preprocess")
    try
      val source=new BufferedImage(num(data("width")),num(data("height")),BufferedImage.TYPE_INT_RGB)
      arr(data("pixels")).zipWithIndex.foreach { (v,i) =>
        val c=arr(v).map(num);source.setRGB(i%source.getWidth,i/source.getWidth,(c(0)<<16)|(c(1)<<8)|c(2))
      }
      val input=root.resolve("input.png");NativePreprocess.write(source,input,Vector("parameters"->"source parameters"))
      val config=root.resolve("scheduler.json");Files.writeString(config,"{}")
      val backend=new NativeDiffusionBackend(root,config)
      assert(backend.supported.contains("preprocess"))
      val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
      val differences=scala.collection.mutable.Map.empty[String,(Int,Int)]
      for item <- arr(data("cases")) do
        val f=obj(item)
        val request=NativeRequest("pre","preprocess",Map("preprocessor"->f("name"),"args"->f("args"),
          "standalone"->f("standalone"),"processor_res"->f("processor_res"),"pixel_perfect"->f("pixel_perfect"),
          "image_path"->Str(input.toString),"output_dir"->Str(root.toString)),None)
        val result=backend.execute(request,context)
        if f("standalone")==Bool(true) then
          assertEquals(result("info"),f("info"));assertEquals(result("notes"),f("notes"))
        val path=Path.of(result("path").asInstanceOf[Str].value)
        val output=ImageIO.read(path.toFile)
        if f("standalone")==Bool(true) then
          assertEquals(NativePreprocess.sourceParameters(path),Some(f("parameters").asInstanceOf[Str].value))
          assertEquals(NativePreprocess.sourceParameters(path,"Source parameters"),Some("source parameters"))
        assertEquals((output.getWidth,output.getHeight),(num(f("width")),num(f("height"))))
        arr(f("pixels")).zipWithIndex.foreach { (v,i) =>
          val expected=arr(v).map(num);val actual=output.getRGB(i%output.getWidth,i/output.getWidth)
          for c <- 0 to 2 do
            val diff=math.abs(((actual>>>(16-8*c))&255)-expected(c))
            val key=s"${f("name")} ${f("standalone")} ${f("processor_res")} ${f("pixel_perfect")}"
            val (maximum,count)=differences.getOrElse(key,(0,0))
            differences(key)=(math.max(maximum,diff),count+(if diff>0 then 1 else 0))
        }
        Files.delete(path)
      val denied=new NativeWorkContext(()=>false,()=>false,(_,_)=>())
      val p=Map[String,NativeJson]("preprocessor"->Str("invert"),"image_path"->Str(input.toString),"output_dir"->Str(root.toString))
      intercept[CancellationException](backend.execute(NativeRequest("cancel","preprocess",p,None),denied))
      val files=Files.list(root);try assertEquals(files.count(),2L) finally files.close()
      differences.foreach { (key,value) =>
        if key.startsWith("Str(canny)") then
          assertEquals(value._2,0,s"Canny differs from reference: $key")
        else assert(value._1<=1,s"$key differs from NUI: $value")
      }
    finally
      val paths=Files.walk(root)
      try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }

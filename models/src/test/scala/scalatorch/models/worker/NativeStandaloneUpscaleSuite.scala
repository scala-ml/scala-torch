package scalatorch.models.worker

import java.awt.image.BufferedImage
import java.nio.file.{Files,Path}
import java.util.concurrent.CancellationException
import javax.imageio.ImageIO
import NativeJson.*

class NativeStandaloneUpscaleSuite extends munit.FunSuite:
  test("normalizes the NUI model-catalog name for the native RRDB upscaler") {
    assertEquals(NativeStandaloneUpscale.canonicalUpscaler("R-ESRGAN 4x+"),Some("R-ESRGAN 4x+"))
    assertEquals(NativeStandaloneUpscale.canonicalUpscaler("RealESRGAN_x4plus"),Some("R-ESRGAN 4x+"))
    assertEquals(NativeStandaloneUpscale.canonicalUpscaler("RealESRGAN_x4plus.pth"),Some("R-ESRGAN 4x+"))
    assertEquals(NativeStandaloneUpscale.canonicalUpscaler("RealESRGAN_x4plus_anime_6B.pth"),Some("R-ESRGAN 4x+"))
    assertEquals(NativeStandaloneUpscale.canonicalUpscaler("4x-UltraSharp"),None)
  }

  test("hardware metadata formats provenance and elapsed time like NUI") {
    assertEquals(NativeHardwareInfo.compact("Intel(R) Core(TM) i7-12700 CPU @ 2.10GHz",32L*1073741824,
      Some("NVIDIA GeForce RTX 3060" -> (12L*1073741824))),"RTX 3060 12GB, i7-12700, 32GB RAM")
    assertEquals(NativeHardwareInfo.compact("AMD64 Family 25, AuthenticAMD",16L*1073741824,None),
      "AMD64 Family 25, AuthenticAMD, 16GB RAM")
    assertEquals(NativeHardwareInfo.duration(2.25),"2.2 sec.")
    assertEquals(NativeHardwareInfo.duration(63.25),"1 min. 3.2 sec.")
    assertEquals(NativeHardwareInfo.profile("cuda:not-a-number"),"unknown")
    intercept[IllegalArgumentException](NativeHardwareInfo.compact("",1,None))
  }
  test("standalone upscale works before model loading and publishes Pillow-compatible nearest pixels") {
    val root=Files.createTempDirectory("native-upscale")
    try
      val scheduler=root.resolve("scheduler.json"); Files.writeString(scheduler,"{}")
      val output=Files.createDirectory(root.resolve("output"))
      val source=new BufferedImage(3,2,BufferedImage.TYPE_INT_RGB)
      for y <- 0 until 2; x <- 0 until 3 do
        val n=10*(1+y*3+x);source.setRGB(x,y,0xff000000 | (n<<16) | (n<<8) | n)
      val imagePath=root.resolve("source.png");assert(ImageIO.write(source,"png",imagePath.toFile))
      val backend=new NativeDiffusionBackend(root,scheduler)
      assert(backend.supported.contains("upscale"))
      val params=Map[String,NativeJson]("image_path"->Str(imagePath.toString),"output_dir"->Str(output.toString),
        "upscaler"->Str("Nearest"),"scale"->Number(BigDecimal("1.5")),"upscaler_paths"->Obj(Map.empty),
        "hardware_info"->Obj(Map("enabled"->Bool(false),"remark"->Str(""))))
      val request=NativeRequest("up","upscale",params,None)
      val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
      val result=backend.execute(request,context)
      assert(!result.contains("hardware_info"))
      val image=ImageIO.read(Path.of(result("path").asInstanceOf[Str].value).toFile)
      assertEquals((image.getWidth,image.getHeight),(4,3))
      assertEquals((for y <- 0 until 3; x <- 0 until 4 yield image.getRGB(x,y)&255).toVector,
        Vector(10,20,20,30,40,50,50,60,40,50,50,60))
      val smaller=backend.execute(request.copy(params=params.updated("scale",Number(BigDecimal("0.5")))),context)
      val downsampled=ImageIO.read(Path.of(smaller("path").asInstanceOf[Str].value).toFile)
      assertEquals((downsampled.getWidth,downsampled.getHeight),(2,1))
      assertEquals(Vector(downsampled.getRGB(0,0)&255,downsampled.getRGB(1,0)&255),Vector(40,60))
      val denied=new NativeWorkContext(()=>false,()=>false,(_,_)=>())
      intercept[CancellationException](backend.execute(request,denied))
      val entries=Files.list(output)
      try assertEquals(entries.count(),2L) finally entries.close()
      intercept[IllegalArgumentException](backend.execute(request.copy(params=params.updated("scale",Number(0))),context))
      intercept[IllegalArgumentException](backend.execute(request.copy(params=params.updated("scale",Number(17))),context))
      intercept[NativeBackendError](backend.execute(request.copy(params=params.updated("upscaler",Str("unknown"))),context))
      val withHardware=backend.execute(request.copy(params=params.updated("hardware_info",
        Obj(Map("enabled"->Bool(true),"remark"->Str("test workstation"))))),context)
      val info=withHardware("hardware_info").asInstanceOf[Obj].fields
      assert(info("hardware_info").asInstanceOf[Str].value.nonEmpty)
      assert(info("time_taken").asInstanceOf[Str].value.matches("(?:[0-9]+ min\\. )?[0-9]+\\.[0-9] sec\\."))
      assertEquals(info("remark"),Str("test workstation"))
      val emptyRemark=backend.execute(request.copy(params=params.updated("hardware_info",
        Obj(Map("enabled"->Bool(true),"remark"->Str(""))))),context)
      assertEquals(emptyRemark("hardware_info").asInstanceOf[Obj].fields("remark"),Null)
    finally
      val paths=Files.walk(root)
      try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }

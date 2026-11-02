package scalatorch.models

import java.nio.file.{Files,Path}

class DiffusionImageInputSuite extends munit.FunSuite:
  test("PNG/JPEG/TIFF EXIF orientations and RGB conversion match NUI Pillow reference") {
    val names=(for i <- 1 to 8; extension <- Seq("png","jpg","tiff") yield s"orientation-$i.$extension") ++ Seq("mode-L.png","mode-LA.png","mode-P.png")
    names.foreach { name =>
      def bytes(resource: String): Array[Byte] =
        val stream=getClass.getResourceAsStream(s"/image-input/$resource")
        try stream.readAllBytes() finally stream.close()
      val path=Files.createTempFile("image-input-",".png")
      try
        Files.write(path,bytes(name))
        check(path,bytes(name+".rgb"),name)
      finally Files.delete(path)
    }
    def check(path: Path,reference: Array[Byte],name: String): Unit =
      val image=DiffusionImageInput.readRgb(path)
      val expected=reference.map(_ & 255).toVector
      val rotated=name.matches("orientation-[5-8].(png|jpg|tiff)")
      assertEquals(image.getWidth,if rotated then 2 else 3)
      assertEquals(image.getHeight,if rotated then 3 else 2)
      val actual=(for y <- 0 until image.getHeight; x <- 0 until image.getWidth; shift <- Seq(16,8,0) yield
        (image.getRGB(x,y) >>> shift)&255).toVector
      assertEquals(actual,expected,name)
      intercept[IllegalArgumentException](DiffusionImageInput.readRgb(path,maxPixels=5))
      intercept[IllegalArgumentException](DiffusionImageInput.readRgb(path,maxBytes=1))
  }
  test("malformed input and out-of-bounds EXIF reject without decoding") {
    val path=Files.createTempFile("bad-image-",".png")
    try
      Files.write(path,Array[Byte](1,2,3,4))
      intercept[IllegalArgumentException](DiffusionImageInput.readRgb(path))
      val stream=getClass.getResourceAsStream("/image-input/orientation-6.png")
      val data=try stream.readAllBytes() finally stream.close()
      val marker=data.sliding(4).indexWhere(_.sameElements(Array[Byte](101,88,73,102)))
      assert(marker>=0)
      for i <- marker+8 until marker+12 do data(i)=0xff.toByte
      Files.write(path,data)
      intercept[IllegalArgumentException](DiffusionImageInput.readRgb(path))
    finally Files.delete(path)
    intercept[IllegalArgumentException](DiffusionImageInput.readRgb(path))
  }

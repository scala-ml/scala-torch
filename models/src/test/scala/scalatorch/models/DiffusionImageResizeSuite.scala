package scalatorch.models

import java.awt.image.BufferedImage

class DiffusionImageResizeSuite extends munit.FunSuite:
  test("RGB Lanczos and all NUI resize modes match independent Pillow fixtures") {
    for (w,h,tw,th) <- Seq((7,5,13,11),(13,11,5,3),(9,4,6,9),(4,9,9,6),(1,7,4,3),(7,1,3,4),(7,5,7,5)); mode <- 0 to 2 do
      val input=new BufferedImage(w,h,BufferedImage.TYPE_INT_RGB)
      for y <- 0 until h; x <- 0 until w do
        input.setRGB(x,y,(((x*71+y*19)%256)<<16)|(((x*13+y*83)%256)<<8)|((x*97+y*41)%256))
      val output=DiffusionImageResize(input,tw,th,mode)
      val stream=getClass.getResourceAsStream(s"/image-resize/$w-$h-$tw-$th-$mode.rgb")
      val expected=try stream.readAllBytes().map(_ & 255).toVector finally stream.close()
      val actual=(for y <- 0 until th; x <- 0 until tw; shift <- Seq(16,8,0) yield (output.getRGB(x,y) >>> shift)&255).toVector
      assertEquals(actual,expected,s"$w,$h -> $tw,$th mode$mode")
    val image=new BufferedImage(2,2,BufferedImage.TYPE_INT_RGB)
    intercept[IllegalArgumentException](DiffusionImageResize(image,0,2,0))
    intercept[IllegalArgumentException](DiffusionImageResize(image,2,2,3))
    intercept[IllegalArgumentException](DiffusionImageResize(image,2,2,0,maxPixels=3))
  }

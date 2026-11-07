package scalatorch.models

import scalatorch.*
import scalatorch.models.worker.NativePreview

class NativePreviewSuite extends munit.FunSuite:
  test("SD1 and SDXL preview pixels match NUI reference vectors") {
    val latent=Tensor.fromFloats((0 until 16).map(i=>(i-8)/8f).toArray,Shape.from(Seq(1,4,2,2)))
    def pixels(sdxl: Boolean): Vector[Int] =
      val image=NativePreview.rgb(latent,sdxl)
      (0 until 4).flatMap { i =>
        val rgb=image.getRGB(i%2,i/2)
        Vector((rgb>>16)&255,(rgb>>8)&255,rgb&255)
      }.toVector
    assertEquals(pixels(false),Vector(66,66,60,68,72,63,70,79,65,73,85,68))
    assertEquals(pixels(true),Vector(72,60,57,70,64,60,69,67,64,67,71,68))
  }
  test("SDXL approximate preview uses SDXL channel projection") {
    val latent=Tensor.fromFloats(Array(1f,0f,0f,0f),Shape.from(Seq(1,4,1,1)))
    val rgb=NativePreview.rgb(latent,sdxl=true)
    assertEquals(rgb.getRGB(0,0)&0xffffff,(171<<16)|(181<<8)|184)
    assertNotEquals(rgb.getRGB(0,0),NativePreview.rgb(latent).getRGB(0,0))
  }
  test("SD15 approximate preview uses channel projection and bounded JPEG output") {
    val latent=Tensor.fromFloats(Array.fill(8)(1f),Shape.from(Seq(1,4,1,2)))
    val rgb=NativePreview.rgb(latent)
    assertEquals(rgb.getRGB(0,0)&0xffffff,(146<<16)|(180<<8)|149)
    val neutral=NativePreview.rgb(Tensor.fromFloats(Array.fill(4)(0f),Shape.from(Seq(1,4,1,1))))
    assertEquals(neutral.getRGB(0,0)&0xffffff,0x808080)
    val jpeg=java.util.Base64.getDecoder.decode(NativePreview.jpegBase64(latent))
    val image=javax.imageio.ImageIO.read(new java.io.ByteArrayInputStream(jpeg))
    assertEquals(image.getWidth,256)
    assertEquals(image.getHeight,128)
    intercept[IllegalArgumentException](NativePreview.rgb(Tensor.fromFloats(Array.fill(4)(Float.NaN),Shape.from(Seq(1,4,1,1)))))
  }

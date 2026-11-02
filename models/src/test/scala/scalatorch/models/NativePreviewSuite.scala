package scalatorch.models

import scalatorch.*
import scalatorch.models.worker.NativePreview

class NativePreviewSuite extends munit.FunSuite:
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

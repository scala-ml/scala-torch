package scalatorch.models.worker

import java.awt.image.BufferedImage

class NativeInpaintSuite extends munit.FunSuite:
  test("mask inversion, resize and blur preserve white-means-repaint semantics") {
    val source=new BufferedImage(12,8,BufferedImage.TYPE_INT_RGB)
    val mask=new BufferedImage(12,8,BufferedImage.TYPE_INT_RGB)
    for y <- 0 until 8; x <- 0 until 12 do
      source.setRGB(x,y,0xff406080)
      mask.setRGB(x,y,if x<6 then 0xff000000 else 0xffffffff)
    val plain=NativeInpaint.prepare(source,mask,12,8,0,0,false,() => false)
    val values=plain.mask.toFloatArray
    assertEquals(values(0),0f)
    assertEquals(values(11),1f)
    assertEquals(plain.image.toFloatArray.toVector,plain.originalImage.toFloatArray.toVector)
    val inverted=NativeInpaint.prepare(source,mask,12,8,0,0,true,() => false).mask.toFloatArray
    assertEquals(inverted(0),1f)
    assertEquals(inverted(11),0f)
    val blurred=NativeInpaint.prepare(source,mask,12,8,0,2,false,() => false).mask.toFloatArray
    assert(blurred(5)>0f && blurred(5)<1f)
    assert(blurred(0)<blurred(5) && blurred(11)>blurred(6))
  }


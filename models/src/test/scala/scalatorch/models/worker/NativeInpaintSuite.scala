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

  test("full-resolution inpainting crops around the mask and restores the original canvas") {
    val source=new BufferedImage(100,60,BufferedImage.TYPE_INT_RGB)
    val mask=new BufferedImage(100,60,BufferedImage.TYPE_INT_RGB)
    for y <- 0 until 60; x <- 0 until 100 do
      source.setRGB(x,y,0xffff0000)
      mask.setRGB(x,y,if x>=40 && x<60 && y>=20 && y<40 then 0xffffffff else 0xff000000)
    val prepared=NativeInpaint.prepare(source,mask,32,32,0,0,false,() => false,fullRes=true,fullResPadding=8)
    assertEquals(prepared.image.shape.dims,Vector(1L,3L,32L,32L))
    assertEquals(prepared.pasteBox,Some((32,12,68,48)))
    val values=Array.fill(3*32*32)(-1f)
    java.util.Arrays.fill(values,32*32,2*32*32,1f)
    val generated=scalatorch.Tensor.fromFloats(values,scalatorch.Shape.from(Seq(1L,3L,32L,32L)))
    val result=prepared.pasteBack(scalatorch.models.DiffusionImageIO.toBufferedImage(generated))
    assertEquals((result.getWidth,result.getHeight),(100,60))
    assertEquals((result.getRGB(0,0)&0xffffff),0xff0000)
    assertEquals((result.getRGB(50,30)&0xffffff),0x00ff00)
    intercept[java.util.concurrent.CancellationException](prepared.pasteBack(
      scalatorch.models.DiffusionImageIO.toBufferedImage(generated),() => true))
  }


package scalatorch.models.worker

import scalatorch.*
import java.awt.image.BufferedImage
import scalatorch.models.worker.NativeJson.*

class NativeInpaintSuite extends munit.FunSuite:
  test("full resolution crops and restores match NUI reference") {
    val stream=getClass.getResourceAsStream("/inpaint-full-reference.json")
    val fixture=try NativeJson.parse(new String(stream.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8)).asInstanceOf[Obj].fields finally stream.close()
    def integer(v: NativeJson)=v.asInstanceOf[Number].value.toInt
    def rgb(v: NativeJson): Int =
      val a=v.asInstanceOf[Arr].values.map(integer)
      (a(0)<<16)|(a(1)<<8)|a(2)
    def image(values: NativeJson,w: Int,h: Int): BufferedImage =
      val out=new BufferedImage(w,h,BufferedImage.TYPE_INT_RGB)
      values.asInstanceOf[Arr].values.zipWithIndex.foreach((v,i)=>out.setRGB(i%w,i/w,rgb(v)))
      out
    val width=integer(fixture("width"));val height=integer(fixture("height"))
    val source=image(fixture("source"),width,height)
    val generated=image(fixture("generated"),16,12)
    fixture("cases").asInstanceOf[Arr].values.foreach { value =>
      val row=value.asInstanceOf[Obj].fields
      val mask=new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB)
      row("mask").asInstanceOf[Arr].values.zipWithIndex.foreach { (v,i) =>
        val n=integer(v);mask.setRGB(i%width,i/width,(n<<16)|(n<<8)|n)
      }
      val result=NativeInpaint.prepare(source,mask,16,12,0,integer(row("blur")),false,() => false,
        fullRes=true,fullResPadding=integer(row("padding")))
      val expectedBox=row("box") match
        case Null => None
        case a: Arr =>
          val v=a.values.map(integer);Some((v(0),v(1),v(2),v(3)))
        case _ => fail("invalid reference box")
      val label=s"${row("name")} blur=${row("blur")} padding=${row("padding")}"
      assertEquals(result.pasteBox,expectedBox,label)
      val actualImage=scalatorch.models.DiffusionImageIO.toBufferedImage(result.image)
      val expectedImage=row("model_image").asInstanceOf[Arr].values.map(rgb)
      val actualPixels=actualImage.getRGB(0,0,16,12,null,0,16).toVector.map(_&0xffffff)
      assertEquals(actualPixels,expectedImage,s"model image $label")
      val actualMask=result.mask.toFloatArray.map(v=>math.round(v*255)).toVector
      val expectedMask=row("model_mask").asInstanceOf[Arr].values.map(integer)
      assertEquals(actualMask,expectedMask,s"model mask $label")
      if expectedBox.nonEmpty then
        val actual=result.pasteBack(generated).getRGB(0,0,width,height,null,0,width).toVector.map(_&0xffffff)
        val expected=row("final").asInstanceOf[Arr].values.map(rgb)
        assertEquals(actual,expected,s"final image $label")
    }
  }
  test("all inpainting fill modes match NUI and Pillow reference pixels") {
    val stream=getClass.getResourceAsStream("/inpaint-fill-reference.json")
    val fixture=try NativeJson.parse(new String(stream.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8)).asInstanceOf[Obj].fields finally stream.close()
    val width=fixture("width").asInstanceOf[Number].value.toInt
    val height=fixture("height").asInstanceOf[Number].value.toInt
    val source=new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB)
    fixture("pixels").asInstanceOf[Arr].values.zipWithIndex.foreach { (value,i) =>
      source.setRGB(i%width,i/width,value.asInstanceOf[Number].value.toInt)
    }
    fixture("cases").asInstanceOf[Arr].values.foreach { value =>
      val row=value.asInstanceOf[Obj].fields
      val mask=new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB)
      row("mask").asInstanceOf[Arr].values.zipWithIndex.foreach { (value,i) =>
        val v=value.asInstanceOf[Number].value.toInt
        mask.setRGB(i%width,i/width,(v<<16)|(v<<8)|v)
      }
      val mode=row("fill").asInstanceOf[Number].value.toInt
      val blur=row("blur").asInstanceOf[Number].value.toInt
      val result=NativeInpaint.prepare(source,mask,width,height,0,blur,false,() => false,fill=mode,seed=42)
      val image=scalatorch.models.DiffusionImageIO.toBufferedImage(result.image)
      val actual=image.getRGB(0,0,width,height,null,0,width).toVector.map(_ & 0xffffff)
      val expected=row("result").asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toInt)
      assertEquals(actual,expected,s"pattern=${row("pattern")} blur=$blur fill=$mode")
    }
  }
  test("inpaint mask blur matches Pillow's reference bytes") {
    val stream=getClass.getResourceAsStream("/inpaint-mask-reference.json")
    val fixture=try NativeJson.parse(new String(stream.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8)).asInstanceOf[Obj].fields finally stream.close()
    val width=fixture("width").asInstanceOf[Number].value.toInt
    val height=fixture("height").asInstanceOf[Number].value.toInt
    val source=new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB)
    fixture("cases").asInstanceOf[Arr].values.foreach { value =>
      val row=value.asInstanceOf[Obj].fields
      val input=row("input").asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toInt)
      val expected=row("output").asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toInt)
      val mask=new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB)
      for i <- input.indices do
        val level=input(i)
        mask.setRGB(i%width,i/width,(level<<16)|(level<<8)|level)
      val radius=row("radius").asInstanceOf[Number].value.toInt
      val actual=NativeInpaint.prepare(source,mask,width,height,0,radius,false,() => false).mask.toFloatArray
        .map(value => math.round(value*255f)).toVector
      assertEquals(actual,expected,s"pattern=${row("pattern")} radius=$radius")
    }
  }
  test("nine-channel masked pixels preserve values below 0.5 and zero values at or above it") {
    val image=Tensor.fromFloats(Array(.2f,.4f,.6f,.3f,.5f,.7f,.4f,.6f,.8f),Shape.from(Seq(1L,3L,1L,3L)))
    val mask=Tensor.fromFloats(Array(.49f,.5f,.51f),Shape.from(Seq(1L,1L,1L,3L)))
    val actual=NativeInpaint.maskedImage(image,mask,() => false).toFloatArray
    assertEquals(actual.toVector,Vector(.2f,0f,0f,.3f,0f,0f,.4f,0f,0f))
  }

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

  test("pixel noise fill matches Python random bytes and gray fill preserves unmasked pixels") {
    val source=new BufferedImage(12,8,BufferedImage.TYPE_INT_RGB)
    val mask=new BufferedImage(12,8,BufferedImage.TYPE_INT_RGB)
    for y <- 0 until 8; x <- 0 until 12 do
      source.setRGB(x,y,0xff112233)
      mask.setRGB(x,y,if x<6 then 0xff000000 else 0xffffffff)
    assertEquals(NativeInpaint.pythonRandomBytes(42,24).map(b=>f"${b&255}%02x").mkString,
      "9d79b1a37f31801cd11a6706fb40d6bd57526846903bb13e")
    assertEquals(NativeInpaint.pythonRandomBytes(42,5).map(b=>f"${b&255}%02x").mkString,"9d79b1a31c")
    assertEquals(NativeInpaint.pythonRandomBytes(9223372036854775807L,16).map(b=>f"${b&255}%02x").mkString,
      "9fa30f51c89413a8cb359aa160a7f721")
    val noise=NativeInpaint.prepare(source,mask,12,8,0,0,false,() => false,fill=2,seed=42)
    val noiseImage=scalatorch.models.DiffusionImageIO.toBufferedImage(noise.image)
    assertEquals((noiseImage.getRGB(0,0)&0xffffff),0x112233)
    assertEquals((noiseImage.getRGB(6,0)&0xffffff),0x684690)
    val gray=NativeInpaint.prepare(source,mask,12,8,0,0,false,() => false,fill=3)
    val grayImage=scalatorch.models.DiffusionImageIO.toBufferedImage(gray.image)
    assertEquals((grayImage.getRGB(0,0)&0xffffff),0x112233)
    assertEquals((grayImage.getRGB(6,0)&0xffffff),0x7f7f7f)
  }

  test("blur fill replaces masked pixels with surrounding color and handles a fully masked image") {
    val source=new BufferedImage(7,7,BufferedImage.TYPE_INT_RGB)
    val mask=new BufferedImage(7,7,BufferedImage.TYPE_INT_RGB)
    for y <- 0 until 7; x <- 0 until 7 do
      source.setRGB(x,y,if x==3 && y==3 then 0xffff0000 else 0xff0000ff)
      mask.setRGB(x,y,if x==3 && y==3 then 0xffffffff else 0xff000000)
    val filled=NativeInpaint.prepare(source,mask,7,7,0,0,false,() => false,fill=0)
    val result=scalatorch.models.DiffusionImageIO.toBufferedImage(filled.image)
    assert((result.getRGB(0,0)&0xffffff)==0x0000ff,
      f"unmasked pixel changed: 0x${result.getRGB(0,0)&0xffffff}%06x")
    assert((result.getRGB(3,3)&0x0000ff)>(result.getRGB(3,3)&0xff0000),
      f"blur-fill did not use surrounding blue: 0x${result.getRGB(3,3)&0xffffff}%06x")
    val allMask=new BufferedImage(7,7,BufferedImage.TYPE_INT_RGB)
    for y <- 0 until 7; x <- 0 until 7 do allMask.setRGB(x,y,0xffffffff)
    val emptyContext=NativeInpaint.prepare(source,allMask,7,7,0,0,false,() => false,fill=0)
    val black=scalatorch.models.DiffusionImageIO.toBufferedImage(emptyContext.image)
    assert((black.getRGB(3,3)&0xffffff)<=0x010101,
      f"fully masked context should be black: 0x${black.getRGB(3,3)&0xffffff}%06x")
  }


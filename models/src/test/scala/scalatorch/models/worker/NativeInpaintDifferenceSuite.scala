package scalatorch.models.worker

import java.awt.image.BufferedImage
import NativeJson.*

class NativeInpaintDifferenceSuite extends munit.FunSuite:
  test("large morphology windows handle thin images and neutral borders") {
    val base=new BufferedImage(1,5,BufferedImage.TYPE_INT_RGB)
    val changed=new BufferedImage(1,5,BufferedImage.TYPE_INT_RGB)
    changed.setRGB(0,2,0xffffff)
    assertEquals(NativeInpaintDifference.mask(base,changed,1,100,0,()=>false).toVector,Vector.fill(5)(255))
    assertEquals(NativeInpaintDifference.mask(base,changed,1,0,100,()=>false).toVector,Vector.fill(5)(0))
    for y <- 0 until 5 do changed.setRGB(0,y,0xffffff)
    assertEquals(NativeInpaintDifference.mask(base,changed,1,0,100,()=>false).toVector,Vector.fill(5)(255))
  }
  test("difference input validation and cancellation remain explicit") {
    val image=new BufferedImage(2,2,BufferedImage.TYPE_INT_RGB)
    intercept[IllegalArgumentException](NativeInpaintDifference.mask(image,image,Double.NaN,0,0,()=>false))
    intercept[IllegalArgumentException](NativeInpaintDifference.mask(image,image,1.0,101,0,()=>false))
    intercept[java.util.concurrent.CancellationException](NativeInpaintDifference.mask(image,image,1.0,0,0,()=>true))
    assert(NativeInpaintDifference.parse(Obj(Map("base_path"->Str("image.png"),"contours"->Bool(true)))).contours)
    intercept[IllegalArgumentException](NativeInpaintDifference.parse(Obj(Map("base_path"->Str("image.png"),"contours"->Str("true")))))
  }
  test("difference thresholds and morphology match NUI reference masks") {
    val stream=this.getClass.getResourceAsStream("/inpaint-difference-reference.json")
    val data=try NativeJson.parse(new String(stream.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8)).asInstanceOf[Obj].fields finally stream.close()
    val width=data("width").asInstanceOf[Number].value.toInt
    val height=data("height").asInstanceOf[Number].value.toInt
    def image(key: String): BufferedImage =
      val bytes=data(key).asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toInt)
      val result=new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB)
      for y <- 0 until height; x <- 0 until width do
        val i=(y*width+x)*3
        result.setRGB(x,y,(bytes(i)<<16)|(bytes(i+1)<<8)|bytes(i+2))
      result
    val base=image("base");val altered=image("altered")
    for entry <- data("cases").asInstanceOf[Arr].values do
      val row=entry.asInstanceOf[Obj].fields
      val threshold=row("threshold").asInstanceOf[Number].value.toDouble
      val dilation=row("dilation").asInstanceOf[Number].value.toInt
      val erosion=row("erosion").asInstanceOf[Number].value.toInt
      val contours=row("contours").asInstanceOf[Bool].value
      val expected=row("pixels").asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toInt)
      val actual=NativeInpaintDifference.mask(base,altered,threshold,dilation,erosion,()=>false,contours)
      assertEquals(actual.toVector,expected,s"threshold=$threshold dilation=$dilation erosion=$erosion contours=$contours")
    val sw=data("small_width").asInstanceOf[Number].value.toInt
    val sh=data("small_height").asInstanceOf[Number].value.toInt
    val values=data("small").asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toInt)
    val small=new BufferedImage(sw,sh,BufferedImage.TYPE_INT_RGB)
    for y <- 0 until sh; x <- 0 until sw do
      val i=(y*sw+x)*3
      small.setRGB(x,y,(values(i)<<16)|(values(i+1)<<8)|values(i+2))
    for entry <- data("resized_cases").asInstanceOf[Arr].values do
      val row=entry.asInstanceOf[Obj].fields
      val threshold=row("threshold").asInstanceOf[Number].value.toDouble
      val expected=row("pixels").asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toInt)
      assertEquals(NativeInpaintDifference.mask(small,altered,threshold,0,0,()=>false).toVector,expected,s"resized threshold=$threshold")
  }

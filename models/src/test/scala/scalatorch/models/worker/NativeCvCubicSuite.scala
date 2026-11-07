package scalatorch.models.worker

import java.awt.image.BufferedImage
import java.nio.charset.StandardCharsets.UTF_8
import NativeJson.*

class NativeCvCubicSuite extends munit.FunSuite:
  private def obj(v: NativeJson)=v.asInstanceOf[Obj].fields
  private def arr(v: NativeJson)=v.asInstanceOf[Arr].values
  private def num(v: NativeJson)=v.asInstanceOf[Number].value.toInt
  test("OpenCV cubic resize preserves replicated boundaries and reference pixels") {
    val stream=getClass.getResourceAsStream("/cv-cubic-reference.json")
    val bytes=try stream.readAllBytes() finally stream.close()
    assertEquals(java.util.HexFormat.of().formatHex(java.security.MessageDigest.getInstance("SHA-256").digest(bytes)),"2ac3f69bdecc9cfdcfad9959b84ce614b39b56ff47bcd2cc9f503770804fd00e")
    for (item,caseIndex) <- arr(obj(NativeJson.parse(new String(bytes,UTF_8)))("cases")).zipWithIndex do
      val f=obj(item);val w=num(f("w"));val h=num(f("h"));val ow=num(f("ow"));val oh=num(f("oh"))
      val image=new BufferedImage(w,h,BufferedImage.TYPE_INT_RGB)
      arr(f("source")).zipWithIndex.foreach { (v,i) => val c=arr(v).map(num);image.setRGB(i%w,i/w,(c(0)<<16)|(c(1)<<8)|c(2)) }
      val output=NativeControlNetImageOps.cvResize(image,ow,oh,"cubic",()=>false)
      var maxDifference=0;var wrong=0
      arr(f("pixels")).zipWithIndex.foreach { (v,i) =>
        val c=arr(v).map(num);val p=output.getRGB(i%ow,i/ow)
        for channel <- 0 to 2 do
          val diff=math.abs(((p>>>(16-8*channel))&255)-c(channel))
          maxDifference=math.max(maxDifference,diff);if diff!=0 then wrong+=1
      }
      assertEquals((maxDifference,wrong),(0,0),s"case $caseIndex")
    intercept[java.util.concurrent.CancellationException] {
      NativeControlNetImageOps.cvResize(new BufferedImage(4,4,BufferedImage.TYPE_INT_RGB),8,8,"cubic",()=>true)
    }
  }

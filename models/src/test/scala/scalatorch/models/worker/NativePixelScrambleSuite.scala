package scalatorch.models.worker

import javax.imageio.ImageIO

class NativePixelScrambleSuite extends munit.FunSuite:
  test("protected PNG validates password and interoperates with Python") {
    val dir=java.nio.file.Files.createTempDirectory("native-protected-png")
    val input=dir.resolve("input.png")
    val stream=getClass.getResourceAsStream("/scramble/protected.png")
    try java.nio.file.Files.copy(stream,input) finally stream.close()
    val password="\u30c6\u30b9\u30c8 pass"
    val decoded=NativeProtectedPng.read(input,Some(password))
    val source=ImageIO.read(getClass.getResourceAsStream("/scramble/source.png"))
    def pixels(image: java.awt.image.BufferedImage)=image.getRaster.getPixels(0,0,4,3,null.asInstanceOf[Array[Int]]).toVector
    assertEquals(pixels(decoded.image),pixels(source))
    assertEquals(decoded.verified,Some(true))
    intercept[IllegalArgumentException](NativeProtectedPng.read(input,None))
    intercept[IllegalArgumentException](NativeProtectedPng.read(input,Some("wrong")))
    val output=dir.resolve("output.png")
    NativeProtectedPng.write(source,output,password,Vector("VAE Tools"->"{}"))
    assertEquals(pixels(NativeProtectedPng.read(output,Some(password)).image),pixels(source))
    assertEquals(NativePreprocess.sourceParameters(output,"Encrypt"),Some("pixel_shuffle_3"))
  }
  test("pixel scrambling matches Python for all versions without losing transparent RGB") {
    def image(name: String)=ImageIO.read(getClass.getResourceAsStream(s"/scramble/$name.png"))
    def pixels(image: java.awt.image.BufferedImage)=
      val raster=image.getRaster
      raster.getPixels(0,0,image.getWidth,image.getHeight,null.asInstanceOf[Array[Int]]).toVector
    val source=image("source")
    val password="\u30c6\u30b9\u30c8 pass"
    assertEquals(NativePixelScramble.passwordMarker(password),"7c193f90737153a366509493f12c43175899ffcada6aa6919eabd304abd9cb3b")
    assert(NativePixelScramble.verifyPassword(password,NativePixelScramble.passwordMarker(password)))
    assert(!NativePixelScramble.verifyPassword("wrong",NativePixelScramble.passwordMarker(password)))
    assert(!NativePixelScramble.verifyPassword(password,"malformed"))
    for version<-1 to 3 do
      val scrambled=image(s"v$version")
      assertEquals(pixels(NativePixelScramble.decode(scrambled,password,version)),pixels(source))
      if version==3 then assertEquals(pixels(NativePixelScramble.encode(source,password)),pixels(scrambled))
  }

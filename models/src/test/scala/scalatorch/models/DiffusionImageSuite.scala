package scalatorch.models

import scalatorch.*
import scalatorch.autograd.Variable
import java.nio.file.Files
import javax.imageio.ImageIO

class DiffusionImageSuite extends munit.FunSuite:
  private def v(xs: Float*)=Variable(Tensor.fromFloats(xs.toArray,Shape.from(Seq(xs.size))))
  test("classifier-free guidance preserves both branch gradients and exact shapes") {
    val u=v(1,2);val c=v(3,1)
    val result=ClassifierFreeGuidance.combine(u,c,2)
    assertEquals(result.data.toFloatArray.toVector,Vector(5f,0f))
    result.sum().backward()
    assertEquals(u.grad.get.toFloatArray.toVector,Vector(-1f,-1f))
    assertEquals(c.grad.get.toFloatArray.toVector,Vector(2f,2f))
    intercept[IllegalArgumentException](ClassifierFreeGuidance.combine(u,v(1),2))
    intercept[IllegalArgumentException](ClassifierFreeGuidance.combine(u,c,Float.NaN))
  }
  test("PNG export maps NCHW colors and preserves an existing destination") {
    val data=Tensor.fromFloats(Array(-1f,1f,0f,0f,1f,-1f),Shape.from(Seq(1,3,1,2)))
    val dir=Files.createTempDirectory("diffusion-png")
    val path=dir.resolve("image.png")
    try
      DiffusionImageIO.writePng(data,path)
      val decoded=ImageIO.read(path.toFile)
      assertEquals(decoded.getWidth,2);assertEquals(decoded.getHeight,1)
      assertEquals(decoded.getRGB(0,0)&0xffffff,0x0080ff)
      assertEquals(decoded.getRGB(1,0)&0xffffff,0xff8000)
      val saved=Files.readAllBytes(path).toVector
      intercept[java.nio.file.FileAlreadyExistsException](DiffusionImageIO.writePng(data,path))
      assertEquals(Files.readAllBytes(path).toVector,saved)
      val invalid=Tensor.fromFloats(Array(Float.NaN,0f,0f),Shape.from(Seq(1,3,1,1)))
      intercept[IllegalArgumentException](DiffusionImageIO.writePng(invalid,dir.resolve("invalid.png")))
      assert(!Files.exists(dir.resolve("invalid.png")))
    finally
      Files.deleteIfExists(path);Files.delete(dir)
  }

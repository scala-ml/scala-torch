package scalatorch.models.worker
import scalatorch.{Shape,Tensor}

class NativeLatentUpscaleSuite extends munit.FunSuite:
  test("area interpolation matches PyTorch adaptive average pooling") {
    val input=Tensor.fromFloats((0 until 12).map(_.toFloat).toArray,Shape.from(Seq(1L,1L,3L,4L)))
    val output=NativeHires.resize(input,3,5,"area")
    assertEquals(output.toFloatArray.toVector,Vector(.5f,1.5f,2.5f,2.5f,3.5f,4.5f,4.5f,5.5f,6.5f,6.5f,7.5f,8.5f,8.5f,9.5f,10.5f))
  }
  test("latent upscale keeps exactly the requested denoising steps") {
    for (steps,strength,expectedCount) <- Vector((10,0.55,19),(10,1.0,10),(7,0.3,24),(1,0.0001,10000)) do
      val (count,effective,active)=NativeLatentUpscale.fullSteps(steps,strength)
      assertEquals(count,expectedCount)
      assertEquals(active,steps)
      assertEquals((count*effective).toInt,steps)
    for strength <- Vector(0.0,-0.1,1.1,Double.NaN,Double.PositiveInfinity,Double.MinPositiveValue) do
      intercept[IllegalArgumentException](NativeLatentUpscale.fullSteps(10,strength))
  }

package scalatorch.models

import scalatorch.*

class GuidanceRescaleSuite extends munit.FunSuite:
  test("rescale matches diffusers torch standard deviation per batch") {
    val shape=Shape.from(Seq(2L,1L,2L,2L))
    val guided=Tensor.fromFloats(Array(1f,-2f,3f,4f,4f,7f,-3f,1f),shape)
    val positive=Tensor.fromFloats(Array(2f,1f,0f,3f,-4f,3f,2f,5f),shape)
    val expected=Array(.64156502f,-1.28313005f,1.92469501f,2.5662601f,3.73847127f,6.54232502f,-2.80385351f,.93461782f)
    val actual=GuidanceRescale(guided,positive,.7f).toFloatArray
    for i <- actual.indices do assert(math.abs(actual(i)-expected(i))<1e-6,s"index $i: ${actual(i)} != ${expected(i)}")
    assertEquals(GuidanceRescale(guided,positive,0f).toFloatArray.toVector,guided.toFloatArray.toVector)
    intercept[IllegalArgumentException](GuidanceRescale(guided,positive,1.1f))
  }

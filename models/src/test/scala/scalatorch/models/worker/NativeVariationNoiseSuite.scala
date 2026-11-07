package scalatorch.models.worker

import scalatorch.{Shape,Tensor}
import scalatorch.models.VariationNoise

class NativeVariationNoiseSuite extends munit.FunSuite:
  test("variation uses channel-wise spherical interpolation") {
    val shape=Shape.from(Seq(1L,2L,1L,1L))
    val low=Tensor.fromFloats(Array(1f,0f),shape)
    val high=Tensor.fromFloats(Array(0f,1f),shape)
    val result=VariationNoise.slerp(low,high,0.25).toFloatArray
    assert(math.abs(result(0)-math.cos(math.Pi/8))<1e-6)
    assert(math.abs(result(1)-math.sin(math.Pi/8))<1e-6)
  }
  test("nearly parallel vectors follow the NUI linear fallback") {
    val shape=Shape.from(Seq(1L,2L,1L,1L))
    val low=Tensor.fromFloats(Array(1f,0f),shape)
    val high=Tensor.fromFloats(Array(2f,0f),shape)
    assertEquals(VariationNoise.slerp(low,high,0.25).toFloatArray.toVector,Vector(1.75f,0f))
  }
  test("multi-pixel channel interpolation matches the PyTorch NUI reference") {
    val shape=Shape.from(Seq(1L,4L,1L,2L))
    val low=Tensor.fromFloats(Array(1f,2f,3f,4f,-1f,1f,2f,-2f),shape)
    val high=Tensor.fromFloats(Array(2f,-1f,1f,3f,1f,2f,-2f,1f),shape)
    val expected=Vector(1.93385291f,0.97560847f,3.05644512f,4.11919308f,-0.28678459f,1.57179236f,0.57356918f,-0.97560847f)
    VariationNoise.slerp(low,high,0.37).toFloatArray.toVector.zip(expected).foreach { (actual,reference) =>
      assert(math.abs(actual-reference)<1e-6)
    }
  }

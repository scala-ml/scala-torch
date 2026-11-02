package scalatorch.models

import scalatorch.*
import scalatorch.autograd.Variable
import scalatorch.nn.*

class DiffusionResamplingSuite extends munit.FunSuite:
  private def conv(stride: Int,pad: Int): Conv2d =
    val w=Array.fill(9)(0f);w(4)=1f
    Conv2d(Parameter(Tensor.fromFloats(w,Shape.from(Seq(1,1,3,3)))),None,
      Conv2dConfig(stride=(stride,stride),padding=(pad,pad)))
  test("odd-size downsampling distinguishes symmetric and bottom-right padded geometry") {
    val input=Tensor.fromFloats((1 to 15).map(_.toFloat).toArray,Shape.from(Seq(1,1,3,5)))
    val x=Variable(input,true)
    val symmetric=DiffusionDownsample2d(conv(2,1))(x)
    assertEquals(symmetric.data.shape,Shape.from(Seq(1,1,2,3)))
    assertEquals(symmetric.data.toFloatArray.toVector,Vector(1f,3f,5f,11f,13f,15f))
    symmetric.sum().backward()
    assertEquals(x.grad.get.toFloatArray.toVector,Vector(1f,0f,1f,0f,1f,0f,0f,0f,0f,0f,1f,0f,1f,0f,1f))
    val asymmetric=DiffusionDownsample2d(conv(2,0))(Variable(input))
    assertEquals(asymmetric.data.shape,Shape.from(Seq(1,1,1,2)))
    assertEquals(asymmetric.data.toFloatArray.toVector,Vector(7f,9f))
    intercept[IllegalArgumentException](DiffusionDownsample2d(conv(1,1)))
  }
  test("upsampling targets odd skip sizes and gradients flow through convolution and resize") {
    val x=Variable(Tensor.fromFloats(Array(1f,2f,3f,4f),Shape.from(Seq(1,1,2,2))),true)
    val layer=DiffusionUpsample2d(conv(1,1))
    val out=layer(x,3,5)
    assertEquals(out.data.shape,Shape.from(Seq(1,1,3,5)))
    assertEquals(out.data.toFloatArray.toVector,Vector(1f,1f,1f,2f,2f,1f,1f,1f,2f,2f,3f,3f,3f,4f,4f))
    out.sum().backward()
    assertEquals(x.grad.get.toFloatArray.toVector,Vector(6f,4f,3f,2f))
    assert(layer.conv.weight.grad.get.toFloatArray.exists(_!=0f))
    intercept[IllegalArgumentException](DiffusionUpsample2d(conv(2,1)))
  }

package scalatorch.autograd

import scalatorch.*

class SpatialPaddingSuite extends munit.FunSuite:
  test("padding plus convolution matches explicit tiling and all input finite differences") {
    val shape=Shape.from(Seq(1,1,2,3))
    val x=Array(1f,2f,3f,4f,5f,6f)
    val weight=Variable(Tensor.fromFloats(Array(1f,2f,3f,4f,5f,6f,7f,8f,9f),Shape.from(Seq(1,1,3,3))))
    for horizontal <- PaddingMode.values; vertical <- PaddingMode.values do
      val c=Padding2d(1,1,1,1,horizontal,vertical)
      val input=Variable(Tensor.fromFloats(x,shape),true)
      val y=input.pad2d(c).conv2d(weight)
      val expected=Array.tabulate(6) { o =>
        var sum=0f
        for ky <- 0 until 3; kx <- 0 until 3 do
          val iy=o/3+ky-1;val ix=o%3+kx-1
          if (iy>=0 && iy<2 || vertical==PaddingMode.Circular) && (ix>=0 && ix<3 || horizontal==PaddingMode.Circular) then
            sum+=x(Math.floorMod(iy,2)*3+Math.floorMod(ix,3))*(ky*3+kx+1)
        sum
      }
      assertEquals(y.data.toFloatArray.toVector,expected.toVector)
      y.sum().backward()
      def score()=Conv2dOps.forward(SpatialPadding.forward(Tensor.fromFloats(x,shape),c),weight.data).toFloatArray.map(_.toDouble).sum
      for i <- x.indices do
        val saved=x(i)
        x(i)=saved+.01f;val plus=score()
        x(i)=saved-.01f;val minus=score()
        x(i)=saved
        assertEqualsDouble(input.grad.get.toFloatArray(i),(plus-minus)/.02,.01)
      assert(!Autograd.noGrad(input.pad2d(c)).requiresGrad)
  }
  test("wrapped duplicate pixels accumulate gradients, including repeated backward") {
    val x=Variable(Tensor.fromFloats(Array(2f),Shape.from(Seq(1,1,1,1))),true)
    val y=x.pad2d(Padding2d(1,1,1,1,PaddingMode.Circular,PaddingMode.Circular)).sum()
    y.backward();y.backward()
    assertEquals(x.grad.get.toFloatArray.toVector,Vector(18f))
  }

package scalatorch.autograd

import scalatorch.*

class Resize2dSuite extends munit.FunSuite:
  test("nearest resize sums repeated gradients and leaves unsampled pixels zero") {
    def input()=Variable(Tensor.fromFloats(Array(1f,2f,3f),Shape.from(Seq(1,1,1,3))),true)
    val up=input()
    up.resizeNearest2d(2,5,NearestMode.Exact).sum().backward()
    assertEquals(up.grad.get.toFloatArray.toVector,Vector(4f,2f,4f))
    val down=input()
    down.resizeNearest2d(1,2,NearestMode.Exact).sum().backward()
    assertEquals(down.grad.get.toFloatArray.toVector,Vector(1f,0f,1f))
    assert(!Autograd.noGrad(down.resizeNearest2d(1,2)).requiresGrad)
  }
  test("both nearest maps match input finite differences for mixed spatial scaling") {
    val shape=Shape.from(Seq(2,2,2,3))
    val x=Array.tabulate(24)(i => (i*.1).toFloat)
    for mode <- NearestMode.values do
      val input=Variable(Tensor.fromFloats(x,shape),true)
      val y=input.resizeNearest2d(3,2,mode)
      val seed=Array.tabulate(24)(i => math.cos(i).toFloat)
      y.backward(Tensor.fromFloats(seed,y.data.shape))
      def score()=Resize2d.nearest(Tensor.fromFloats(x,shape),3,2,mode).toFloatArray.zip(seed).map((a,b) => a.toDouble*b).sum
      for i <- x.indices do
        val saved=x(i)
        x(i)=saved+.001f;val plus=score()
        x(i)=saved-.001f;val minus=score()
        x(i)=saved
        assertEqualsDouble(input.grad.get.toFloatArray(i),(plus-minus)/.002,.0002)
  }

package scalatorch.autograd

import scalatorch.*

class QuickGeluSuite extends munit.FunSuite:
  test("QuickGELU values and derivatives match sigmoid-weighted definition") {
    val values=Array(-8f,-2f,-.5f,0f,.5f,2f,8f)
    val x=Variable(Tensor.fromFloats(values,Shape.from(Seq(values.length.toLong))),true)
    val y=x.quickGelu()
    y.sum().backward()
    for i <- values.indices do
      val v=values(i).toDouble
      val s=1/(1+math.exp(-1.702*v))
      assertEqualsDouble(y.data.toFloatArray(i),v*s,1e-6)
      assertEqualsDouble(x.grad.get.toFloatArray(i),s+1.702*v*s*(1-s),1e-6)
  }
  test("QuickGELU preserves finite extremes, limits and logical views") {
    val values=Array(-Float.MaxValue,Float.MaxValue,Float.NegativeInfinity,Float.PositiveInfinity,Float.NaN,0f)
    val x=Variable(Tensor.fromFloats(values,Shape.from(Seq(2,3))).transpose(0,1),true)
    val y=x.quickGelu()
    y.sum().backward()
    val ys=y.data.toFloatArray;val ds=x.grad.get.toFloatArray
    assertEquals(ys(0),-0f);assertEquals(ds(0),0f)
    assertEquals(ys(1),Float.PositiveInfinity);assertEquals(ds(1),1f)
    assertEquals(ys(2),Float.MaxValue);assertEquals(ds(2),1f)
    assert(ys(3).isNaN && ds(3).isNaN)
    assertEquals(ys(4),-0f);assertEquals(ds(4),0f)
    assertEquals(ys(5),0f);assertEquals(ds(5),.5f)
  }

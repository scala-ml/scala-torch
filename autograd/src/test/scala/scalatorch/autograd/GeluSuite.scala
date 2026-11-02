package scalatorch.autograd

import scalatorch.*

class GeluSuite extends munit.FunSuite:
  private def variable(values: Array[Float])=Variable(Tensor.fromFloats(values,Shape.from(Seq(values.length.toLong))),true)
  test("erf GELU matches Gaussian values and derivatives including the negative tail") {
    val x=variable(Array(-3f,-1f,0f,1f,3f,-10f))
    val y=x.gelu()
    val expected=Array(-.00404969409489028,-.15865525393145705,0.0,.8413447460685429,2.99595030590511)
    y.data.toFloatArray.take(5).zip(expected).foreach((a,e) => assertEqualsDouble(a,e,1e-7))
    assertEqualsDouble(y.data.toFloatArray.last / -7.61985302416047e-23,1.0,1e-6)
    y.sum().backward()
    val derivative=Array(-.011945647204184,-.0833154705876863,.5,1.0833154705876863,1.011945647204184)
    x.grad.get.toFloatArray.take(5).zip(derivative).foreach((a,e) => assertEqualsDouble(a,e,1e-7))
  }
  test("explicit tanh mode is distinct and both pullbacks match finite differences") {
    val values=Array(-4f,-2f,-.3f,0f,.7f,1f,3f)
    for mode <- GeluApproximation.values do
      val x=variable(values)
      x.gelu(mode).sum().backward()
      for i <- values.indices do
        val saved=values(i);val e=.001f
        values(i)=saved+e;val plus=variable(values).gelu(mode).data.toFloatArray(i)
        values(i)=saved-e;val minus=variable(values).gelu(mode).data.toFloatArray(i)
        values(i)=saved
        assertEqualsDouble(x.grad.get.toFloatArray(i),(plus.toDouble-minus)/(2*e),.0002)
    val exact=variable(Array(1f)).gelu().data.toFloatArray.head
    val approximate=variable(Array(1f)).gelu(GeluApproximation.Tanh).data.toFloatArray.head
    assert(math.abs(exact-approximate)>1e-4)
  }
  test("views, empty values, noGrad and nonfinite limits are explicit") {
    val raw=Tensor.fromFloats(Array(-1f,2f,0f,3f),Shape.from(Seq(2,2))).transpose(0,1)
    assertEquals(Variable(raw).gelu().data.toFloatArray.toVector,Variable(raw.contiguous).gelu().data.toFloatArray.toVector)
    assertEquals(variable(Array.emptyFloatArray).gelu().data.shape,Shape.from(Seq(0)))
    for mode <- GeluApproximation.values do
      val x=variable(Array(Float.NegativeInfinity,Float.PositiveInfinity,Float.NaN))
      val y=x.gelu(mode)
      assertEquals(y.data.toFloatArray(0),-0f)
      assertEquals(y.data.toFloatArray(1),Float.PositiveInfinity)
      assert(y.data.toFloatArray(2).isNaN)
      y.sum().backward()
      assertEquals(x.grad.get.toFloatArray.take(2).toVector,Vector(0f,1f))
      assert(x.grad.get.toFloatArray(2).isNaN)
      assert(!Autograd.noGrad(x.gelu(mode)).requiresGrad)
  }

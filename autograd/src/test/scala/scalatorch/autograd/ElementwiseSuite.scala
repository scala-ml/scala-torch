package scalatorch.autograd

import scalatorch.*

class ElementwiseSuite extends munit.FunSuite:
  private def tensor(xs: Float*)=Tensor.fromFloats(xs.toArray,Shape.from(Seq(xs.size)))
  test("exp preserves logical views and differentiates with the exponential") {
    val input=Variable(Tensor.fromFloats(Array(-2f,-1f,0f,.5f,1f,2f),Shape.from(Seq(2,3))))
    val output=input.transpose(0,1).exp()
    val expected=Array(-2f,.5f,-1f,1f,0f,2f).map(x => math.exp(x.toDouble).toFloat)
    output.data.toFloatArray.zip(expected).foreach((a,b) => assertEqualsFloat(a,b,1e-6f))
    output.sum().backward()
    input.grad.get.toFloatArray.zip(input.data.toFloatArray).foreach((g,x) => assertEqualsFloat(g,math.exp(x.toDouble).toFloat,1e-6f))
  }
  test("clamp passes gradients at both bounds and zeros them outside") {
    val input=Variable(tensor(-31,-30,-1,20,21))
    val output=input.clamp(-30,20)
    assertEquals(output.data.toFloatArray.toVector,Vector(-30f,-30f,-1f,20f,20f))
    output.backward(tensor(Float.PositiveInfinity,2,3,4,Float.NaN))
    assertEquals(input.grad.get.toFloatArray.toVector,Vector(0f,2f,3f,4f,0f))
    intercept[IllegalArgumentException](input.clamp(2,1))
    intercept[IllegalArgumentException](input.clamp(Float.NaN,1))
  }
  test("exp and clamp support empty, frozen and noGrad execution") {
    val empty=Variable(tensor())
    empty.exp().clamp(0,1).sum().backward()
    assertEquals(empty.grad.get.shape,empty.data.shape)
    val input=Variable(tensor(1))
    assert(!Autograd.noGrad(input.exp().clamp(0,3)).requiresGrad)
    assert(!input.detach.exp().clamp(0,3).requiresGrad)
  }
  test("exp and clamp compose with finite difference gradients away from bounds") {
    val xs=Array(-2f,-.4f,.3f,2f)
    def value(a: Array[Float])=Variable(Tensor.fromFloats(a,Shape.from(Seq(4))))
    val input=value(xs)
    input.clamp(-1,1).exp().sum().backward()
    for i <- xs.indices do
      val plus=xs.clone();plus(i)+=.001f
      val minus=xs.clone();minus(i)-=.001f
      val numeric=(value(plus).clamp(-1,1).exp().sum().data.toFloatArray.head-
        value(minus).clamp(-1,1).exp().sum().data.toFloatArray.head)/.002f
      assertEqualsFloat(input.grad.get.toFloatArray(i),numeric,.0003f)
  }

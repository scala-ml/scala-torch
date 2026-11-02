package scalatorch.nn

import scalatorch.*
import scalatorch.autograd.{Variable,Autograd}

class GroupNormSuite extends munit.FunSuite:
  test("GroupNorm affine input, scale and bias derivatives match finite differences") {
    val shape=Shape.from(Seq(2,4,2,3))
    val x=Array.tabulate(shape.numel.toInt)(i => math.sin(i*.4).toFloat)
    val scale=Array(.7f,-.4f,1.2f,.3f);val bias=Array(.1f,.2f,-.1f,.5f)
    val layer=GroupNorm(2,4,Some(Parameter(Tensor.fromFloats(scale,Shape.from(Seq(4))))),
      Some(Parameter(Tensor.fromFloats(bias,Shape.from(Seq(4))))),.01)
    val input=Variable(Tensor.fromFloats(x,shape),true)
    val y=layer(input)
    val seed=Array.tabulate(x.length)(i => math.cos(i*.3).toFloat)
    y.backward(Tensor.fromFloats(seed,shape))
    def score(): Double =
      val norm=GroupNormOps.forward(Tensor.fromFloats(x,shape),2,.01).toFloatArray
      norm.indices.map(i => (norm(i).toDouble*scale(i/6%4)+bias(i/6%4))*seed(i)).sum
    for (values,gradient) <- Seq(x -> input.grad.get,scale -> layer.weight.get.grad.get,bias -> layer.bias.get.grad.get) do
      val actual=gradient.toFloatArray
      for i <- values.indices do
        val saved=values(i);val eps=.001f
        values(i)=saved+eps;val plus=score()
        values(i)=saved-eps;val minus=score()
        values(i)=saved
        assertEqualsDouble(actual(i),(plus-minus)/(2*eps),.0005)
    assert(!Autograd.noGrad(layer(input)).requiresGrad)
  }
  test("identity initialization, affine-free mode and empty batch gradients") {
    val layer=GroupNorm.init(4,2)
    assertEquals(Module.namedParameters(layer).map(_._1),Vector("weight","bias"))
    val x=Variable(Tensor.fromFloats(Array.emptyFloatArray,Shape.from(Seq(0,4,2,2))),true)
    layer(x).sum().backward()
    assert(layer.weight.get.grad.get.toFloatArray.forall(_==0f))
    assert(layer.bias.get.grad.get.toFloatArray.forall(_==0f))
    assertEquals(x.grad.get.shape,x.data.shape)
    val plain=GroupNorm.init(4,2,affine=false)
    assertEquals(Module.parameters(plain).size,0)
    val sample=Variable(Tensor.fromFloats(Array(1f,2f,4f,8f),Shape.from(Seq(1,4))),true)
    assertEquals(layer(sample).data.toFloatArray.toVector,plain(sample).data.toFloatArray.toVector)
    intercept[IllegalArgumentException](GroupNorm.init(3,2))
  }

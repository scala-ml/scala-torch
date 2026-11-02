package scalatorch.models

import scalatorch.*
import scalatorch.autograd.{Variable,Autograd}
import scalatorch.nn.*

class DiffusionTransformerSuite extends munit.FunSuite:
  private def attention(context: Int,seed: Long)=DiffusionAttention(Linear.init(2,2,seed),Linear.init(context,2,seed+1),
    Linear.init(context,2,seed+2),Linear.init(2,2,seed+3),1)
  private def block()=DiffusionTransformerBlock(LayerNorm.init(2),attention(2,1),LayerNorm.init(2),attention(3,5),
    LayerNorm.init(2),DiffusionFeedForward(Linear.init(2,6,9),Linear.init(3,2,10)))
  test("GEGLU splits value before gate and preserves leading dimensions") {
    def linear(values: Array[Float],out: Int,in: Int)=Linear(Parameter(Tensor.fromFloats(values,Shape.from(Seq(out.toLong,in.toLong)))),None)
    val ff=DiffusionFeedForward(linear(Array(1f,0f,0f,1f,3f,0f,0f,2f),4,2),linear(Array(1f,0f,0f,1f),2,2))
    val x=Variable(Tensor.fromFloats(Array(1f,2f),Shape.from(Seq(1,1,2))))
    val result=ff(x)
    assertEquals(result.data.shape,x.data.shape)
    result.data.toFloatArray.zip(Array(2.99595030590511,7.99974663006534)).foreach((a,e) => assertEqualsDouble(a,e,5e-7))
    intercept[IllegalArgumentException](DiffusionFeedForward(Linear.init(2,5,1),Linear.init(2,2,2)))
  }
  test("full Transformer gradients match finite differences through all parameters and both inputs") {
    val model=block()
    val xs=Shape.from(Seq(1,2,2));val cs=Shape.from(Seq(1,3,3))
    val xa=Array(.2f,-.7f,.5f,.8f);val ca=Array.tabulate(9)(i => math.sin(i*.3).toFloat)
    val x=Variable(Tensor.fromFloats(xa,xs),true);val context=Variable(Tensor.fromFloats(ca,cs),true)
    val seed=Tensor.fromFloats(Array(.3f,-.2f,.4f,.7f),xs)
    model(x,context).backward(seed)
    val parameters=Module.parameters(model)
    val grads=parameters.map(p => p -> p.grad.get.toFloatArray).toMap
    def score()=Autograd.noGrad(model(Variable(Tensor.fromFloats(xa,xs)),Variable(Tensor.fromFloats(ca,cs))))
      .data.toFloatArray.zip(seed.toFloatArray).map((a,b) => a.toDouble*b).sum
    for (a,g) <- Seq(xa -> x.grad.get.toFloatArray,ca -> context.grad.get.toFloatArray) do
      for i <- a.indices do
        val saved=a(i);val e=.001f
        a(i)=saved+e;val plus=score()
        a(i)=saved-e;val minus=score()
        a(i)=saved
        assertEqualsDouble(g(i),(plus-minus)/(2*e),.003)
    for p <- parameters do
      val original=p.data;val values=original.toFloatArray
      for i <- values.indices do
        val saved=values(i);val e=.001f
        values(i)=saved+e;p.load(Tensor.fromFloats(values,original.shape));val plus=score()
        values(i)=saved-e;p.load(Tensor.fromFloats(values,original.shape));val minus=score()
        values(i)=saved
        assertEqualsDouble(grads(p)(i),(plus-minus)/(2*e),.003)
      p.load(original)
    intercept[IllegalArgumentException](model.copy(norm2=LayerNorm.init(3)))
  }
  test("zero output branches leave the identity residual and its gradient") {
    val model=block()
    for layer <- Seq(model.attn1.toOut,model.attn2.toOut,model.feedForward.output) do
      layer.weight.load(Tensor.fromFloats(Array.fill(layer.weight.data.shape.numel.toInt)(0f),layer.weight.data.shape))
      layer.bias.foreach(p => p.load(Tensor.fromFloats(Array.fill(p.data.shape.numel.toInt)(0f),p.data.shape)))
    val x=Variable(Tensor.fromFloats(Array(.2f,.4f),Shape.from(Seq(1,1,2))),true)
    val context=Variable(Tensor.fromFloats(Array(.1f,.2f,.3f),Shape.from(Seq(1,1,3))),true)
    val y=model(x,context)
    assertEquals(y.data.toFloatArray.toVector,x.data.toFloatArray.toVector)
    y.sum().backward()
    assertEquals(x.grad.get.toFloatArray.toVector,Vector(1f,1f))
    assert(context.grad.get.toFloatArray.forall(_==0f))
  }

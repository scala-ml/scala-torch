package scalatorch.models

import scalatorch.*
import scalatorch.autograd.{Variable,Autograd}
import scalatorch.nn.*

class DiffusionSpatialSuite extends munit.FunSuite:
  private def conv(values: Array[Float])=Conv2d(Parameter(Tensor.fromFloats(values,Shape.from(Seq(2,2,1,1)))))
  private def attention(context: Int,seed: Long)=DiffusionAttention(Linear.init(2,2,seed),Linear.init(context,2,seed+1),
    Linear.init(context,2,seed+2),Linear.init(2,2,seed+3),1)
  private def block()=DiffusionTransformerBlock(LayerNorm.init(2),attention(2,1),LayerNorm.init(2),attention(3,5),
    LayerNorm.init(2),DiffusionFeedForward(Linear.init(2,6,9),Linear.init(3,2,10)))
  private def model()=DiffusionSpatialTransformer(GroupNorm.init(2,1,epsilon=.01),conv(Array(.2f,.3f,-.1f,.4f)),
    Vector(block()),conv(Array(.5f,-.2f,.3f,.1f)))
  private def zeroOutputs(b: DiffusionTransformerBlock): Unit =
    for linear <- Seq(b.attn1.toOut,b.attn2.toOut,b.feedForward.output) do
      for p <- Module.parameters(linear) do p.load(Tensor.fromFloats(Array.fill(p.data.shape.numel.toInt)(0f),p.data.shape))
  test("rectangular image token round trip preserves channel and spatial order") {
    val b=block();zeroOutputs(b)
    val identity=Array(1f,0f,0f,1f)
    val m=DiffusionSpatialTransformer(GroupNorm.init(2,1,epsilon=1.0),conv(identity),Vector(b),conv(identity))
    val input=Tensor.fromFloats((1 to 12).map(_.toFloat).toArray,Shape.from(Seq(1,2,2,3)))
    val context=Variable(Tensor.fromFloats(Array(.1f,.2f,.3f),Shape.from(Seq(1,1,3))))
    val out=m(Variable(input),context).data
    assertEquals(out.shape,input.shape)
    val mean=6.5;val variance=143.0/12
    out.toFloatArray.zip(input.toFloatArray).foreach((a,x) =>
      assertEqualsDouble(a,x+(x-mean)/math.sqrt(variance+1),1e-6))
    intercept[IllegalArgumentException](m.copy(blocks=Vector.empty))
    intercept[IllegalArgumentException] {
      m.copy(projIn=Conv2d(m.projIn.weight,None,Conv2dConfig(stride=(2,2))))
    }
  }
  test("spatial Transformer differentiates every image, context and parameter value") {
    val m=model()
    val xs=Shape.from(Seq(2,2,1,2));val cs=Shape.from(Seq(2,2,3))
    val xa=Array.tabulate(8)(i => math.sin(i*.6).toFloat)
    val ca=Array.tabulate(12)(i => math.cos(i*.4).toFloat)
    val x=Variable(Tensor.fromFloats(xa,xs),true);val c=Variable(Tensor.fromFloats(ca,cs),true)
    val seed=Tensor.fromFloats(Array.tabulate(8)(i => math.cos(i*.7).toFloat),xs)
    m(x,c).backward(seed)
    val params=Module.parameters(m);val grads=params.map(p => p -> p.grad.get.toFloatArray).toMap
    def score()=Autograd.noGrad(m(Variable(Tensor.fromFloats(xa,xs)),Variable(Tensor.fromFloats(ca,cs))))
      .data.toFloatArray.zip(seed.toFloatArray).map((a,b) => a.toDouble*b).sum
    for (values,grad) <- Seq(xa -> x.grad.get.toFloatArray,ca -> c.grad.get.toFloatArray) do
      for i <- values.indices do
        val saved=values(i);val e=.001f
        values(i)=saved+e;val plus=score()
        values(i)=saved-e;val minus=score()
        values(i)=saved
        assertEqualsDouble(grad(i),(plus-minus)/(2*e),.003)
    for p <- params do
      val original=p.data;val values=original.toFloatArray
      for i <- values.indices do
        val saved=values(i);val e=.001f
        values(i)=saved+e;p.load(Tensor.fromFloats(values,original.shape));val plus=score()
        values(i)=saved-e;p.load(Tensor.fromFloats(values,original.shape));val minus=score()
        values(i)=saved
        assertEqualsDouble(grads(p)(i),(plus-minus)/(2*e),.003)
      p.load(original)
  }
  test("zero spatial output projection preserves image residual and gradient") {
    val m=model()
    m.projOut.weight.load(Tensor.fromFloats(Array.fill(4)(0f),Shape.from(Seq(2,2,1,1))))
    val x=Variable(Tensor.fromFloats(Array(.1f,.2f,.3f,.4f),Shape.from(Seq(1,2,1,2))),true)
    val c=Variable(Tensor.fromFloats(Array(.2f,.3f,.4f),Shape.from(Seq(1,1,3))),true)
    val out=m(x,c)
    assertEquals(out.data.toFloatArray.toVector,x.data.toFloatArray.toVector)
    out.sum().backward()
    assert(x.grad.get.toFloatArray.forall(_==1f))
    assert(c.grad.get.toFloatArray.forall(_==0f))
  }

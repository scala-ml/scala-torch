package scalatorch.nn

import scalatorch.*
import scalatorch.autograd.{Variable,Autograd}

class Conv2dModuleSuite extends munit.FunSuite:
  private def tensor(a: Array[Float],dims: Long*) = Tensor.fromFloats(a,Shape.from(dims))
  test("grouped convolution input, weight and bias gradients match finite differences") {
    val xs=Shape.from(Seq(2,4,4,5));val ws=Shape.from(Seq(4,2,2,2))
    val x=Array.tabulate(xs.numel.toInt)(i => (math.sin(i*.7)*.3).toFloat)
    val w=Array.tabulate(ws.numel.toInt)(i => (math.cos(i*.4)*.2).toFloat)
    val b=Array(.1f,-.2f,.3f,-.4f)
    val c=Conv2dConfig(stride=(2,1),padding=(1,1),dilation=(1,2),groups=2)
    val input=Variable(Tensor.fromFloats(x,xs),true)
    val layer=Conv2d(Parameter(Tensor.fromFloats(w,ws)),Some(Parameter(tensor(b,4))),c)
    assertEquals(Module.namedParameters(layer).map(_._1),Vector("weight","bias"))
    val y=layer(input)
    val seed=Array.tabulate(y.data.shape.numel.toInt)(i => math.sin(i*.3).toFloat)
    y.backward(Tensor.fromFloats(seed,y.data.shape))
    def score(): Double =
      val out=Conv2dOps.forward(Tensor.fromFloats(x,xs),Tensor.fromFloats(w,ws),c)
      val area=(out.shape.dims(2)*out.shape.dims(3)).toInt
      out.toFloatArray.indices.map(i => (out.toFloatArray(i).toDouble+b((i/area)%4))*seed(i)).sum
    for (values,gradient) <- Seq(x -> input.grad.get,w -> layer.weight.grad.get,b -> layer.bias.get.grad.get) do
      val actual=gradient.toFloatArray
      for i <- values.indices do
        val saved=values(i);val epsilon=.001f
        values(i)=saved+epsilon;val plus=score()
        values(i)=saved-epsilon;val minus=score()
        values(i)=saved
        assertEqualsDouble(actual(i),(plus-minus)/(2*epsilon),.0002)
    assert(!Autograd.noGrad(layer(input)).requiresGrad)
  }
  test("empty batches have zero parameter gradients and replaced weights invalidate saved graphs") {
    val layer=Conv2d(Parameter(tensor(Array(2f),1,1,1,1)),Some(Parameter(tensor(Array(3f),1))))
    val empty=Variable(tensor(Array.emptyFloatArray,0,1,2,2),true)
    layer(empty).sum().backward()
    assertEquals(layer.weight.grad.get.toFloatArray.toVector,Vector(0f))
    assertEquals(layer.bias.get.grad.get.toFloatArray.toVector,Vector(0f))
    assertEquals(empty.grad.get.shape,empty.data.shape)
    val saved=layer(Variable(tensor(Array(1f),1,1,1,1),true)).sum()
    layer.weight.load(tensor(Array(4f),1,1,1,1))
    intercept[IllegalArgumentException](saved.backward())
  }

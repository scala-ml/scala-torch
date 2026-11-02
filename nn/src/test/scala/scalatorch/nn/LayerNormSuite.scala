package scalatorch.nn

import scalatorch.*
import scalatorch.autograd.{Variable,Autograd}

class LayerNormSuite extends munit.FunSuite:
  test("final-axis normalization isolates tokens and preserves strided and rank-one inputs") {
    val input=Tensor.fromFloats(Array(1f,3f,10f,14f,5f,7f,20f,24f),Shape.from(Seq(2,2,2)))
    val layer=LayerNorm.init(2,epsilon=1.0)
    val expected=Vector(-1/math.sqrt(2),1/math.sqrt(2),-2/math.sqrt(5),2/math.sqrt(5),
      -1/math.sqrt(2),1/math.sqrt(2),-2/math.sqrt(5),2/math.sqrt(5))
    layer(Variable(input)).data.toFloatArray.zip(expected).foreach((a,e) => assertEqualsDouble(a,e,1e-7))
    val view=input.transpose(1,2)
    assertEquals(layer(Variable(view)).data.toFloatArray.toVector,layer(Variable(view.contiguous)).data.toFloatArray.toVector)
    val one=Variable(Tensor.fromFloats(Array(1f,3f),Shape.from(Seq(2))),true)
    assertEquals(layer(one).data.shape,one.data.shape)
    val plain=LayerNorm.init(2,epsilon=1.0,affine=false)
    assertEquals(plain(one).data.toFloatArray.toVector,layer(one).data.toFloatArray.toVector)
    intercept[IllegalArgumentException](plain(Variable(Tensor.fromFloats(Array(1f),Shape.from(Seq(1))))))
  }
  test("all input, scale and bias gradients match finite differences") {
    val shape=Shape.from(Seq(2,2,3))
    val values=Array.tabulate(12)(i => math.sin(i*.4).toFloat)
    val scale=Array(.4f,-.2f,.8f);val bias=Array(.1f,.3f,-.1f)
    val layer=LayerNorm(3,Some(Parameter(Tensor.fromFloats(scale,Shape.from(Seq(3))))),
      Some(Parameter(Tensor.fromFloats(bias,Shape.from(Seq(3))))),.01)
    val x=Variable(Tensor.fromFloats(values,shape),true)
    val seed=Tensor.fromFloats(Array.tabulate(12)(i => math.cos(i*.3).toFloat),shape)
    layer(x).backward(seed)
    val dx=x.grad.get.toFloatArray
    val dw=layer.weight.get.grad.get.toFloatArray;val db=layer.bias.get.grad.get.toFloatArray
    def score(): Double =
      val candidate=LayerNorm(3,Some(Parameter(Tensor.fromFloats(scale,Shape.from(Seq(3))))),
        Some(Parameter(Tensor.fromFloats(bias,Shape.from(Seq(3))))),.01)
      Autograd.noGrad(candidate(Variable(Tensor.fromFloats(values,shape)))).data.toFloatArray.zip(seed.toFloatArray)
        .map((a,b) => a.toDouble*b).sum
    for (array,grad) <- Seq(values -> dx,scale -> dw,bias -> db) do
      for i <- array.indices do
        val saved=array(i);val e=.001f
        array(i)=saved+e;val plus=score()
        array(i)=saved-e;val minus=score()
        array(i)=saved
        assertEqualsDouble(grad(i),(plus-minus)/(2*e),.0005)
    assert(!Autograd.noGrad(layer(x)).requiresGrad)
  }
  test("empty leading dimensions have zero affine gradients and stale graphs reject") {
    val layer=LayerNorm.init(3)
    val empty=Variable(Tensor.fromFloats(Array.emptyFloatArray,Shape.from(Seq(2,0,3))),true)
    layer(empty).sum().backward()
    assertEquals(empty.grad.get.shape,empty.data.shape)
    assert(layer.weight.get.grad.get.toFloatArray.forall(_==0))
    assert(layer.bias.get.grad.get.toFloatArray.forall(_==0))
    val x=Variable(Tensor.fromFloats(Array(1f,2f,4f),Shape.from(Seq(3))),true)
    val graph=layer(x).sum()
    layer.weight.get.load(Tensor.fromFloats(Array(2f,2f,2f),Shape.from(Seq(3))))
    intercept[IllegalArgumentException](graph.backward())
    intercept[IllegalArgumentException](LayerNorm.init(0))
    intercept[IllegalArgumentException](LayerNorm.init(3,epsilon=Double.NaN))
    intercept[IllegalArgumentException](Variable(Tensor.fromFloats(Array(1f),Shape.from(Seq.empty))).layerNorm())
  }

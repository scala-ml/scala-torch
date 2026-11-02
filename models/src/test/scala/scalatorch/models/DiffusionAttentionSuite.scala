package scalatorch.models

import scalatorch.*
import scalatorch.autograd.{Variable,Autograd}
import scalatorch.nn.*

class DiffusionAttentionSuite extends munit.FunSuite:
  private def model()=DiffusionAttention(Linear.init(3,4,1),Linear.init(5,4,2),
    Linear.init(5,4,3),Linear.init(4,3,4),2)
  test("cross attention propagates through all projected parameters") {
    val m=model()
    val x=Variable(Tensor.fromFloats(Array.tabulate(6)(i => math.sin(i).toFloat),Shape.from(Seq(1,2,3))),true)
    val c=Variable(Tensor.fromFloats(Array.tabulate(15)(i => math.cos(i).toFloat),Shape.from(Seq(1,3,5))),true)
    val y=m(x,Some(c))
    assertEquals(y.data.shape,x.data.shape)
    val seed=Tensor.fromFloats(Array(.1f,.2f,-.3f,.4f,-.5f,.6f),y.data.shape)
    y.backward(seed)
    assert(x.grad.get.toFloatArray.forall(_.isFinite))
    assert(c.grad.get.toFloatArray.exists(g => math.abs(g)>1e-7f))
    for (original,other,query) <- Seq((x,c,true),(c,x,false)) do
      val values=original.data.toFloatArray
      val grad=original.grad.get.toFloatArray
      def inputScore(): Double =
        val candidate=Variable(Tensor.fromFloats(values,original.data.shape))
        val out=if query then Autograd.noGrad(m(candidate,Some(other))) else Autograd.noGrad(m(other,Some(candidate)))
        out.data.toFloatArray.zip(seed.toFloatArray).map((a,b) => a.toDouble*b).sum
      for i <- values.indices do
        val saved=values(i);val e=.001f
        values(i)=saved+e;val plus=inputScore()
        values(i)=saved-e;val minus=inputScore()
        values(i)=saved
        assertEqualsDouble(grad(i),(plus-minus)/(2*e),.0003)
    intercept[IllegalArgumentException](m.copy(maxScoreElements=11)(x,Some(c)))
    val grads=Module.parameters(m).map(p => p -> p.grad.get.toFloatArray).toMap
    def score()=Autograd.noGrad(m(x,Some(c))).data.toFloatArray.zip(seed.toFloatArray).map((a,b) => a.toDouble*b).sum
    for p <- Module.parameters(m) do
      val original=p.data;val values=original.toFloatArray
      for i <- values.indices do
        val saved=values(i);val e=.001f
        values(i)=saved+e;p.load(Tensor.fromFloats(values,original.shape));val plus=score()
        values(i)=saved-e;p.load(Tensor.fromFloats(values,original.shape));val minus=score()
        values(i)=saved
        assertEqualsDouble(grads(p)(i),(plus-minus)/(2*e),.0003)
      p.load(original)
    intercept[IllegalArgumentException](m(x))
    intercept[IllegalArgumentException](m.copy(heads=3))
  }
  test("default self context agrees with explicit context and shared inputs accumulate") {
    val m=DiffusionAttention(Linear.init(3,4,1),Linear.init(3,4,2),Linear.init(3,4,3),Linear.init(4,3,4),2)
    val tensor=Tensor.fromFloats(Array.tabulate(6)(_.toFloat),Shape.from(Seq(1,2,3)))
    val shared=Variable(tensor,true)
    val explicit=Variable(tensor,true)
    val a=m(shared);val b=m(explicit,Some(explicit))
    assertEquals(a.data.toFloatArray.toVector,b.data.toFloatArray.toVector)
    a.sum().backward();b.sum().backward()
    assertEquals(shared.grad.get.toFloatArray.toVector,explicit.grad.get.toFloatArray.toVector)
  }

package scalatorch.models

import scalatorch.*
import scalatorch.autograd.{Variable,Autograd}
import scalatorch.nn.*

class DiffusionTimeSuite extends munit.FunSuite:
  private def times(values: Float*)=Tensor.fromFloats(values.toArray,Shape.from(Seq(values.size.toLong)))
  test("fractional sinusoidal features respect frequencies, order, phase scale and odd padding") {
    val input=times(0f,.5f,-2f)
    val y=DiffusionTimesteps.sinusoidal(input,5,frequencyShift=0,phaseScale=2,maxPeriod=100)
    for row <- 0 until 3 do
      val t=input.toFloatArray(row)*2.0
      val expected=Vector(math.sin(t),math.sin(t*.1),math.cos(t),math.cos(t*.1),0.0)
      y.toFloatArray.slice(row*5,row*5+5).zip(expected).foreach((a,e) => assertEqualsDouble(a,e,1e-7))
    val flipped=DiffusionTimesteps.sinusoidal(input,4,flipSinToCos=true,frequencyShift=1,maxPeriod=100)
    val t=.5
    flipped.toFloatArray.slice(4,8).zip(Vector(math.cos(t),math.cos(t*.01),math.sin(t),math.sin(t*.01)))
      .foreach((a,e) => assertEqualsDouble(a,e,1e-7))
  }
  test("features validate finite values, geometry and allocation budget including empty batches") {
    assertEquals(DiffusionTimesteps.sinusoidal(times(),4).shape,Shape.from(Seq(0,4)))
    intercept[IllegalArgumentException](DiffusionTimesteps.sinusoidal(times(1),1))
    intercept[IllegalArgumentException](DiffusionTimesteps.sinusoidal(times(1),4,frequencyShift=2))
    intercept[IllegalArgumentException](DiffusionTimesteps.sinusoidal(times(Float.NaN),4))
    intercept[IllegalArgumentException](DiffusionTimesteps.sinusoidal(times(1),4,phaseScale=Double.PositiveInfinity))
    intercept[IllegalArgumentException](DiffusionTimesteps.sinusoidal(times(1),4,maxPeriod=0))
    intercept[IllegalArgumentException](DiffusionTimesteps.sinusoidal(times(1,2),4,maxElements=7))
  }
  test("learned timestep MLP has finite-difference gradients for all parameters") {
    val model=DiffusionTimeEmbedding(Linear.init(4,5,12),Linear.init(5,3,13))
    val features=Variable(DiffusionTimesteps.sinusoidal(times(.5f,10f),4))
    val seed=Tensor.fromFloats(Array(.2f,-.1f,.3f,.4f,-.2f,.5f),Shape.from(Seq(2,3)))
    model(features).backward(seed)
    val saved=Module.parameters(model).map(p => p -> p.grad.get.toFloatArray).toMap
    def score()=Autograd.noGrad(model(features)).data.toFloatArray.zip(seed.toFloatArray).map((a,b) => a.toDouble*b).sum
    for p <- Module.parameters(model) do
      val original=p.data;val values=original.toFloatArray
      for i <- values.indices do
        val value=values(i);val e=.001f
        values(i)=value+e;p.load(Tensor.fromFloats(values,original.shape));val plus=score()
        values(i)=value-e;p.load(Tensor.fromFloats(values,original.shape));val minus=score()
        values(i)=value
        assertEqualsDouble(saved(p)(i),(plus-minus)/(2*e),.0002)
      p.load(original)
    assertEquals(Module.namedParameters(model).map(_._1),Vector("linear1.weight","linear1.bias","linear2.weight","linear2.bias"))
    intercept[IllegalArgumentException](DiffusionTimeEmbedding(Linear.init(4,5,1),Linear.init(6,3,2)))
  }
  test("image residual loss propagates through the learned timestep MLP") {
    val mlp=DiffusionTimeEmbedding(Linear.init(4,5,12),Linear.init(5,3,13))
    def conv(seed: Int)=Conv2d(Parameter(Tensor.fromFloats(
      Array.tabulate(36)(i => (math.sin(i+seed)*.1).toFloat),Shape.from(Seq(2,2,3,3)))),
      None,Conv2dConfig(padding=(1,1)))
    val block=DiffusionResnet2d(GroupNorm.init(2,1),conv(1),Some(Linear.init(3,2,16)),
      GroupNorm.init(2,1),conv(2))
    val input=Variable(Tensor.fromFloats(Array.tabulate(16)(i => math.sin(i*.3).toFloat),Shape.from(Seq(2,2,2,2))))
    val embedding=mlp(Variable(DiffusionTimesteps.sinusoidal(times(.5f,10f),4)))
    val image=block(input,Some(embedding))
    (image*image).mean().backward()
    val gradients=Module.parameters(mlp).flatMap(_.grad.get.toFloatArray)
    assert(gradients.forall(_.isFinite))
    assert(gradients.exists(g => math.abs(g)>1e-7f))
    val empty=Variable(DiffusionTimesteps.sinusoidal(times(),4))
    assertEquals(mlp(empty).data.shape,Shape.from(Seq(0,3)))
  }

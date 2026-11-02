package scalatorch.models

import scalatorch.*
import scalatorch.autograd.{Variable,Autograd}
import scalatorch.nn.*

class DiffusionResnetSuite extends munit.FunSuite:
  private def conv(in: Int,out: Int,kernel: Int,zero: Boolean = false): Conv2d =
    val shape=Shape.from(Seq(out.toLong,in.toLong,kernel.toLong,kernel.toLong))
    val values=Array.tabulate(shape.numel.toInt)(i => if zero then 0f else (math.sin(i*.7+1)*.1).toFloat)
    Conv2d(Parameter(Tensor.fromFloats(values,shape)),Some(Parameter(Tensor.fromFloats(Array.fill(out)(.03f),Shape.from(Seq(out.toLong))))),
      Conv2dConfig(padding=(kernel/2,kernel/2)))
  private def block()=DiffusionResnet2d(GroupNorm.init(2,1,epsilon=.01),conv(2,4,3),
    Some(Linear.init(3,4,17)),GroupNorm.init(4,2,epsilon=.01),conv(4,4,3),Some(conv(2,4,1)),2.0)
  test("conditioned block differentiates all inputs, embeddings and parameters") {
    val model=block()
    val shape=Shape.from(Seq(2,2,2,2));val timeShape=Shape.from(Seq(2,3))
    val xs=Array.tabulate(16)(i => math.cos(i*.3).toFloat)
    val ts=Array(.2f,-.4f,.3f,.1f,.7f,-.2f)
    val x=Variable(Tensor.fromFloats(xs,shape),true)
    val time=Variable(Tensor.fromFloats(ts,timeShape),true)
    val y=model(x,Some(time))
    assertEquals(y.data.shape,Shape.from(Seq(2,4,2,2)))
    val seed=Array.tabulate(32)(i => math.sin(i*.4).toFloat)
    y.backward(Tensor.fromFloats(seed,y.data.shape))
    val named=Module.namedParameters(model)
    assert(named.exists(_._1=="timeEmbProj.weight"))
    assert(named.exists(_._1=="convShortcut.weight"))
    val gradients=named.map((_,p) => p -> p.grad.get.toFloatArray).toMap
    def score(): Double = Autograd.noGrad {
      model(Variable(Tensor.fromFloats(xs,shape)),Some(Variable(Tensor.fromFloats(ts,timeShape)))).data.toFloatArray
        .zip(seed).map((a,b) => a.toDouble*b).sum
    }
    for (values,gradient) <- Seq(xs -> x.grad.get.toFloatArray,ts -> time.grad.get.toFloatArray) do
      for i <- values.indices do
        val saved=values(i);val e=.001f
        values(i)=saved+e;val plus=score()
        values(i)=saved-e;val minus=score()
        values(i)=saved
        assertEqualsDouble(gradient(i),(plus-minus)/(2*e),.002)
    for p <- Module.parameters(model) do
      val original=p.data;val values=original.toFloatArray
      for i <- values.indices do
        val saved=values(i);val e=.001f
        values(i)=saved+e;p.load(Tensor.fromFloats(values,original.shape));val plus=score()
        values(i)=saved-e;p.load(Tensor.fromFloats(values,original.shape));val minus=score()
        values(i)=saved
        assertEqualsDouble(gradients(p)(i),(plus-minus)/(2*e),.002)
      p.load(original)
    assert(!Autograd.noGrad(model(x,Some(time))).requiresGrad)
  }
  test("zero residual preserves the scaled shortcut and invalid conditioning rejects") {
    val c=conv(2,2,3,true)
    c.bias.get.load(Tensor.fromFloats(Array(0f,0f),Shape.from(Seq(2))))
    val model=DiffusionResnet2d(GroupNorm.init(2,1),conv(2,2,3),None,GroupNorm.init(2,1),c,None,2.0)
    val x=Variable(Tensor.fromFloats(Array(1f,2f,3f,4f,5f,6f,7f,8f),Shape.from(Seq(1,2,2,2))),true)
    assertEquals(model(x).data.toFloatArray.toVector,x.data.toFloatArray.map(_*.5f).toVector)
    model(x).sum().backward()
    assertEquals(x.grad.get.toFloatArray.toVector,Vector.fill(8)(.5f))
    intercept[IllegalArgumentException](block()(x))
    intercept[IllegalArgumentException](model(x,Some(Variable(Tensor.fromFloats(Array(1f),Shape.from(Seq(1,1)))))))
    intercept[IllegalArgumentException](block().copy(convShortcut=None))
    intercept[IllegalArgumentException](model.copy(outputScale=Double.NaN))
    intercept[IllegalArgumentException](model.copy(conv1=conv(2,2,1)))
  }
  test("timestep projection is added before second normalization and activation") {
    val first=conv(2,2,3,true)
    first.bias.get.load(Tensor.fromFloats(Array(0f,0f),Shape.from(Seq(2))))
    val values=new Array[Float](36)
    values(4)=1f;values(31)=1f
    val second=Conv2d(Parameter(Tensor.fromFloats(values,Shape.from(Seq(2,2,3,3)))),None,Conv2dConfig(padding=(1,1)))
    val projection=Linear(Parameter(Tensor.fromFloats(Array(1f,0f,0f,1f),Shape.from(Seq(2,2)))),None)
    val model=DiffusionResnet2d(GroupNorm.init(2,1),first,Some(projection),GroupNorm.init(2,1,epsilon=.1),second)
    val x=Variable(Tensor.fromFloats(Array.fill(4)(0f),Shape.from(Seq(1,2,1,2))))
    val time=Variable(Tensor.fromFloats(Array(0f,2f),Shape.from(Seq(1,2))))
    val delta=(2.0/(1+math.exp(-2)))/2
    val positive=delta/math.sqrt(delta*delta+.1)
    def silu(v: Double)=v/(1+math.exp(-v))
    val expected=Vector(silu(-positive),silu(-positive),silu(positive),silu(positive))
    model(x,Some(time)).data.toFloatArray.zip(expected).foreach((a,e) => assertEqualsDouble(a,e,1e-6))
    val wrongBatch=Variable(Tensor.fromFloats(Array.fill(4)(0f),Shape.from(Seq(2,2))))
    intercept[IllegalArgumentException](model(x,Some(wrongBatch)))
  }

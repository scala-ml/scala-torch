package scalatorch.models

import scalatorch.*
import scalatorch.autograd.{Variable,Autograd}

class VaePosteriorSuite extends munit.FunSuite:
  private def moments(xs: Array[Float],batch: Int=1)=Variable(Tensor.fromFloats(xs,Shape.from(Seq(batch,2,1,xs.length/(batch*2)))))
  test("posterior separates mean and clipped variance with explicit noise") {
    val input=moments(Array(1f,2f,3f,4f,-40f,0f,math.log(4).toFloat,30f))
    val p=VaePosterior(input)
    assertEquals(p.mean.data.toFloatArray.toVector,Vector(1f,2f,3f,4f))
    assertEquals(p.mode.data.toFloatArray.toVector,p.mean.data.toFloatArray.toVector)
    assertEquals(p.logVariance.data.toFloatArray.toVector,Vector(-30f,0f,math.log(4).toFloat,20f))
    val variance=p.variance.data.toFloatArray
    assertEqualsFloat(variance(0),math.exp(-30).toFloat,1e-19f)
    assertEqualsFloat(variance(2),4f,1e-6f)
    val noise=Tensor.fromFloats(Array(0f,1f,-1f,0f),p.mean.data.shape)
    assertEquals(p.sample(noise).data.toFloatArray.toVector,Vector(1f,3f,1f,4f))
  }
  test("KL reduces each image independently and reparameterization gradients reach moments") {
    val input=moments(Array(1f,2f,0f,0f,3f,4f,0f,0f),2)
    val p=VaePosterior(input)
    assertEquals(p.kl().data.shape,Shape.from(Seq(2)))
    assertEquals(p.kl().data.toFloatArray.toVector,Vector(2.5f,12.5f))
    val noise=Tensor.fromFloats(Array(2f,-2f,4f,-4f),p.mean.data.shape)
    (p.sample(noise).sum()+p.kl().sum()).backward()
    assertEquals(input.grad.get.toFloatArray.toVector,Vector(2f,3f,1f,-1f,4f,5f,2f,-2f))
  }
  test("posterior sample passes finite differences including clipped tails and KL has analytical gradients") {
    val xs=Array(.1f,.2f,-.4f,.3f,-31f,-.4f,.7f,21f)
    val noise=Tensor.fromFloats(Array(.2f,.4f,-.3f,0f),Shape.from(Seq(1,1,1,4)))
    // Keep noise zero on the positive clipped tail to isolate its zero derivative.
    val input=moments(xs)
    val p=VaePosterior(input)
    p.sample(noise).sum().backward()
    for i <- xs.indices do
      val plus=xs.clone();plus(i)+=.001f
      val minus=xs.clone();minus(i)-=.001f
      val numeric=(VaePosterior(moments(plus)).sample(noise).sum().data.toFloatArray.head-
        VaePosterior(moments(minus)).sample(noise).sum().data.toFloatArray.head)/.002f
      assertEqualsFloat(input.grad.get.toFloatArray(i),numeric,.0001f)
    val ordinary=moments(Array(.2f,-.3f,.4f,-.5f))
    VaePosterior(ordinary).kl().sum().backward()
    val expected=Vector(.2f,-.3f,((math.exp(.4)-1)*.5).toFloat,((math.exp(-.5)-1)*.5).toFloat)
    ordinary.grad.get.toFloatArray.zip(expected).foreach((a,b) => assertEqualsFloat(a,b,1e-6f))
  }
  test("seeded posterior sampling is reproducible and supplied noise is exact shape") {
    val p=VaePosterior(moments(Array.fill(12)(0f)))
    assertEquals(p.sample(42L).data.toFloatArray.toVector,p.sample(42L).data.toFloatArray.toVector)
    assert(p.sample(42L).data.toFloatArray.toVector!=p.sample(43L).data.toFloatArray.toVector)
    intercept[IllegalArgumentException](p.sample(Tensor.fromFloats(Array(1f),Shape.from(Seq(1)))))
    intercept[IllegalArgumentException](VaePosterior(Variable(Tensor.fromFloats(Array(1f,2f,3f),Shape.from(Seq(1,3,1,1))))))
    intercept[IllegalArgumentException](VaePosterior(Variable(Tensor.fromFloats(Array(1f,2f),Shape.from(Seq(2))))))
    val frozen=Autograd.noGrad(VaePosterior(moments(Array(1f,0f))).sample(42L))
    assert(!frozen.requiresGrad)
    val empty=VaePosterior(Variable(Tensor.fromFloats(Array.emptyFloatArray,Shape.from(Seq(0,2,1,1)))))
    assertEquals(empty.kl().data.shape,Shape.from(Seq(0)))
    assertEquals(empty.sample(42L).data.shape,Shape.from(Seq(0,1,1,1)))
  }

package scalatorch.autograd

import scalatorch.*

class LossSuite extends munit.FunSuite:
  private def tensor(ds: Seq[Long], values: Float*) = Tensor.fromFloats(values.toArray,Shape.from(ds))
  private def targets(ds: Seq[Long], values: Long*) = Tensor.fromLongs(values.toArray,Shape.from(ds))
  private def near(actual: Array[Float], expected: Double*): Unit =
    assertEquals(actual.length,expected.size)
    actual.zip(expected).foreach((a,e) => assertEqualsDouble(a,e,1e-6))
  test("known loss and unreduced upstream gradients") {
    val x = Variable(tensor(Seq(2,2),math.log(2).toFloat,0,0,0))
    val y = x.crossEntropy(targets(Seq(2),0,1),LossReduction.None)
    assertEquals(y.data.shape,Shape.from(Seq(2L)))
    near(y.data.toFloatArray,math.log(1.5),math.log(2))
    y.backward(tensor(Seq(2),2,-3))
    near(x.grad.get.toFloatArray,-2.0/3,2.0/3,-1.5,1.5)
  }
  test("ignored rows do not contribute to sum mean or gradients") {
    val x = Variable(tensor(Seq(3,3),Seq.fill(9)(0f)*))
    val ids = targets(Seq(3),1,-100,2)
    val mean = x.crossEntropy(ids)
    near(mean.data.toFloatArray,math.log(3))
    near(x.crossEntropy(ids,LossReduction.Sum).data.toFloatArray,2*math.log(3))
    mean.backward()
    near(x.grad.get.toFloatArray,1.0/6,-1.0/3,1.0/6,0,0,0,1.0/6,1.0/6,-1.0/3)
  }
  test("strided batched logits and targets match every-coordinate central differences") {
    val values = Array(.2f,-.3f,.8f,.4f,-.6f,.7f,.1f,-.2f,.5f,.9f,-.4f,.3f)
    val shape = Shape.from(Seq(2L,3L,2L))
    val ids = targets(Seq(2,2),0,1,2,-7).transpose(0,1)
    val seed = tensor(Seq(2,2),.3f,.7f,-.2f,.5f)
    val x = Variable(Tensor.fromFloats(values,shape))
    x.transpose(1,2).crossEntropy(ids,LossReduction.None,ignoreIndex = -7).backward(seed)
    val gradient = x.grad.get.toFloatArray
    def objective(v: Array[Float]): Double =
      val loss = Variable(Tensor.fromFloats(v,shape),requiresGrad=false).transpose(1,2).crossEntropy(ids,LossReduction.None,ignoreIndex = -7)
      loss.data.toFloatArray.zip(seed.toFloatArray).map((a,b) => a.toDouble*b).sum
    values.indices.foreach { i =>
      val plus=values.clone(); val minus=values.clone(); plus(i)+=.001f; minus(i)-=.001f
      assertEqualsDouble(gradient(i),(objective(plus)-objective(minus))/.002,2e-4)
    }
  }
  test("huge equal logits retain normalization and singleton classes have zero loss") {
    val x = Variable(tensor(Seq(2),Float.MaxValue,Float.MaxValue))
    val loss = x.crossEntropy(targets(Seq.empty,1))
    near(loss.data.toFloatArray,math.log(2)); loss.backward()
    near(x.grad.get.toFloatArray,.5,-.5)
    near(Variable(tensor(Seq(1),-1000)).crossEntropy(targets(Seq.empty,0)).data.toFloatArray,0)
  }
  test("empty and all ignored reductions have explicit values and zero gradients") {
    val x = Variable(tensor(Seq(2,2),1,2,3,4))
    val ids = targets(Seq(2),-100,-100)
    val mean=x.crossEntropy(ids)
    assert(mean.data.toFloatArray.head.isNaN); mean.backward()
    near(x.grad.get.toFloatArray,0,0,0,0)
    near(x.crossEntropy(ids,LossReduction.Sum).data.toFloatArray,0)
    near(x.crossEntropy(ids,LossReduction.None).data.toFloatArray,0,0)
    val empty=Variable(tensor(Seq(0,3)))
    val e=targets(Seq(0))
    val em=empty.crossEntropy(e);assert(em.data.toFloatArray.head.isNaN);em.backward()
    assertEquals(empty.grad.get.toFloatArray.length,0)
    near(empty.crossEntropy(e,LossReduction.Sum).data.toFloatArray,0)
    assertEquals(empty.crossEntropy(e,LossReduction.None).data.shape,Shape.from(Seq(0L)))
  }
  test("invalid shapes targets and nonfinite logits reject; noGrad and versions remain enforced") {
    val x=Variable(tensor(Seq(2,2),1,2,3,4))
    for ids <- Seq(targets(Seq(2),0,2),targets(Seq(2),Long.MaxValue,0),targets(Seq(2),-1,0),targets(Seq(1),0)) do
      intercept[IllegalArgumentException](x.crossEntropy(ids))
    intercept[IllegalArgumentException](Variable(tensor(Seq.empty,1)).crossEntropy(targets(Seq.empty,0)))
    intercept[IllegalArgumentException](Variable(tensor(Seq(2,0))).crossEntropy(targets(Seq(2),0,0)))
    intercept[IllegalArgumentException](Variable(tensor(Seq(1,2),Float.NaN,0)).crossEntropy(targets(Seq(1),-100)))
    val ids=targets(Seq(2),0,1)
    assert(!Autograd.noGrad { x.crossEntropy(ids) }.requiresGrad)
    val saved=x.crossEntropy(ids)
    x.replaced(x.data)
    intercept[IllegalArgumentException](saved.backward())
  }

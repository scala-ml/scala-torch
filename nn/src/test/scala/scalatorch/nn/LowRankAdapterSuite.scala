package scalatorch.nn

import munit.FunSuite
import scalatorch.*
import scalatorch.autograd.Variable

class LowRankAdapterSuite extends FunSuite:
  test("convolution adapter applies a spatial down kernel and restores scope") {
    def tensor4(values: Array[Float],dims: Long*)=Tensor.fromFloats(values,Shape.from(dims))
    val weight=Parameter(tensor4(Array.fill(9)(0f),1,1,3,3),false)
    val down=Parameter(tensor4(Array.fill(9)(1f),1,1,3,3),false)
    val up=Parameter(tensor4(Array(2f),1,1,1,1),false)
    val layer=Conv2d(weight,None,Conv2dConfig(padding=(1,1)))
    val input=Variable(tensor4(Array.fill(9)(1f),1,1,3,3),false)
    val adapter=LowRankAdapter(down,up,1.0)
    val actual=LinearAdapterScope.withAdapters(Map(weight->Vector(adapter->0.5f)))(layer(input).data.toFloatArray.toVector)
    assertEquals(actual,Vector(4f,6f,4f,6f,9f,6f,4f,6f,4f))
    assert(layer(input).data.toFloatArray.forall(_==0f))
  }
  private def tensor(values: Float*)(rows: Int, cols: Int) =
    Tensor.fromFloats(values.toArray, Shape.from(Seq(rows.toLong, cols.toLong)))

  test("scaled low-rank branch has analytical input and factor gradients") {
    val a = Parameter(tensor(2f, -1f)(1, 2))
    val b = Parameter(tensor(3f, 4f)(2, 1))
    val adapter = LowRankAdapter(a, b, 2.0)
    val layer = Linear(Parameter(tensor(1f, 0f, 0f, 1f)(2, 2), false), None, Some(adapter))
    val x = Variable(tensor(5f, 2f)(1, 2), true)
    val y = layer(x)
    assertEquals(y.data.toFloatArray.toSeq, Seq(53f, 66f))
    y.sum().backward()
    assertEquals(x.grad.get.toFloatArray.toSeq, Seq(29f, -13f))
    assertEquals(a.grad.get.toFloatArray.toSeq, Seq(70f, 28f))
    assertEquals(b.grad.get.toFloatArray.toSeq, Seq(16f, 16f))
    assertEquals(layer.weight.grad, None)
  }

  test("zero initialization preserves output and optimizer only updates adapters") {
    val base = Linear.init(2, 3, 9L)
    val layer = base.withLoRA(2, 4.0, 42L)
    val x = Variable(tensor(1f, 2f)(1, 2), false)
    assertEquals(layer(x).data.toFloatArray.toSeq, base(x).data.toFloatArray.toSeq)
    val original = layer.weight.data.toFloatArray.toSeq
    val bias = layer.bias.get.data.toFloatArray.toSeq
    val optimizer = new AdamW(Module.parameters(layer), learningRate = 0.01)
    layer(x).sum().backward()
    optimizer.step()
    assertEquals(layer.weight.data.toFloatArray.toSeq, original)
    assertEquals(layer.bias.get.data.toFloatArray.toSeq, bias)
    assert(base.weight.trainable)
    assert(layer.weight ne base.weight)
    assert(layer.lora.get.b.data.toFloatArray.exists(_ != 0f))
    assertEquals(Module.parameters(layer).count(_.trainable), 2)
    intercept[IllegalArgumentException](layer.withLoRA(1, 1.0, 0L))
  }

  test("invalid shapes, dimensions and unrepresentable scales are rejected") {
    for alpha <- Seq(0.0, -1.0, Double.NaN, Double.PositiveInfinity, Double.MaxValue, Double.MinPositiveValue) do
      intercept[IllegalArgumentException](LowRankAdapter.init(2, 3, 1, alpha, 0L))
    intercept[IllegalArgumentException](LowRankAdapter.init(2, 3, 0, 1.0, 0L))
    intercept[IllegalArgumentException](LowRankAdapter.init(Int.MaxValue, 3, 2, 1.0, 0L))
    val adapter = LowRankAdapter.init(2, 3, 1, 1.0, 0L)
    intercept[IllegalArgumentException](Linear(Linear.init(4, 3, 0L).weight, None, Some(adapter)))
    intercept[IllegalArgumentException](LowRankAdapter(adapter.a, Parameter(tensor(1f, 2f)(1, 2)), 1.0))
  }

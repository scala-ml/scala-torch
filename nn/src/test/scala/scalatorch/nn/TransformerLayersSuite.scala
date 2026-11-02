package scalatorch.nn

import scalatorch.*
import scalatorch.autograd.*

class TransformerLayersSuite extends munit.FunSuite:
  private def tensor(ds: Seq[Long], xs: Float*) = Tensor.fromFloats(xs.toArray, Shape.from(ds))
  private def ids(ds: Seq[Long], xs: Long*) = Tensor.fromLongs(xs.toArray, Shape.from(ds))
  test("embedding preserves token shape and sums repeated row gradients") {
    val table = Parameter(tensor(Seq(3, 2), 1, 2, 3, 4, 5, 6))
    val layer = Embedding(table)
    val output = layer(ids(Seq(2, 2), 2, 0, 2, 1))
    assertEquals(output.data.shape, Shape.from(Seq(2L, 2L, 2L)))
    assertEquals(output.data.toFloatArray.toSeq, Seq(5f, 6f, 1f, 2f, 5f, 6f, 3f, 4f))
    output.sum().backward()
    assertEquals(table.grad.get.toFloatArray.toSeq, Seq(1f, 1f, 1f, 1f, 2f, 2f))
    table.zeroGrad()
    output.backward(tensor(Seq(2, 2, 2), 1, 2, 3, 4, 5, 6, 7, 8))
    assertEquals(table.grad.get.toFloatArray.toSeq, Seq(3f, 4f, 7f, 8f, 6f, 8f))
    assertEquals(Module.namedParameters(layer).map(_._1), Vector("weight"))
  }
  test("embedding scalar empty views and invalid Int64 IDs") {
    val table = Parameter(tensor(Seq(2, 3), 1, 2, 3, 4, 5, 6).transpose(0, 1))
    val layer = Embedding(table)
    assertEquals(layer(ids(Seq.empty, 1)).data.toFloatArray.toSeq, Seq(2f, 5f))
    val empty = layer(ids(Seq(2, 0)))
    assertEquals(empty.data.shape, Shape.from(Seq(2L, 0L, 2L)))
    empty.sum().backward()
    assertEquals(table.grad.get.toFloatArray.toSeq, Seq.fill(6)(0f))
    for index <- Seq(-1L, 3L, Long.MaxValue) do
      intercept[IllegalArgumentException](layer(ids(Seq(1), index)))
    assertEquals(table.grad.get.toFloatArray.toSeq, Seq.fill(6)(0f))
    assertEquals(Embedding.init(3, 2, 7).weight.data.toFloatArray.toSeq, Embedding.init(3, 2, 7).weight.data.toFloatArray.toSeq)
  }
  test("RMSNorm input and scale gradients agree with central differences") {
    val values = Array(0.4f, -0.7f, 1.2f, 0.8f, 0.3f, -0.2f)
    val weights = Array(1.1f, 0.9f, 1.3f)
    val seed = tensor(Seq(2, 3), 0.2f, -0.3f, 0.7f, 0.5f, 0.8f, -0.4f)
    val x = Variable(Tensor.fromFloats(values, Shape.from(Seq(2L, 3L))))
    val w = Variable(Tensor.fromFloats(weights, Shape.from(Seq(3L))))
    x.rmsNorm(w, 1e-5).backward(seed)
    def objective(xs: Array[Float], ws: Array[Float]): Double = Autograd.noGrad {
      val input = Variable(Tensor.fromFloats(xs, x.data.shape))
      val scale = Variable(Tensor.fromFloats(ws, w.data.shape))
      input.rmsNorm(scale, 1e-5).data.toFloatArray.zip(seed.toFloatArray).map((a, b) => a.toDouble * b).sum
    }
    for (source, gradient, isInput) <- Seq((values, x.grad.get.toFloatArray, true), (weights, w.grad.get.toFloatArray, false)) do
      source.indices.foreach { i =>
        val plus = source.clone(); plus(i) += 0.001f
        val minus = source.clone(); minus(i) -= 0.001f
        val numerical = if isInput then (objective(plus, weights) - objective(minus, weights)) / 0.002
          else (objective(values, plus) - objective(values, minus)) / 0.002
        assertEqualsDouble(gradient(i), numerical, 0.0002)
      }
  }
  test("RMSNorm handles empty batches zero rows large inputs and strided views") {
    val layer = RMSNorm.init(2, epsilon = 1e-6)
    assertEquals(Module.namedParameters(layer).map(_._1), Vector("weight"))
    val empty = Variable(tensor(Seq(0, 2)))
    layer(empty).sum().backward()
    assertEquals(layer.weight.grad.get.toFloatArray.toSeq, Seq(0f, 0f))
    val zero = Variable(tensor(Seq(1, 2), 0, 0))
    layer(zero).sum().backward()
    zero.grad.get.toFloatArray.foreach(x => assertEqualsDouble(x, 1000.0, 0.001))
    val big = Variable(tensor(Seq(1, 2), 1e30f, -1e30f))
    val bigOutput = layer(big)
    val normalized = bigOutput.data.toFloatArray
    assertEqualsDouble(normalized(0), 1.0, 1e-6)
    assertEqualsDouble(normalized(1), -1.0, 1e-6)
    bigOutput.backward(tensor(Seq(1, 2), 1, 0))
    big.grad.get.toFloatArray.foreach { gradient =>
      assert(gradient.isFinite)
      assertEqualsDouble(gradient.toDouble / 1e-30, 0.5, 1e-5)
    }
    val strided = Variable(tensor(Seq(2, 3), 1, 2, 3, 4, 5, 6).transpose(0, 1))
    assertEquals(layer(strided).data.shape, Shape.from(Seq(3L, 2L)))
    intercept[IllegalArgumentException](RMSNorm.init(2, epsilon = 0))
    intercept[IllegalArgumentException](layer(Variable(tensor(Seq(1, 3), 1, 2, 3))))
  }
  test("SiLU finite differences and stable extreme limits") {
    val xs = Array(-3f, -0.5f, 0f, 0.8f, 4f)
    val x = Variable(Tensor.fromFloats(xs, Shape.from(Seq(5L))))
    x.silu().sum().backward()
    xs.indices.foreach { i =>
      def f(v: Double) = v / (1 + math.exp(-v))
      val numerical = (f(xs(i) + 0.001) - f(xs(i) - 0.001)) / 0.002
      assertEqualsDouble(x.grad.get.toFloatArray(i), numerical, 1e-6)
    }
    val extreme = Variable(tensor(Seq(4), Float.NegativeInfinity, -1000, 1000, Float.PositiveInfinity))
    val y = extreme.silu()
    assertEquals(y.data.toFloatArray.toSeq, Seq(-0f, -0f, 1000f, Float.PositiveInfinity))
    y.sum().backward()
    assertEquals(extreme.grad.get.toFloatArray.toSeq, Seq(0f, 0f, 1f, 1f))
  }
  test("new operations preserve frozen parameter invalidation") {
    val x = Parameter(tensor(Seq(1, 2), 1, 2))
    val scale = Parameter(tensor(Seq(2), 1, 1), trainable = false)
    val y = x.value.rmsNorm(scale.value).silu().sum()
    scale.load(tensor(Seq(2), 2, 2))
    intercept[IllegalArgumentException](y.backward())
    assertEquals(x.grad, None)
  }
  case class TokenModel(embedding: Embedding, norm: RMSNorm, head: Linear) derives Module:
    def apply(tokens: Tensor[Long]): Variable = head(norm(embedding(tokens)).silu())
  test("embedding norm SiLU and Linear train a token model together") {
    val model = TokenModel(Embedding.init(3, 4, 11), RMSNorm.init(4), Linear.init(4, 1, 12))
    val optimizer = AdamW(Module.parameters(model), 0.03, weightDecay = 0)
    val tokens = ids(Seq(3), 0, 1, 2)
    val target = Variable(tensor(Seq(3, 1), -1, 0.5f, 2), false)
    var mse = Float.PositiveInfinity
    for _ <- 0 until 250 do
      optimizer.zeroGrad()
      val residual = model(tokens) - target
      val loss = (residual * residual).mean()
      mse = loss.data.toFloatArray(0)
      loss.backward()
      optimizer.step()
    assert(mse < 0.001f, s"token model MSE=$mse")
  }
  test("empty embedding avoids narrowing unused oversized kernel dimensions") {
    val width = Int.MaxValue.toLong + 1L
    val table = Parameter(Tensor.fromFloats(Array.emptyFloatArray, Shape.from(Seq(0L, width))))
    val output = Embedding(table)(ids(Seq(0)))
    assertEquals(output.data.shape, Shape.from(Seq(0L, width)))
    output.sum().backward()
    assertEquals(table.grad.get.toFloatArray.length, 0)
  }
  test("attention and rotary retain frozen saved-parameter invalidation") {
    for changeKey <- Seq(true, false) do
      val q = Parameter(tensor(Seq(1,1,2,2), 1,2,3,4))
      val k = Parameter(tensor(Seq(1,1,2,2), 2,3,4,5), trainable = false)
      val v = Parameter(tensor(Seq(1,1,2,2), 3,4,5,6), trainable = false)
      val loss = q.value.rope().causalAttention(k.value.rope(), v.value).sum()
      val changed = if changeKey then k else v
      changed.load(tensor(Seq(1,1,2,2), 7,8,9,10))
      intercept[IllegalArgumentException](loss.backward())
      assertEquals(q.grad, None)
  }

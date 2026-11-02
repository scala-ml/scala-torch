package scalatorch.nn

import scalatorch.*
import scalatorch.autograd.*

class TrainingSuite extends munit.FunSuite:
  private def tensor(ds: Seq[Long], xs: Float*) = Tensor.fromFloats(xs.toArray, Shape.from(ds))
  private def scalar(x: Float) = tensor(Seq.empty, x)
  case class Pair(left: Parameter, right: Parameter) derives Module
  case class MLP(first: Linear, second: Linear) derives Module:
    def apply(x: Variable): Variable = second(first(x).relu())

  test("Scala derived parameter names retain aliases while optimizer lists deduplicate") {
    val p = Parameter(scalar(2))
    val model = Pair(p, p)
    assertEquals(Module.namedParameters(model).map(_._1), Vector("left", "right"))
    assertEquals(Module.parameters(model), Vector(p))
    (p.value * p.value).backward()
    SGD(Seq(p, p), learningRate = 0.1).step()
    assertEqualsDouble(p.data.toFloatArray(0), 1.6, 1e-6)
    assertEquals(p.grad.get.toFloatArray.toSeq, Seq(4f))
    p.zeroGrad()
    assertEquals(p.grad, None)
  }
  test("Linear follows out-in weight convention and initialization is seeded") {
    val layer = Linear(Parameter(tensor(Seq(2, 3), 1, 2, 3, 4, 5, 6)), Some(Parameter(tensor(Seq(2), 10, 20))))
    val y = layer(Variable(tensor(Seq(1, 3), 1, 2, 3), requiresGrad = false))
    assertEquals(y.data.toFloatArray.toSeq, Seq(24f, 52f))
    assertEquals(Module.namedParameters(layer).map(_._1), Vector("weight", "bias"))
    val a = Linear.init(2, 4, seed = 7)
    val b = Linear.init(2, 4, seed = 7)
    assertEquals(a.weight.data.toFloatArray.toSeq, b.weight.data.toFloatArray.toSeq)
    assertEquals(Linear.init(2, 4, seed = 8, bias = false).bias, None)
    intercept[IllegalArgumentException](Linear.init(0, 4, seed = 1))
  }
  test("AdamW bias correction and decay match known first two steps") {
    val p = Parameter(scalar(1))
    val optimizer = AdamW(Seq(p), learningRate = 0.1, weightDecay = 0.1)
    (p.value * 2f).backward()
    optimizer.step()
    assertEqualsDouble(p.data.toFloatArray(0), 0.89, 1e-6)
    optimizer.step() // same retained gradient, per-parameter step 2
    assertEqualsDouble(p.data.toFloatArray(0), 0.7811, 1e-6)
    optimizer.zeroGrad()
    val before = p.data
    optimizer.step()
    assert(p.data eq before) // no gradient means no decay or moment advancement
  }
  test("old graphs reject updated trainable and saved frozen parameters atomically") {
    val p = Parameter(scalar(2))
    val frozen = Parameter(scalar(3), trainable = false)
    val loss = p.value * frozen.value
    loss.backward()
    SGD(Seq(p), learningRate = 0.1).step()
    p.zeroGrad()
    intercept[IllegalArgumentException](loss.backward())
    assertEquals(p.grad, None)
    val nextLoss = p.value * frozen.value
    frozen.load(scalar(4))
    intercept[IllegalArgumentException](nextLoss.backward())
    assertEquals(p.grad, None)
  }
  test("invalid optimizer gradients and hyperparameters do not partially update weights") {
    val p = Parameter(scalar(1))
    val q = Parameter(scalar(2))
    (p.value + q.value * Float.NaN).backward()
    val optimizer = SGD(Seq(p, q), learningRate = 0.1)
    intercept[IllegalArgumentException](optimizer.step())
    assertEquals(p.data.toFloatArray.toSeq, Seq(1f))
    assertEquals(q.data.toFloatArray.toSeq, Seq(2f))
    intercept[IllegalArgumentException](SGD(Seq(p), learningRate = Double.NaN))
    intercept[IllegalArgumentException](AdamW(Seq(p), learningRate = 0.1, beta1 = 1.0))
  }
  test("state dict snapshots are immutable and load validates all aliases before mutation") {
    val p = Parameter(scalar(1))
    val model = Pair(p, p)
    val saved = Module.stateDict(model)
    p.load(scalar(3))
    assertEquals(saved("left").toFloatArray.toSeq, Seq(1f))
    Module.loadStateDict(model, saved)
    assertEquals(p.data.toFloatArray.toSeq, Seq(1f))
    intercept[IllegalArgumentException](Module.loadStateDict(model, saved.updated("right", scalar(9))))
    assertEquals(p.data.toFloatArray.toSeq, Seq(1f))
    intercept[IllegalArgumentException](Module.loadStateDict(model, saved - "left"))
    intercept[IllegalArgumentException](Module.loadStateDict(model, saved.updated("left", tensor(Seq(1), 1))))
  }
  test("derived two-layer MLP learns XOR using AdamW") {
    val model = MLP(Linear.init(2, 8, seed = 41), Linear.init(8, 1, seed = 42))
    val optimizer = AdamW(Module.parameters(model), learningRate = 0.03, weightDecay = 0.0)
    val x = Variable(tensor(Seq(4, 2), 0, 0, 0, 1, 1, 0, 1, 1), requiresGrad = false)
    val target = Variable(tensor(Seq(4, 1), 0, 1, 1, 0), requiresGrad = false)
    var initial = 0f
    var last = 0f
    for i <- 0 until 500 do
      optimizer.zeroGrad()
      val residual = model(x) - target
      val loss = (residual * residual).mean()
      last = loss.data.toFloatArray(0)
      if i == 0 then initial = last
      loss.backward()
      optimizer.step()
    assert(last < 0.001f && last < initial * 0.01f, s"XOR MSE $initial -> $last")
    val prediction = Autograd.noGrad { model(x).data.toFloatArray }
    prediction.zip(Array(0f, 1f, 1f, 0f)).foreach((p, y) => assertEqualsDouble(p, y, 0.05))
  }
  test("AdamW changing gradients and rejected steps preserve moment state") {
    val p = Parameter(scalar(1))
    val q = Parameter(scalar(2))
    val optimizer = AdamW(Seq(p, q), learningRate = 0.1, weightDecay = 0.0)
    (p.value + q.value * Float.NaN).backward()
    intercept[IllegalArgumentException](optimizer.step())
    optimizer.zeroGrad()
    (p.value * 2f + q.value * 3f).backward()
    optimizer.step()
    assertEqualsDouble(p.data.toFloatArray(0), 0.9, 1e-6)
    optimizer.zeroGrad()
    optimizer.step() // absent gradients must not advance either parameter's counter
    (p.value * 4f).backward()
    optimizer.step()
    val firstMoment = 0.9 * 0.2 + 0.1 * 4
    val secondMoment = 0.999 * 0.004 + 0.001 * 16
    val expected = 0.9 - 0.1 * (firstMoment / 0.19) / (math.sqrt(secondMoment / 0.001999) + 1e-8)
    assertEqualsDouble(p.data.toFloatArray(0), expected, 1e-6)
    assertEqualsDouble(q.data.toFloatArray(0), 1.9, 1e-6)
  }
  test("transformed frozen weights retain stale-graph validation") {
    val input = Parameter(tensor(Seq(1, 2), 1, 2))
    val frozen = Parameter(tensor(Seq(1, 2), 3, 4), trainable = false)
    val layer = Linear(frozen, None)
    val loss = layer(input.value).sum()
    frozen.load(tensor(Seq(1, 2), 5, 6))
    intercept[IllegalArgumentException](loss.backward())
    assertEquals(input.grad, None)
  }
  test("explicit noGrad snapshots intentionally detach frozen dependency lineage") {
    val input = Parameter(tensor(Seq(1, 2), 1, 2))
    val frozen = Parameter(tensor(Seq(1, 2), 3, 4), trainable = false)
    val snapshot = Autograd.noGrad { frozen.value.transpose(0, 1) }
    val loss = input.value.matmul(snapshot).sum()
    frozen.load(tensor(Seq(1, 2), 5, 6))
    loss.backward()
    assertEquals(input.grad.get.toFloatArray.toSeq, Seq(3f, 4f))
  }

package scalatorch.autograd

import scalatorch.*

class AutogradSuite extends munit.FunSuite:
  private def tensor(ds: Seq[Long], xs: Float*) = Tensor.fromFloats(xs.toArray, Shape.from(ds))
  private def scalar(x: Float) = tensor(Seq.empty, x)
  private def values(x: Variable) = x.grad.get.toFloatArray.toSeq
  test("shared edges and fresh backward passes accumulate leaf gradients") {
    val x = Variable(scalar(3))
    val y = x * x + x
    y.backward()
    assertEquals(values(x), Seq(7f))
    y.backward()
    assertEquals(values(x), Seq(14f))
    assertEquals(y.grad, None)
    x.zeroGrad()
    assertEquals(x.grad, None)
    y.backward()
    assertEquals(values(x), Seq(7f))
  }
  test("broadcast gradients reduce singleton scalar and empty dimensions") {
    val a = Variable(tensor(Seq(2, 1), 2, 3))
    val b = Variable(tensor(Seq(3), 4, 5, 6))
    (a * b).sum().backward()
    assertEquals(values(a), Seq(15f, 15f))
    assertEquals(values(b), Seq(5f, 5f, 5f))
    val s = Variable(scalar(2))
    val empty = Variable(tensor(Seq(0, 3)))
    (s + empty).sum().backward()
    assertEquals(values(s), Seq(0f))
    assertEquals(values(empty), Seq.empty[Float])
  }
  test("explicit vector seed and invalid backward contracts") {
    val x = Variable(tensor(Seq(2), 2, 3))
    val y = x * x
    intercept[IllegalArgumentException](y.backward())
    intercept[IllegalArgumentException](y.backward(scalar(1)))
    assertEquals(x.grad, None)
    y.backward(tensor(Seq(2), 2, 4))
    assertEquals(values(x), Seq(8f, 24f))
    intercept[IllegalArgumentException](Variable(scalar(1), requiresGrad = false).backward())
  }
  test("transpose reshape and broadcast retain correct pullbacks") {
    val x = Variable(tensor(Seq(2, 3), 1, 2, 3, 4, 5, 6))
    x.transpose(0, 1).reshape(Shape.from(Seq(6L))).backward(tensor(Seq(6), 1, 2, 3, 4, 5, 6))
    assertEquals(values(x), Seq(1f, 3f, 5f, 2f, 4f, 6f))
    val b = Variable(tensor(Seq(1), 7))
    b.broadcastTo(Shape.from(Seq(2L, 3L))).sum().backward()
    assertEquals(values(b), Seq(6f))
  }
  test("GEMM bias and ReLU gradients agree with central differences") {
    val xs = Array(0.5f, -1f, 2f, 0.25f)
    val ws = Array(0.4f, -0.3f, 0.2f, 0.8f, -0.5f, 0.6f)
    val bs = Array(0.7f, 0.2f, -0.4f)
    def loss(x: Array[Float], w: Array[Float], b: Array[Float]): Variable =
      val a = Variable(Tensor.fromFloats(x, Shape.from(Seq(2L, 2L))))
      val weight = Variable(Tensor.fromFloats(w, Shape.from(Seq(2L, 3L))))
      val bias = Variable(Tensor.fromFloats(b, Shape.from(Seq(3L))))
      (a.matmul(weight) + bias).relu().mean()
    val x = Variable(Tensor.fromFloats(xs, Shape.from(Seq(2L, 2L))))
    val w = Variable(Tensor.fromFloats(ws, Shape.from(Seq(2L, 3L))))
    val b = Variable(Tensor.fromFloats(bs, Shape.from(Seq(3L))))
    (x.matmul(w) + b).relu().mean().backward()
    for (input, grad, slot) <- Seq((xs, values(x), 0), (ws, values(w), 1), (bs, values(b), 2)) do
      input.indices.foreach { i =>
        val plus = input.clone(); plus(i) += 0.001f
        val minus = input.clone(); minus(i) -= 0.001f
        def evaluate(v: Array[Float]) = Autograd.noGrad {
          loss(if slot == 0 then v else xs, if slot == 1 then v else ws, if slot == 2 then v else bs).data.toFloatArray(0)
        }
        val numerical = (evaluate(plus) - evaluate(minus)) / 0.002f
        assertEqualsDouble(grad(i).toDouble, numerical.toDouble, 0.0002)
      }
  }
  test("detach and nested noGrad do not track and restore after exceptions") {
    val x = Variable(scalar(3))
    assert(!x.detach.requiresGrad)
    val y = Autograd.noGrad { Autograd.noGrad { x * x } }
    assert(!y.requiresGrad)
    intercept[IllegalStateException] { Autograd.noGrad { throw IllegalStateException("stop") } }
    (x * x.detach).backward()
    assertEquals(values(x), Seq(3f))
    assert((x + x).requiresGrad)
  }
  test("iterative traversal supports deep graphs") {
    val x = Variable(scalar(1))
    var y = x
    for _ <- 0 until 12000 do y = y + Variable(scalar(1), requiresGrad = false)
    y.backward()
    assertEquals(values(x), Seq(1f))
  }
  test("noGrad is local to its thread") {
    val tracked = new java.util.concurrent.atomic.AtomicBoolean(false)
    val x = Variable(scalar(2))
    Autograd.noGrad {
      val worker = new Thread(() => tracked.set((x + x).requiresGrad))
      worker.start()
      worker.join(5000)
      assert(!worker.isAlive)
      assert(!(x + x).requiresGrad)
    }
    assert(tracked.get())
  }
  test("automatic gradients train a linear regressor without Python") {
    val x = Variable(tensor(Seq(4, 1), -1, 0, 1, 2), requiresGrad = false)
    val target = Variable(tensor(Seq(4, 1), -1, 1, 3, 5), requiresGrad = false)
    var w = Variable(tensor(Seq(1, 1), 0))
    var b = Variable(scalar(0))
    var first = 0f
    var last = 0f
    for iteration <- 0 until 200 do
      val residual = x.matmul(w) + b - target
      val loss = (residual * residual).mean()
      last = loss.data.toFloatArray(0)
      if iteration == 0 then first = last
      loss.backward()
      def updated(v: Variable): Variable =
        val next = v.data.toFloatArray.zip(v.grad.get.toFloatArray).map((value, grad) => value - 0.05f * grad)
        Variable(Tensor.fromFloats(next, v.data.shape))
      w = updated(w)
      b = updated(b)
    assertEqualsDouble(first.toDouble, 9.0, 1e-6)
    assert(last < 1e-6f, s"final MSE=$last")
    assertEqualsDouble(w.data.toFloatArray(0).toDouble, 2.0, 0.001)
    assertEqualsDouble(b.data.toFloatArray(0).toDouble, 1.0, 0.001)
  }
  test("inactive ReLU branches return zero for nonfinite upstream gradients") {
    val x = Variable(tensor(Seq(3), -1, 0, 1))
    x.relu().backward(tensor(Seq(3), Float.PositiveInfinity, Float.NaN, 2))
    assertEquals(values(x), Seq(0f, 0f, 2f))
  }

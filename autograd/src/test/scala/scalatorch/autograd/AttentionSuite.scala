package scalatorch.autograd

import scalatorch.*

class AttentionSuite extends munit.FunSuite:
  private def t(ds: Seq[Long], xs: Float*) = Tensor.fromFloats(xs.toArray, Shape.from(ds))
  private def v(ds: Seq[Long], xs: Float*) = Variable(t(ds, xs*))
  private def near(actual: Array[Float], expected: Seq[Double], tolerance: Double = 1e-6): Unit =
    assertEquals(actual.length, expected.size)
    actual.zip(expected).foreach((a, e) => assertEqualsDouble(a, e, tolerance))

  test("causal uniform scores and value gradients have known prefix means") {
    val q = v(Seq(1, 1, 3, 1), 0, 0, 0)
    val k = v(Seq(1, 1, 3, 1), 0, 0, 0)
    val values = v(Seq(1, 1, 3, 1), 2, 4, 8)
    val y = q.causalAttention(k, values)
    near(y.data.toFloatArray, Seq(2.0, 3.0, 14.0 / 3))
    y.sum().backward()
    near(values.grad.get.toFloatArray, Seq(1 + 0.5 + 1.0 / 3, 0.5 + 1.0 / 3, 1.0 / 3))
  }
  test("masked future NaNs never contaminate output or gradient") {
    val q = v(Seq(1, 1, 1, 1), 1)
    val k = v(Seq(1, 1, 2, 1), 1, Float.NaN)
    val values = v(Seq(1, 1, 2, 1), 3, Float.NaN)
    val y = q.causalAttention(k, values)
    near(y.data.toFloatArray, Seq(3.0))
    y.sum().backward()
    near(q.grad.get.toFloatArray, Seq(0.0))
    near(k.grad.get.toFloatArray, Seq(0.0, 0.0))
    near(values.grad.get.toFloatArray, Seq(1.0, 0.0))
  }
  test("grouped heads sum K/V gradients and decode offset matches prefill") {
    val q = v(Seq(1, 4, 2, 1), Seq.fill(8)(0f)*)
    val k = v(Seq(1, 2, 2, 1), 0, 0, 0, 0)
    val values = v(Seq(1, 2, 2, 1), 2, 4, 10, 14)
    val full = q.causalAttention(k, values)
    near(full.data.toFloatArray, Seq(2.0, 3.0, 2.0, 3.0, 10.0, 12.0, 10.0, 12.0))
    full.sum().backward()
    near(values.grad.get.toFloatArray, Seq(3.0, 1.0, 3.0, 1.0))
    val decode = v(Seq(1, 4, 1, 1), 0, 0, 0, 0).causalAttention(k, values, queryStart = 1)
    near(decode.data.toFloatArray, Seq(3.0, 3.0, 12.0, 12.0))
  }
  test("q k v gradients with GQA and unequal value width match central differences") {
    val qs = Array(.2f, .3f, .4f, -.5f, .6f, -.7f, .8f, .9f)
    val ks = Array(.3f, -.2f, .7f, .4f)
    val vs = Array(.5f, -.3f, .8f, .1f, .2f, -.4f)
    val shapes = Seq(Shape.from(Seq(1L,2L,2L,2L)), Shape.from(Seq(1L,1L,2L,2L)), Shape.from(Seq(1L,1L,2L,3L)))
    val inputs = Seq(qs, ks, vs).zip(shapes).map((a, s) => Variable(Tensor.fromFloats(a, s)))
    val seed = t(Seq(1,2,2,3), .1f,.2f,.3f,.4f,.5f,.6f,.7f,.8f,.9f,1f,1.1f,1.2f)
    inputs(0).causalAttention(inputs(1), inputs(2)).backward(seed)
    def objective(arrays: Seq[Array[Float]]): Double = Autograd.noGrad {
      val a = arrays.zip(shapes).map((x, s) => Variable(Tensor.fromFloats(x, s), false))
      a(0).causalAttention(a(1), a(2)).data.toFloatArray.zip(seed.toFloatArray).map((x, g) => x.toDouble * g).sum
    }
    Seq(qs, ks, vs).zipWithIndex.foreach { (source, slot) => source.indices.foreach { i =>
      val plus = source.clone(); plus(i) += .001f
      val minus = source.clone(); minus(i) -= .001f
      val arrays = Seq(qs, ks, vs)
      val numerical = (objective(arrays.updated(slot, plus)) - objective(arrays.updated(slot, minus))) / .002
      assertEqualsDouble(inputs(slot).grad.get.toFloatArray(i), numerical, .0002)
    }}
  }
  test("RoPE uses Llama half pairing and inverse rotation for gradients") {
    val x = v(Seq(1,1,2,4), 1,2,3,4,1,2,3,4)
    val y = x.rope()
    val c = math.cos(1.0); val s = math.sin(1.0)
    val c2 = math.cos(.01); val s2 = math.sin(.01)
    near(y.data.toFloatArray, Seq(1.0,2.0,3.0,4.0,c-3*s,2*c2-4*s2,3*c+s,4*c2+2*s2))
    y.sum().backward()
    near(x.grad.get.toFloatArray, Seq(1.0,1.0,1.0,1.0,c+s,c2+s2,c-s,c2-s2))
    val chunk = v(Seq(1,1,1,4), 1,2,3,4).rope(positionStart = 1)
    near(chunk.data.toFloatArray, y.data.toFloatArray.drop(4).map(_.toDouble).toSeq)
  }
  test("empty inputs invalid shapes and workspace are explicit") {
    val q = v(Seq(1,1,0,2))
    val k = v(Seq(1,1,2,2), 1,2,3,4)
    val values = v(Seq(1,1,2,3), 1,2,3,4,5,6)
    val empty = q.causalAttention(k, values)
    assertEquals(empty.data.shape.dims, Vector(1L,1L,0L,3L))
    empty.sum().backward()
    near(k.grad.get.toFloatArray, Seq.fill(4)(0.0))
    near(values.grad.get.toFloatArray, Seq.fill(6)(0.0))
    assertEquals(q.rope().data.shape, q.data.shape)
    val normal = v(Seq(1,1,2,2), 1,2,3,4)
    intercept[IllegalArgumentException](normal.causalAttention(k, values, maxScoreElements = 3))
    intercept[IllegalArgumentException](normal.causalAttention(k, values, queryStart = -1))
    intercept[IllegalArgumentException](normal.causalAttention(v(Seq(1,1,0,2)), v(Seq(1,1,0,2))))
    intercept[IllegalArgumentException](v(Seq(1,3,1,2), Seq.fill(6)(0f)*).causalAttention(v(Seq(1,2,1,2), 0,0,0,0), v(Seq(1,2,1,2),0,0,0,0)))
    intercept[IllegalArgumentException](v(Seq(1,1,1,3),1,2,3).rope())
    intercept[IllegalArgumentException](normal.rope(theta = 0))
    assertEquals(normal.grad, None)
  }
  test("batch offsets and strided values preserve nonuniform value gradients") {
    val q = v(Seq(2,1,2,1), 0,0,0,0)
    val k = v(Seq(2,1,2,1), 0,0,0,0)
    val values = Variable(t(Seq(2,1,2,2), 1,10,3,30,2,20,4,40).transpose(2,3))
    val y = q.causalAttention(k, values)
    near(y.data.toFloatArray, Seq(1.0,3.0,5.5,16.5,2.0,4.0,11.0,22.0))
    y.backward(t(Seq(2,1,2,2), 1,2,3,4,5,6,7,8))
    near(values.grad.get.toFloatArray, Seq(2.5,4.0,1.5,2.0,8.5,10.0,3.5,4.0))
  }
  test("large finite logits use stable softmax") {
    val q = v(Seq(1,1,2,1), 1e30f,1e30f)
    val k = v(Seq(1,1,2,1), 1e30f,1e30f)
    val values = v(Seq(1,1,2,1), 2,4)
    near(q.causalAttention(k, values).data.toFloatArray, Seq(2.0,3.0))
  }
  test("batched RoPE nonuniform gradients use Long position offsets") {
    val values = Array.tabulate(16)(i => (i - 3) * .25f)
    val seeds = Array.tabulate(16)(i => (i + 1) * .2f)
    val shape = Shape.from(Seq(2L,2L,2L,2L))
    val x = Variable(Tensor.fromFloats(values, shape))
    val y = x.rope(positionStart = Int.MaxValue)
    y.backward(Tensor.fromFloats(seeds, shape))
    val expected = (0 until 8).flatMap { row =>
      val angle = Int.MaxValue.toLong + row % 2
      val c = math.cos(angle.toDouble); val s = math.sin(angle.toDouble)
      Seq(seeds(row * 2).toDouble * c + seeds(row * 2 + 1).toDouble * s,
        seeds(row * 2 + 1).toDouble * c - seeds(row * 2).toDouble * s)
    }
    near(x.grad.get.toFloatArray, expected)
    val decode = v(Seq(1,1,1,1), 0).causalAttention(v(Seq(1,1,2,1),0,0), v(Seq(1,1,2,1),2,4), queryStart = Int.MaxValue)
    near(decode.data.toFloatArray, Seq(3.0))
  }
  test("empty oversized unused dimensions avoid all kernel narrowing") {
    val huge = Int.MaxValue.toLong + 1
    val q = v(Seq(0,huge,1,2))
    val k = v(Seq(0,1,1,2))
    val values = v(Seq(0,1,1,3))
    val y = q.rope().causalAttention(k, values, maxScoreElements = 0)
    assertEquals(y.data.shape.dims, Vector(0L,huge,1L,3L))
    y.sum().backward()
    assertEquals(q.grad.get.shape, q.data.shape)
    assertEquals(values.grad.get.toFloatArray.length, 0)
  }

  test("inference backend rotary and GQA preserve existing autograd forward semantics") {
    import scalatorch.cpu.CpuFloatBackend
    def values(shape: Shape,seed: Int) = Tensor.fromFloats(Array.tabulate(shape.numel.toInt)(i => ((i*7+seed)%19-9).toFloat/5),shape)
    val q=values(Shape.from(Seq(2L,4L,6L,3L)),1).transpose(2,3)
    val k=values(Shape.from(Seq(2L,2L,5L,6L)),2)
    val v=values(Shape.from(Seq(2L,2L,5L,4L)),3)
    assertEquals(CpuFloatBackend.rope(q,7,100000).toFloatArray.toVector,
      Variable(q,false).rope(positionStart=7,theta=100000).data.toFloatArray.toVector)
    assertEquals(CpuFloatBackend.causalAttention(q,k,v,2).toFloatArray.toVector,
      Variable(q,false).causalAttention(Variable(k,false),Variable(v,false),queryStart=2).data.toFloatArray.toVector)
  }

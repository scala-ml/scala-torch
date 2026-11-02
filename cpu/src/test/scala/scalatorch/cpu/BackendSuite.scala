package scalatorch.cpu

import scalatorch.*

class BackendSuite extends munit.FunSuite:
  private def t(ds: Seq[Long], xs: Float*) = Tensor.fromFloats(xs.toArray, Shape.from(ds))
  test("add broadcasts views and sum reduces axes") {
    val a = t(Seq(2, 3), 1, 2, 3, 4, 5, 6)
    val b = t(Seq(3), 10, 20, 30)
    assertEquals(Ops.add(a, b).toFloatArray.toSeq, Seq(11f, 22f, 33f, 14f, 25f, 36f))
    assertEquals(Ops.sum(a, Seq(0), false).toFloatArray.toSeq, Seq(5f, 7f, 9f))
    val all = Ops.sum(a, Seq(0, 1), true)
    assertEquals(all.shape, Shape.from(Seq(1L, 1L)))
    assertEquals(all.toFloatArray.toSeq, Seq(21f))
    assertEquals(Ops.sum(a.transpose(0, 1), Seq(1), false).toFloatArray.toSeq, Seq(5f, 7f, 9f))
    val empty = t(Seq(2, 0))
    assertEquals(Ops.sum(empty, Seq(1), false).toFloatArray.toSeq, Seq(0f, 0f))
    intercept[IllegalArgumentException](Ops.sum(a, Seq(0, 0), false))
    intercept[IllegalArgumentException](Ops.sum(a, Seq(2), false))
    intercept[IllegalArgumentException](Ops.add(a, t(Seq(2), 1, 2)))
  }
  test("reference and numscala GEMM agree for transpose slice and empty inner dimension") {
    val a = t(Seq(2, 3), 1, 2, 3, 4, 5, 6)
    val b = t(Seq(3, 2), 7, 8, 9, 10, 11, 12)
    val expected = Seq(58f, 64f, 139f, 154f)
    assertEquals(Ops.matmul(a, b).toFloatArray.toSeq, expected)
    assertEquals(NumscalaBackend.matmul(a, b).toFloatArray.toSeq, expected)
    assertEquals(NumscalaBackend.matmul(a.transpose(0, 1), a).toFloatArray.toSeq,
      Ops.matmul(a.transpose(0, 1), a).toFloatArray.toSeq)
    val sliced = a.slice(1, 0, 3, 2)
    assertEquals(NumscalaBackend.matmul(sliced, sliced.transpose(0, 1)).toFloatArray.toSeq,
      Ops.matmul(sliced, sliced.transpose(0, 1)).toFloatArray.toSeq)
    assertEquals(NumscalaBackend.matmul(t(Seq(2, 0)), t(Seq(0, 3))).toFloatArray.toSeq, Seq.fill(6)(0f))
    intercept[IllegalArgumentException](Ops.matmul(a, a))
    intercept[IllegalArgumentException](NumscalaBackend.matmul(a, a))
    intercept[IllegalArgumentException](Ops.matmul(t(Seq(2), 1, 2), a))
  }
  test("GEMM propagates NaN rather than skipping zero factors") {
    val a = t(Seq(1, 1), 0f)
    val b = t(Seq(1, 1), Float.NaN)
    assert(Ops.matmul(a, b).toFloatArray(0).isNaN)
    assert(NumscalaBackend.matmul(a, b).toFloatArray(0).isNaN)
  }
  test("empty tensors never truncate large dimensions during operations") {
    val huge = Int.MaxValue.toLong + 1
    val a = t(Seq(huge, 0))
    intercept[IllegalArgumentException](Ops.sum(a, Seq(1), false))
    val b = t(Seq(0, 0))
    assertEquals(Ops.matmul(a, b).shape, Shape.from(Seq(huge, 0L)))
    assertEquals(NumscalaBackend.matmul(a, b).shape, Shape.from(Seq(huge, 0L)))
  }

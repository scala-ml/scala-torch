package scalatorch

class TensorViewSuite extends munit.FunSuite:
  private def shape(ds: Long*) = Shape.from(ds)
  private def matrix = Tensor.fromFloats(Array(1f, 2f, 3f, 4f, 5f, 6f), shape(2, 3))
  test("transpose shares storage and reads logical order") {
    val x = matrix
    val y = x.transpose(0, 1)
    assert(y.storage eq x.storage)
    assertEquals(y.shape, shape(3, 2))
    assertEquals(y.toFloatArray.toSeq, Seq(1f, 4f, 2f, 5f, 3f, 6f))
    assertEquals(y.transpose(0, 1).toFloatArray.toSeq, x.toFloatArray.toSeq)
    intercept[IllegalArgumentException](x.transpose(0, 2))
  }
  test("reshape requires a contiguous layout and contiguous copies views") {
    val x = matrix
    assert(x.contiguous eq x)
    assert(x.reshape(shape(6)).storage eq x.storage)
    val t = x.transpose(0, 1)
    intercept[IllegalArgumentException](t.reshape(shape(6)))
    val c = t.contiguous
    assert(!(c.storage eq x.storage))
    assertEquals(c.reshape(shape(6)).toFloatArray.toSeq, Seq(1f, 4f, 2f, 5f, 3f, 6f))
    intercept[IllegalArgumentException](x.reshape(shape(7)))
  }
  test("broadcast handles scalars singleton and zero dimensions") {
    val x = Tensor.fromFloats(Array(7f), shape())
    assertEquals(x.broadcastTo(shape(2, 3)).toFloatArray.toSeq, Seq.fill(6)(7f))
    val y = Tensor.fromFloats(Array(1f, 2f), shape(2, 1))
    assertEquals(y.broadcastTo(shape(2, 3)).toFloatArray.toSeq, Seq(1f, 1f, 1f, 2f, 2f, 2f))
    val empty = Tensor.fromFloats(Array.emptyFloatArray, shape(0, 1)).broadcastTo(shape(0, 3))
    assertEquals(empty.toFloatArray.length, 0)
    assertEquals(x.broadcastTo(shape(0, 3)).toFloatArray.length, 0)
    intercept[IllegalArgumentException](matrix.broadcastTo(shape(2, 4)))
    intercept[IllegalArgumentException](matrix.broadcastTo(shape(3)))
  }
  test("slice preserves alias and supports strides including reversal") {
    val x = matrix
    val y = x.slice(1, 0, 3, 2)
    assert(y.storage eq x.storage)
    assertEquals(y.shape, shape(2, 2))
    assertEquals(y.toFloatArray.toSeq, Seq(1f, 3f, 4f, 6f))
    assertEquals(x.slice(1, 2, -1, -1).toFloatArray.toSeq, Seq(3f, 2f, 1f, 6f, 5f, 4f))
    assertEquals(x.slice(0, 2, 2).toFloatArray.length, 0)
    intercept[IllegalArgumentException](x.slice(1, 0, 3, 0))
    intercept[IllegalArgumentException](x.slice(1, 0, 4))
  }
  test("layout rejects invalid bounds and overflow before reading storage") {
    intercept[IllegalArgumentException](Layout.checked(shape(2), Vector(1L), 2L, 3L))
    intercept[IllegalArgumentException](Layout.checked(shape(2), Vector(Long.MaxValue), 1L, 3L))
    intercept[IllegalArgumentException](Layout.checked(shape(1), Vector(1L), -1L, 3L))
  }
  test("empty shapes do not require representable unused suffix strides") {
    val x = Tensor.fromFloats(Array.emptyFloatArray, shape(0, Long.MaxValue, 2))
    assertEquals(x.toFloatArray.length, 0)
    assertEquals(x.transpose(0, 2).shape, shape(2, Long.MaxValue, 0))
    assertEquals(x.reshape(shape(Long.MaxValue, 2, 0)).toFloatArray.length, 0)
  }

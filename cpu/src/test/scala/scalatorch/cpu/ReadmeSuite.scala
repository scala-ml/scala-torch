package scalatorch.cpu

import scalatorch.*

class ReadmeSuite extends munit.FunSuite:
  test("README Gram matrix example") {
    val x = Tensor.fromFloats(Array(1f, 2f, 3f, 4f, 5f, 6f), Shape.from(Seq(2L, 3L)))
    val gram = NumscalaBackend.matmul(x, x.transpose(0, 1))
    assertEquals(gram.toFloatArray.toSeq, Seq(14f, 32f, 32f, 77f))
  }

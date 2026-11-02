package scalatorch.ffm

class CudaProbeSuite extends munit.FunSuite:
  test("real CUDA driver copies host data to device and back") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"), "set SCALATORCH_CUDA_TESTS=1 for real GPU test")
    val values = Array(1f, -2f, 0f, Float.PositiveInfinity, 42.5f)
    val result = CudaProbe.roundTrip(values)
    assert(result.driverVersion > 0)
    assert(result.deviceName.nonEmpty)
    assertEquals(result.values.toSeq, values.toSeq)
  }
  test("GEMM validates input before invoking native code") {
    intercept[IllegalArgumentException](CudaProbe.gemm(Array(1f), Array(2f), 2, 1, 1))
    intercept[IllegalArgumentException](CudaProbe.gemm(Array.emptyFloatArray, Array.emptyFloatArray, -1, 2, 3))
    intercept[IllegalArgumentException](CudaProbe.gemm(Array.emptyFloatArray, Array.emptyFloatArray, Int.MaxValue, 2, 1))
  }
  test("real GPU GEMM matches CPU across blocks and completes before release") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"), "real GPU test disabled")
    import scalatorch.*
    for (m, n, k) <- Seq((2, 2, 3), (5, 17, 7), (3, 4, 0)) do
      val a = Array.tabulate(m * k)(i => (i % 11 - 5).toFloat / 8)
      val b = Array.tabulate(k * n)(i => (i % 7 - 3).toFloat / 4)
      val expected = Ops.matmul(Tensor.fromFloats(a, Shape.from(Seq(m.toLong, k.toLong))),
        Tensor.fromFloats(b, Shape.from(Seq(k.toLong, n.toLong)))).toFloatArray
      val result = CudaProbe.gemm(a, b, m, n, k)
      result.values.zip(expected).foreach { (actual, wanted) => assertEqualsDouble(actual.toDouble, wanted.toDouble, 1e-5) }
      assertEquals(result.values.length, expected.length)
  }

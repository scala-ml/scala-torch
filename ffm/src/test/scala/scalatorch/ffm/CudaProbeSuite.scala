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

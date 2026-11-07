package scalatorch.ffm

import java.lang.foreign.ValueLayout.JAVA_BYTE

class CudaHostStagingBufferSuite extends munit.FunSuite:
  test("staging memory is reused at the same or smaller size and replaced only when it grows") {
    val staging=new CudaHostStagingBuffer
    try
      val first=staging.segment(128)
      first.set(JAVA_BYTE,0,42.toByte)
      val smaller=staging.segment(32)
      assert(smaller.address()==first.address(),"smaller slice reuses existing memory")
      assert(smaller.byteSize()==32L,"slice exposes requested size")
      assert(smaller.get(JAVA_BYTE,0)==42.toByte,"reused memory keeps contents until overwritten")

      val grown=staging.segment(256)
      assert(grown.address()!=first.address(),"growth replaces the backing memory")
      assert(grown.byteSize()==256L,"grown slice exposes requested size")
      grown.set(JAVA_BYTE,255,7.toByte)
      val reused=staging.segment(256)
      assert(reused.address()==grown.address(),"same-size request reuses grown memory")
      assert(reused.get(JAVA_BYTE,255)==7.toByte,"grown buffer remains reusable")
    finally staging.close()
  }

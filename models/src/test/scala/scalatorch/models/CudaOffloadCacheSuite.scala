package scalatorch.models

import scalatorch.*
import scalatorch.autograd.Variable
import scalatorch.ffm.CudaSession
import scalatorch.nn.*

class CudaOffloadCacheSuite extends munit.FunSuite:
  private def gpu(): Unit=assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"real GPU test disabled")
  private def checkOutput(actual: Array[Float],expected: Array[Float]): Unit =
    assertEquals(actual.length,expected.length)
    actual.zip(expected).foreach((a,e) => assertEqualsDouble(a,e,1e-5))

  test("offloaded linear weights stay cached across calls and are released with the runtime") {
    gpu()
    val input=Tensor.fromFloats(Array(.2f,-.7f),Shape.from(Seq(1L,2L)))
    val layer=Linear.init(2,3,seed=17,bias=false)
    val expected=layer(Variable(input)).data.toFloatArray
    val weightBytes=layer.weight.data.shape.numel*4
    CudaSession.withDevice(maxBytes=2048) { session =>
      val runtime=CudaDiffusionRuntime.upload(session,layer,offloadWeights=true)
      val deviceInput=session.upload(input)
      val inputBytes=session.allocatedBytes
      for _ <- 0 until 2 do
        val output=runtime.linear(deviceInput,layer)
        checkOutput(output.toCpu.toFloatArray,expected)
        output.close()
        assertEquals(session.allocatedBytes,inputBytes+weightBytes)
      runtime.close()
      assertEquals(session.allocatedBytes,inputBytes)
      deviceInput.close()
      assertEquals(session.allocatedBytes,0L)
    }
  }

  test("offload cache retains first fitting weights when later weights do not fit") {
    gpu()
    val input=Tensor.fromFloats(Array.fill(8)(.25f),Shape.from(Seq(1L,8L)))
    val a=Linear.init(8,5,seed=1,bias=false) // 160 bytes
    val b=Linear.init(8,6,seed=2,bias=false) // 192 bytes
    val c=Linear.init(8,11,seed=3,bias=false) // 352 bytes
    val expected=Vector(a,b,c).map(layer => layer(Variable(input)).data.toFloatArray)
    CudaSession.withDevice(maxBytes=2048) { session =>
      val runtime=CudaDiffusionRuntime.upload(session,Vector(a.weight,b.weight,c.weight),offloadWeights=true)
      val deviceInput=session.upload(input)
      val inputBytes=session.allocatedBytes
      def run(layer: Linear,index: Int): Unit =
        val output=runtime.linear(deviceInput,layer)
        checkOutput(output.toCpu.toFloatArray,expected(index))
        output.close()
      run(a,0);run(b,1);run(a,0);run(c,2)
      // A and B fit first (352 bytes); C remains transient without evicting either.
      assertEquals(session.allocatedBytes,inputBytes+352L)
      run(a,0)
      run(b,1)
      assertEquals(session.allocatedBytes,inputBytes+352L)
      runtime.close()
      assertEquals(session.allocatedBytes,inputBytes)
      deviceInput.close()
      assertEquals(session.allocatedBytes,0L)
    }
  }

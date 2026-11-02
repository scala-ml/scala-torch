package scalatorch.ffm

import scalatorch.*

class CudaSessionSuite extends munit.FunSuite:
  private def gpu(): Unit = assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"real GPU test disabled")
  private def matrix(values: Array[Float], m: Int, n: Int) = Tensor.fromFloats(values,Shape.from(Seq(m.toLong,n.toLong)))

  test("persistent device GEMMs chain without CPU intermediates and upload logical views") {
    gpu()
    CudaSession.withDevice() { session =>
      val a=matrix(Array(1f,2f,3f,4f,5f,6f),3,2).transpose(0,1)
      val b=matrix(Array(1f,2f,3f,4f,5f,6f),3,2)
      val expected=Ops.matmul(Ops.matmul(a,b),matrix(Array(1f,0f,0f,2f),2,2))
      val ad=session.upload(a);val bd=session.upload(b)
      val first=ad.matmul(bd)
      val finalResult=first.matmul(session.upload(matrix(Array(1f,0f,0f,2f),2,2)))
      assertEquals(finalResult.toCpu.toFloatArray.toVector,expected.toFloatArray.toVector)
      first.close();first.close()
      intercept[IllegalStateException](first.toCpu)
      assertEquals(ad.toCpu.toFloatArray.toVector,a.toFloatArray.toVector)
    }
  }

  test("budget rejection releases no live inputs and explicit close makes space reusable") {
    gpu()
    CudaSession.withDevice(maxBytes=16) { session =>
      val a=session.upload(matrix(Array(1f,2f,3f,4f),2,2))
      assertEquals(session.allocatedBytes,16L)
      intercept[IllegalArgumentException](a.matmul(a))
      assertEquals(session.allocatedBytes,16L)
      a.close()
      assertEquals(session.allocatedBytes,0L)
      val b=session.upload(matrix(Array(5f,6f,7f,8f),2,2))
      assertEquals(b.toCpu.toFloatArray.toVector,Vector(5f,6f,7f,8f))
    }
  }

  test("scope exit, exceptional exit, foreign sessions and wrong threads reject stale handles") {
    gpu()
    val escaped=CudaSession.withDevice() { session =>
      val a=session.upload(matrix(Array(1f),1,1))
      CudaSession.withDevice() { other =>
        val b=other.upload(matrix(Array(2f),1,1))
        intercept[IllegalArgumentException](a.matmul(b))
      }
      val error=new java.util.concurrent.atomic.AtomicReference[Throwable]()
      val thread=new Thread(() => { try a.toCpu catch case t: Throwable => error.set(t); () })
      thread.start();thread.join()
      assert(error.get().isInstanceOf[IllegalStateException])
      // Both the original and this shared-allocation view must close at scope exit.
      session.reshape(a,Shape.from(Seq(1L)))
    }
    intercept[IllegalStateException](escaped.toCpu)
    escaped.close()
    var failed: Option[CudaTensor]=None
    intercept[RuntimeException] {
      CudaSession.withDevice() { session =>
        failed=Some(session.upload(matrix(Array(1f),1,1)))
        throw new RuntimeException("body failed")
      }
    }
    intercept[IllegalStateException](failed.get.toCpu)
  }

  test("zero inner dimension produces zero output and empty shapes roundtrip") {
    gpu()
    CudaSession.withDevice() { session =>
      val a=session.upload(matrix(Array.emptyFloatArray,3,0))
      val b=session.upload(matrix(Array.emptyFloatArray,0,2))
      assertEquals(a.matmul(b).toCpu.toFloatArray.toVector,Vector.fill(6)(0f))
      val empty=session.upload(matrix(Array.emptyFloatArray,0,3))
      assertEquals(empty.toCpu.shape,Shape.from(Seq(0L,3L)))
    }
  }

  test("scope cleanup rebinds its context after external native code changes it") {
    gpu()
    import java.lang.foreign.MemorySegment
    import java.lang.foreign.ValueLayout.ADDRESS
    CudaDriver.withDevice(0) { (driver,_,_) =>
      val before=driver.pointerResult("cuCtxGetCurrent",Seq.empty)
      val escaped=CudaSession.withDevice() { session =>
        val tensor=session.upload(matrix(Array(9f),1,1))
        driver.call("cuCtxSetCurrent",Seq(ADDRESS),MemorySegment.NULL)
        intercept[IllegalStateException](tensor.toCpu)
        tensor
      }
      intercept[IllegalStateException](escaped.toCpu)
      escaped.close()
      assertEquals(driver.pointerResult("cuCtxGetCurrent",Seq.empty).address(),before.address())
    }
  }

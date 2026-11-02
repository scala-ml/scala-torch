package scalatorch.models

import scalatorch.*
import scalatorch.cpu.CpuFloatBackend
import scalatorch.ffm.CudaSession
import scala.util.Using

class BackendAdapterSuite extends munit.FunSuite:
  private def check[T](backend: FloatBackend[T], bytes: () => Long): Unit =
    val base=LlamaModel.init(LlamaConfig(8,8,16,2,2,1,maxPositions=16,initializerRange=.1,tieWordEmbeddings=true),7)
    val adapted=LlamaLoRA.attach(base,2,3,19,Set("q_proj","v_proj","down_proj"))
    adapted.adapters.foreach { (_,adapter) =>
      val shape=adapter.b.data.shape
      adapter.b.load(Tensor.fromFloats(Array.tabulate(shape.numel.toInt)(i => (i%7-3)*.07f),shape))
    }
    val input=Tensor.fromLongs(Array(1L,2L,3L,4L,3L,2L),Shape.from(Seq(2L,3L)))
    val expected=adapted.model.logits(input).toFloatArray
    assert(!expected.sameElements(base.logits(input).toFloatArray))
    def compare(actual: Array[Float],reference: Array[Float]): Unit =
      assertEquals(actual.length,reference.length)
      actual.zip(reference).foreach((a,e) => assertEqualsDouble(a.toDouble,e.toDouble,3e-6))
    Using.resource(BackendLlama.prepare(adapted.model,backend)) { prepared =>
      val resident=bytes()
      compare(prepared.logits(input).toFloatArray,expected)
      assertEquals(bytes(),resident)
      Using.resource(prepared.newCache(batchSize=2,capacity=4)) { cache =>
        val cached=bytes()
        for position <- 0 until 3 do
          val chunk=Tensor.fromLongs(Array(input.toLongArray(position),input.toLongArray(3+position)),Shape.from(Seq(2L,1L)))
          compare(cache.decode(chunk).toFloatArray,(expected.slice(position*8,(position+1)*8) ++ expected.slice((3+position)*8,(4+position)*8)))
          assertEquals(bytes(),cached)
        val a=adapted.adapters.head._2.a
        a.load(a.data)
        intercept[IllegalArgumentException](prepared.logits(input))
        intercept[IllegalArgumentException](cache.decode(input))
      }
      assertEquals(bytes(),resident)
    }
    assertEquals(bytes(),0L)

  test("prepared CPU adapters match autograd inference and cached batches") {
    check(CpuFloatBackend,() => 0L)
  }
  test("resident CUDA adapters match autograd inference and release intermediates") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA gate disabled")
    CudaSession.withDevice()(session => check(session,() => session.allocatedBytes))
  }
  test("CUDA adapter preparation and intermediate allocation failures release owned handles") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA gate disabled")
    val base=LlamaModel.init(LlamaConfig(8,8,16,1,2,1,maxPositions=16),7)
    val adapted=LlamaLoRA.attach(base,2,3,19)
    var resident=0L
    CudaSession.withDevice() { session =>
      Using.resource(BackendLlama.prepare(adapted.model,session))(_ => resident=session.allocatedBytes)
    }
    CudaSession.withDevice(maxBytes=resident-1) { session =>
      intercept[IllegalArgumentException](BackendLlama.prepare(adapted.model,session))
      assertEquals(session.allocatedBytes,0L)
    }
    // This budget admits the embedding, normalization, base projection and low-rank
    // projection and delta, then rejects its scaling while the linear scope owns handles.
    CudaSession.withDevice(maxBytes=resident+288) { session =>
      Using.resource(BackendLlama.prepare(adapted.model,session)) { prepared =>
        val input=Tensor.fromLongs(Array(1L,2L),Shape.from(Seq(1L,2L)))
        intercept[IllegalArgumentException](prepared.logits(input))
        assertEquals(session.allocatedBytes,resident)
      }
      assertEquals(session.allocatedBytes,0L)
    }
  }

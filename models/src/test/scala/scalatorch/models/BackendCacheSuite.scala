package scalatorch.models

import scalatorch.*
import scalatorch.cpu.CpuFloatBackend
import scalatorch.ffm.CudaSession
import scala.util.Using

class BackendCacheSuite extends munit.FunSuite:
  private val config=LlamaConfig(8,8,16,2,2,1,maxPositions=16,initializerRange=.1)
  private val tokens=Tensor.fromLongs(Array(1L,2L,3L,4L,2L,3L,4L,1L),Shape.from(Seq(2L,4L)))
  private def compare(a: Array[Float],b: Array[Float]): Unit =
    assertEquals(a.length,b.length)
    a.zip(b).foreach((x,y) => assertEqualsDouble(x,y,3e-6))
  private def exercise[T](backend: FloatBackend[T]): Unit =
    val model=LlamaModel.init(config,9)
    Using.resource(BackendLlama.prepare(model,backend)) { prepared =>
      Using.resource(prepared.newCache(batchSize=2,capacity=8)) { cache =>
        val first=cache.decode(tokens.slice(1,0,2))
        compare(first.toFloatArray,prepared.logits(tokens.slice(1,0,2)).toFloatArray)
        assertEquals(cache.position,2)
        val second=cache.decode(tokens.slice(1,2,4))
        compare(second.toFloatArray,prepared.logits(tokens).slice(1,2,4).toFloatArray)
        assertEquals(cache.position,4)
        val retained=second.toFloatArray.toVector
        intercept[IllegalArgumentException](cache.decode(Tensor.fromLongs(Array(-1L,-1L),Shape.from(Seq(2L,1L)))))
        assertEquals(cache.position,4)
        cache.reset()
        compare(cache.decode(tokens).toFloatArray,prepared.logits(tokens).toFloatArray)
        assertEquals(second.toFloatArray.toVector,retained)
        model.norm.weight.load(model.norm.weight.data)
        intercept[IllegalArgumentException](cache.decode(tokens))
      }
    }
  test("prepared CPU cache matches batched full prefixes and preserves rejected positions") {
    exercise(CpuFloatBackend)
  }
  test("prepared CUDA cache matches batched prefixes and releases its resident buffers") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA cache gate disabled")
    var weightBytes=0L
    CudaSession.withDevice() { session =>
      exercise(session)
      assertEquals(session.allocatedBytes,0L)
      val model=LlamaModel.init(config,7)
      Using.resource(BackendLlama.prepare(model,session)) { prepared =>
        val weights=session.allocatedBytes
        weightBytes=weights
        val cache=prepared.newCache(capacity=8)
        val resident=session.allocatedBytes
        for id <- Seq(1L,2L,3L) do
          cache.decode(Tensor.fromLongs(Array(id),Shape.from(Seq(1L,1L))))
          assertEquals(session.allocatedBytes,resident)
        cache.close();cache.close()
        assertEquals(session.allocatedBytes,weights)
        intercept[IllegalStateException](cache.decode(tokens))
        assertEquals(prepared.generateIds(Vector(1L,3L),3),prepared.generateIds(Vector(1L,3L),3,useCache=false))
        val forgotten=prepared.newCache(capacity=4)
        prepared.close()
        intercept[IllegalStateException](forgotten.decode(tokens))
      }
      assertEquals(session.allocatedBytes,0L)
    }
    CudaSession.withDevice(maxBytes=weightBytes+300) { session =>
      Using.resource(BackendLlama.prepare(LlamaModel.init(config,7),session)) { prepared =>
        // One layer cache fits, the next does not; the first must be released.
        intercept[IllegalArgumentException](prepared.newCache(capacity=8))
        assertEquals(session.allocatedBytes,weightBytes)
      }
    }
  }
  test("cache budgets and late forward failures never advance committed position") {
    val model=LlamaModel.init(config,7,limits=LlamaLimits(maxLogitsElements=16))
    Using.resource(BackendLlama.prepare(model,CpuFloatBackend)) { prepared =>
      intercept[IllegalArgumentException](prepared.newCache(capacity=8,maxBytes=1))
      Using.resource(prepared.newCache(capacity=4)) { cache =>
        cache.decode(Tensor.fromLongs(Array(1L),Shape.from(Seq(1L,1L))))
        intercept[IllegalArgumentException](cache.decode(Tensor.fromLongs(Array(2L,3L,4L),Shape.from(Seq(1L,3L)))))
        assertEquals(cache.position,1)
        cache.decode(Tensor.fromLongs(Array(2L),Shape.from(Seq(1L,1L))))
        assertEquals(cache.position,2)
        intercept[IllegalArgumentException](cache.decode(Tensor.fromLongs(Array.fill(4)(1L),Shape.from(Seq(1L,4L)))))
        assertEquals(cache.position,2)
        intercept[IllegalArgumentException](cache.decode(Tensor.fromLongs(Array.emptyLongArray,Shape.from(Seq(1L,0L)))))
      }
    }
    class FailedDownload extends FloatBackend[Tensor[Float]]:
      export CpuFloatBackend.{upload,shape,release,reshape,transpose,embedding,matmul,add,multiply,silu,rmsNorm,rope,causalAttention,attentionCache}
      private var fail=true
      def download(value: Tensor[Float]): Tensor[Float] =
        if fail then
          fail=false
          throw IllegalStateException("injected download failure after all layer writes")
        value
    val backend=new FailedDownload
    val base=LlamaModel.init(config,5)
    Using.resource(BackendLlama.prepare(base,backend)) { prepared =>
      Using.resource(prepared.newCache(batchSize=2,capacity=8)) { cache =>
        intercept[IllegalStateException](cache.decode(tokens))
        assertEquals(cache.position,0)
        compare(cache.decode(tokens).toFloatArray,base.logits(tokens).toFloatArray)
        assertEquals(cache.position,4)
      }
    }
  }

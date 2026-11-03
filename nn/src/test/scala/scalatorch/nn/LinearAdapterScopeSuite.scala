package scalatorch.nn

import scalatorch.*
import scalatorch.autograd.Variable

class LinearAdapterScopeSuite extends munit.FunSuite:
  private val weight=Parameter(Tensor.fromFloats(Array(1f,0f,0f,1f),Shape.from(Seq(2L,2L))),false)
  private val linear=Linear(weight,None)
  private val input=Variable(Tensor.fromFloats(Array(1f,2f),Shape.from(Seq(1L,2L))))
  private val first=LowRankAdapter(
    Parameter(Tensor.fromFloats(Array(1f,0f),Shape.from(Seq(1L,2L))),false),
    Parameter(Tensor.fromFloats(Array(2f,3f),Shape.from(Seq(2L,1L))),false),1.0)
  private val second=LowRankAdapter(
    Parameter(Tensor.fromFloats(Array(0f,1f),Shape.from(Seq(1L,2L))),false),
    Parameter(Tensor.fromFloats(Array(1f,-1f),Shape.from(Seq(2L,1L))),false),1.0)
  private def output=linear(input).data.toFloatArray.toVector

  test("scoped adapters add independently and nested or failed scopes restore prior adapters") {
    val adapters=Map(weight->Vector(first->0.5f,second->2.0f))
    assertEquals(output,Vector(1f,2f))
    assertEquals(LinearAdapterScope.withAdapters(adapters)(output),Vector(6f,-0.5f))
    assertEquals(LinearAdapterScope.withAdapters(adapters) {
      val nested=LinearAdapterScope.withAdapters(Map(weight->Vector(second->2.0f)))(output)
      assertEquals(nested,Vector(5f,-2f))
      output
    },Vector(6f,-0.5f))
    intercept[IllegalStateException] {
      LinearAdapterScope.withAdapters(adapters) { throw IllegalStateException("request failed") }
    }
    assertEquals(output,Vector(1f,2f))
  }

  test("adapter scopes are thread local and reject nonfinite multipliers") {
    val entered=new java.util.concurrent.CountDownLatch(1)
    val release=new java.util.concurrent.CountDownLatch(1)
    val result=new java.util.concurrent.atomic.AtomicReference[Vector[Float]]()
    val thread=new Thread(() => {
      LinearAdapterScope.withAdapters(Map(weight->Vector(first->1f))) {
        entered.countDown()
        release.await()
        result.set(output)
      }
    })
    thread.start()
    entered.await()
    assertEquals(output,Vector(1f,2f),"another request must not observe this thread's adapter")
    release.countDown()
    thread.join()
    assertEquals(result.get(),Vector(3f,5f))
    intercept[IllegalArgumentException] {
      LinearAdapterScope.withAdapters(Map(weight->Vector(first->Float.NaN)))(())
    }
    assertEquals(output,Vector(1f,2f))
  }

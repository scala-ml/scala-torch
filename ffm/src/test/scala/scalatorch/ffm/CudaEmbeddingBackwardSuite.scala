package scalatorch.ffm

import scalatorch.*

class CudaEmbeddingBackwardSuite extends munit.FunSuite:
  private def gpu(): Unit = assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA gate disabled")
  test("embedding pullback preserves repeated-row order and isolates nonfinite rows") {
    gpu()
    val ids=Tensor.fromLongs(Array(2L,2L,2L,1L,3L),Shape.from(Seq(5L)))
    val gradient=Tensor.fromFloats(Array(1e8f,2f,1f,3f,-1e8f,4f,5f,6f,Float.NaN,7f),Shape.from(Seq(5L,2L)))
    CudaSession.withDevice() { session =>
      val g=session.upload(gradient)
      val out=session.embeddingBackward(Shape.from(Seq(5L,2L)),ids,g)
      g.close()
      val values=out.toCpu.toFloatArray
      assertEquals(values.take(6).toVector,Vector(0f,0f,5f,6f,0f,9f))
      assert(values(6).isNaN)
      assertEquals(values.drop(7).toVector,Vector(7f,0f,0f))
      out.close()
      val emptyIds=Tensor.fromLongs(Array.emptyLongArray,Shape.from(Seq(0L)))
      val empty=session.fill(Shape.from(Seq(0L,2L)),0f)
      val zero=session.embeddingBackward(Shape.from(Seq(5L,2L)),emptyIds,empty)
      assertEquals(zero.toCpu.toFloatArray.toVector,Vector.fill(10)(0f))
      val noRows=session.embeddingBackward(Shape.from(Seq(0L,2L)),emptyIds,empty)
      assertEquals(noRows.toCpu.shape,Shape.from(Seq(0L,2L)))
      noRows.close();zero.close();empty.close()
      assertEquals(session.allocatedBytes,0L)
    }
  }
  test("embedding pullback validates metadata and preflights complete scratch budget") {
    gpu()
    val shape=Shape.from(Seq(4L,3L))
    val ids=Tensor.fromLongs(Array(1L,1L),Shape.from(Seq(2L)))
    CudaSession.withDevice(maxBytes=80) { session =>
      val g=session.fill(Shape.from(Seq(2L,3L)),1f)
      val baseline=session.allocatedBytes
      // Output48 + heads16 + next8 exceeds the remaining56 bytes.
      intercept[IllegalArgumentException](session.embeddingBackward(shape,ids,g))
      for bad <- Seq(-1L,4L,Long.MaxValue) do
        intercept[IllegalArgumentException](session.embeddingBackward(shape,Tensor.fromLongs(Array(1L,bad),ids.shape),g))
      intercept[IllegalArgumentException](session.embeddingBackward(Shape.from(Seq(4L,2L)),ids,g))
      intercept[IllegalArgumentException](session.embeddingBackward(Shape.from(Seq(4L)),ids,g))
      intercept[IllegalArgumentException](session.embeddingBackward(Shape.from(Seq(Int.MaxValue.toLong,3L)),ids,g))
      intercept[IllegalArgumentException](session.embeddingBackward(Shape.from(Seq(4L,0L)),ids,g))
      assertEquals(session.allocatedBytes,baseline)
      g.close()
      intercept[IllegalStateException](session.embeddingBackward(shape,ids,g))
      assertEquals(session.allocatedBytes,0L)
    }
    // The exact budget includes input24, output48, heads16 and next8.
    CudaSession.withDevice(maxBytes=96) { session =>
      val g=session.fill(Shape.from(Seq(2L,3L)),1f)
      val out=session.embeddingBackward(shape,ids,g)
      assertEquals(session.allocatedBytes,72L)
      assertEquals(out.toCpu.toFloatArray.slice(3,6).toVector,Vector(2f,2f,2f))
      out.close();g.close()
      assertEquals(session.allocatedBytes,0L)
    }
  }

package scalatorch.ffm

import scalatorch.*

class CudaLayoutSuite extends munit.FunSuite:
  private def gpu(): Unit = assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"real GPU test disabled")
  test("reshape shares allocation budget and independently retained handles survive either close order") {
    gpu()
    CudaSession.withDevice(maxBytes=24) { session =>
      val cpu=Tensor.fromFloats(Array(1f,2f,3f,4f,5f,6f),Shape.from(Seq(2L,3L)))
      val input=session.upload(cpu)
      val view=session.reshape(input,Shape.from(Seq(6L)))
      val nested=session.reshape(view,Shape.from(Seq(3L,2L)))
      assertEquals(session.allocatedBytes,24L)
      view.close();input.close()
      assertEquals(nested.toCpu.toFloatArray.toVector,cpu.toFloatArray.toVector)
      assertEquals(session.allocatedBytes,24L)
      nested.close();nested.close()
      assertEquals(session.allocatedBytes,0L)
      val again=session.upload(cpu)
      session.reshape(again,Shape.from(Seq(6L))).close()
      assertEquals(again.toCpu.toFloatArray.toVector,cpu.toFloatArray.toVector)
      intercept[IllegalArgumentException](session.reshape(again,Shape.from(Seq(5L))))
      again.close()
      intercept[IllegalStateException](session.reshape(again,cpu.shape))
    }
  }
  test("transpose packs all axis pairs and preserves empty layouts without overflow") {
    gpu()
    CudaSession.withDevice() { session =>
      for dims <- Seq(Seq(2L,3L),Seq(2L,3L,4L),Seq(2L,3L,2L,5L),Seq(0L,Long.MaxValue,2L)) do
        val shape=Shape.from(dims)
        val cpu=Tensor.fromFloats(Array.tabulate(shape.numel.toInt)(_.toFloat),shape)
        val input=session.upload(cpu)
        for i <- dims.indices; j <- dims.indices do
          val output=session.transpose(input,i,j)
          assertEquals(output.toCpu.shape,cpu.transpose(i,j).shape)
          assertEquals(output.toCpu.toFloatArray.toVector,cpu.transpose(i,j).toFloatArray.toVector)
          output.close()
        intercept[IllegalArgumentException](session.transpose(input,-1,0))
        input.close()
      assertEquals(session.allocatedBytes,0L)
    }
  }
  test("GPU embedding gathers validated integer IDs and cleans metadata after budget rejection") {
    gpu()
    val table=Tensor.fromFloats(Array(1f,2f,3f,4f,5f,6f),Shape.from(Seq(3L,2L)))
    val ids=Tensor.fromLongs(Array(2L,0L,2L,1L),Shape.from(Seq(2L,2L)))
    CudaSession.withDevice() { session =>
      val device=session.upload(table)
      val result=session.embedding(device,ids)
      assertEquals(result.toCpu.toFloatArray.toVector,Vector(5f,6f,1f,2f,5f,6f,3f,4f))
      assertEquals(result.shape,Shape.from(Seq(2L,2L,2L)))
      assertEquals(session.allocatedBytes,56L)
      val scalar=session.embedding(device,Tensor.fromLongs(Array(1L),Shape.from(Seq.empty)))
      assertEquals(scalar.toCpu.toFloatArray.toVector,Vector(3f,4f))
      val empty=session.embedding(device,Tensor.fromLongs(Array.emptyLongArray,Shape.from(Seq(0L))))
      assertEquals(empty.toCpu.shape,Shape.from(Seq(0L,2L)))
      val before=session.allocatedBytes
      for id <- Seq(-1L,3L,Long.MaxValue) do
        intercept[IllegalArgumentException](session.embedding(device,Tensor.fromLongs(Array(id),Shape.from(Seq(1L)))))
      assertEquals(session.allocatedBytes,before)
    }
    CudaSession.withDevice(maxBytes=40) { session =>
      val device=session.upload(table)
      intercept[IllegalArgumentException](session.embedding(device,ids))
      assertEquals(session.allocatedBytes,24L)
      intercept[IllegalArgumentException](session.transpose(device,0,1))
      assertEquals(session.allocatedBytes,24L)
    }
  }

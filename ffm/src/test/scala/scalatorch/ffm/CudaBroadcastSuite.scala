package scalatorch.ffm

import scalatorch.*

class CudaBroadcastSuite extends munit.FunSuite:
  test("broadcast addition matches CPU bias and timestep layouts without expanded inputs") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"real GPU test disabled")
    CudaSession.withDevice() { s =>
      for (ad,bd) <- Seq((Seq(2L,3L,4L,5L),Seq(1L,3L,1L,1L)),
          (Seq(2L,3L,4L,5L),Seq(2L,3L,1L,1L)),(Seq(2L,3L),Seq(3L)),
          (Seq(2L,1L),Seq(1L,3L)),(Seq.empty[Long],Seq(2L,3L)),(Seq(0L,3L),Seq(1L,3L))) do
        def data(d: Seq[Long]) =
          val shape=Shape.from(d)
          Tensor.fromFloats(Array.tabulate(shape.numel.toInt)(i => (i-5).toFloat/3),shape)
        val a=data(ad);val b=data(bd);val da=s.upload(a);val db=s.upload(b)
        val out=s.addBroadcast(da,db)
        assertEquals(out.shape,Ops.add(a,b).shape)
        assertEquals(out.toCpu.toFloatArray.toVector,Ops.add(a,b).toFloatArray.toVector)
        out.close();db.close();da.close()
      assertEquals(s.allocatedBytes,0L)
    }
    CudaSession.withDevice(maxBytes=48) { s =>
      val a=s.fill(Shape.from(Seq(2L,3L)),1f);val b=s.fill(Shape.from(Seq(3L)),1f)
      intercept[IllegalArgumentException](s.addBroadcast(a,b))
      assertEquals(s.allocatedBytes,36L)
      val incompatible=s.fill(Shape.from(Seq(2L)),0f)
      intercept[IllegalArgumentException](s.addBroadcast(a,incompatible))
      incompatible.close()
      assertEquals(s.allocatedBytes,36L)
      b.close();a.close()
      intercept[IllegalStateException](s.addBroadcast(a,b))
    }
  }

package scalatorch.ffm

import scalatorch.*

class CudaAttentionBackwardSuite extends munit.FunSuite:
  test("attention pullback preflights score and exact workspace budgets") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA gate disabled")
    for budget <- Seq(399L,400L) do
      CudaSession.withDevice(maxBytes=budget) { session =>
        val q=session.fill(Shape.from(Seq(1L,2L,2L,2L)),.3f)
        val k=session.fill(Shape.from(Seq(1L,1L,3L,2L)),.5f)
        val v=session.fill(k.shape,.7f);val g=session.fill(q.shape,1f)
        val before=session.allocatedBytes
        intercept[IllegalArgumentException](session.causalAttentionBackward(q,k,v,g,maxScoreElements=11))
        intercept[IllegalArgumentException](session.causalAttentionBackward(q,k,v,g,queryStart = -1))
        val wrong=session.reshape(g,Shape.from(Seq(2L,4L)))
        intercept[IllegalArgumentException](session.causalAttentionBackward(q,k,v,wrong))
        wrong.close()
        if budget==399 then intercept[IllegalArgumentException](session.causalAttentionBackward(q,k,v,g))
        else
          val (dq,dk,dv)=session.causalAttentionBackward(q,k,v,g)
          assertEquals(session.allocatedBytes,before+80)
          assert(dq.toCpu.toFloatArray.forall(_.isFinite))
          assert(dk.toCpu.toFloatArray.forall(_.isFinite))
          assert(dv.toCpu.toFloatArray.forall(_.isFinite))
          dq.close();dk.close();dv.close()
        assertEquals(session.allocatedBytes,before)
        g.close();v.close();k.close();q.close()
        intercept[IllegalStateException](session.causalAttentionBackward(q,k,v,g))
        assertEquals(session.allocatedBytes,0L)
      }
  }

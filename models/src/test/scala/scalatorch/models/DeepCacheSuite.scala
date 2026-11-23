package scalatorch.models

import munit.FunSuite

class DeepCacheSuite extends FunSuite:
  test("full steps first, then one full call per interval; call slots are tracked separately") {
    val state=new DeepCacheState(DeepCacheConfig(interval=3,branch=0,fullFirst=2))
    val dims=Seq(1L,4L,8L,8L)
    // Two calls per step (conditional and unconditional) share a timestep.
    val pattern=(0 until 8).map { step =>
      val t=1000f-step
      val (slotA,reuseA)=state.beginCall(t,dims)
      assertEquals(slotA,0)
      if !reuseA then state.record(0,dims,null)
      val (slotB,reuseB)=state.beginCall(t,dims)
      assertEquals(slotB,1)
      if !reuseB then state.record(1,dims,null)
      assertEquals(reuseA,reuseB)
      reuseA
    }
    // steps 0,1 full (full_first); then (step-2)%3==0 -> 2 full, 3,4 reuse, 5 full, 6,7 reuse
    assertEquals(pattern.toList,List(false,false,false,true,true,false,true,true))
    assertEquals(state.fullCalls+state.cachedCalls,16)
  }

  test("a rising timestep starts a new sampling: nothing recorded before is reused and the full-first steps count again") {
    val state=new DeepCacheState(DeepCacheConfig(interval=3,branch=0,fullFirst=1))
    val dims=Seq(2L,4L,8L,8L)
    def sampling(): List[Boolean] = (0 until 5).map { step =>
      val (slot,reuse)=state.beginCall(900f-step*200,dims)
      assertEquals(slot,0)
      if !reuse then state.record(0,dims,null)
      reuse
    }.toList
    // step 0 full (full_first), step 1 full ((1-1)%3==0), 2 and 3 reuse, 4 full
    val first=sampling()
    assertEquals(first,List(false,false,true,true,false))
    // The second image of a batch starts again at the highest timestep (its tensors belong to another session).
    assertEquals(sampling(),first)
    assertEquals(sampling(),first)
  }

  test("a changed input shape forces a full call") {
    val state=new DeepCacheState(DeepCacheConfig(2,0,0))
    assert(!state.beginCall(10f,Seq(1L,4L,8L,8L))._2)
    state.record(0,Seq(1L,4L,8L,8L),null)
    assert(!state.beginCall(9f,Seq(1L,4L,16L,16L))._2)
  }

  test("both CFG slots refresh after a low to high resolution transition") {
    val state=new DeepCacheState(DeepCacheConfig(interval=4,branch=0,fullFirst=0))
    val low=Seq(1L,4L,8L,8L)
    val high=Seq(1L,4L,16L,16L)
    def pair(t: Float,dims: Seq[Long]): Vector[Boolean] =
      (0 to 1).map { expectedSlot =>
        val (slot,reuse)=state.beginCall(t,dims)
        assertEquals(slot,expectedSlot)
        if !reuse then state.record(slot,dims,null)
        reuse
      }.toVector
    try
      assertEquals(pair(1000f,low),Vector(false,false))
      assertEquals(pair(900f,low),Vector(true,true))
      assertEquals(pair(800f,high),Vector(false,false))
      assertEquals(pair(700f,high),Vector(true,true))
    finally state.close()
  }

  test("config validation") {
    intercept[IllegalArgumentException](DeepCacheConfig(1,0,0))
    intercept[IllegalArgumentException](DeepCacheConfig(3,4,0))
    intercept[IllegalArgumentException](DeepCacheConfig(3,0,51))
    intercept[IllegalArgumentException](DeepCacheConfig(3,0,5,-1))
    intercept[IllegalArgumentException](DeepCacheConfig(3,0,5,3))
    assertEquals(DeepCacheConfig(3,0,5,2).order,2)
  }

  test("TaylorSeer order 1 and 2 forecast recovers linear and quadratic feature trajectories on GPU") {
    val hasCuda=try scalatorch.ffm.CudaProbe.memoryInfo(0).totalBytes>0 catch case _: Throwable => false
    assume(hasCuda,"CUDA device required")
    val shape=scalatorch.Shape.from(Seq(1L,4L,8L,8L))
    def vals(f: Float)=scalatorch.Tensor.fromFloats(Array.fill(256)(f),shape)
    // Quadratic trajectory y(s) = 1.0 + 0.5 * s + 0.25 * s^2 -> s=0: 1.0, s=1: 1.75, s=2: 3.0 -> s=3: 4.75, s=4: 7.0
    // Linear trajectory z(s) = 2.0 + 1.5 * s -> s=0: 2.0, s=1: 3.5 -> s=2: 5.0, s=3: 6.5
    scalatorch.ffm.CudaSession.withDevice(maxBytes=1L<<30) { session =>
      val state1=new DeepCacheState(DeepCacheConfig(interval=4,branch=0,fullFirst=2,order=1))
      try
        state1.beginCall(1000f,shape.dims)
        scala.util.Using.resource(session.uploadHalf(vals(2.0f)))(t => state1.recordWithSession(0,shape.dims,t,session))
        state1.beginCall(900f,shape.dims)
        scala.util.Using.resource(session.uploadHalf(vals(3.5f)))(t => state1.recordWithSession(0,shape.dims,t,session))
        state1.beginCall(800f,shape.dims)
        val out1=scala.util.Using.resource(state1.forecast(0,session))(_.toCpu.toFloatArray)
        assert(out1.forall(v => math.abs(v-5.0f)<1e-2),s"order 1 got ${out1.head}")
      finally state1.close()

      val state2=new DeepCacheState(DeepCacheConfig(interval=4,branch=0,fullFirst=3,order=2))
      try
        for (t,s) <- Seq((1000f,0),(900f,1),(800f,2)) do
          state2.beginCall(t,shape.dims)
          val y=(1.0+0.5*s+0.25*s*s).toFloat
          scala.util.Using.resource(session.uploadHalf(vals(y)))(ten => state2.recordWithSession(0,shape.dims,ten,session))
        state2.beginCall(700f,shape.dims)
        val out2=scala.util.Using.resource(state2.forecast(0,session))(_.toCpu.toFloatArray)
        assert(out2.forall(v => math.abs(v-4.75f)<1e-2),s"order 2 got ${out2.head}")
      finally state2.close()
    }
  }

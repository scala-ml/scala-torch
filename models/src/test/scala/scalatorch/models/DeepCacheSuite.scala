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

  test("a changed input shape forces a full call") {
    val state=new DeepCacheState(DeepCacheConfig(2,0,0))
    assert(!state.beginCall(10f,Seq(1L,4L,8L,8L))._2)
    state.record(0,Seq(1L,4L,8L,8L),null)
    assert(!state.beginCall(9f,Seq(1L,4L,16L,16L))._2)
  }

  test("config validation") {
    intercept[IllegalArgumentException](DeepCacheConfig(1,0,0))
    intercept[IllegalArgumentException](DeepCacheConfig(3,4,0))
    intercept[IllegalArgumentException](DeepCacheConfig(3,0,51))
  }

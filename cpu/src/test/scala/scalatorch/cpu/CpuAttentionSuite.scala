package scalatorch.cpu

import scalatorch.*

class CpuAttentionSuite extends munit.FunSuite:
  test("split-half rotation and causal grouped heads match hand calculations") {
    val x=Tensor.fromFloats(Array(1f,2f,3f,4f),Shape.from(Seq(1L,1L,1L,4L)))
    val rotated=CpuFloatBackend.rope(x,1,10000).toFloatArray
    val angles=Array(1.0,.01)
    for d <- 0 until 2 do
      assertEqualsDouble(rotated(d),(x.toFloatArray(d)*math.cos(angles(d))-x.toFloatArray(d+2)*math.sin(angles(d))).toFloat,1e-6)
    val q=Tensor.fromFloats(new Array[Float](8),Shape.from(Seq(1L,2L,2L,2L)))
    val k=Tensor.fromFloats(new Array[Float](4),Shape.from(Seq(1L,1L,2L,2L)))
    val v=Tensor.fromFloats(Array(2f,4f,6f,8f,10f,12f),Shape.from(Seq(1L,1L,2L,3L)))
    assertEquals(CpuFloatBackend.causalAttention(q,k,v).toFloatArray.toVector,
      Vector(2f,4f,6f,5f,7f,9f,2f,4f,6f,5f,7f,9f))
    assertEquals(CpuFloatBackend.causalAttention(q,k,v,queryStart=1).toFloatArray.toVector,
      Vector(5f,7f,9f,5f,7f,9f,5f,7f,9f,5f,7f,9f))
  }
  test("attention and rotary reject invalid shapes, offsets and budgets") {
    val x=Tensor.fromFloats(Array(1f,2f),Shape.from(Seq(1L,1L,1L,2L)))
    intercept[IllegalArgumentException](CpuFloatBackend.rope(x,-1))
    intercept[IllegalArgumentException](CpuFloatBackend.rope(x,theta=Double.NaN))
    intercept[IllegalArgumentException](CpuFloatBackend.causalAttention(x,x,x,maxScoreElements=0))
    intercept[IllegalArgumentException](CpuFloatBackend.causalAttention(x,x,x,queryStart = -1))
  }

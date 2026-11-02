package scalatorch.cpu

import scalatorch.*

class CpuFloatBackendSuite extends munit.FunSuite:
  private def tensor(values: Float*) = Tensor.fromFloats(values.toArray,Shape.from(Seq(2L,2L)))
  test("backend arithmetic and stable normalization match hand calculations") {
    val backend: FloatBackend[Tensor[Float]]=CpuFloatBackend
    val a=tensor(1,2,3,4);val b=tensor(2,3,4,5)
    assertEquals(backend.add(a,b).toFloatArray.toVector,Vector(3f,5f,7f,9f))
    assertEquals(backend.multiply(a,b).toFloatArray.toVector,Vector(2f,6f,12f,20f))
    assertEquals(backend.matmul(a,b).toFloatArray.toVector,Vector(10f,13f,22f,29f))
    val norm=backend.rmsNorm(tensor(3,4,0,0),Tensor.fromFloats(Array(2f,3f),Shape.from(Seq(2L))),.5)
    val expected=Vector((6/math.sqrt(13)).toFloat,(12/math.sqrt(13)).toFloat,0f,0f)
    assertEquals(norm.toFloatArray.toVector,expected)
    assertEquals(backend.shape(a),a.shape)
    backend.release(a)
    assertEquals(backend.download(backend.upload(a)).toFloatArray.toVector,a.toFloatArray.toVector)
  }
  test("SiLU handles signs, infinities and NaN without overflow") {
    val result=CpuFloatBackend.silu(Tensor.fromFloats(Array(-1f,0f,1f,Float.NegativeInfinity,Float.PositiveInfinity,Float.NaN),Shape.from(Seq(6L)))).toFloatArray
    assertEqualsDouble(result(0),-.26894142137,1e-7)
    assertEqualsDouble(result(2),.73105857863,1e-7)
    assertEquals(java.lang.Float.floatToRawIntBits(result(3)),0x80000000)
    assertEquals(result(4),Float.PositiveInfinity)
    assert(result(5).isNaN)
  }
  test("backend rejects implicit broadcasting and invalid normalization") {
    val a=tensor(1,2,3,4)
    val w=Tensor.fromFloats(Array(1f,1f),Shape.from(Seq(2L)))
    intercept[IllegalArgumentException](CpuFloatBackend.add(a,w))
    intercept[IllegalArgumentException](CpuFloatBackend.multiply(a,w))
    for epsilon <- Seq(0.0,Double.NaN,Double.PositiveInfinity) do
      intercept[IllegalArgumentException](CpuFloatBackend.rmsNorm(a,w,epsilon))
    intercept[IllegalArgumentException](CpuFloatBackend.rmsNorm(a,a,1e-6))
  }

  test("embedding gathers repeated scalar and batched IDs with checked bounds") {
    val table=Tensor.fromFloats(Array(1f,2f,3f,4f,5f,6f),Shape.from(Seq(3L,2L)))
    val ids=Tensor.fromLongs(Array(2L,0L,2L,1L),Shape.from(Seq(2L,2L)))
    val result=CpuFloatBackend.embedding(table,ids)
    assertEquals(result.shape,Shape.from(Seq(2L,2L,2L)))
    assertEquals(result.toFloatArray.toVector,Vector(5f,6f,1f,2f,5f,6f,3f,4f))
    assertEquals(CpuFloatBackend.embedding(table,Tensor.fromLongs(Array(1L),Shape.from(Seq.empty))).toFloatArray.toVector,Vector(3f,4f))
    for id <- Seq(-1L,3L,Long.MaxValue) do
      intercept[IllegalArgumentException](CpuFloatBackend.embedding(table,Tensor.fromLongs(Array(id),Shape.from(Seq(1L)))))
  }

  test("backend reshape follows logical transpose order and rejects size changes") {
    val value=Tensor.fromFloats(Array(1f,2f,3f,4f,5f,6f),Shape.from(Seq(2L,3L)))
    val transposed=CpuFloatBackend.transpose(value,0,1)
    val result=CpuFloatBackend.reshape(transposed,Shape.from(Seq(6L)))
    assertEquals(result.toFloatArray.toVector,Vector(1f,4f,2f,5f,3f,6f))
    intercept[IllegalArgumentException](CpuFloatBackend.reshape(value,Shape.from(Seq(5L))))
  }

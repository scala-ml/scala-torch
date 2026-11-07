package scalatorch.io

import scalatorch.*

class SafeTensorMemorySuite extends munit.FunSuite:
  test("single float tensor round trips exactly and rejects malformed payloads") {
    val values=Array(0f,-0f,1.25f,Float.MinPositiveValue)
    val tensor=Tensor.fromFloats(values,Shape.from(Seq(1,4,1,1)))
    val data=SafeTensorMemory.writeFloat(tensor)
    val decoded=SafeTensorMemory.readFloat(data)
    assertEquals(decoded.shape,tensor.shape)
    assertEquals(decoded.toFloatArray.map(java.lang.Float.floatToRawIntBits).toVector,
      values.map(java.lang.Float.floatToRawIntBits).toVector)
    intercept[IllegalArgumentException](SafeTensorMemory.readFloat(data.dropRight(1)))
    intercept[IllegalArgumentException](SafeTensorMemory.readFloat(new Array[Byte](8)))
  }

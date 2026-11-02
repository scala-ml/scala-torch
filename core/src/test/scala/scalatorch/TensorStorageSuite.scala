package scalatorch

class TensorStorageSuite extends munit.FunSuite:
  test("contiguous offset extraction preserves raw bits and independent primitive buffers") {
    val bits=Array(0x7fc01234,0x80000000,0x7f800000,0x3f800000)
    val floats=Tensor.fromFloats(bits.map(java.lang.Float.intBitsToFloat),Shape.from(Seq(2L,2L)))
    val view=floats.slice(0,1,2)
    assertEquals(view.toFloatArray.map(java.lang.Float.floatToRawIntBits).toVector,bits.drop(2).toVector)
    val copy=floats.toFloatArray
    assertEquals(copy.map(java.lang.Float.floatToRawIntBits).toVector,bits.toVector)
    copy(0)=0f
    assertEquals(java.lang.Float.floatToRawIntBits(floats.toFloatArray(0)),bits(0))
    val longs=Tensor.fromLongs(Array(0L,Long.MinValue,Long.MaxValue),Shape.from(Seq(3L)))
    assertEquals(longs.slice(0,1,3).toLongArray.toVector,Vector(Long.MinValue,Long.MaxValue))
    val doubles=Tensor.fromDoubles(Array(0d,-0d,Double.NaN),Shape.from(Seq(3L)))
    assertEquals(doubles.slice(0,1,2).toDoubleArray.map(java.lang.Double.doubleToRawLongBits).toVector,Vector(Long.MinValue))
    val bools=Tensor.fromBooleans(Array(false,true,false),Shape.from(Seq(3L)))
    val extracted=bools.slice(0,1,3).toBooleanArray
    assertEquals(extracted.toVector,Vector(true,false))
    extracted(0)=false
    assert(bools.toBooleanArray(1))
  }
  test("shape validates dimensions and checked size") {
    assertEquals(Shape.from(Seq.empty).numel, 1L)
    assertEquals(Shape.from(Seq(2L, 0L, 3L)).numel, 0L)
    intercept[IllegalArgumentException](Shape.from(Seq(-1L)))
    intercept[IllegalArgumentException](Shape.from(Seq(Long.MaxValue, 2L)))
    assertEquals(Shape.from(Seq(Long.MaxValue, 2L, 0L)).numel, 0L)
  }
  test("construction and extraction copy buffers") {
    val data = Array(1f, 2f, 3f, 4f)
    val x = Tensor.fromFloats(data, Shape.from(Seq(2L, 2L)))
    data(0) = 99f
    assertEquals(x.toFloatArray.toSeq, Seq(1f, 2f, 3f, 4f))
    val copy = x.toFloatArray
    copy(1) = 99f
    assertEquals(x.toFloatArray.toSeq, Seq(1f, 2f, 3f, 4f))
    assertEquals(x.dtype, DType.Float32)
    assertEquals(x.device, Device.Cpu)
    intercept[IllegalArgumentException](Tensor.fromFloats(data, Shape.from(Seq(3L))))
  }
  test("all initial dtypes retain their primitive values") {
    val shape = Shape.from(Seq(2L))
    val d = Tensor.fromDoubles(Array(1.0, Double.NaN), shape)
    assertEquals(d.dtype, DType.Float64)
    assert(d.toDoubleArray(1).isNaN)
    val l = Tensor.fromLongs(Array(Long.MaxValue, Long.MinValue), shape)
    assertEquals(l.toLongArray.toSeq, Seq(Long.MaxValue, Long.MinValue))
    assertEquals(l.dtype, DType.Int64)
    val b = Tensor.fromBooleans(Array(true, false), shape)
    assertEquals(b.toBooleanArray.toSeq, Seq(true, false))
    assertEquals(b.dtype, DType.Bool)
  }

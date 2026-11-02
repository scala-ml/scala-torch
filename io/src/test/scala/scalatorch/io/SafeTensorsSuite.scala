package scalatorch.io

import java.nio.{ByteBuffer, ByteOrder}
import java.nio.file.{Files, Path}
import java.nio.charset.StandardCharsets.UTF_8
import scala.util.Using

class SafeTensorsSuite extends munit.FunSuite:
  private def archive[A](header: String, payload: Array[Byte] = Array.emptyByteArray)(f: Path => A): A =
    val json = header.getBytes(UTF_8)
    val prefix = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(json.length.toLong).array()
    val path = Files.createTempFile("scala-torch-safetensors-", ".safetensors")
    Files.write(path, prefix ++ json ++ payload)
    try f(path)
    finally Files.deleteIfExists(path)
  private def info(dtype: String, shape: String, begin: Long, end: Long): String =
    s"""{"dtype":"$dtype","shape":$shape,"data_offsets":[$begin,$end]}"""
  private def bad(header: String, bytes: Array[Byte] = Array.emptyByteArray): Unit = archive(header, bytes) { path =>
    intercept[IllegalArgumentException](SafeTensors.open(path))
  }
  test("typed scalar and empty tensors with metadata") {
    val payload = ByteBuffer.allocate(21).order(ByteOrder.LITTLE_ENDIAN)
      .putFloat(1.5f).putDouble(-2.25).putLong(9007199254740993L).put(1.toByte).array()
    val header = s"""{"__metadata__":{"format":"pt"},"f":${info("F32", "[]", 0, 4)},"d":${info("F64", "[]", 4, 12)},"l":${info("I64", "[]", 12, 20)},"b":${info("BOOL", "[]", 20, 21)},"z":${info("F32", "[2,0]", 21, 21)}}   """
    archive(header, payload) { path => Using.resource(SafeTensors.open(path)) { file =>
      assertEquals(file.metadata, Map("format" -> "pt"))
      assertEquals(file.readFloat("f").toFloatArray.toSeq, Seq(1.5f))
      assertEquals(file.readDouble("d").toDoubleArray.toSeq, Seq(-2.25))
      assertEquals(file.readLong("l").toLongArray.toSeq, Seq(9007199254740993L))
      assertEquals(file.readBoolean("b").toBooleanArray.toSeq, Seq(true))
      assertEquals(file.readFloat("z").shape.dims, Vector(2L, 0L))
      intercept[IllegalArgumentException](file.readFloat("l"))
      intercept[IllegalArgumentException](file.readFloat("missing"))
    }}
  }
  test("FP16 and BF16 decode subnormals signed zero and nonfinite values") {
    val half = Array(0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3c00, 0xc000, 0x7bff, 0x7c00, 0xfc00, 0x7e01)
    val bf = Array(0x3f80, 0xc000, 0x0001, 0x8000, 0x7f80, 0xff80, 0x7fc1)
    val payload = ByteBuffer.allocate((half.length + bf.length) * 2).order(ByteOrder.LITTLE_ENDIAN)
    (half ++ bf).foreach(x => payload.putShort(x.toShort))
    archive(s"""{"h":${info("F16", s"[${half.length}]", 0, half.length * 2)},"b":${info("BF16", s"[${bf.length}]", half.length * 2, payload.capacity())}}""", payload.array()) { path =>
      Using.resource(SafeTensors.open(path)) { file =>
        val h = file.readFloat("h").toFloatArray
        assertEquals(h.take(8).toSeq, Seq(0f, -0f, math.pow(2, -24).toFloat, (1023 * math.pow(2, -24)).toFloat, math.pow(2, -14).toFloat, 1f, -2f, 65504f))
        assertEquals(java.lang.Float.floatToRawIntBits(h(1)), 0x80000000)
        assertEquals(h(8), Float.PositiveInfinity); assertEquals(h(9), Float.NegativeInfinity); assert(h(10).isNaN)
        val b = file.readFloat("b").toFloatArray
        assertEquals(b.take(3).toSeq, Seq(1f, -2f, math.pow(2, -133).toFloat))
        assertEquals(java.lang.Float.floatToRawIntBits(b(3)), 0x80000000)
        assertEquals(b(4), Float.PositiveInfinity); assertEquals(b(5), Float.NegativeInfinity); assert(b(6).isNaN)
      }
    }
  }
  test("header validation rejects duplicates wrong types unknown dtype and fractional integers") {
    bad("""{"a":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},"a":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}""")
    bad("""{"a":{"dtype":"F32","dtype":"F16","shape":[0],"data_offsets":[0,0]}}""")
    bad("""{"__metadata__":{"x":1}}""")
    bad(s"""{"x":${info("F8_UNKNOWN", "[0]", 0, 0)}}""")
    bad("""{"x":{"dtype":"F32","shape":[0.0],"data_offsets":[0,0]}}""")
    bad("""{"x":{"dtype":"F32","shape":[0],"data_offsets":[0,0.0]}}""")
    bad("{} {}")
    bad(" {}")
  }
  test("offsets shapes and file length validate fully before reads") {
    bad(s"""{"x":${info("F32", "[-1]", 0, 0)}}""")
    bad(s"""{"x":${info("F32", "[9223372036854775807,2]", 0, 0)}}""")
    bad(s"""{"x":${info("F64", "[9223372036854775807]", 0, 0)}}""")
    bad(s"""{"x":${info("F32", "[1]", 0, 3)}}""", new Array[Byte](3))
    bad(s"""{"a":${info("F32", "[1]", 0, 4)},"b":${info("F32", "[1]", 0, 4)}}""", new Array[Byte](4))
    bad(s"""{"x":${info("F32", "[1]", 1, 5)}}""", new Array[Byte](5))
    bad(s"""{"x":${info("F32", "[1]", 0, 4)}}""", new Array[Byte](3))
    bad("{}", Array[Byte](0))
    bad("""{"x":{"dtype":"F32","shape":[0],"data_offsets":[9007199254740993,9007199254740993]}}""")
  }
  test("limits lifetime and returned Tensor ownership") {
    archive(s"""{"x":${info("F32", "[2]", 0, 8)}}""", new Array[Byte](8)) { path =>
      intercept[IllegalArgumentException](SafeTensors.open(path, ReadLimits(maxHeaderBytes = 2)))
      intercept[IllegalArgumentException](SafeTensors.open(path, ReadLimits(maxTensors = 0)))
      intercept[IllegalArgumentException](SafeTensors.open(path, ReadLimits(maxRank = 0)))
      val limited = SafeTensors.open(path, ReadLimits(maxTensorBytes = 4))
      try intercept[IllegalArgumentException](limited.readFloat("x"))
      finally limited.close()
      val file = SafeTensors.open(path)
      val tensor = file.readFloat("x")
      file.close(); file.close()
      intercept[IllegalStateException](file.readFloat("x"))
      assertEquals(tensor.toFloatArray.toSeq, Seq(0f, 0f))
    }
    archive(s"""{"x":${info("F16", "[1]", 0, 2)}}""", Array[Byte](0, 0x3c)) { path =>
      Using.resource(SafeTensors.open(path, ReadLimits(maxTensorBytes = 2))) { file =>
        assertEquals(file.tensors("x").end, 2L)
        intercept[IllegalArgumentException](file.readFloat("x")) // decoded Float32 needs 4 bytes
      }
    }
  }
  test("payload decoding crosses chunk boundaries correctly") {
    val n = 20000
    val payload = ByteBuffer.allocate(n * 4).order(ByteOrder.LITTLE_ENDIAN)
    for i <- 0 until n do payload.putFloat(i.toFloat / 8)
    archive(s"""{"x":${info("F32", s"[$n]", 0, n * 4)}}""", payload.array()) { path =>
      Using.resource(SafeTensors.open(path)) { file =>
        assertEquals(file.readFloat("x").toFloatArray.toSeq, (0 until n).map(_.toFloat / 8))
      }
    }
  }
  test("upstream Rust fixture decodes without Python or Rust at runtime") {
    val path = Files.createTempFile("scala-torch-upstream-", ".safetensors")
    try
      Using.resource(getClass.getResourceAsStream("/safetensors/upstream-0.6.2.safetensors")) { input =>
        assert(input != null)
        Files.copy(input, path, java.nio.file.StandardCopyOption.REPLACE_EXISTING)
      }
      Using.resource(SafeTensors.open(path)) { file =>
        assertEquals(file.readFloat("weight").toFloatArray.toSeq, Seq(1f, 2f, 3f, 4f, 5f, 6f))
        assertEquals(file.readFloat("bias").toFloatArray.toSeq, Seq(10f, 20f))
        assertEquals(file.readFloat("half").toFloatArray.toSeq, Seq(1f, -2f, math.pow(2, -24).toFloat, 65504f))
        assert(file.readFloat("bfloat").toFloatArray.last.isNaN)
        assertEquals(file.readLong("long").toLongArray.toSeq, Seq(9007199254740993L, -7L))
        assertEquals(file.readDouble("double").toDoubleArray.toSeq, Seq(1.25, -2.5))
        assertEquals(file.readBoolean("bool").toBooleanArray.toSeq, Seq(false, true))
        assertEquals(file.readFloat("empty").shape.dims, Vector(0L, 3L))
      }
    finally Files.deleteIfExists(path)
  }
  test("invalid UTF8 surrogates nesting and unsigned lengths reject") {
    bad("""{"\ud800":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}""")
    bad("""{"x":{"dtype":"F32","shape":[18446744073709551615],"data_offsets":[0,0]}}""")
    bad("""{"x":{"dtype":"F32","shape":[[[[0]]]],"data_offsets":[0,0]}}""")
    archive("{}") { path =>
      for bytes <- Seq(Array.emptyByteArray, Array.fill[Byte](8)(-1), Array[Byte](2,0,0,0,0,0,0,0,0xc0.toByte,0xaf.toByte)) do
        Files.write(path, bytes)
        intercept[IllegalArgumentException](SafeTensors.open(path))
    }
  }
  test("other dtype decoders cross chunk boundaries") {
    val n = 40000
    val cases = Seq(
      ("F16", Array(0x3c00, 0xc000, 0x0001, 0x7bff), Seq(1f, -2f, math.pow(2, -24).toFloat, 65504f)),
      ("BF16", Array(0x3f80, 0xc000, 0x0001, 0x7f7f), Seq(1f, -2f, math.pow(2, -133).toFloat, ((2.0 - 1.0 / 128) * math.pow(2, 127)).toFloat)))
    cases.foreach { (dtype, words, expected) =>
      val payload = ByteBuffer.allocate(n * 2).order(ByteOrder.LITTLE_ENDIAN)
      for i <- 0 until n do payload.putShort(words(i % words.length).toShort)
      archive(s"""{"x":${info(dtype, s"[$n]", 0, n * 2)}}""", payload.array()) { path =>
        Using.resource(SafeTensors.open(path)) { file =>
          assertEquals(file.readFloat("x").toFloatArray.toSeq, (0 until n).map(i => expected(i % expected.size)))
        }
      }
    }
    val count = 10000
    val doubles = ByteBuffer.allocate(count * 8).order(ByteOrder.LITTLE_ENDIAN)
    for i <- 0 until count do doubles.putDouble(i.toDouble + 0.125)
    archive(s"""{"x":${info("F64", s"[$count]", 0, count * 8)}}""", doubles.array()) { path =>
      Using.resource(SafeTensors.open(path)) { file =>
        assertEquals(file.readDouble("x").toDoubleArray.toSeq, (0 until count).map(_.toDouble + 0.125))
      }
    }
  }
  test("metadata duplicates and malformed UTF8 within JSON reject") {
    bad("""{"__metadata__":{"a":"1","a":"2"}}""")
    bad("""{"__metadata__":{},"__metadata__":{}}""")
    archive("""{"__metadata__":{"a":"X"}}""") { path =>
      val bytes = Files.readAllBytes(path)
      val index = bytes.indexOf('X'.toByte, 8)
      assert(index >= 8)
      bytes(index) = 0xff.toByte
      Files.write(path, bytes)
      intercept[IllegalArgumentException](SafeTensors.open(path))
    }
  }
  test("shape integers above Double precision remain exact") {
    val size = 9007199254740993L
    archive(s"""{"x":${info("F32", s"[$size,0]", 0, 0)}}""") { path =>
      Using.resource(SafeTensors.open(path)) { file =>
        assertEquals(file.tensors("x").shape.dims, Vector(size, 0L))
        assertEquals(file.readFloat("x").shape.dims, Vector(size, 0L))
      }
    }
  }

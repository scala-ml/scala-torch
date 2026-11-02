package scalatorch.io

import java.nio.{ByteBuffer, ByteOrder}
import java.nio.file.{Files, Path}
import scala.util.Using

class ShardSuite extends munit.FunSuite:
  private def directory[A](f: Path => A): A =
    val dir = Files.createTempDirectory("scala-torch-shards-")
    try f(dir)
    finally
      Using.resource(Files.list(dir))(_.forEach(p => Files.deleteIfExists(p)))
      Files.deleteIfExists(dir)
  private def write(dir: Path, file: String, name: String, value: Float): Unit =
    val header = s"""{"$name":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}""".getBytes(java.nio.charset.StandardCharsets.UTF_8)
    val bytes = ByteBuffer.allocate(8+header.length+4).order(ByteOrder.LITTLE_ENDIAN).putLong(header.length).put(header).putFloat(value).array()
    Files.write(dir.resolve(file),bytes)
  private def index(dir: Path, map: String = "\"a\":\"a.safetensors\",\"b\":\"b.safetensors\"", metadata: String = "\"total_size\":8"): Unit =
    Files.writeString(dir.resolve("model.safetensors.index.json"),s"""{"metadata":{$metadata},"weight_map":{$map}}""")
  private def fixture(dir: Path): Unit =
    write(dir,"a.safetensors","a",1.25f); write(dir,"b.safetensors","b",-2f); index(dir)
  test("indexed float reads route correctly and copied output outlives close") {
    directory { dir =>
      fixture(dir)
      val set = SafeTensorSet.open(dir)
      assertEquals(set.tensors.keySet,Set("a","b"))
      val a = set.readFloat("a")
      assertEquals(set.readFloat("b").toFloatArray.toVector,Vector(-2f))
      intercept[IllegalArgumentException](set.readFloat("missing"))
      intercept[IllegalArgumentException](set.readLong("a"))
      set.close(); set.close()
      assertEquals(a.toFloatArray.toVector,Vector(1.25f))
      intercept[IllegalStateException](set.readFloat("a"))
      index(dir,metadata="")
      Using.resource(SafeTensorSet.open(dir))(s => assertEquals(s.tensors.size,2))
    }
  }
  test("single archive uses the same interface and ambiguity rejects") {
    directory { dir =>
      write(dir,"model.safetensors","a",3f)
      Using.resource(SafeTensorSet.open(dir))(s => assertEquals(s.readFloat("a").toFloatArray.toVector,Vector(3f)))
      index(dir)
      intercept[IllegalArgumentException](SafeTensorSet.open(dir))
    }
  }
  test("indexed typed reads preserve upstream fixture dtypes") {
    directory { dir =>
      val input = getClass.getResourceAsStream("/safetensors/upstream-0.6.2.safetensors")
      try Files.copy(input,dir.resolve("typed.safetensors")) finally input.close()
      val names = Using.resource(SafeTensors.open(dir.resolve("typed.safetensors")))(_.tensors.keys.toVector)
      index(dir,names.map(n => s"\"$n\":\"typed.safetensors\"").mkString(","),metadata="")
      Using.resource(SafeTensorSet.open(dir)) { set =>
        assertEquals(set.readDouble("double").toDoubleArray.toVector,Vector(1.25,-2.5))
        assertEquals(set.readLong("long").toLongArray.toVector,Vector(9007199254740993L,-7L))
        assertEquals(set.readBoolean("bool").toBooleanArray.toVector,Vector(false,true))
        assertEquals(set.readFloat("half").toFloatArray.head,1f)
      }
    }
  }
  test("index assignments names total bytes and aggregate limits validate") {
    directory { dir =>
      fixture(dir)
      for map <- Vector("\"a\":\"a.safetensors\",\"b\":\"a.safetensors\"", "\"a\":\"b.safetensors\",\"b\":\"a.safetensors\"", "\"x\":\"a.safetensors\",\"b\":\"b.safetensors\"", "\"a\":\"missing.safetensors\"") do
        index(dir,map)
        intercept[Exception](SafeTensorSet.open(dir))
      index(dir,metadata="\"total_size\":9")
      intercept[IllegalArgumentException](SafeTensorSet.open(dir))
      index(dir)
      intercept[IllegalArgumentException](SafeTensorSet.open(dir,ReadLimits(maxTensors=1)))
      intercept[IllegalArgumentException](SafeTensorSet.open(dir,shardLimits=ShardLimits(maxShards=1)))
      intercept[IllegalArgumentException](SafeTensorSet.open(dir,shardLimits=ShardLimits(maxIndexBytes=8)))
      write(dir,"b.safetensors","a",2f)
      intercept[IllegalArgumentException](SafeTensorSet.open(dir))
    }
  }
  test("index rejects malformed JSON metadata duplicate fields and escaping paths") {
    directory { dir =>
      fixture(dir)
      for file <- Vector("../a.safetensors","/tmp/a.safetensors","x/a.safetensors","x\\\\a.safetensors","C:a.safetensors","bad.bin","") do
        index(dir,s"\"a\":\"$file\"")
        intercept[IllegalArgumentException](SafeTensorSet.open(dir))
      for raw <- Vector("{}", "{", "{\"weight_map\":{}}", "{\"weight_map\":{\"a\":1}}", "{\"weight_map\":{\"a\":\"a.safetensors\",\"a\":\"a.safetensors\"}}", "{\"metadata\":{\"total_size\":8.0},\"weight_map\":{}}", "{\"metadata\":{\"total_size\":-1},\"weight_map\":{}}") do
        Files.writeString(dir.resolve("model.safetensors.index.json"),raw)
        intercept[IllegalArgumentException](SafeTensorSet.open(dir))
    }
  }
  test("later shard failure does not retain open descriptors") {
    directory { dir =>
      fixture(dir)
      Files.writeString(dir.resolve("b.safetensors"),"bad")
      val fds = Path.of("/proc/self/fd")
      assume(Files.isDirectory(fds),"Linux descriptor gate")
      def openForFixture: Long = Using.resource(Files.list(fds)) { stream =>
        stream.filter(p => try Files.readSymbolicLink(p).toString.startsWith(dir.toString) catch case _: java.io.IOException => false).count()
      }
      val before = openForFixture
      for _ <- 0 until 12 do intercept[Exception](SafeTensorSet.open(dir))
      assertEquals(openForFixture,before)
    }
  }

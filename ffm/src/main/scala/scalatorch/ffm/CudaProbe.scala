package scalatorch.ffm

import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.{ADDRESS, JAVA_INT, JAVA_LONG}
import java.nio.charset.StandardCharsets
import scala.util.Using

/** Windows/Linux 64-bit driver-only feasibility probes; not a Tensor CUDA backend. */
object CudaProbe:
  final case class Result(driverVersion: Int, deviceName: String, values: Array[Float])
  final case class MemoryInfo(freeBytes: Long,totalBytes: Long,driverVersion: Int,deviceName: String):
    def usedBytes: Long=totalBytes-freeBytes

  def roundTrip(values: Array[Float], deviceIndex: Int = 0): Result =
    require(values.nonEmpty, "probe requires nonempty input")
    CudaDriver.withDevice(deviceIndex) { (driver, version, name) =>
      val address = driver.upload(values)
      Result(version, name, driver.download(address, values.length))
    }

  /** Reads device-wide free and total VRAM through the CUDA Driver API. */
  def memoryInfo(deviceIndex: Int = 0): MemoryInfo =
    CudaDriver.withDevice(deviceIndex) { (driver,version,name) =>
      val (free,total)=driver.memoryInfo()
      MemoryInfo(free,total,version,name)
    }

  /** Packed row-major GEMM. Every invocation owns its context reference, buffers, stream and event. */
  def gemm(a: Array[Float], b: Array[Float], m: Int, n: Int, k: Int, deviceIndex: Int = 0): Result =
    require(m >= 0 && n >= 0 && k >= 0, "negative GEMM dimension")
    val count = m.toLong * n
    require(count <= Int.MaxValue, "GEMM output exceeds Int indexing")
    require(a.length.toLong == m.toLong * k && b.length.toLong == k.toLong * n, "GEMM input length mismatch")
    CudaDriver.withDevice(deviceIndex) { (driver, version, name) =>
      import driver.*
      val ad = upload(a)
      val bd = upload(b)
      val cd = allocateDevice(count * 4)
      val ptx = Using.resource(CudaProbe.getClass.getResourceAsStream("/scalatorch/gemm.ptx")) { input =>
        require(input != null, "GEMM PTX resource missing")
        new String(input.readAllBytes(), StandardCharsets.UTF_8)
      }
      val module = pointerResult("cuModuleLoadData", Seq(ADDRESS), arena.allocateFrom(ptx))
      cleanup { call("cuModuleUnload", Seq(ADDRESS), module) }
      val function = pointerResult("cuModuleGetFunction", Seq(ADDRESS, ADDRESS), module, arena.allocateFrom("gemm"))
      val stream = pointerResult("cuStreamCreate", Seq(JAVA_INT), Int.box(1))
      cleanup { call("cuStreamDestroy_v2", Seq(ADDRESS), stream) }
      val event = pointerResult("cuEventCreate", Seq(JAVA_INT), Int.box(2))
      cleanup { call("cuEventDestroy_v2", Seq(ADDRESS), event) }
      // Registered last: drain submitted work before any dependent resource is released, also on failure.
      cleanup { call("cuStreamSynchronize", Seq(ADDRESS), stream) }
      if count > 0 then
        val pointers = Seq(ad, bd, cd).map(v => arena.allocateFrom(JAVA_LONG, v))
        val dimensions = Seq(m, n, k).map(v => arena.allocateFrom(JAVA_INT, v))
        val params = arena.allocate(ADDRESS, 6)
        (pointers ++ dimensions).zipWithIndex.foreach { (p, i) => params.setAtIndex(ADDRESS, i.toLong, p) }
        call("cuLaunchKernel", Seq(ADDRESS) ++ Seq.fill(7)(JAVA_INT) ++ Seq.fill(3)(ADDRESS),
          function, Int.box(((count + 63) / 64).toInt), Int.box(1), Int.box(1),
          Int.box(64), Int.box(1), Int.box(1), Int.box(0), stream, params, MemorySegment.NULL)
      call("cuEventRecord", Seq(ADDRESS, ADDRESS), event, stream)
      call("cuEventSynchronize", Seq(ADDRESS), event)
      Result(version, name, download(cd, count.toInt))
    }

  def main(args: Array[String]): Unit =
    val result = gemm(Array(1f, 2f, 3f, 4f, 5f, 6f), Array(7f, 8f, 9f, 10f, 11f, 12f), 2, 2, 3)
    require(result.values.sameElements(Array(58f, 64f, 139f, 154f)), "CUDA GEMM mismatch")
    println(s"CUDA driver=${result.driverVersion} device=${result.deviceName} shape=2x3x2 GEMM=PASS event=PASS cleanup=PASS")

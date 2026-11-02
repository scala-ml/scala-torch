package scalatorch.ffm

import java.lang.foreign.{Arena, FunctionDescriptor, Linker, MemoryLayout, MemorySegment, SymbolLookup}
import java.lang.foreign.ValueLayout.{ADDRESS, JAVA_FLOAT, JAVA_INT, JAVA_LONG}
import scala.util.Using

/** Linux x86_64 driver-only feasibility probe; not a Tensor CUDA backend. */
object CudaProbe:
  final case class Result(driverVersion: Int, deviceName: String, values: Array[Float])

  def roundTrip(values: Array[Float], deviceIndex: Int = 0): Result =
    require(values.nonEmpty, "probe requires nonempty input")
    require(deviceIndex >= 0, "negative device index")
    require(System.getProperty("os.name").toLowerCase.contains("linux") && ADDRESS.byteSize() == 8,
      "probe currently supports Linux 64-bit only")
    Using.Manager { use =>
      val arena = use(Arena.ofConfined())
      val lookup = SymbolLookup.libraryLookup("libcuda.so.1", arena)
      val linker = Linker.nativeLinker()
      def call(name: String, layouts: Seq[MemoryLayout], args: AnyRef*): Unit =
        val symbol = lookup.find(name).orElseThrow(() => UnsatisfiedLinkError(s"CUDA symbol missing: $name"))
        val handle = linker.downcallHandle(symbol, FunctionDescriptor.of(JAVA_INT, layouts*))
        val code = handle.invokeWithArguments(args*).asInstanceOf[Integer].intValue()
        if code != 0 then throw IllegalStateException(s"$name failed with CUDA status $code")
      def cleanup(action: => Unit): Unit =
        use(new AutoCloseable { def close(): Unit = action })
        ()
      call("cuInit", Seq(JAVA_INT), Int.box(0))
      val version = arena.allocate(JAVA_INT)
      call("cuDriverGetVersion", Seq(ADDRESS), version)
      val device = arena.allocate(JAVA_INT)
      call("cuDeviceGet", Seq(ADDRESS, JAVA_INT), device, Int.box(deviceIndex))
      val dev = device.get(JAVA_INT, 0L)
      val name = arena.allocate(256)
      call("cuDeviceGetName", Seq(ADDRESS, JAVA_INT, JAVA_INT), name, Int.box(256), Int.box(dev))
      val previous = arena.allocate(ADDRESS)
      call("cuCtxGetCurrent", Seq(ADDRESS), previous)
      val context = arena.allocate(ADDRESS)
      call("cuDevicePrimaryCtxRetain", Seq(ADDRESS, JAVA_INT), context, Int.box(dev))
      cleanup { call("cuDevicePrimaryCtxRelease_v2", Seq(JAVA_INT), Int.box(dev)) }
      call("cuCtxSetCurrent", Seq(ADDRESS), context.get(ADDRESS, 0L))
      cleanup { call("cuCtxSetCurrent", Seq(ADDRESS), previous.get(ADDRESS, 0L)) }
      val bytes = values.length.toLong * JAVA_FLOAT.byteSize()
      val pointer = arena.allocate(JAVA_LONG)
      call("cuMemAlloc_v2", Seq(ADDRESS, JAVA_LONG), pointer, Long.box(bytes))
      val address = pointer.get(JAVA_LONG, 0L)
      cleanup { call("cuMemFree_v2", Seq(JAVA_LONG), Long.box(address)) }
      val host = arena.allocateFrom(JAVA_FLOAT, values*)
      val out = arena.allocate(bytes, JAVA_FLOAT.byteAlignment())
      call("cuMemcpyHtoD_v2", Seq(JAVA_LONG, ADDRESS, JAVA_LONG), Long.box(address), host, Long.box(bytes))
      call("cuMemcpyDtoH_v2", Seq(ADDRESS, JAVA_LONG, JAVA_LONG), out, Long.box(address), Long.box(bytes))
      // Synchronous copies complete before device memory and host arena are released.
      Result(version.get(JAVA_INT, 0L), name.getString(0L), out.toArray(JAVA_FLOAT))
    }.get

  def main(args: Array[String]): Unit =
    val input = Array(1f, -2f, 3f, 4f)
    val result = roundTrip(input)
    require(result.values.sameElements(input), "CUDA copy round-trip mismatch")
    println(s"CUDA driver=${result.driverVersion} device=${result.deviceName} bytes=${input.length * 4} roundTrip=PASS cleanup=PASS")

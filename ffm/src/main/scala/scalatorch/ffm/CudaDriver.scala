package scalatorch.ffm

import java.lang.foreign.{Arena, FunctionDescriptor, Linker, MemoryLayout, MemorySegment, SymbolLookup}
import java.lang.foreign.ValueLayout.{ADDRESS, JAVA_FLOAT, JAVA_INT, JAVA_LONG}
import scala.util.Using

/** Internal, thread-confined driver session. Resources release in reverse acquisition order. */
private[ffm] final class CudaDriver private (
    val arena: Arena, register: (() => Unit) => Unit
):
  private val lookup = SymbolLookup.libraryLookup("libcuda.so.1", arena)
  private val linker = Linker.nativeLinker()
  def call(name: String, layouts: Seq[MemoryLayout], args: AnyRef*): Unit =
    val symbol = lookup.find(name).orElseThrow(() => UnsatisfiedLinkError(s"CUDA symbol missing: $name"))
    val handle = linker.downcallHandle(symbol, FunctionDescriptor.of(JAVA_INT, layouts*))
    val code = handle.invokeWithArguments(args*).asInstanceOf[Integer].intValue()
    if code != 0 then throw IllegalStateException(s"$name failed with CUDA status $code")
  def cleanup(action: => Unit): Unit = register(() => action)
  def pointerResult(name: String, layouts: Seq[MemoryLayout], args: AnyRef*): MemorySegment =
    val pointer = arena.allocate(ADDRESS)
    call(name, ADDRESS +: layouts, (pointer +: args)*)
    pointer.get(ADDRESS, 0L)
  def allocateDevice(bytes: Long): Long =
    val pointer = arena.allocate(JAVA_LONG)
    call("cuMemAlloc_v2", Seq(ADDRESS, JAVA_LONG), pointer, Long.box(math.max(bytes, 1L)))
    val address = pointer.get(JAVA_LONG, 0L)
    cleanup { call("cuMemFree_v2", Seq(JAVA_LONG), Long.box(address)) }
    address
  def upload(values: Array[Float]): Long =
    val address = allocateDevice(values.length.toLong * 4)
    if values.nonEmpty then
      val host = arena.allocateFrom(JAVA_FLOAT, values*)
      copyHostToDevice(address,host,values.length.toLong*4)
    address
  /** Pageable synchronous HtoD may return before DMA completes on the default stream.
    * Complete that transfer before a nonblocking compute stream can consume it.
    */
  def copyHostToDevice(address: Long,host: MemorySegment,bytes: Long): Unit =
    call("cuMemcpyHtoD_v2",Seq(JAVA_LONG,ADDRESS,JAVA_LONG),Long.box(address),host,Long.box(bytes))
    call("cuStreamSynchronize",Seq(ADDRESS),MemorySegment.NULL)
  def download(address: Long, count: Int): Array[Float] =
    val out = arena.allocate(count.toLong * 4, 4)
    if count > 0 then call("cuMemcpyDtoH_v2", Seq(ADDRESS, JAVA_LONG, JAVA_LONG), out, Long.box(address), Long.box(count.toLong * 4))
    out.toArray(JAVA_FLOAT)

private[ffm] object CudaDriver:
  def withDevice[A](index: Int)(body: (CudaDriver, Int, String) => A): A =
    require(index >= 0, "negative device index")
    require(System.getProperty("os.name").toLowerCase.contains("linux") && ADDRESS.byteSize() == 8,
      "probe currently supports Linux 64-bit only")
    Using.Manager { use =>
      val arena = use(Arena.ofConfined())
      val driver = new CudaDriver(arena, action => {
        use(new AutoCloseable { def close(): Unit = action() }); ()
      })
      import driver.*
      call("cuInit", Seq(JAVA_INT), Int.box(0))
      val version = arena.allocate(JAVA_INT)
      call("cuDriverGetVersion", Seq(ADDRESS), version)
      val device = arena.allocate(JAVA_INT)
      call("cuDeviceGet", Seq(ADDRESS, JAVA_INT), device, Int.box(index))
      val dev = device.get(JAVA_INT, 0L)
      val name = arena.allocate(256)
      call("cuDeviceGetName", Seq(ADDRESS, JAVA_INT, JAVA_INT), name, Int.box(256), Int.box(dev))
      val previous = pointerResult("cuCtxGetCurrent", Seq.empty)
      val context = pointerResult("cuDevicePrimaryCtxRetain", Seq(JAVA_INT), Int.box(dev))
      cleanup { call("cuDevicePrimaryCtxRelease_v2", Seq(JAVA_INT), Int.box(dev)) }
      call("cuCtxSetCurrent", Seq(ADDRESS), context)
      cleanup { call("cuCtxSetCurrent", Seq(ADDRESS), previous) }
      body(driver, version.get(JAVA_INT, 0L), name.getString(0L))
    }.get

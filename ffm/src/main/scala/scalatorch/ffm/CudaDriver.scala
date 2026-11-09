package scalatorch.ffm

import java.lang.foreign.{Arena, FunctionDescriptor, Linker, MemoryLayout, MemorySegment, SymbolLookup}
import java.lang.foreign.ValueLayout.{ADDRESS, JAVA_FLOAT, JAVA_INT, JAVA_LONG}
import scala.util.Using

/** Internal, thread-confined driver session. Resources release in reverse acquisition order. */
private[ffm] final class CudaDriver private (
    val arena: Arena, register: (() => Unit) => Unit
):
  private val lookup = SymbolLookup.libraryLookup(CudaDriver.libraryName, arena)
  private val linker = Linker.nativeLinker()
  private def result(name: String, layouts: Seq[MemoryLayout], args: AnyRef*): Int =
    val symbol = lookup.find(name).orElseThrow(() => UnsatisfiedLinkError(s"CUDA symbol missing: $name"))
    val handle = linker.downcallHandle(symbol, FunctionDescriptor.of(JAVA_INT, layouts*))
    handle.invokeWithArguments(args*).asInstanceOf[Integer].intValue()
  def call(name: String, layouts: Seq[MemoryLayout], args: AnyRef*): Unit =
    val code=result(name,layouts,args*)
    if code != 0 then throw IllegalStateException(s"$name failed with CUDA status $code")
  def loadModule(ptx: String): MemorySegment =
    val output=arena.allocate(ADDRESS)
    val options=arena.allocate(2L*JAVA_INT.byteSize(),JAVA_INT.byteAlignment())
    options.setAtIndex(JAVA_INT,0,5) // CU_JIT_ERROR_LOG_BUFFER
    options.setAtIndex(JAVA_INT,1,6) // CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
    val errorLog=arena.allocate(8192,1)
    val errorLogSize=arena.allocate(JAVA_INT)
    errorLogSize.set(JAVA_INT,0,8192)
    val values=arena.allocate(2L*ADDRESS.byteSize(),ADDRESS.byteAlignment())
    values.setAtIndex(ADDRESS,0,errorLog)
    values.setAtIndex(ADDRESS,1,errorLogSize)
    val image=arena.allocateFrom(ptx)
    val code=result("cuModuleLoadDataEx",Seq(ADDRESS,ADDRESS,JAVA_INT,ADDRESS,ADDRESS),
      output,image,Int.box(2),options,values)
    if code != 0 then
      val detail=errorLog.getString(0)
      throw IllegalStateException(s"cuModuleLoadDataEx failed with CUDA status $code${if detail.nonEmpty then s": $detail" else ""}")
    output.get(ADDRESS,0)
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
  def memoryInfo(): (Long,Long) =
    val free=arena.allocate(JAVA_LONG)
    val total=arena.allocate(JAVA_LONG)
    call("cuMemGetInfo_v2",Seq(ADDRESS,ADDRESS),free,total)
    val available=free.get(JAVA_LONG,0L)
    val capacity=total.get(JAVA_LONG,0L)
    require(available>=0 && capacity>0 && available<=capacity,"CUDA returned invalid device memory information")
    (available,capacity)
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
  private[ffm] def copyHostToDeviceProfiled(address: Long,host: MemorySegment,bytes: Long)
      (record: (String,Long) => Unit): Unit =
    var started=System.nanoTime()
    call("cuMemcpyHtoD_v2",Seq(JAVA_LONG,ADDRESS,JAVA_LONG),Long.box(address),host,Long.box(bytes))
    record("HtoD call",System.nanoTime()-started)
    started=System.nanoTime()
    call("cuStreamSynchronize",Seq(ADDRESS),MemorySegment.NULL)
    record("HtoD stream sync",System.nanoTime()-started)
  def download(address: Long, count: Int): Array[Float] =
    val out = arena.allocate(count.toLong * 4, 4)
    if count > 0 then call("cuMemcpyDtoH_v2", Seq(ADDRESS, JAVA_LONG, JAVA_LONG), out, Long.box(address), Long.box(count.toLong * 4))
    out.toArray(JAVA_FLOAT)

private[ffm] object CudaDriver:
  private def libraryName: String =
    val os = System.getProperty("os.name").toLowerCase(java.util.Locale.ROOT)
    require(ADDRESS.byteSize() == 8, "CUDA requires a 64-bit JVM")
    if os.startsWith("windows") then "nvcuda.dll"
    else if os == "linux" then "libcuda.so.1"
    else throw IllegalArgumentException(s"CUDA requires Windows or Linux, got $os")

  def withDevice[A](index: Int)(body: (CudaDriver, Int, String) => A): A =
    require(index >= 0, "negative device index")
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
      CudaContextKeeper.enter(dev)
      cleanup { CudaContextKeeper.exit(dev) }
      call("cuCtxSetCurrent", Seq(ADDRESS), context)
      cleanup { call("cuCtxSetCurrent", Seq(ADDRESS), previous) }
      body(driver, version.get(JAVA_INT, 0L), name.getString(0L))
    }.get
/** Keeps the CUDA primary context (and the cuBLAS handles created on it) alive between back-to-back sessions. A generation opens several
  * sessions (text encoder, U-Net, VAE); creating and destroying the context costs about 90 ms and the cuBLAS handle about 150 ms each time.
  * Once no session has been open for `SCALATORCH_CUDA_KEEP_CONTEXT_SECONDS` (default 20, 0 = never keep) the handles and the extra context
  * reference are dropped, so the device memory of the context is returned when no generation is running. */
private[ffm] object CudaContextKeeper:
  val seconds: Long = sys.env.get("SCALATORCH_CUDA_KEEP_CONTEXT_SECONDS").flatMap(_.trim.toLongOption).getOrElse(20L)
  def enabled: Boolean = seconds > 0
  private final class State(var context: MemorySegment, var active: Int, var lastExit: Long, val closers: java.util.ArrayList[() => Unit])
  private val states = new java.util.HashMap[Int, State]()
  private val scheduler = java.util.concurrent.Executors.newSingleThreadScheduledExecutor { r =>
    val t = new Thread(r, "scala-torch-cuda-context-keeper"); t.setDaemon(true); t
  }
  private lazy val handles: (java.lang.invoke.MethodHandle, java.lang.invoke.MethodHandle, java.lang.invoke.MethodHandle) =
    val os = System.getProperty("os.name").toLowerCase(java.util.Locale.ROOT)
    val lookup = SymbolLookup.libraryLookup(if os.startsWith("windows") then "nvcuda.dll" else "libcuda.so.1", Arena.global())
    val linker = Linker.nativeLinker()
    def find(symbol: String, d: FunctionDescriptor) = linker.downcallHandle(lookup.find(symbol).orElseThrow(), d)
    (find("cuDevicePrimaryCtxRetain", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT)),
      find("cuDevicePrimaryCtxRelease_v2", FunctionDescriptor.of(JAVA_INT, JAVA_INT)),
      find("cuCtxSetCurrent", FunctionDescriptor.of(JAVA_INT, ADDRESS)))

  /** A session starts on `device` (its own context reference is already taken): make sure our extra reference exists. */
  def enter(device: Int): Unit =
    if !enabled then return
    try states.synchronized {
      val st = states.get(device) match
        case null =>
          val created = new State(MemorySegment.NULL, 0, System.nanoTime(), new java.util.ArrayList[() => Unit]())
          states.put(device, created); created
        case existing => existing
      if st.context == MemorySegment.NULL then
        Using.resource(Arena.ofConfined()) { arena =>
          val out = arena.allocate(ADDRESS)
          if handles._1.invokeWithArguments(out, Int.box(device)).asInstanceOf[Integer].intValue() == 0 then st.context = out.get(ADDRESS, 0L)
        }
      st.active += 1
    } catch case _: Throwable => ()

  def exit(device: Int): Unit =
    if !enabled then return
    try states.synchronized {
      val st = states.get(device)
      if st != null then
        st.active = math.max(0, st.active - 1)
        st.lastExit = System.nanoTime()
        scheduler.schedule(new Runnable { def run(): Unit = release(device) }, seconds, java.util.concurrent.TimeUnit.SECONDS)
    } catch case _: Throwable => ()

  /** Registers a resource that lives on the kept context (for example a cuBLAS handle); closed before the context is released. */
  def onRelease(device: Int, closer: () => Unit): Unit =
    states.synchronized { val st = states.get(device); if st != null then st.closers.add(closer) else closer() }

  private def release(device: Int): Unit =
    states.synchronized {
      val st = states.get(device)
      if st != null && st.active == 0 && st.context != MemorySegment.NULL &&
          System.nanoTime() - st.lastExit >= seconds * 1_000_000_000L - 100_000_000L then
        try
          handles._3.invokeWithArguments(st.context)
          st.closers.forEach(c => try c() catch case _: Throwable => ())
          st.closers.clear()
          handles._3.invokeWithArguments(MemorySegment.NULL)
        catch case _: Throwable => ()
        try handles._2.invokeWithArguments(Int.box(device)) catch case _: Throwable => ()
        st.context = MemorySegment.NULL
    }

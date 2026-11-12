package scalatorch.ffm

import java.lang.foreign.{Arena, FunctionDescriptor, Linker, MemorySegment, SymbolLookup}
import java.lang.foreign.ValueLayout.{ADDRESS, JAVA_FLOAT, JAVA_INT, JAVA_LONG}
import java.lang.invoke.MethodHandle
import java.nio.file.{Files, Path, Paths}

/** Optional cuBLAS SGEMM, loaded at run time from the CUDA Toolkit. Absent or unusable libraries return None
  * so callers keep the built-in kernels. TF32 tensor-core math is opt-in (`SCALATORCH_CUDA_TF32=1`).
  */
private[ffm] final class Cublas private (handle: MemorySegment,sgemmHandle: MethodHandle,batchedHandle: MethodHandle,gemmExHandle: MethodHandle,gemmExBatchedHandle: MethodHandle,arena: Arena,setStreamHandle: MethodHandle):
  /** Points the (shared) handle at the stream of the session that is about to use it. */
  def bindStream(stream: MemorySegment): Unit =
    require(setStreamHandle.invokeWithArguments(handle,stream).asInstanceOf[Integer].intValue()==0,"cublasSetStream_v2 failed")
  private val alpha=arena.allocate(JAVA_FLOAT); alpha.set(JAVA_FLOAT,0,1f)
  private val beta=arena.allocate(JAVA_FLOAT); beta.set(JAVA_FLOAT,0,0f)
  /** Row-major C[m,n] = A[m,k] * B[k,n] with leading dimensions lda, ldb, ldc (column-major trick). */
  def sgemmRowMajor(m: Int,n: Int,k: Int,a: Long,lda: Int,b: Long,ldb: Int,c: Long,ldc: Int): Unit =
    val status=sgemmHandle.invokeWithArguments(handle,Int.box(0),Int.box(0),Int.box(n),Int.box(m),Int.box(k),alpha,
      MemorySegment.ofAddress(b),Int.box(ldb),MemorySegment.ofAddress(a),Int.box(lda),beta,
      MemorySegment.ofAddress(c),Int.box(ldc)).asInstanceOf[Integer].intValue()
    if status!=0 then throw IllegalStateException(s"cublasSgemm_v2 failed with status $status")
  /** Column-major strided-batched SGEMM: C = alpha * op(A) * op(B). Transpose flags use cuBLAS codes 0 (N) / 1 (T). */
  def sgemmStridedBatched(transA: Int,transB: Int,m: Int,n: Int,k: Int,alpha: Float,
      a: Long,lda: Int,strideA: Long,b: Long,ldb: Int,strideB: Long,c: Long,ldc: Int,strideC: Long,batch: Int): Unit =
    val scale=arena.allocate(JAVA_FLOAT); scale.set(JAVA_FLOAT,0,alpha)
    val status=batchedHandle.invokeWithArguments(handle,Int.box(transA),Int.box(transB),Int.box(m),Int.box(n),Int.box(k),scale,
      MemorySegment.ofAddress(a),Int.box(lda),Long.box(strideA),MemorySegment.ofAddress(b),Int.box(ldb),Long.box(strideB),beta,
      MemorySegment.ofAddress(c),Int.box(ldc),Long.box(strideC),Int.box(batch)).asInstanceOf[Integer].intValue()
    if status!=0 then throw IllegalStateException(s"cublasSgemmStridedBatched failed with status $status")

  /** Row-major C[m,n] = A[m,k] * W[n,k]^T in Float32. */
  def sgemmRowMajorBT(m: Int,n: Int,k: Int,a: Long,w: Long,c: Long): Unit =
    val status=sgemmHandle.invokeWithArguments(handle,Int.box(1),Int.box(0),Int.box(n),Int.box(m),Int.box(k),alpha,
      MemorySegment.ofAddress(w),Int.box(k),MemorySegment.ofAddress(a),Int.box(k),beta,
      MemorySegment.ofAddress(c),Int.box(n)).asInstanceOf[Integer].intValue()
    if status!=0 then throw IllegalStateException(s"cublasSgemm_v2 (transposed) failed with status $status")

  /** Row-major C[m,n] (Float32) = A[m,k] * W[n,k]^T with Float16 operands and Float32 accumulation on tensor cores. */
  def gemmExHalfBT(m: Int,n: Int,k: Int,a: Long,w: Long,c: Long): Unit =
    val status=gemmExHandle.invokeWithArguments(handle,Int.box(1),Int.box(0),Int.box(n),Int.box(m),Int.box(k),alpha,
      MemorySegment.ofAddress(w),Int.box(2),Int.box(k),MemorySegment.ofAddress(a),Int.box(2),Int.box(k),beta,
      MemorySegment.ofAddress(c),Int.box(0),Int.box(n),Int.box(68),Int.box(99)).asInstanceOf[Integer].intValue()
    if status!=0 then throw IllegalStateException(s"cublasGemmEx failed with status $status")

  /** Row-major C[m,n] = A[m,k] * B[k,n], all Float16, Float32 accumulation (tensor cores). */
  def gemmHalfRowMajor(m: Int,n: Int,k: Int,a: Long,lda: Int,b: Long,ldb: Int,c: Long,ldc: Int): Unit =
    val status: Int=gemmExHandle.invokeExact(handle,0,0,n,m,k,alpha,
      MemorySegment.ofAddress(b),2,ldb,MemorySegment.ofAddress(a),2,lda,beta,
      MemorySegment.ofAddress(c),2,ldc,68,99)
    if status!=0 then throw IllegalStateException(s"cublasGemmEx (half) failed with status $status")

  /** Row-major C[m,n] = A[m,k] * W[n,k]^T, all Float16, Float32 accumulation. */
  def gemmHalfRowMajorBT(m: Int,n: Int,k: Int,a: Long,w: Long,c: Long,accumulate: Boolean=false): Unit =
    // invokeExact: no boxing and no argument array (invokeWithArguments cost tens of microseconds per call; a U-Net pass makes hundreds)
    val status: Int=gemmExHandle.invokeExact(handle,1,0,n,m,k,alpha,
      MemorySegment.ofAddress(w),2,k,MemorySegment.ofAddress(a),2,k,if accumulate then alpha else beta,
      MemorySegment.ofAddress(c),2,n,68,99)
    if status!=0 then throw IllegalStateException(s"cublasGemmEx (half, transposed) failed with status $status")

  /** Column-major strided-batched GEMM, all Float16 with Float32 accumulation. Transpose codes 0 (N) / 1 (T). */
  def gemmHalfStridedBatched(transA: Int,transB: Int,m: Int,n: Int,k: Int,alphaValue: Float,
      a: Long,lda: Int,strideA: Long,b: Long,ldb: Int,strideB: Long,c: Long,ldc: Int,strideC: Long,batch: Int): Unit =
    alpha.set(JAVA_FLOAT,0,alphaValue)
    try
      val status=gemmExBatchedHandle.invokeWithArguments(handle,Int.box(transA),Int.box(transB),Int.box(m),Int.box(n),Int.box(k),alpha,
        MemorySegment.ofAddress(a),Int.box(2),Int.box(lda),Long.box(strideA),
        MemorySegment.ofAddress(b),Int.box(2),Int.box(ldb),Long.box(strideB),beta,
        MemorySegment.ofAddress(c),Int.box(2),Int.box(ldc),Long.box(strideC),Int.box(batch),Int.box(68),Int.box(99))
        .asInstanceOf[Integer].intValue()
      if status!=0 then throw IllegalStateException(s"cublasGemmStridedBatchedEx failed with status $status")
    finally alpha.set(JAVA_FLOAT,0,1f)

private[ffm] object Cublas:
  /** Where cuBLAS is looked for: the search path first (bare names), then `SCALATORCH_CUDNN_PATH` (a PyTorch `torch/lib` holds
    * cuBLAS next to cuDNN), the CUDA Toolkit of `CUDA_PATH` / `CUDA_HOME`, and on Windows every Toolkit under Program Files. Before,
    * only the search path and `CUDA_PATH` were tried, and a PC without them had no cuBLAS at all. */
  private[ffm] def candidates(variables: Map[String,String]=sys.env): Seq[String] =
    // Windows does not distinguish the case of variable names, and a parent process may pass PROGRAMFILES or PATH in capitals.
    val env=new scala.collection.immutable.TreeMap[String,String]()(using Ordering.by[String,String](_.toLowerCase(java.util.Locale.ROOT)))++variables
    val windows=System.getProperty("os.name").toLowerCase(java.util.Locale.ROOT).startsWith("windows")
    val names=if windows then Seq("cublas64_13.dll","cublas64_12.dll","cublas64_11.dll")
      else Seq("libcublas.so.13","libcublas.so.12","libcublas.so.11","libcublas.so")
    def path(text: String): Option[Path] = try Some(Paths.get(text.trim)) catch case _: Throwable => None
    def children(dir: Path): Seq[Path] =
      try
        val stream=Files.list(dir)
        try { val b=Seq.newBuilder[Path];stream.forEach(p => b+=p);b.result().sortBy(_.toString).reverse } finally stream.close()
      catch case _: Throwable => Seq.empty
    val explicit=env.get("SCALATORCH_CUDNN_PATH").toSeq.flatMap(path)
    val toolkit=(env.get("CUDA_PATH").toSeq ++ env.get("CUDA_HOME").toSeq).flatMap(path)
      .flatMap(r => Seq(r.resolve("bin"),r.resolve("bin").resolve("x64"),r.resolve("lib64")))
    val installed=if !windows then Seq.empty else env.get("ProgramFiles").toSeq.flatMap(path)
      .flatMap(r => children(r.resolve("NVIDIA GPU Computing Toolkit").resolve("CUDA"))).map(_.resolve("bin"))
    val full=for dir <- (explicit++toolkit++installed).distinct; name <- names; p=dir.resolve(name)
      if (try Files.isRegularFile(p) catch case _: Throwable => false) yield p.toAbsolutePath.toString
    names ++ full.distinct

  /** Loads the companion library of an absolute cuBLAS path first: cuBLAS needs cublasLt, and Windows does not search the
    * directory of a library that was loaded by its full path. */
  private def preloadCompanion(name: String,arena: Arena): Unit =
    try
      val file=Paths.get(name)
      if file.isAbsolute then
        val base=file.getFileName.toString
        val companion=file.resolveSibling(base.replace("cublas64_","cublasLt64_").replace("libcublas.so","libcublasLt.so"))
        if companion!=file && Files.isRegularFile(companion) then SymbolLookup.libraryLookup(companion.toString,arena)
    catch case _: Throwable => ()

  /** True when some cuBLAS can be loaded in this process (checked once). Float16 execution needs it. */
  lazy val loadable: Boolean =
    !sys.env.get("SCALATORCH_CUDA_CUBLAS").contains("0") && candidates().exists { name =>
      try { preloadCompanion(name,Arena.global());SymbolLookup.libraryLookup(name,Arena.global());true }
      catch case error: Throwable =>
        if sys.env.get("SCALATORCH_CUDA_LIBRARY_LOG").contains("1") then System.err.println(s"[scala-torch] cuBLAS candidate $name: $error")
        false
    }

  private val shared=new java.util.concurrent.ConcurrentHashMap[(Int,Thread),Cublas]()

  /** Per (device, thread) cuBLAS handle that outlives the session (see [[CudaContextKeeper]]); creating one costs about 150 ms.
    * None when keeping is disabled, so the caller opens a private handle as before. */
  def openShared(device: Int,stream: MemorySegment): Option[Cublas] =
    if !CudaContextKeeper.enabled then return None
    val key=(device,Thread.currentThread())
    val existing=shared.get(key)
    if existing!=null then { existing.bindStream(stream); return Some(existing) }
    val created=open(Arena.global(),stream,destroy => CudaContextKeeper.onRelease(device,() => { shared.remove(key);destroy() }))
    created.foreach(shared.put(key,_))
    created

  /** Opens cuBLAS bound to `stream`; None when disabled or unavailable. `register` frees the handle. */
  def open(arena: Arena,stream: MemorySegment,register: (() => Unit) => Unit): Option[Cublas] =
    if sys.env.get("SCALATORCH_CUDA_CUBLAS").contains("0") then return None
    val linker=Linker.nativeLinker()
    candidates().iterator.map { name =>
      try
        preloadCompanion(name,arena)
        val lookup=SymbolLookup.libraryLookup(name,arena)
        def down(symbol: String,descriptor: FunctionDescriptor): MethodHandle =
          linker.downcallHandle(lookup.find(symbol).orElseThrow(() => UnsatisfiedLinkError(symbol)),descriptor)
        val create=down("cublasCreate_v2",FunctionDescriptor.of(JAVA_INT,ADDRESS))
        val destroy=down("cublasDestroy_v2",FunctionDescriptor.of(JAVA_INT,ADDRESS))
        val setStream=down("cublasSetStream_v2",FunctionDescriptor.of(JAVA_INT,ADDRESS,ADDRESS))
        val setMath=down("cublasSetMathMode",FunctionDescriptor.of(JAVA_INT,ADDRESS,JAVA_INT))
        val sgemm=down("cublasSgemm_v2",FunctionDescriptor.of(JAVA_INT,ADDRESS,JAVA_INT,JAVA_INT,JAVA_INT,JAVA_INT,JAVA_INT,
          ADDRESS,ADDRESS,JAVA_INT,ADDRESS,JAVA_INT,ADDRESS,ADDRESS,JAVA_INT))
        val batched=down("cublasSgemmStridedBatched",FunctionDescriptor.of(JAVA_INT,ADDRESS,JAVA_INT,JAVA_INT,JAVA_INT,JAVA_INT,JAVA_INT,
          ADDRESS,ADDRESS,JAVA_INT,JAVA_LONG,ADDRESS,JAVA_INT,JAVA_LONG,ADDRESS,ADDRESS,JAVA_INT,JAVA_LONG,JAVA_INT))
        val gemmEx=down("cublasGemmEx",FunctionDescriptor.of(JAVA_INT,ADDRESS,JAVA_INT,JAVA_INT,JAVA_INT,JAVA_INT,JAVA_INT,
          ADDRESS,ADDRESS,JAVA_INT,JAVA_INT,ADDRESS,JAVA_INT,JAVA_INT,ADDRESS,ADDRESS,JAVA_INT,JAVA_INT,JAVA_INT,JAVA_INT))
        val gemmExBatched=down("cublasGemmStridedBatchedEx",FunctionDescriptor.of(JAVA_INT,ADDRESS,JAVA_INT,JAVA_INT,JAVA_INT,JAVA_INT,JAVA_INT,
          ADDRESS,ADDRESS,JAVA_INT,JAVA_INT,JAVA_LONG,ADDRESS,JAVA_INT,JAVA_INT,JAVA_LONG,ADDRESS,ADDRESS,JAVA_INT,JAVA_INT,JAVA_LONG,
          JAVA_INT,JAVA_INT,JAVA_INT))
        val out=arena.allocate(ADDRESS)
        require(create.invokeWithArguments(out).asInstanceOf[Integer].intValue()==0,"cublasCreate_v2 failed")
        val handle=out.get(ADDRESS,0)
        register(() => { destroy.invokeWithArguments(handle); () })
        require(setStream.invokeWithArguments(handle,stream).asInstanceOf[Integer].intValue()==0,"cublasSetStream_v2 failed")
        if sys.env.get("SCALATORCH_CUDA_TF32").contains("1") then
          setMath.invokeWithArguments(handle,Int.box(3)) // CUBLAS_TF32_TENSOR_OP_MATH
        Some(new Cublas(handle,sgemm,batched,gemmEx,gemmExBatched,arena,setStream))
      catch case _: Throwable => None
    }.collectFirst { case Some(value) => value }

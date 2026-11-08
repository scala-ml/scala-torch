package scalatorch.ffm

import java.lang.foreign.{Arena, FunctionDescriptor, Linker, MemorySegment, SymbolLookup}
import java.lang.foreign.ValueLayout.{ADDRESS, JAVA_FLOAT, JAVA_INT, JAVA_LONG}
import java.lang.invoke.MethodHandle
import java.nio.file.{Files, Path, Paths}

/** Optional cuBLAS SGEMM, loaded at run time from the CUDA Toolkit. Absent or unusable libraries return None
  * so callers keep the built-in kernels. TF32 tensor-core math is opt-in (`SCALATORCH_CUDA_TF32=1`).
  */
private[ffm] final class Cublas private (handle: MemorySegment,sgemmHandle: MethodHandle,batchedHandle: MethodHandle,gemmExHandle: MethodHandle,gemmExBatchedHandle: MethodHandle,arena: Arena):
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
    val status=gemmExHandle.invokeWithArguments(handle,Int.box(0),Int.box(0),Int.box(n),Int.box(m),Int.box(k),alpha,
      MemorySegment.ofAddress(b),Int.box(2),Int.box(ldb),MemorySegment.ofAddress(a),Int.box(2),Int.box(lda),beta,
      MemorySegment.ofAddress(c),Int.box(2),Int.box(ldc),Int.box(68),Int.box(99)).asInstanceOf[Integer].intValue()
    if status!=0 then throw IllegalStateException(s"cublasGemmEx (half) failed with status $status")

  /** Row-major C[m,n] = A[m,k] * W[n,k]^T, all Float16, Float32 accumulation. */
  def gemmHalfRowMajorBT(m: Int,n: Int,k: Int,a: Long,w: Long,c: Long): Unit =
    val status=gemmExHandle.invokeWithArguments(handle,Int.box(1),Int.box(0),Int.box(n),Int.box(m),Int.box(k),alpha,
      MemorySegment.ofAddress(w),Int.box(2),Int.box(k),MemorySegment.ofAddress(a),Int.box(2),Int.box(k),beta,
      MemorySegment.ofAddress(c),Int.box(2),Int.box(n),Int.box(68),Int.box(99)).asInstanceOf[Integer].intValue()
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
  private def candidates: Seq[String] =
    val os=System.getProperty("os.name").toLowerCase(java.util.Locale.ROOT)
    val names=if os.startsWith("windows") then Seq("cublas64_13.dll","cublas64_12.dll","cublas64_11.dll")
      else Seq("libcublas.so.13","libcublas.so.12","libcublas.so.11","libcublas.so")
    val dirs=(sys.env.get("CUDA_PATH").toSeq ++ sys.env.get("CUDA_HOME").toSeq).flatMap { root =>
      Seq(Paths.get(root,"bin"),Paths.get(root,"bin","x64"),Paths.get(root,"lib64")) }
    val full=for dir <- dirs; name <- names; p=dir.resolve(name) if Files.exists(p) yield p.toString
    names ++ full

  /** Opens cuBLAS bound to `stream`; None when disabled or unavailable. `register` frees the handle. */
  def open(arena: Arena,stream: MemorySegment,register: (() => Unit) => Unit): Option[Cublas] =
    if sys.env.get("SCALATORCH_CUDA_CUBLAS").contains("0") then return None
    val linker=Linker.nativeLinker()
    candidates.iterator.map { name =>
      try
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
        Some(new Cublas(handle,sgemm,batched,gemmEx,gemmExBatched,arena))
      catch case _: Throwable => None
    }.collectFirst { case Some(value) => value }

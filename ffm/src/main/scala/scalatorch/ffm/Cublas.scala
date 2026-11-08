package scalatorch.ffm

import java.lang.foreign.{Arena, FunctionDescriptor, Linker, MemorySegment, SymbolLookup}
import java.lang.foreign.ValueLayout.{ADDRESS, JAVA_FLOAT, JAVA_INT, JAVA_LONG}
import java.lang.invoke.MethodHandle
import java.nio.file.{Files, Path, Paths}

/** Optional cuBLAS SGEMM, loaded at run time from the CUDA Toolkit. Absent or unusable libraries return None
  * so callers keep the built-in kernels. TF32 tensor-core math is opt-in (`SCALATORCH_CUDA_TF32=1`).
  */
private[ffm] final class Cublas private (handle: MemorySegment,sgemmHandle: MethodHandle,batchedHandle: MethodHandle,arena: Arena):
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
        val out=arena.allocate(ADDRESS)
        require(create.invokeWithArguments(out).asInstanceOf[Integer].intValue()==0,"cublasCreate_v2 failed")
        val handle=out.get(ADDRESS,0)
        register(() => { destroy.invokeWithArguments(handle); () })
        require(setStream.invokeWithArguments(handle,stream).asInstanceOf[Integer].intValue()==0,"cublasSetStream_v2 failed")
        if sys.env.get("SCALATORCH_CUDA_TF32").contains("1") then
          setMath.invokeWithArguments(handle,Int.box(3)) // CUBLAS_TF32_TENSOR_OP_MATH
        Some(new Cublas(handle,sgemm,batched,arena))
      catch case _: Throwable => None
    }.collectFirst { case Some(value) => value }

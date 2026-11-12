package scalatorch.ffm

import java.lang.foreign.{Arena, FunctionDescriptor, Linker, MemorySegment, SymbolLookup}
import java.lang.foreign.ValueLayout.{ADDRESS, JAVA_FLOAT, JAVA_INT, JAVA_LONG}
import java.lang.invoke.MethodHandle
import java.nio.file.{Files, Path, Paths}

/** Optional cuDNN convolution (Float16, NCHW, tensor-core math), loaded at run time when the library is found. Absent or unusable
  * libraries give None and the caller keeps its own im2col + GEMM convolution.
  *
  * Search order: `SCALATORCH_CUDNN_PATH` (a directory), the directories of `CUDA_PATH` / `CUDA_HOME`, the cuDNN installer's directory
  * under Program Files, then the directories of the search path. It is not shipped with the library.
  * `SCALATORCH_CUDNN=0` disables it. */
private[ffm] final class Cudnn private (handle: MemorySegment, arena: Arena, h: Cudnn.Handles):
  private val one = arena.allocate(JAVA_FLOAT); one.set(JAVA_FLOAT, 0, 1f)
  private val zero = arena.allocate(JAVA_FLOAT); zero.set(JAVA_FLOAT, 0, 0f)
  private val sizeCell = arena.allocate(JAVA_LONG)

  import Cudnn.Plan
  private val plans = new java.util.HashMap[Seq[Int], Plan]()

  private def check(code: AnyRef, what: String): Unit =
    val status = code.asInstanceOf[Integer].intValue()
    if status != 0 then throw IllegalStateException(s"$what failed with cuDNN status $status")

  private def descriptor(create: MethodHandle, what: String): MemorySegment =
    val out = arena.allocate(ADDRESS)
    check(create.invokeWithArguments(out), what)
    out.get(ADDRESS, 0L)

  def bindStream(stream: MemorySegment): Unit = check(h.setStream.invokeWithArguments(handle, stream), "cudnnSetStream")

  /** The plan for a Float16 NCHW cross-correlation, or None when cuDNN has no usable algorithm for it. */
  def plan(n: Int, ci: Int, hh: Int, ww: Int, co: Int, cg: Int, kh: Int, kw: Int, oh: Int, ow: Int,
      stride: (Int, Int), padding: (Int, Int), dilation: (Int, Int), groups: Int, workspaceLimit: Long = Long.MaxValue): Option[Plan] =
    val key = Seq(n, ci, hh, ww, co, cg, kh, kw, oh, ow, stride._1, stride._2, padding._1, padding._2, dilation._1, dilation._2, groups)
    val known = plans.get(key)
    if known != null then return Option(known).filter(_.algo >= 0)
    val built =
      try
        val x = descriptor(h.createTensor, "cudnnCreateTensorDescriptor")
        check(h.setTensor.invokeWithArguments(x, Int.box(0), Int.box(2), Int.box(n), Int.box(ci), Int.box(hh), Int.box(ww)), "cudnnSetTensor4dDescriptor")
        val y = descriptor(h.createTensor, "cudnnCreateTensorDescriptor")
        check(h.setTensor.invokeWithArguments(y, Int.box(0), Int.box(2), Int.box(n), Int.box(co), Int.box(oh), Int.box(ow)), "cudnnSetTensor4dDescriptor")
        val w = descriptor(h.createFilter, "cudnnCreateFilterDescriptor")
        check(h.setFilter.invokeWithArguments(w, Int.box(2), Int.box(0), Int.box(co), Int.box(cg), Int.box(kh), Int.box(kw)), "cudnnSetFilter4dDescriptor")
        val conv = descriptor(h.createConv, "cudnnCreateConvolutionDescriptor")
        check(h.setConv.invokeWithArguments(conv, Int.box(padding._1), Int.box(padding._2), Int.box(stride._1), Int.box(stride._2),
          Int.box(dilation._1), Int.box(dilation._2), Int.box(1), Int.box(2)), "cudnnSetConvolution2dDescriptor")
        check(h.setMath.invokeWithArguments(conv, Int.box(1)), "cudnnSetConvolutionMathType")
        if groups != 1 then check(h.setGroups.invokeWithArguments(conv, Int.box(groups)), "cudnnSetConvolutionGroupCount")
        // cuDNN's own ranking for this geometry (the heuristic PyTorch uses without benchmarking). One fixed algorithm was up to ten
        // times slower on some shapes (16 x 1280 x 32 x 32: 16 ms, a convolution to 4 channels: 5.8 ms). Entries are 48 bytes:
        // algo, status, time, memory (at 16), determinism, mathType (at 28), reserved.
        val ranked =
          try
            val entries = 8
            val perf = arena.allocate(48L * entries, 8)
            val returned = arena.allocate(JAVA_INT)
            val status = h.ranking.invokeWithArguments(handle, x, w, conv, y, Int.box(entries), returned, perf).asInstanceOf[Integer].intValue()
            if status != 0 then Seq.empty
            else (0 until returned.get(JAVA_INT, 0L)).map { i =>
              val at = 48L * i
              (perf.get(JAVA_INT, at), perf.get(JAVA_INT, at + 4), perf.get(JAVA_LONG, at + 16), perf.get(JAVA_INT, at + 28))
            }.collect { case (algo, 0, memory, math) => (algo, memory, math) }
          catch case _: Throwable => Seq.empty
        ranked.find(_._2 <= workspaceLimit) match
          case Some((algo, memory, math)) =>
            check(h.setMath.invokeWithArguments(conv, Int.box(math)), "cudnnSetConvolutionMathType")
            new Plan(x, w, conv, y, algo, memory)
          case None =>
            // IMPLICIT_PRECOMP_GEMM (1) uses the tensor cores; IMPLICIT_GEMM (0) needs no workspace and always exists.
            Seq(1, 0).iterator.map { algo =>
              val status = h.workspace.invokeWithArguments(handle, x, w, conv, y, Int.box(algo), sizeCell).asInstanceOf[Integer].intValue()
              if status == 0 && sizeCell.get(JAVA_LONG, 0L) <= workspaceLimit then new Plan(x, w, conv, y, algo, sizeCell.get(JAVA_LONG, 0L)) else null
            }.find(_ != null).getOrElse(new Plan(x, w, conv, y, -1, 0L))
      catch case _: Throwable => new Plan(MemorySegment.NULL, MemorySegment.NULL, MemorySegment.NULL, MemorySegment.NULL, -1, 0L)
    plans.put(key, built)
    Option(built).filter(_.algo >= 0)

  def forward(plan: Plan, input: Long, weight: Long, workspace: Long, output: Long): Unit =
    check(h.forward.invokeWithArguments(handle, one, plan.x, MemorySegment.ofAddress(input), plan.w, MemorySegment.ofAddress(weight), plan.conv,
      Int.box(plan.algo), MemorySegment.ofAddress(workspace), Long.box(plan.workspace), zero, plan.y, MemorySegment.ofAddress(output)),
      "cudnnConvolutionForward")

  def close(): Unit = try h.destroy.invokeWithArguments(handle) catch case _: Throwable => ()

private[ffm] object Cudnn:
  /** Descriptors and algorithm of one convolution geometry; built once and reused. */
  final class Plan(val x: MemorySegment, val w: MemorySegment, val conv: MemorySegment, val y: MemorySegment, val algo: Int, val workspace: Long)
  final class Handles(val setStream: MethodHandle, val createTensor: MethodHandle, val setTensor: MethodHandle, val createFilter: MethodHandle,
      val setFilter: MethodHandle, val createConv: MethodHandle, val setConv: MethodHandle, val setMath: MethodHandle, val setGroups: MethodHandle,
      val workspace: MethodHandle, val forward: MethodHandle, val destroy: MethodHandle, val ranking: MethodHandle)

  val enabled: Boolean = !sys.env.get("SCALATORCH_CUDNN").contains("0")

  /** Complete cuDNN installations as absolute paths of the front library. A front library whose parts are missing aborts the process
    * when it is used, so a candidate counts only when its parts sit next to it; bare names from the search path are never tried. */
  private[ffm] def candidates(env: Map[String,String]=sys.env): Seq[String] =
    val windows = System.getProperty("os.name").toLowerCase(java.util.Locale.ROOT).startsWith("windows")
    // (front library, parts that must exist beside it)
    val layouts: Seq[(String, Seq[String])] =
      if windows then Seq(
        ("cudnn64_9.dll", Seq("cudnn_ops64_9.dll", "cudnn_cnn64_9.dll", "cudnn_graph64_9.dll")),
        ("cudnn64_8.dll", Seq("cudnn_ops_infer64_8.dll", "cudnn_cnn_infer64_8.dll")))
      else Seq(
        ("libcudnn.so.9", Seq("libcudnn_ops.so.9", "libcudnn_cnn.so.9", "libcudnn_graph.so.9")),
        ("libcudnn.so.8", Seq("libcudnn_ops_infer.so.8", "libcudnn_cnn_infer.so.8")))
    def children(dir: Path): Seq[Path] =
      try
        val stream = Files.list(dir)
        try { val b = Seq.newBuilder[Path]; stream.forEach(p => b += p); b.result().sortBy(_.toString).reverse } finally stream.close()
      catch case _: Throwable => Seq.empty
    def path(text: String): Option[Path] = try Some(Paths.get(text.trim)) catch case _: Throwable => None
    val explicit = env.get("SCALATORCH_CUDNN_PATH").toSeq.flatMap(path)
    val toolkit = (env.get("CUDA_PATH").toSeq ++ env.get("CUDA_HOME").toSeq).flatMap(path)
      .flatMap(r => Seq(r.resolve("bin"), r.resolve("bin").resolve("x64"), r.resolve("lib64")))
    // The cuDNN installer for Windows puts the libraries under Program Files/NVIDIA/CUDNN/v9.x/bin/<CUDA version>/
    val installed = if !windows then Seq.empty else env.get("ProgramFiles").toSeq.flatMap(path)
      .flatMap(r => children(r.resolve("NVIDIA").resolve("CUDNN"))).map(_.resolve("bin")).flatMap(bin => children(bin) :+ bin)
    val searchPath = env.get(if windows then "Path" else "LD_LIBRARY_PATH").orElse(env.get("PATH")).toSeq
      .flatMap(_.split(java.io.File.pathSeparator).toSeq).filter(_.trim.nonEmpty).flatMap(path)
    val found = for
      dir <- (explicit ++ toolkit ++ installed ++ searchPath).distinct
      (front, parts) <- layouts
      file = dir.resolve(front)
      if (try Files.isRegularFile(file) && parts.forall(part => Files.isRegularFile(dir.resolve(part))) catch case _: Throwable => false)
    yield file.toAbsolutePath.toString
    found.distinct

  /** Opens cuDNN on the current context and binds it to `stream`; `register` receives the cleanup action. */
  def open(arena: Arena, stream: MemorySegment, register: (() => Unit) => Unit): Option[Cudnn] =
    if !enabled then return None
    val linker = Linker.nativeLinker()
    candidates().iterator.map { name =>
      try
        // cuDNN 9 is a small front library that loads its parts by bare name. Next to an explicit path they are not on the search
        // path, and a part that cannot be found aborts the process: put its directory on the DLL search path first.
        val file = Paths.get(name)
        if System.getProperty("os.name").toLowerCase(java.util.Locale.ROOT).startsWith("windows") then
          try
            val kernel = SymbolLookup.libraryLookup("kernel32.dll", arena)
            val setDirectory = linker.downcallHandle(kernel.find("SetDllDirectoryW").orElseThrow(), FunctionDescriptor.of(JAVA_INT, ADDRESS))
            val wide = (file.getParent.toString + " ").getBytes(java.nio.charset.StandardCharsets.UTF_16LE)
            val text = arena.allocate(wide.length.toLong, 2)
            MemorySegment.copy(MemorySegment.ofArray(wide), 0, text, 0, wide.length.toLong)
            setDirectory.invokeWithArguments(text)
          catch case _: Throwable => ()
        val lookup = SymbolLookup.libraryLookup(name, arena)
        def down(symbol: String, layouts: java.lang.foreign.MemoryLayout*): MethodHandle =
          linker.downcallHandle(lookup.find(symbol).orElseThrow(() => UnsatisfiedLinkError(symbol)), FunctionDescriptor.of(JAVA_INT, layouts*))
        val create = down("cudnnCreate", ADDRESS)
        val handles = new Handles(
          down("cudnnSetStream", ADDRESS, ADDRESS),
          down("cudnnCreateTensorDescriptor", ADDRESS),
          down("cudnnSetTensor4dDescriptor", ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT),
          down("cudnnCreateFilterDescriptor", ADDRESS),
          down("cudnnSetFilter4dDescriptor", ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT),
          down("cudnnCreateConvolutionDescriptor", ADDRESS),
          down("cudnnSetConvolution2dDescriptor", ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT),
          down("cudnnSetConvolutionMathType", ADDRESS, JAVA_INT),
          down("cudnnSetConvolutionGroupCount", ADDRESS, JAVA_INT),
          down("cudnnGetConvolutionForwardWorkspaceSize", ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, JAVA_INT, ADDRESS),
          down("cudnnConvolutionForward", ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS, ADDRESS),
          down("cudnnDestroy", ADDRESS),
          down("cudnnGetConvolutionForwardAlgorithm_v7", ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, ADDRESS))
        val out = arena.allocate(ADDRESS)
        if create.invokeWithArguments(out).asInstanceOf[Integer].intValue() != 0 then null
        else
          val library = new Cudnn(out.get(ADDRESS, 0L), arena, handles)
          library.bindStream(stream)
          register(() => library.close())
          library
      catch case _: Throwable => null
    }.find(_ != null)

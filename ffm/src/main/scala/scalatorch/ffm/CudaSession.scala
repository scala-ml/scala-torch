package scalatorch.ffm

import scalatorch.*
import java.lang.foreign.{Arena, MemorySegment}
import java.lang.foreign.ValueLayout.{ADDRESS, JAVA_DOUBLE, JAVA_FLOAT, JAVA_INT, JAVA_LONG}
import java.nio.charset.StandardCharsets.UTF_8
import scala.collection.mutable
import scala.util.Using

private[ffm] final class CudaAllocation(val address: Long,val bytes: Long,val arena: CudaArena):
  var references=1L
  /** Owned by [[CudaWeightPool]]: shared between sessions and threads, never freed or counted by a session. */
  @volatile var pooled=false
  /** Open sessions that use this pooled allocation; guarded by the pool's lock. */
  var users=0

/** Independently releasable packed Float32 handle; reshape may share an allocation. */
final class CudaTensor private[ffm] (private[ffm] val session: CudaSession,
    val shape: Shape, private[ffm] val allocation: CudaAllocation,val half: Boolean=false) extends AutoCloseable:
  private[ffm] def address: Long = allocation.address
  private[ffm] var released=false
  def toCpu: Tensor[Float] = session.download(this)
  def matmul(other: CudaTensor): CudaTensor = session.matmul(this,other)
  def close(): Unit = session.release(this)

/** Thread-confined resident Float32 storage and synchronous CUDA kernels. */
final class CudaSession private[ffm] (driver: CudaDriver, val deviceIndex: Int,
    val deviceName: String, val driverVersion: Int, val maxBytes: Long) extends AdamWBackend[CudaTensor]:
  private val owner=Thread.currentThread()
  private var closed=false
  private var used=0L
  // Device memory comes from an arena of large chunks (see [[CudaArena]]): steady-state inference does no cuMemAlloc/cuMemFree
  // (cuMemFree synchronizes the whole device). Reuse is safe because every kernel and cuBLAS call runs on one stream.
  private val arena=new CudaArena(CudaSession.chunkBytes)
  private lazy val weightArena=CudaWeightPool.newArena(deviceIndex)
  private var weightArenaUsed=false
  private var peakUsed=0L
  // SCALATORCH_CUDA_LAUNCH_STATS=1: time spent on the CPU inside cuLaunchKernel and in preparing its arguments.
  private val launchStats=sys.env.get("SCALATORCH_CUDA_LAUNCH_STATS").contains("1")
  private val opened=System.nanoTime()
  private var launchNanos=0L
  private var launchPrepNanos=0L
  private var launchCount=0L
  private var peakOwned=0L
  /** Set while a kept weight is uploaded: its blocks come from the arena of the kept weights, which outlives the session. */
  private var sharedAllocation=false
  private val memoryLog=sys.env.get("SCALATORCH_CUDA_MEMORY_LOG").contains("1")
  private val held=CudaWeightPool.heldCounter(deviceIndex)
  private val touchedShared: java.util.Set[CudaAllocation]=java.util.Collections.newSetFromMap(new java.util.IdentityHashMap[CudaAllocation,java.lang.Boolean]())
  private val rawFree: (Long,Long) => Unit = (address,bytes) =>
    hostSync() // kernels that still use the chunk may be queued on this session's stream
    driver.call("cuMemFree_v2",Seq(JAVA_LONG),Long.box(address))
    held.addAndGet(-bytes)
  private val rawAllocate: Long => Long = bytes => Using.resource(Arena.ofConfined()) { scope =>
    val result=scope.allocate(JAVA_LONG)
    driver.call("cuMemAlloc_v2",Seq(ADDRESS,JAVA_LONG),result,Long.box(bytes))
    held.addAndGet(bytes)
    result.get(JAVA_LONG,0)
  }
  private def trimPool(): Long = arena.trim(rawFree)
  /** Returns the chunks that hold no tensor to the device (for example after the Float32 staging blocks of a weight upload). */
  def trimFreed(): Unit = { check();hostSync();trimPool();CudaWeightPool.trimArenas(deviceIndex,rawFree) }
  private def deviceFree(): Long = CudaWeightPool.governedFree(deviceIndex,driver.freeBytes())
  /** Called when the session is short of memory, with the number of bytes that are missing. It may close tensors of this session that
    * can be rebuilt (offloaded weights kept for speed) and returns true when it closed something. */
  var pressure: Long => Boolean = _ => false
  /** Device memory this session could still get: what is free now plus the kept weights that would be evicted for it. */
  def availableBytes: Long = { check();deviceFree()+CudaWeightPool.unusedBytes(deviceIndex)+(arena.ownedBytes-arena.liveBytes) }
  /** Bytes of the kept weights for `keys` that another open session holds on the device (no upload needed, not evictable). */
  def sharedInUseBytes(keys: Iterable[AnyRef],half: Boolean): Long = CudaWeightPool.residentInUse(deviceIndex,keys,half)
  private val stamp=CudaWeightPool.clock.incrementAndGet()
  /** Evicts unused kept weights, oldest first, until the device has `needed` bytes free or nothing is left to evict. */
  private def evictSharedWeights(needed: Long): Boolean =
    var any=false
    var more=true
    while more do
      val short=needed-deviceFree()
      if short<=0 then more=false
      else
        // A chunk only returns to the device when all its blocks are gone, so ask for a chunk more than is missing.
        val evicted=CudaWeightPool.evictUnused(deviceIndex,(if short>(Long.MaxValue>>2) then Long.MaxValue else short+67108864L))
        if evicted.isEmpty then more=false
        else
          any=true
          evicted.foreach(e => e.allocation.arena.release(e.allocation.address))
          CudaWeightPool.trimArenas(deviceIndex,rawFree)
    any
  /** Before a new chunk: when the device is short of `bytes`, give back this session's empty chunks, then the kept weights that no open
    * session uses. On Windows an over-full device does not fail, it silently spills into system memory and becomes very slow. */
  private val makeRoom: Long => Unit = bytes =>
    val needed=bytes+67108864L
    if deviceFree()<needed then
      hostSync()
      trimPool();CudaWeightPool.trimArenas(deviceIndex,rawFree)
      if deviceFree()<needed then evictSharedWeights(needed)
      if deviceFree()<needed && pressure(needed-deviceFree()) then trimPool()
  private val rawAllocateOrEvict: Long => Long = bytes =>
    try rawAllocate(bytes)
    catch case error: Throwable =>
      hostSync()
      val freed=trimPool()+CudaWeightPool.trimArenas(deviceIndex,rawFree)
      if !evictSharedWeights(Long.MaxValue>>1) && freed==0L then throw error
      rawAllocate(bytes)
  private def hostSync(): Unit = driver.synchronize(stream)
  private def dtoh(host: MemorySegment,address: java.lang.Long,bytes: java.lang.Long): Unit =
    hostSync()
    driver.call("cuMemcpyDtoH_v2",Seq(ADDRESS,JAVA_LONG,JAVA_LONG),host,address,bytes)
  private def syncedCopyToDevice(address: Long,host: MemorySegment,bytes: Long): Unit =
    hostSync()
    driver.copyHostToDevice(address,host,bytes)
  private val profileEnabled=sys.env.get("SCALATORCH_CUDA_PROFILE").contains("1")
  private val fusedDenseAttentionEnabled=sys.env.get("SCALATORCH_CUDA_FUSED_ATTENTION").contains("1")
  private val cublasAttentionEnabled= !sys.env.get("SCALATORCH_CUDA_CUBLAS_ATTENTION").contains("0")
  private val multiQueryAttentionEnabled=sys.env.get("SCALATORCH_CUDA_MULTI_QUERY_ATTENTION").contains("1")
  private val profile=mutable.Map.empty[String,(Long,Long)]
  private val uploadStaging=new CudaHostStagingBuffer
  private val syncEveryLaunch=sys.env.get("SCALATORCH_CUDA_PROFILE").contains("1") || sys.env.get("SCALATORCH_CUDA_SYNC").contains("1")
  private def record(label: String,started: Long): Unit = if profileEnabled then
    val (count,nanos)=profile.getOrElse(label,(0L,0L))
    profile.update(label,(count+1,nanos+System.nanoTime()-started))
  private def recordElapsed(label: String,nanos: Long): Unit = if profileEnabled then
    val (count,total)=profile.getOrElse(label,(0L,0L))
    profile.update(label,(count+1,total+nanos))
  private val live=mutable.LinkedHashSet.empty[CudaTensor]
  private val context=driver.pointerResult("cuCtxGetCurrent",Seq.empty)
  private val ptx=Using.resource(getClass.getResourceAsStream("/scalatorch/gemm.ptx")) { input =>
    require(input != null,"GEMM PTX resource missing")
    new String(input.readAllBytes(),UTF_8)
  }
  private val module=driver.loadModule(ptx)
  driver.cleanup { driver.call("cuModuleUnload",Seq(ADDRESS),module) }
  private val function=driver.pointerResult("cuModuleGetFunction",Seq(ADDRESS,ADDRESS),module,driver.arena.allocateFrom("gemm"))
  private val tiledFunction=driver.pointerResult("cuModuleGetFunction",Seq(ADDRESS,ADDRESS),module,driver.arena.allocateFrom("gemm_tiled"))
  private val convN32Function=driver.pointerResult("cuModuleGetFunction",Seq(ADDRESS,ADDRESS),module,driver.arena.allocateFrom("gemm_conv_n32"))
  private val opsPtx=Using.resource(getClass.getResourceAsStream("/scalatorch/transformer.ptx")) { input =>
    require(input != null,"transformer PTX resource missing")
    new String(input.readAllBytes(),UTF_8)
  }
  private val opsModule=driver.loadModule(opsPtx)
  driver.cleanup { driver.call("cuModuleUnload",Seq(ADDRESS),opsModule) }
  private val kernels=Seq("geglu","geglu_fast","f32_to_f16","broadcast_add5","transpose5","softmax_par","gelu_erf","quick_gelu","broadcast_add","concat_copy","resize_nearest","groupnorm","groupnorm_stats","groupnorm_stats_par","groupnorm_apply","conv2d","conv2d_fast","im2col","adamw_update","finite_status","cross_entropy_stats","cross_entropy_output","cross_entropy_gradient","binary","scale","fill","sum","expand_scalar","silu","leaky_relu","rmsnorm","rms_input_backward","rms_inverse","rms_weight_backward","transpose","embedding","embedding_backward","rope","attention","attention_scores","attention_prepare","attention_output","attention_fused_fp32","attention_fused_multiquery_fp32","attention_scores_fp32","attention_prepare_fp32","attention_output_fp32","attention_backward","attention_gradient_cast","cache_append").map { name =>
    name -> driver.pointerResult("cuModuleGetFunction",Seq(ADDRESS,ADDRESS),opsModule,driver.arena.allocateFrom(name))
  }.toMap
  private lazy val halfModule=
    val halfPtx=Using.resource(getClass.getResourceAsStream("/scalatorch/half.ptx")) { input =>
      require(input != null,"half PTX resource missing")
      new String(input.readAllBytes(),UTF_8)
    }
    val loaded=driver.loadModule(halfPtx)
    driver.cleanup { driver.call("cuModuleUnload",Seq(ADDRESS),loaded) }
    loaded
  private lazy val halfKernels=Seq("h_f32_to_f16","h_f16_to_f32","h_fill","h_add","h_add5","h_scale","h_silu","h_geglu","h_transpose5",
      "h_concat","h_resize_nearest","h_groupnorm_stats","h_groupnorm_apply","h_im2col","h_im2col_batch","h_attention_fused_r1c3","h_attention_fused_r2c3","h_attention_fused_r4c3","h_attention_fused_r1c4","h_attention_fused_r2c4","h_attention_fused_r4c4","h_attention_fused_r1c5","h_attention_fused_r2c5","h_fill_rows","h_wmma_layout","h_softmax","h_gelu_erf","h_quick_gelu","h_softmax_causal").map { name =>
    name -> driver.pointerResult("cuModuleGetFunction",Seq(ADDRESS,ADDRESS),halfModule,driver.arena.allocateFrom(name))
  }.toMap
  private val stream=driver.pointerResult("cuStreamCreate",Seq(JAVA_INT),Int.box(1))
  driver.cleanup { driver.call("cuStreamDestroy_v2",Seq(ADDRESS),stream) }
  private val cublas: Option[Cublas]=Cublas.openShared(deviceIndex,stream).orElse(Cublas.open(driver.arena,stream,action => driver.cleanup(action())))
  // cuDNN is opened on first use: creating its handle is slow and most sessions (text encoders) never convolve.
  private lazy val cudnn: Option[Cudnn]=
    try Cudnn.open(driver.arena,stream,action => driver.cleanup(action())) catch case _: Throwable => None
  def hasCudnn: Boolean = cudnn.isDefined
  private def cublasGemm(m: Int,n: Int,k: Int,a: Long,lda: Int,b: Long,ldb: Int,c: Long,ldc: Int): Boolean =
    cublas match
      case Some(lib) if m>0 && n>0 && k>0 =>
        val started=if profileEnabled then System.nanoTime() else 0L
        lib.sgemmRowMajor(m,n,k,a,lda,b,ldb,c,ldc)
        driver.synchronize(stream)
        if profileEnabled then record(s"cublas_sgemm[$m,$n,$k]",started)
        true
      case _ => false
  driver.cleanup { shutdown() }

  private def threadCheck(): Unit =
    if Thread.currentThread() ne owner then throw IllegalStateException("CUDA session used from another thread")
  private def check(): Unit =
    threadCheck()
    if closed then throw IllegalStateException("CUDA session is closed")
    Using.resource(Arena.ofConfined()) { arena =>
      val current=arena.allocate(ADDRESS)
      driver.call("cuCtxGetCurrent",Seq(ADDRESS),current)
      if current.get(ADDRESS,0).address() != context.address() then
        throw IllegalStateException("CUDA session context is not current")
    }
  private def validate(tensor: CudaTensor): Unit =
    check()
    require(tensor.session eq this,"tensor belongs to another CUDA session")
    if tensor.released then throw IllegalStateException("CUDA tensor is closed")
  def allocatedBytes: Long = { check();used }
  def shape(value: CudaTensor): Shape = { validate(value);value.shape }

  private def allocate(shape: Shape,half: Boolean=false): CudaTensor =
    check()
    require(shape.numel <= Int.MaxValue,"CUDA tensor exceeds Int indexing")
    val bytes=math.max(shape.numel*(if half then 2L else 4L),1L)
    while bytes>maxBytes-used && pressure(bytes-(maxBytes-used)) do ()
    require(bytes <= maxBytes-used,"CUDA session allocation budget exceeded")
    val from=if sharedAllocation then { weightArenaUsed=true;weightArena } else arena
    val address=from.allocate(bytes,makeRoom,rawAllocateOrEvict,rawFree)
    val tensor=new CudaTensor(this,shape,new CudaAllocation(address,bytes,from),half)
    live+=tensor;used+=bytes
    if used>peakUsed then peakUsed=used
    if arena.ownedBytes>peakOwned then peakOwned=arena.ownedBytes
    tensor

  def reshape(value: CudaTensor,shape: Shape): CudaTensor =
    validate(value)
    require(value.shape.numel==shape.numel,"reshape changes element count")
    val view=new CudaTensor(this,shape,value.allocation,value.half)
    if !value.allocation.pooled then value.allocation.references+=1
    live+=view
    view

  /** Private integer staging allocation, charged to the same device byte budget. */
  private def withInts[A](values: Array[Int])(body: Long => A): A =
    val buffer=allocate(Shape.from(Seq(values.length.toLong)))
    Using.resource(buffer) { _ =>
      Using.resource(Arena.ofConfined()) { arena =>
        if values.nonEmpty then
          val host=arena.allocateFrom(JAVA_INT,values*)
          syncedCopyToDevice(buffer.address,host,values.length.toLong*4)
      }
      body(buffer.address)
    }

  private final class ResidentAdamState(val belongsTo: CudaSession,val buffer: CudaTensor,
      originalShape: Shape,counter: Long) extends AdamWState[CudaTensor]:
    def shape: Shape = { validate(buffer);originalShape }
    def step: Long = { validate(buffer);counter }
    def snapshot(): AdamWMoments =
      validate(buffer)
      val pairs=Using.resource(Arena.ofConfined()) { arena =>
        val bytes=originalShape.numel*16
        val host=arena.allocate(bytes,8)
        if bytes>0 then dtoh(host,Long.box(buffer.address),Long.box(bytes))
        host.toArray(JAVA_DOUBLE)
      }
      val n=originalShape.numel.toInt
      AdamWMoments(counter,Tensor.fromDoubles(Array.tabulate(n)(i => pairs(i*2)),originalShape),
        Tensor.fromDoubles(Array.tabulate(n)(i => pairs(i*2+1)),originalShape))
    def close(): Unit = buffer.close()

  def restoreAdamW(saved: AdamWMoments): AdamWState[CudaTensor] =
    check();saved.validate()
    require(saved.shape.numel*4<=Int.MaxValue,"Adam Double state exceeds Int indexing")
    val buffer=allocate(Shape.from(Seq(saved.shape.numel,4L)))
    try
      if saved.shape.numel>0 then Using.resource(Arena.ofConfined()) { arena =>
        val first=saved.first.toDoubleArray;val second=saved.second.toDoubleArray
        val pairs=new Array[Double](first.length*2)
        first.indices.foreach(i => { pairs(i*2)=first(i);pairs(i*2+1)=second(i) })
        val host=arena.allocateFrom(JAVA_DOUBLE,pairs*)
        syncedCopyToDevice(buffer.address,host,saved.shape.numel*16)
      }
      new ResidentAdamState(this,buffer,saved.shape,saved.step)
    catch case error: Throwable =>
      try buffer.close() catch case cleanup: Throwable => error.addSuppressed(cleanup)
      throw error

  def adamW(weight: CudaTensor,gradient: CudaTensor,previous: Option[AdamWState[CudaTensor]],config: AdamWConfig): AdamWUpdate[CudaTensor] =
    validate(weight);validate(gradient)
    require(weight.shape==gradient.shape,"Adam gradient shape mismatch")
    val old=previous.map {
      case state: ResidentAdamState if state.belongsTo eq this =>
        require(state.shape==weight.shape,"Adam state shape mismatch")
        state
      case _ => throw IllegalArgumentException("Adam state belongs to another backend/session")
    }
    val step=Math.addExact(old.fold(0L)(_.step),1L)
    val n=weight.shape.numel
    require(n*4<=Int.MaxValue,"Adam Double state exceeds Int indexing")
    require(math.max(n*16,1L)+math.max(n*4,1L)+4<=maxBytes-used,"Adam update allocation budget exceeded")
    var moments: CudaTensor=null
    var result: CudaTensor=null
    try
      moments=allocate(Shape.from(Seq(n,4L)))
      result=allocate(weight.shape)
      Using.resource(fill(Shape.from(Seq.empty),0f)) { status =>
        launch(n,kernels("adamw_update")) { arena =>
          Seq(weight.address,gradient.address,old.fold(0L)(_.buffer.address),result.address,moments.address,status.address)
            .map(x => arena.allocateFrom(JAVA_LONG,x)) ++ Seq(arena.allocateFrom(JAVA_INT,n.toInt)) ++
            Seq(config.learningRate,config.beta1,config.beta2,config.epsilon,config.weightDecay,
              1.0-math.pow(config.beta1,step.toDouble),1.0-math.pow(config.beta2,step.toDouble))
              .map(x => arena.allocateFrom(JAVA_DOUBLE,x))
        }
        require(download(status).toFloatArray.head==0f,"nonfinite Adam input/update")
      }
      AdamWUpdate(result,new ResidentAdamState(this,moments,weight.shape,step))
    catch case error: Throwable =>
      Seq(result,moments).filter(_!=null).foreach { value =>
        try value.close() catch case cleanup: Throwable => error.addSuppressed(cleanup)
      }
      throw error

  def allFinite(value: CudaTensor): Boolean =
    validate(value)
    Using.resource(fill(Shape.from(Seq.empty),0f)) { status =>
      launch(value.shape.numel,kernels("finite_status")) { arena =>
        Seq(value.address,status.address).map(x => arena.allocateFrom(JAVA_LONG,x)) :+
          arena.allocateFrom(JAVA_INT,value.shape.numel.toInt)
      }
      download(status).toFloatArray.head==0f
    }

  def crossEntropy(logits: CudaTensor,targets: Tensor[Long],reduction: LossReduction = LossReduction.Mean,ignoreIndex: Long = -100L): CudaTensor =
    crossEntropyRun(logits,targets,None,reduction,ignoreIndex)
  def crossEntropyBackward(logits: CudaTensor,targets: Tensor[Long],gradient: CudaTensor,
      reduction: LossReduction = LossReduction.Mean,ignoreIndex: Long = -100L): CudaTensor =
    validate(gradient)
    require(gradient.shape==CrossEntropyLayout.output(targets,reduction),"cross entropy gradient shape mismatch")
    crossEntropyRun(logits,targets,Some(gradient),reduction,ignoreIndex)
  private def crossEntropyRun(logits: CudaTensor,targets: Tensor[Long],gradient: Option[CudaTensor],
      reduction: LossReduction,ignoreIndex: Long): CudaTensor =
    validate(logits)
    val ids=CrossEntropyLayout.targets(logits.shape,targets,ignoreIndex)
    val rows=ids.length.toLong
    val output=if gradient.isDefined then logits.shape else CrossEntropyLayout.output(targets,reduction)
    require(rows*6<=Int.MaxValue,"cross entropy Double workspace exceeds Int indexing")
    val bytes=math.max(rows*4,1L)+math.max(rows*24,1L)+4+math.max(output.numel*4,1L)
    require(bytes<=maxBytes-used,"cross entropy workspace budget exceeded")
    var result: CudaTensor=null
    try
      withInts(ids) { targetAddress =>
        Using.Manager { use =>
          val stats=use(allocate(Shape.from(Seq(rows,6L))))
          val invalid=use(fill(Shape.from(Seq.empty),0f))
          launch(rows,kernels("cross_entropy_stats")) { arena =>
            Seq(logits.address,targetAddress,stats.address,invalid.address).map(x => arena.allocateFrom(JAVA_LONG,x)) ++
              Seq(rows,logits.shape.dims.last).map(x => arena.allocateFrom(JAVA_INT,x.toInt))
          }
          // Four-byte control transfer only: logits, statistics and gradients stay resident.
          require(download(invalid).toFloatArray.head==0f,"cross entropy requires finite logits")
          val kernel=if gradient.isDefined then "cross_entropy_gradient" else "cross_entropy_output"
          result=execute(output,output.numel,kernels(kernel)) { (arena,result) =>
            Seq(logits.address,targetAddress,stats.address,gradient.fold(0L)(_.address),result.address).map(x => arena.allocateFrom(JAVA_LONG,x)) ++
              Seq(rows,logits.shape.dims.last,ids.count(_>=0).toLong,reduction.ordinal.toLong,output.numel).map(x => arena.allocateFrom(JAVA_INT,x.toInt))
          }
          result
        }.get
      }
    catch case error: Throwable =>
      if result!=null then
        try result.close() catch case cleanup: Throwable => error.addSuppressed(cleanup)
      throw error


  def rope(value: CudaTensor,positionStart: Int = 0,theta: Double = 10000.0): CudaTensor =
    rotate(value,positionStart,theta,inverse=false)
  def ropeBackward(gradient: CudaTensor,positionStart: Int = 0,theta: Double = 10000.0): CudaTensor =
    rotate(gradient,positionStart,theta,inverse=true)
  private def rotate(value: CudaTensor,positionStart: Int,theta: Double,inverse: Boolean): CudaTensor =
    validate(value)
    val count=RotaryCoefficients.count(value.shape,positionStart,theta)
    if count==0 then return reshape(value,value.shape)
    require(count*2<=Int.MaxValue && count*8+value.shape.numel*4<=maxBytes-used,"rotary workspace budget exceeded")
    val coefficients=RotaryCoefficients.values(value.shape,positionStart,theta)
    if inverse then
      var i=1
      while i<coefficients.length do
        coefficients(i) = -coefficients(i);i+=2
    Using.resource(allocate(Shape.from(Seq(count*2)))) { buffer =>
      Using.resource(Arena.ofConfined()) { arena =>
        val host=arena.allocateFrom(JAVA_DOUBLE,coefficients*)
        syncedCopyToDevice(buffer.address,host,count*8)
      }
      execute(value.shape,value.shape.numel/2,kernels("rope")) { (arena,result) =>
        Seq(value.address,result.address,buffer.address).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
          Seq((value.shape.numel/2).toInt,value.shape.dims(2).toInt,value.shape.dims(3).toInt).map(v => arena.allocateFrom(JAVA_INT,v))
      }
    }

  def causalAttention(query: CudaTensor,key: CudaTensor,value: CudaTensor,queryStart: Int = 0,
      maxScoreElements: Long = 16777216L): CudaTensor =
    validate(query);validate(key);validate(value)
    // Preserve rank validation before inspecting the token dimension.
    AttentionLayout.validate(query.shape,key.shape,value.shape,queryStart,maxScoreElements)
    attentionFromStorage(query,key,value,queryStart,key.shape.dims(2),key.shape.dims(2),maxScoreElements)

  /** Noncausal attention with query slices sharing bounded Double score scratch.
    * maxScoreElements still limits total work; maxWorkspaceBytes limits resident scratch.
    */
  def scaledDotProductAttention(query: CudaTensor,key: CudaTensor,value: CudaTensor,
      maxScoreElements: Long=16777216L,maxWorkspaceBytes: Long=16777216L): CudaTensor =
    validate(query);validate(key);validate(value)
    require(maxWorkspaceBytes>=0,"negative attention workspace budget")
    AttentionLayout.validate(query.shape,key.shape,value.shape,0,maxScoreElements,!query.half)
    attentionFromStorage(query,key,value,0,key.shape.dims(2),key.shape.dims(2),maxScoreElements,
      causal=false,maxWorkspaceBytes=maxWorkspaceBytes)

  private def attentionFromStorage(query: CudaTensor,key: CudaTensor,value: CudaTensor,queryStart: Int,
      tokens: Long,storageTokens: Long,maxScores: Long,causal: Boolean=true,
      maxWorkspaceBytes: Long=Long.MaxValue): CudaTensor =
    val keyShape=Shape.from(key.shape.dims.updated(2,tokens))
    val valueShape=Shape.from(value.shape.dims.updated(2,tokens))
    val (output,scores)=AttentionLayout.validate(query.shape,keyShape,valueShape,queryStart,maxScores,!query.half)
    if output.numel==0 then return allocate(output,query.half)
    if query.half then return attentionHalf(query,key,value,output,causal,queryStart)
    if causal then require(scores*2<=Int.MaxValue,"attention Double workspace exceeds Int indexing")
    val q=query.shape.dims;val k=keyShape.dims;val v=valueShape.dims
    val rows=output.numel/v(3)
    if !causal && cublasAttentionEnabled && cublas.isDefined && q(1)==k(1) && q(1)==v(1) && q(0)==k(0) && q(0)==v(0) &&
        storageTokens==k(2) && q(3)==k(3) && scores*4<=math.min(2147483648L,maxBytes-used-output.numel*4) && scores<=Int.MaxValue then
      val result=allocate(output)
      try
        Using.resource(allocate(Shape.from(Seq(scores)))) { scratch =>
          val batch=(q(0)*q(1)).toInt
          val qt=q(2).toInt;val kt=k(2).toInt;val d=q(3).toInt;val dv=v(3).toInt
          val started=if profileEnabled then System.nanoTime() else 0L
          cublas.get.sgemmStridedBatched(1,0,kt,qt,d,(1.0/math.sqrt(d.toDouble)).toFloat,
            key.address,d,kt.toLong*d,query.address,d,qt.toLong*d,scratch.address,kt,qt.toLong*kt,batch)
          driver.synchronize(stream)
          if profileEnabled then record("cublas_attention_scores",started)
          launch(batch.toLong*qt,kernels("softmax_par"),threadsPerBlock=256,threadsPerOutput=256,label="softmax_par") { arena =>
            Seq(arena.allocateFrom(JAVA_LONG,scratch.address),arena.allocateFrom(JAVA_INT,(batch.toLong*qt).toInt),
              arena.allocateFrom(JAVA_INT,kt))
          }
          val started2=if profileEnabled then System.nanoTime() else 0L
          cublas.get.sgemmStridedBatched(0,0,dv,qt,kt,1f,
            value.address,dv,kt.toLong*dv,scratch.address,kt,qt.toLong*kt,result.address,dv,qt.toLong*dv,batch)
          driver.synchronize(stream)
          if profileEnabled then record("cublas_attention_output",started2)
        }
        return result
      catch case error: Throwable =>
        try result.close() catch case cleanup: Throwable => error.addSuppressed(cleanup)
        throw error
    if !causal && fusedDenseAttentionEnabled && multiQueryAttentionEnabled && k(2)<=4096 && v(3)<=256 &&
        q(2)%2==0 && rows%2==0 then
      val result=allocate(output)
      try
        launch(rows/2,kernels("attention_fused_multiquery_fp32"),threadsPerBlock=256,threadsPerOutput=256,
            label="attention_fused_multiquery_fp32") { arena =>
          Seq(query.address,key.address,value.address,result.address).map(x => arena.allocateFrom(JAVA_LONG,x)) ++
            Seq(rows.toInt,q(1).toInt,k(1).toInt,q(2).toInt,k(2).toInt,q(3).toInt,v(3).toInt,storageTokens.toInt).map(x => arena.allocateFrom(JAVA_INT,x)) :+
            arena.allocateFrom(JAVA_FLOAT,(1.0/math.sqrt(q(3).toDouble)).toFloat)
        }
        return result
      catch case error: Throwable =>
        try result.close() catch case cleanup: Throwable => error.addSuppressed(cleanup)
        throw error
    if !causal && fusedDenseAttentionEnabled && k(2)<=4096 && v(3)<=256 then
      val result=allocate(output)
      try
        launch(rows,kernels("attention_fused_fp32"),threadsPerBlock=256,threadsPerOutput=256,label="attention_fused_fp32") { arena =>
          Seq(query.address,key.address,value.address,result.address).map(x => arena.allocateFrom(JAVA_LONG,x)) ++
            Seq(rows.toInt,q(1).toInt,k(1).toInt,q(2).toInt,k(2).toInt,q(3).toInt,v(3).toInt,storageTokens.toInt).map(x => arena.allocateFrom(JAVA_INT,x)) :+
            arena.allocateFrom(JAVA_FLOAT,(1.0/math.sqrt(q(3).toDouble)).toFloat)
        }
        return result
      catch case error: Throwable =>
        try result.close() catch case cleanup: Throwable => error.addSuppressed(cleanup)
        throw error
    val sliceRows=if causal then rows else
      val available=maxBytes-used-output.numel*4
      val budget=math.min(maxWorkspaceBytes,available)
      require(budget>=k(2)*4,"attention workspace cannot hold one query row and output")
      math.min(rows,math.min(budget/(k(2)*4),Int.MaxValue.toLong/k(2)))
    require(sliceRows>0,"attention workspace exceeds Int indexing")
    val scoreElements=sliceRows*k(2)*(if causal then 2 else 1)
    Using.resource(allocate(Shape.from(Seq(scoreElements)))) { scratch =>
      val result=allocate(output)
      try
        var rowOffset=0L
        while rowOffset<rows do
          val count=math.min(sliceRows,rows-rowOffset)
          if causal then
            launch(count,kernels("attention")) { arena =>
              Seq(query.address,key.address,value.address,result.address,scratch.address).map(x => arena.allocateFrom(JAVA_LONG,x)) ++
                Seq(count,q(1),k(1),q(2),k(2),q(3),v(3),queryStart.toLong,storageTokens).map(x => arena.allocateFrom(JAVA_INT,x.toInt)) :+
                arena.allocateFrom(JAVA_DOUBLE,1.0/math.sqrt(q(3).toDouble)) :+
                arena.allocateFrom(JAVA_INT,1) :+ arena.allocateFrom(JAVA_INT,rowOffset.toInt)
            }
          else
            launch(count*k(2),kernels("attention_scores_fp32")) { arena =>
              Seq(query.address,key.address,scratch.address).map(x => arena.allocateFrom(JAVA_LONG,x)) ++
                Seq(count*k(2),q(1),k(1),q(2),k(2),q(3),storageTokens,rowOffset).map(x => arena.allocateFrom(JAVA_INT,x.toInt)) :+
                arena.allocateFrom(JAVA_FLOAT,(1.0/math.sqrt(q(3).toDouble)).toFloat)
            }
            launch(count,kernels("attention_prepare_fp32")) { arena =>
              Seq(arena.allocateFrom(JAVA_LONG,scratch.address),arena.allocateFrom(JAVA_INT,count.toInt),
                arena.allocateFrom(JAVA_INT,k(2).toInt))
            }
            launch(count*v(3),kernels("attention_output_fp32")) { arena =>
              Seq(value.address,scratch.address,result.address).map(x => arena.allocateFrom(JAVA_LONG,x)) ++
                Seq(count*v(3),q(1),k(1),q(2),k(2),v(3),storageTokens,rowOffset).map(x => arena.allocateFrom(JAVA_INT,x.toInt))
            }
          rowOffset+=count
        result
      catch case error: Throwable =>
        try result.close() catch case cleanup: Throwable => error.addSuppressed(cleanup)
        throw error
    }

  def causalAttentionBackward(query: CudaTensor,key: CudaTensor,value: CudaTensor,gradient: CudaTensor,
      queryStart: Int = 0,maxScoreElements: Long = 16777216L): (CudaTensor,CudaTensor,CudaTensor) =
    validate(query);validate(key);validate(value);validate(gradient)
    val (output,_)=AttentionLayout.validate(query.shape,key.shape,value.shape,queryStart,maxScoreElements)
    require(gradient.shape==output,"attention gradient shape mismatch")
    val shapes=Vector(query.shape,key.shape,value.shape)
    val outputs=mutable.ArrayBuffer.empty[CudaTensor]
    def keep(t: CudaTensor): CudaTensor = { outputs+=t;t }
    try
      if query.shape.numel==0 then
        require(shapes.map(s => math.max(s.numel*4,1L)).sum<=maxBytes-used,"attention gradient allocation budget exceeded")
        (keep(fill(query.shape,0f)),keep(fill(key.shape,0f)),keep(fill(value.shape,0f)))
      else
        val q=query.shape.dims;val k=key.shape.dims;val v=value.shape.dims
        val groups=q(0)*k(1);val rowScores=groups*k(2)
        require((shapes.map(_.numel) :+ rowScores).forall(_*2<=Int.MaxValue),"attention gradient Double workspace exceeds Int indexing")
        require(shapes.map(_.numel).sum*12+rowScores*16<=maxBytes-used,"attention gradient workspace budget exceeded")
        Using.Manager { use =>
          val sums=shapes.map(s => use(fill(Shape.from(Seq(s.numel,2L)),0f)))
          val probabilities=use(allocate(Shape.from(Seq(rowScores,2L))))
          val dp=use(allocate(Shape.from(Seq(rowScores,2L))))
          launch(groups,kernels("attention_backward")) { arena =>
            Seq(query.address,key.address,value.address,gradient.address,sums(0).address,sums(1).address,sums(2).address,
              probabilities.address,dp.address).map(x => arena.allocateFrom(JAVA_LONG,x)) ++
              Seq(groups,q(1),k(1),q(2),k(2),q(3),v(3),queryStart.toLong).map(x => arena.allocateFrom(JAVA_INT,x.toInt)) :+
              arena.allocateFrom(JAVA_DOUBLE,1.0/math.sqrt(q(3).toDouble))
          }
          val result=shapes.zip(sums).map { (shape,sum) =>
            keep(execute(shape,shape.numel,kernels("attention_gradient_cast")) { (arena,out) =>
              Seq(sum.address,out.address).map(x => arena.allocateFrom(JAVA_LONG,x)) :+
                arena.allocateFrom(JAVA_INT,shape.numel.toInt)
            })
          }
          (result(0),result(1),result(2))
        }.get
    catch case error: Throwable =>
      outputs.reverseIterator.foreach { out =>
        try out.close() catch case cleanup: Throwable => error.addSuppressed(cleanup)
      }
      throw error

  def attentionCache(batchSize: Int,heads: Int,capacity: Int,keyWidth: Int,valueWidth: Int,
      maxBytes: Long = 268435456L): AttentionCache[CudaTensor] =
    check()
    val layout=CacheLayout(batchSize,heads,capacity,keyWidth,valueWidth,maxBytes)
    require(layout.bytes<=this.maxBytes-used,"CUDA cache allocation budget exceeded")
    val keys=allocate(layout.keyShape(capacity))
    try
      val values=allocate(layout.valueShape(capacity))
      new AttentionCache[CudaTensor]:
        private var closed=false
        private var written=0
        def attend(query: CudaTensor,key: CudaTensor,value: CudaTensor,position: Int,
            maxScoreElements: Long = 16777216L): CudaTensor =
          check()
          if closed then throw IllegalStateException("attention cache is closed")
          validate(query);validate(key);validate(value);validate(keys);validate(values)
          val (_,_,end)=layout.validate(query.shape,key.shape,value.shape,position,written,maxScoreElements)
          def append(source: CudaTensor,target: CudaTensor,width: Int): Unit =
            launch(source.shape.numel,kernels("cache_append")) { arena =>
              Seq(source.address,target.address).map(x => arena.allocateFrom(JAVA_LONG,x)) ++
                Seq(source.shape.numel.toInt,width,key.shape.dims(2).toInt,capacity,position).map(x => arena.allocateFrom(JAVA_INT,x))
            }
          append(key,keys,keyWidth);append(value,values,valueWidth)
          val result=attentionFromStorage(query,keys,values,position,end,capacity,maxScoreElements)
          written=end
          result
        def close(): Unit =
          threadCheck()
          if !closed then
            closed=true
            Using.Manager { use => use(keys);use(values);() }.get
    catch case error: Throwable =>
      try keys.close() catch case cleanup: Throwable => error.addSuppressed(cleanup)
      throw error

  /** Concatenate packed tensors on any axis, retaining no input handles. */
  def cat(inputs: Seq[CudaTensor],dim: Int): CudaTensor =
    check()
    val values=inputs.toVector
    values.foreach(validate)
    val output=ConcatLayout.outputShape(values.map(_.shape),dim)
    val half=values.nonEmpty && values.head.half
    require(values.forall(_.half==half),"cat needs one element type")
    val result=allocate(output,half)
    try
      if output.numel>0 then
        val suffix=Shape.from(output.dims.drop(dim+1)).numel
        val rowWidth=output.dims(dim)*suffix
        var offset=0L
        values.foreach { value =>
          val chunk=value.shape.dims(dim)*suffix
          if value.shape.numel>0 then
            launch(value.shape.numel,if half then halfKernels("h_concat") else kernels("concat_copy"),label=if half then "h_concat" else "concat_copy") { arena =>
              Seq(value.address,result.address).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
                Seq(value.shape.numel.toInt,chunk.toInt,rowWidth.toInt,offset.toInt)
                  .map(v => arena.allocateFrom(JAVA_INT,v))
            }
          offset+=chunk
        }
      result
    catch case error: Throwable =>
      try result.close() catch case cleanup: Throwable => error.addSuppressed(cleanup)
      throw error

  /** Packed NCHW nearest resize; copies Float32 bits without arithmetic on values. */
  def resizeNearest2d(input: CudaTensor,height: Int,width: Int,mode: NearestMode=NearestMode.Floor): CudaTensor =
    validate(input)
    val output=Resize2d.outputShape(input.shape,height,width)
    if input.half then
      require(mode==NearestMode.Floor,"Float16 resize supports floor mode only")
      return hexecute(output,output.numel,"h_resize_nearest") { (arena,result) =>
        Seq(input.address,result.address).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
          Seq(output.numel.toInt,input.shape.dims(2).toInt,input.shape.dims(3).toInt,height,width).map(v => arena.allocateFrom(JAVA_INT,v))
      }
    execute(output,output.numel,kernels("resize_nearest")) { (arena,result) =>
      Seq(input.address,result.address).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
        Seq(output.numel.toInt,input.shape.dims(2).toInt,input.shape.dims(3).toInt,height,width,
          if mode==NearestMode.Exact then 1 else 0).map(v => arena.allocateFrom(JAVA_INT,v))
    }

  def transpose(value: CudaTensor,dim0: Int,dim1: Int): CudaTensor =
    validate(value)
    val dims=value.shape.dims
    require(dim0>=0 && dim0<dims.size && dim1>=0 && dim1<dims.size,"transpose dimension out of range")
    if dim0==dim1 then return reshape(value,value.shape)
    def swap[A](v: Vector[A]) = v.updated(dim0,v(dim1)).updated(dim1,v(dim0))
    val output=Shape.from(swap(dims))
    if output.numel==0 then return allocate(output,value.half)
    // Nonempty allocated tensors have Int-bounded dimensions and contiguous strides. Swapping axes lo<hi of
    // [outer,X,mid,Y,inner] gives [outer,Y,mid,X,inner]; only the sizes are passed, so no device metadata is needed.
    val lo=math.min(dim0,dim1);val hi=math.max(dim0,dim1)
    val x=dims(lo);val y=dims(hi)
    val mid=dims.slice(lo+1,hi).product;val inner=dims.drop(hi+1).product
    val transposeArgs=(arena: Arena,result: CudaTensor) =>
      Seq(value.address,result.address).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
        Seq(output.numel,x,mid,y,inner).map(v => arena.allocateFrom(JAVA_INT,v.toInt))
    if value.half then hexecute(output,output.numel,"h_transpose5")(transposeArgs)
    else execute(output,output.numel,kernels("transpose5"),threadsPerBlock=256)(transposeArgs)

  def embedding(table: CudaTensor,indices: Tensor[Long]): CudaTensor =
    validate(table)
    require(table.shape.rank==2 && table.shape.dims(1)>0,"embedding requires a matrix with positive width")
    val ids=indices.toLongArray
    require(ids.forall(id => id>=0 && id<table.shape.dims(0)),"embedding token ID out of bounds")
    val output=Shape.from(indices.shape.dims :+ table.shape.dims(1))
    require(output.numel<=Int.MaxValue,"embedding output exceeds Int indexing")
    if output.numel==0 then return allocate(output)
    withInts(ids.map(_.toInt)) { address =>
      execute(output,output.numel,kernels("embedding")) { (arena,result) =>
        Seq(table.address,result.address,address).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
          Seq(output.numel.toInt,table.shape.dims(1).toInt).map(v => arena.allocateFrom(JAVA_INT,v))
      }
    }

  def embeddingBackward(tableShape: Shape,indices: Tensor[Long],gradient: CudaTensor): CudaTensor =
    validate(gradient)
    val ids=EmbeddingGradient.indices(tableShape,indices,gradient.shape)
    if ids.isEmpty then return fill(tableShape,0f)
    val rows=tableShape.dims(0).toInt;val width=tableShape.dims(1).toInt
    val bytes=(tableShape.numel+rows.toLong+ids.length.toLong)*4
    require(bytes<=maxBytes-used,"embedding gradient workspace budget exceeded")
    val heads=Array.fill(rows)(-1);val next=new Array[Int](ids.length)
    // Reverse insertion preserves the original logical token order in each row.
    var i=ids.length-1
    while i>=0 do
      next(i)=heads(ids(i));heads(ids(i))=i;i-=1
    withInts(heads) { headAddress =>
      withInts(next) { nextAddress =>
        execute(tableShape,tableShape.numel,kernels("embedding_backward")) { (arena,result) =>
          Seq(gradient.address,headAddress,nextAddress,result.address).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
            Seq(tableShape.numel.toInt,width).map(v => arena.allocateFrom(JAVA_INT,v))
        }
      }
    }

  // ---- Float16 activations: storage __half, Float32 arithmetic (kernels in half.cu) ----
  private def hexecute(output: Shape,count: Long,name: String,threadsPerBlock: Int=256)
      (arguments: (Arena,CudaTensor) => Seq[MemorySegment]): CudaTensor =
    val result=allocate(output,half=true)
    try
      launch(count,halfKernels(name),threadsPerBlock,1,name)(arena => arguments(arena,result))
      result
    catch case error: Throwable =>
      try result.close() catch case cleanup: Throwable => error.addSuppressed(cleanup)
      throw error

  /** Float32 to Float16 copy of the same shape. */
  def castToHalf(value: CudaTensor): CudaTensor =
    validate(value)
    require(!value.half,"tensor is already Float16")
    hexecute(value.shape,value.shape.numel,"h_f32_to_f16") { (arena,result) =>
      Seq(value.address,result.address).map(v => arena.allocateFrom(JAVA_LONG,v)) :+ arena.allocateFrom(JAVA_INT,value.shape.numel.toInt)
    }

  /** Float16 to Float32 copy of the same shape. */
  def castToFloat(value: CudaTensor): CudaTensor =
    validate(value)
    require(value.half,"tensor is not Float16")
    val result=allocate(value.shape)
    try
      launch(value.shape.numel,halfKernels("h_f16_to_f32"),256,1,"h_f16_to_f32") { arena =>
        Seq(value.address,result.address).map(v => arena.allocateFrom(JAVA_LONG,v)) :+ arena.allocateFrom(JAVA_INT,value.shape.numel.toInt)
      }
      result
    catch case error: Throwable =>
      try result.close() catch case cleanup: Throwable => error.addSuppressed(cleanup)
      throw error

  /** Uploads Float32 host data and stores it on the device as Float16. */
  def uploadHalf(source: Tensor[Float]): CudaTensor = Using.resource(upload(source))(castToHalf)

  def fillHalf(shape: Shape,value: Float): CudaTensor =
    hexecute(shape,shape.numel,"h_fill") { (arena,result) =>
      Seq(arena.allocateFrom(JAVA_LONG,result.address),arena.allocateFrom(JAVA_INT,shape.numel.toInt),arena.allocateFrom(JAVA_FLOAT,value))
    }

  /** Non-causal attention over [batch, heads, tokens, width] Float16 tensors via cuBLAS and a block softmax. */
  private def attentionHalf(query: CudaTensor,key: CudaTensor,value: CudaTensor,output: Shape,causal: Boolean,queryStart: Int): CudaTensor =
    require(cublas.isDefined,"Float16 attention requires cuBLAS")
    require(key.half && value.half,"Float16 attention needs Float16 key and value")
    val q=query.shape.dims;val k=key.shape.dims;val v=value.shape.dims
    require(q(1)==k(1) && q(1)==v(1) && q(0)==k(0) && q(0)==v(0) && q(3)==k(3),"Float16 attention needs equal head counts")
    val batch=(q(0)*q(1)).toInt;val qt=q(2).toInt;val kt=k(2).toInt;val d=q(3).toInt;val dv=v(3).toInt
    val result=allocate(output,half=true)
    try
      // Tensor-core kernel that keeps the scores in shared memory (see h_attention_fused): no score matrix in device memory at all.
      // one kernel per number of 16-wide chunks (3 = width 33..48, 4 = 49..64, 5 = 65..80); other widths use cuBLAS
      val fusedChunks=(d+15)/16
      if !causal && CudaSession.fusedHalfAttention && wmmaStandardLayout && qt%16==0 && kt%16==0 && qt>=16 && kt>=CudaSession.fusedHalfAttentionMinKeys &&
          d>=33 && d<=80 && (dv+15)/16==fusedChunks && d%8==0 && dv%8==0 then // fragment loads need 16-byte aligned rows
        val started=if profileEnabled then System.nanoTime() else 0L
        // query tiles per warp: more of them per read of the keys and values, as far as the width leaves registers for
        val tiles=Seq(4,2,1).find(t => t<=CudaSession.fusedHalfAttentionTiles && qt%(16*t)==0 && (t<4 || fusedChunks<=4)).get
        val kernelName=s"h_attention_fused_r${tiles}c$fusedChunks"
        launch(batch.toLong*(qt/(16*tiles)),halfKernels(kernelName),32,32,kernelName) { arena =>
          Seq(query.address,key.address,value.address,result.address).map(x => arena.allocateFrom(JAVA_LONG,x)) ++
            Seq(batch,qt,kt,d,dv).map(x => arena.allocateFrom(JAVA_INT,x)) :+
            arena.allocateFrom(JAVA_FLOAT,(1.0/math.sqrt(d.toDouble)).toFloat)
        }
        return result
      // The score matrix never exists as a whole: heads are processed in groups and, when even one head does not fit the scratch
      // budget (very large images), the queries in blocks. `SCALATORCH_CUDA_ATTENTION_SCRATCH_MB` (default 256) trades VRAM for speed.
      val budget=math.min(CudaSession.attentionScratchBytes,math.max(1L,maxBytes-used))
      val perBatch=qt.toLong*kt*2
      val chunk=math.max(1L,math.min(batch.toLong,budget/perBatch)).toInt
      val queryBlock=if perBatch<=budget || causal then qt else math.max(1,math.min(qt.toLong,budget/(kt.toLong*2)).toInt)
      val scale=(1.0/math.sqrt(d.toDouble)).toFloat
      Using.resource(allocate(Shape.from(Seq(chunk.toLong*queryBlock*kt)),half=true)) { scratch =>
        var start=0
        while start<batch do
          val n=math.min(chunk,batch-start)
          var qs=0
          while qs<qt do
            val qb=math.min(queryBlock,qt-qs)
            val qa=query.address+start.toLong*qt*d*2+qs.toLong*d*2
            val ka=key.address+start.toLong*kt*d*2
            val va=value.address+start.toLong*kt*dv*2
            val ra=result.address+start.toLong*qt*dv*2+qs.toLong*dv*2
            val started=if profileEnabled then System.nanoTime() else 0L
            cublas.get.gemmHalfStridedBatched(1,0,kt,qb,d,scale,ka,d,kt.toLong*d,qa,d,qt.toLong*d,scratch.address,kt,qb.toLong*kt,n)
            if profileEnabled then
              driver.synchronize(stream)
              record("half_attention_scores",started)
            if causal then
              launch(n.toLong*qb,halfKernels("h_softmax_causal"),256,256,"h_softmax_causal") { arena =>
                Seq(arena.allocateFrom(JAVA_LONG,scratch.address),arena.allocateFrom(JAVA_INT,(n.toLong*qb).toInt),
                  arena.allocateFrom(JAVA_INT,kt),arena.allocateFrom(JAVA_INT,qb),arena.allocateFrom(JAVA_INT,queryStart))
              }
            else
              launch(n.toLong*qb,halfKernels("h_softmax"),256,256,"h_softmax") { arena =>
                Seq(arena.allocateFrom(JAVA_LONG,scratch.address),arena.allocateFrom(JAVA_INT,(n.toLong*qb).toInt),arena.allocateFrom(JAVA_INT,kt))
              }
            val started2=if profileEnabled then System.nanoTime() else 0L
            cublas.get.gemmHalfStridedBatched(0,0,dv,qb,kt,1f,va,dv,kt.toLong*dv,scratch.address,kt,qb.toLong*kt,ra,dv,qt.toLong*dv,n)
            if profileEnabled then
              driver.synchronize(stream)
              record("half_attention_output",started2)
            qs+=qb
          start+=n
      }
      result
    catch case error: Throwable =>
      try result.close() catch case cleanup: Throwable => error.addSuppressed(cleanup)
      throw error

  def upload(source: Tensor[Float]): CudaTensor =
    val started=if profileEnabled then System.nanoTime() else 0L
    val allocationStarted=if profileEnabled then System.nanoTime() else 0L
    val result=allocate(source.shape)
    record("upload device alloc",allocationStarted)
    try
      if source.shape.numel > 0 then
        val bytes=source.shape.numel*4
        val stagingStarted=if profileEnabled then System.nanoTime() else 0L
        val allocationStarted=if profileEnabled then System.nanoTime() else 0L
        val host=uploadStaging.segment(bytes)
        record("upload staging alloc",allocationStarted)
        val copyStarted=if profileEnabled then System.nanoTime() else 0L
        if source.layout.isContiguous then
          MemorySegment.copy(MemorySegment.ofArray(source.storage.data),source.layout.offset*4,host,0,bytes)
        else
          MemorySegment.copy(MemorySegment.ofArray(source.toFloatArray),0,host,0,bytes)
        record("upload host copy",copyStarted)
        record("upload host staging",stagingStarted)
        hostSync()
        if profileEnabled then
          driver.copyHostToDeviceProfiled(result.address,host,bytes) { (label,nanos) => recordElapsed(label,nanos) }
        else driver.copyHostToDevice(result.address,host,bytes)
      record("upload including allocate",started)
      result
    catch case error: Throwable =>
      try result.close() catch case cleanup: Throwable => error.addSuppressed(cleanup)
      throw error

  /** Weights that stay on the device between sessions and are shared by the sessions of all threads (see [[CudaWeightPool]]): the first call
    * uploads `source`, later calls return a view of the same device memory. `key` identifies the parameter (by identity), `source` is checked by
    * identity as well, so replaced data is uploaded again. The memory is freed after the context keeper's idle time or by `releaseSharedWeights`.
    * Falls back to a private upload when keeping is disabled. */
  def sharedUpload(key: AnyRef,source: Tensor[Float],half: Boolean,halfBits: Option[() => MemorySegment]=None): CudaTensor =
    check()
    if !CudaContextKeeper.enabled then return (if half then uploadHalf(source) else upload(source))
    def view(entry: CudaWeightPool.Entry): CudaTensor =
      val v=new CudaTensor(this,entry.shape,entry.allocation,half)
      live+=v
      v
    val hit=CudaWeightPool.acquire(deviceIndex,key,half,source,touchedShared,stamp)
    if hit!=null then return view(hit)
    CudaWeightPool.lock.synchronized {
      val again=CudaWeightPool.acquire(deviceIndex,key,half,source,touchedShared,stamp)
      if again!=null then view(again)
      else
        // Only the kept tensor comes from the shared arena. The Float32 staging block of a half upload stays in this session's arena:
        // a block freed there is reused on this stream only, so the queued conversion cannot race with another session.
        def kept(body: => CudaTensor): CudaTensor = { sharedAllocation=true;try body finally sharedAllocation=false }
        // Where weights get evicted they are uploaded again and again: send the cached Float16 host copy (half the bytes, no staging).
        val tensor=
          if half && halfBits.nonEmpty && CudaWeightPool.evictedOnce then kept(uploadHalfBits(source.shape,halfBits.get()))
          else if half then Using.resource(upload(source))(staged => kept(castToHalf(staged)))
          else kept(upload(source))
        tensor.allocation.pooled=true
        used-=tensor.allocation.bytes
        CudaWeightPool.put(deviceIndex,key,new CudaWeightPool.Entry(tensor.allocation,tensor.shape,half,source),touchedShared,stamp)
        tensor
    }

  /** Uploads Float16 data that already sits in native memory as IEEE half bits (2 bytes per element, `bits` must be a native segment).
    * Used for offloaded weights: half the bytes of a Float32 upload and no device-side conversion. */
  def uploadHalfBits(shape: Shape,bits: MemorySegment): CudaTensor =
    val result=allocate(shape,half=true)
    try
      if shape.numel>0 then
        require(bits.byteSize()>=shape.numel*2,"Float16 host buffer is too small")
        hostSync()
        driver.copyHostToDevice(result.address,bits,shape.numel*2)
      result
    catch case error: Throwable =>
      try result.close() catch case cleanup: Throwable => error.addSuppressed(cleanup)
      throw error

  def download(tensor: CudaTensor): Tensor[Float] =
    validate(tensor)
    if tensor.half then return Using.resource(castToFloat(tensor))(download)
    val values=Using.resource(Arena.ofConfined()) { arena =>
      val host=arena.allocate(tensor.shape.numel*4,4)
      if tensor.shape.numel > 0 then
        dtoh(host,Long.box(tensor.address),Long.box(tensor.shape.numel*4))
      host.toArray(JAVA_FLOAT)
    }
    Tensor.fromFloats(values,tensor.shape)

  def matmul(a: CudaTensor,b: CudaTensor): CudaTensor =
    validate(a);validate(b)
    require(a.shape.rank==2 && b.shape.rank==2 && a.shape.dims(1)==b.shape.dims(0),"CUDA GEMM shape mismatch")
    val m=a.shape.dims(0);val k=a.shape.dims(1);val n=b.shape.dims(1)
    require(Seq(m,n,k).forall(_ <= Int.MaxValue),"CUDA GEMM dimension exceeds Int indexing")
    val output=Shape.from(Seq(m,n))
    if cublas.isDefined && m>0 && n>0 && k>0 then
      val result=allocate(output)
      try
        cublasGemm(m.toInt,n.toInt,k.toInt,a.address,k.toInt,b.address,n.toInt,result.address,n.toInt)
        return result
      catch case error: Throwable =>
        try result.close() catch case cleanup: Throwable => error.addSuppressed(cleanup)
        throw error
    val tiles=((m+15)/16)*((n+15)/16)
    execute(output,tiles,tiledFunction,threadsPerBlock=256,threadsPerOutput=256,label=s"gemm_tiled[$m,$n,$k]") { (arena,result) =>
      Seq(a.address,b.address,result.address).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
        Seq(m,n,k,n).map(v => arena.allocateFrom(JAVA_INT,v.toInt))
    }

  def hasCublas: Boolean = cublas.isDefined

  /** x[m,k] * w[n,k]^T without materialising the transpose; requires cuBLAS (see hasCublas). */
  def matmulTransposedB(x: CudaTensor,w: CudaTensor): CudaTensor =
    validate(x);validate(w)
    require(cublas.isDefined,"matmulTransposedB requires cuBLAS")
    require(x.shape.rank==2 && w.shape.rank==2 && x.shape.dims(1)==w.shape.dims(1),"CUDA GEMM shape mismatch")
    val m=x.shape.dims(0).toInt;val k=x.shape.dims(1).toInt;val n=w.shape.dims(0).toInt
    if x.half then
      require(w.half,"Float16 GEMM needs a Float16 weight")
      val half=allocate(Shape.from(Seq(m.toLong,n.toLong)),half=true)
      try
        if m>0 && n>0 && k>0 then
          val started=if profileEnabled then System.nanoTime() else 0L
          cublas.get.gemmHalfRowMajorBT(m,n,k,x.address,w.address,half.address)
          if profileEnabled then
            driver.synchronize(stream)
            record(s"half_gemm_bt[$m,$n,$k]",started)
        return half
      catch case error: Throwable =>
        try half.close() catch case cleanup: Throwable => error.addSuppressed(cleanup)
        throw error
    val result=allocate(Shape.from(Seq(m.toLong,n.toLong)))
    try
      if m>0 && n>0 && k>0 then
        val started=if profileEnabled then System.nanoTime() else 0L
        cublas.get.sgemmRowMajorBT(m,n,k,x.address,w.address,result.address)
        driver.synchronize(stream)
        if profileEnabled then record(s"cublas_sgemm_bt[$m,$n,$k]",started)
      result
    catch case error: Throwable =>
      try result.close() catch case cleanup: Throwable => error.addSuppressed(cleanup)
      throw error

  /** Float16 `x [m,k] * w [n,k]^T + bias [n]` as one pass over the output: the rows are filled with the bias and cuBLAS accumulates
    * the product on top (Float32 accumulation, one rounding). A separate broadcast add read the product and wrote it again.
    * None when this does not apply (not Float16, no cuBLAS, other shapes); the caller then multiplies and adds. */
  def linearWithBias(x: CudaTensor,w: CudaTensor,bias: CudaTensor): Option[CudaTensor] =
    validate(x);validate(w);validate(bias)
    if !(CudaSession.fusedLinearBias && cublas.isDefined && x.half && w.half && bias.half && x.shape.rank==2 && w.shape.rank==2 &&
        x.shape.dims(1)==w.shape.dims(1) && bias.shape.rank==1 && bias.shape.dims(0)==w.shape.dims(0)) then return None
    val m=x.shape.dims(0).toInt;val k=x.shape.dims(1).toInt;val n=w.shape.dims(0).toInt
    if m<=0 || n<=0 || k<=0 || m.toLong*n>Int.MaxValue then return None
    val result=allocate(Shape.from(Seq(m.toLong,n.toLong)),half=true)
    try
      launch(m.toLong*n,halfKernels("h_fill_rows"),label="h_fill_rows",threadsPerBlock=256) { arena =>
        Seq(bias.address,result.address).map(v => arena.allocateFrom(JAVA_LONG,v)) ++ Seq(m*n,n).map(v => arena.allocateFrom(JAVA_INT,v))
      }
      val started=if profileEnabled then System.nanoTime() else 0L
      cublas.get.gemmHalfRowMajorBT(m,n,k,x.address,w.address,result.address,accumulate=true)
      if profileEnabled then
        driver.synchronize(stream)
        record(s"half_gemm_bt[$m,$n,$k]",started)
      Some(result)
    catch case error: Throwable =>
      try result.close() catch case cleanup: Throwable => error.addSuppressed(cleanup)
      throw error

  /** True when the fragments of this device have the element positions h_attention_fused is written for: accumulator and matrix_a
    * alike, lane L (g = L / 4, t = L % 4) holding row g in slots 0, 1, 4, 5 and row g + 8 in slots 2, 3, 6, 7, columns
    * 2t, 2t+1, 2t+8, 2t+9. Measured once per session; the kernel is not used otherwise. */
  private lazy val wmmaStandardLayout: Boolean =
    try
      val layout=wmmaLayout()
      (0 until 2).forall { kind => (0 until 32).forall { lane => (0 until 8).forall { slot =>
        val g=lane/4;val t=lane%4
        val row=if Set(0,1,4,5)(slot) then g else g+8
        val column=2*t+(slot&1)+(if slot>=4 then 8 else 0)
        layout((kind*32+lane)*8+slot)==row*16+column
      } } }
    catch case _: Throwable => false

  /** Element positions (row * 16 + column) of the 16x16 tensor-core fragments on this device, per kind, lane and slot:
    * `(kind * 32 + lane) * 8 + slot` with kind 0 accumulator, 1 matrix_a, 2 matrix_b column major, 3 matrix_b row major. */
  private[ffm] def wmmaLayout(): Array[Int] =
    Using.resource(allocate(Shape.from(Seq(1024L)))) { out =>
      launch(1,halfKernels("h_wmma_layout"),32,32,"h_wmma_layout") { arena => Seq(arena.allocateFrom(JAVA_LONG,out.address)) }
      download(out).toFloatArray.map(_.round)
    }

  /** Opaque Float16 copy of a Float32 tensor (2 bytes per element). Only matmulTransposedBHalf may read it. */
  def toHalf(value: CudaTensor): CudaTensor =
    validate(value)
    val packed=allocate(Shape.from(Seq((value.shape.numel+1)/2)))
    try
      launch(value.shape.numel,kernels("f32_to_f16"),threadsPerBlock=256,label="f32_to_f16") { arena =>
        Seq(arena.allocateFrom(JAVA_LONG,value.address),arena.allocateFrom(JAVA_LONG,packed.address),
          arena.allocateFrom(JAVA_INT,value.shape.numel.toInt))
      }
      packed
    catch case error: Throwable =>
      try packed.close() catch case cleanup: Throwable => error.addSuppressed(cleanup)
      throw error

  /** x[m,k] (Float32, cast to Float16 internally) * wHalf[n,k]^T on tensor cores; Float32 output. */
  def matmulTransposedBHalf(x: CudaTensor,wHalf: CudaTensor,n: Int,k: Int): CudaTensor =
    validate(x);validate(wHalf)
    require(cublas.isDefined,"FP16 GEMM requires cuBLAS")
    require(x.shape.rank==2 && x.shape.dims(1)==k,"CUDA GEMM shape mismatch")
    val m=x.shape.dims(0).toInt
    Using.resource(toHalf(x)) { xHalf =>
      val result=allocate(Shape.from(Seq(m.toLong,n.toLong)))
      try
        if m>0 && n>0 && k>0 then
          val started=if profileEnabled then System.nanoTime() else 0L
          cublas.get.gemmExHalfBT(m,n,k,xHalf.address,wHalf.address,result.address)
          driver.synchronize(stream)
          if profileEnabled then record(s"cublas_gemm_fp16[$m,$n,$k]",started)
        result
      catch case error: Throwable =>
        try result.close() catch case cleanup: Throwable => error.addSuppressed(cleanup)
        throw error
    }

  private def matmulInto(a: Long,b: Long,c: Long,m: Int,n: Int,k: Int,ldc: Int): Unit =
    if cublasGemm(m,n,k,a,k,b,n,c,ldc) then return
    val tiles=((m.toLong+15)/16)*((n.toLong+15)/16)
    launch(tiles,tiledFunction,threadsPerBlock=256,threadsPerOutput=256,label=s"gemm_tiled[$m,$n,$k]") { arena =>
      Seq(a,b,c).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
        Seq(m,n,k,ldc).map(v => arena.allocateFrom(JAVA_INT,v))
    }

  private def matmulConvN32Into(a: Long,b: Long,c: Long,m: Int,n: Int,k: Int,ldc: Int,ldb: Int): Unit =
    if cublasGemm(m,n,k,a,k,b,ldb,c,ldc) then return
    val tiles=((m.toLong+15)/16)*((n.toLong+31)/32)
    launch(tiles,convN32Function,threadsPerBlock=256,threadsPerOutput=256,label=s"gemm_conv_n32[$m,$n,$k]") { arena =>
      Seq(a,b,c).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
        Seq(m,n,k,ldc,ldb).map(v => arena.allocateFrom(JAVA_INT,v))
    }

  private def binary(a: CudaTensor,b: CudaTensor,op: Int): CudaTensor =
    validate(a);validate(b)
    require(a.shape==b.shape,"backend arithmetic requires equal shapes")
    require(a.half==b.half,"arithmetic needs one element type")
    if a.half then
      require(op==0,"Float16 supports addition only")
      return hexecute(a.shape,a.shape.numel,"h_add") { (arena,result) =>
        Seq(a.address,b.address,result.address).map(v => arena.allocateFrom(JAVA_LONG,v)) :+ arena.allocateFrom(JAVA_INT,a.shape.numel.toInt)
      }
    execute(a.shape,a.shape.numel,kernels("binary")) { (arena,result) =>
      Seq(a.address,b.address,result.address).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
        Seq(a.shape.numel.toInt,op).map(v => arena.allocateFrom(JAVA_INT,v))
    }
  def add(a: CudaTensor,b: CudaTensor): CudaTensor = binary(a,b,0)
  /** NumPy-style trailing-axis broadcasting, without materializing expanded operands. */
  def addBroadcast(a: CudaTensor,b: CudaTensor): CudaTensor =
    validate(a);validate(b)
    require(a.half==b.half,"broadcast add needs one element type")
    if a.shape==b.shape then return add(a,b)
    val rank=math.max(a.shape.rank,b.shape.rank)
    def padded(s: Shape)=Vector.fill(rank-s.rank)(1L)++s.dims
    val ad=padded(a.shape);val bd=padded(b.shape)
    val output=Shape.from(ad.zip(bd).map { (x,y) =>
      require(x==y || x==1 || y==1,"incompatible broadcast dimensions")
      if x==1 then y else x
    })
    require(output.numel<=Int.MaxValue,"broadcast output exceeds Int indexing")
    if output.numel==0 then return allocate(output,a.half)
    def strides(ds: Vector[Long])=ds.scanRight(1L)(_ * _).tail.zip(ds).map((s,d) => if d==1 then 0L else s)
    if rank<=5 then
      def lead(v: Vector[Long],fill: Long)=Vector.fill(5-rank)(fill)++v
      val dimsP=lead(output.dims,1L);val aS=lead(strides(ad),0L);val bS=lead(strides(bd),0L)
      val broadcastArgs=(arena: Arena,result: CudaTensor) =>
        Seq(a.address,b.address,result.address).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
          (Vector(output.numel)++dimsP++aS++bS).map(v => arena.allocateFrom(JAVA_INT,v.toInt))
      return if a.half then hexecute(output,output.numel,"h_add5")(broadcastArgs)
        else execute(output,output.numel,kernels("broadcast_add5"),threadsPerBlock=256)(broadcastArgs)
    require(!a.half,"Float16 broadcast supports up to five axes")
    withInts((output.dims++strides(ad)++strides(bd)).map(_.toInt).toArray) { metadata =>
      execute(output,output.numel,kernels("broadcast_add")) { (arena,result) =>
        Seq(a.address,b.address,result.address,metadata).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
          Seq(output.numel.toInt,rank).map(v => arena.allocateFrom(JAVA_INT,v))
      }
    }
  /** Final-axis split: value half times Gaussian-error GELU of the gate half. */
  def geglu(input: CudaTensor,fast: Boolean=false): CudaTensor =
    validate(input)
    require(input.shape.rank>0 && input.shape.dims.last>0 && input.shape.dims.last%2==0 &&
      input.shape.dims.last<=Int.MaxValue,"GEGLU requires a positive even final width")
    val width=input.shape.dims.last/2
    val output=Shape.from(input.shape.dims.updated(input.shape.rank-1,width))
    if input.half then return hexecute(output,output.numel,"h_geglu") { (arena,result) =>
      Seq(input.address,result.address).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
        Seq(output.numel.toInt,width.toInt).map(v => arena.allocateFrom(JAVA_INT,v))
    }
    execute(output,output.numel,kernels(if fast then "geglu_fast" else "geglu")) { (arena,result) =>
      Seq(input.address,result.address).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
        Seq(output.numel.toInt,width.toInt).map(v => arena.allocateFrom(JAVA_INT,v))
    }
  def multiply(a: CudaTensor,b: CudaTensor): CudaTensor = binary(a,b,1)
  def fill(shape: Shape,value: Float): CudaTensor =
    execute(shape,shape.numel,kernels("fill")) { (arena,result) =>
      Seq(arena.allocateFrom(JAVA_LONG,result.address),arena.allocateFrom(JAVA_INT,shape.numel.toInt),arena.allocateFrom(JAVA_FLOAT,value))
    }
  def sum(value: CudaTensor): CudaTensor =
    validate(value)
    execute(Shape.from(Seq.empty),1,kernels("sum")) { (arena,result) =>
      Seq(value.address,result.address).map(v => arena.allocateFrom(JAVA_LONG,v)) :+
        arena.allocateFrom(JAVA_INT,value.shape.numel.toInt)
    }
  def expandScalar(value: CudaTensor,shape: Shape): CudaTensor =
    validate(value)
    require(value.shape.rank==0,"expandScalar requires rank zero")
    execute(shape,shape.numel,kernels("expand_scalar")) { (arena,result) =>
      Seq(value.address,result.address).map(v => arena.allocateFrom(JAVA_LONG,v)) :+
        arena.allocateFrom(JAVA_INT,shape.numel.toInt)
    }
  def scale(value: CudaTensor,scalar: Float): CudaTensor =
    validate(value)
    if value.half then return hexecute(value.shape,value.shape.numel,"h_scale") { (arena,result) =>
      Seq(value.address,result.address).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
        Seq(arena.allocateFrom(JAVA_INT,value.shape.numel.toInt),arena.allocateFrom(JAVA_FLOAT,scalar))
    }
    execute(value.shape,value.shape.numel,kernels("scale")) { (arena,result) =>
      Seq(value.address,result.address).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
        Seq(arena.allocateFrom(JAVA_INT,value.shape.numel.toInt),arena.allocateFrom(JAVA_FLOAT,scalar))
    }
  /** NCHW/OIHW direct grouped cross-correlation with the exact FP64 accumulation order. */
  def conv2d(input: CudaTensor,weight: CudaTensor,config: Conv2dConfig=Conv2dConfig()): CudaTensor =
    conv2d(input,weight,config,fast=false)

  /** FP32 FMA convolution for inference, matching the arithmetic class used by CUDA DL libraries. */
  def conv2dFast(input: CudaTensor,weight: CudaTensor,config: Conv2dConfig=Conv2dConfig()): CudaTensor =
    conv2d(input,weight,config,fast=true)

  private def conv2d(input: CudaTensor,weight: CudaTensor,config: Conv2dConfig,fast: Boolean): CudaTensor =
    validate(input);validate(weight)
    val g=Conv2dOps.geometry(input.shape,weight.shape,config)
    val dimensions=Array(g.ci,g.h,g.w,g.co,g.cg,g.kh,g.kw,g.oh,g.ow,
      config.stride._1,config.stride._2,config.padding._1,config.padding._2,
      config.dilation._1,config.dilation._2,config.groups,g.n)
    if fast && input.half then
      require(weight.half && cublas.isDefined,"Float16 convolution needs a Float16 weight and cuBLAS")
      val spatial=g.oh.toLong*g.ow
      val k=g.cg.toLong*g.kh*g.kw
      val outputChannelsPerGroup=g.co/config.groups
      val direct=g.kh==1 && g.kw==1 && config.stride==((1,1)) && config.padding==((0,0)) && config.dilation==((1,1))
      val output=allocate(g.shape,half=true)
      try
        // cuDNN convolves without the im2col copy and with tensor-core kernels made for convolution.
        // 1x1 convolutions go there as well when there is more than one sample: the built-in path calls cuBLAS once per sample.
        val planned=if direct && g.n==1 then None else cudnn.flatMap(_.plan(g.n,g.ci,g.h,g.w,g.co,g.cg,g.kh,g.kw,g.oh,g.ow,
          config.stride,config.padding,config.dilation,config.groups,math.max(0L,(maxBytes-used)/2)))
        if planned.nonEmpty then
          val plan=planned.get
          val started=if profileEnabled then System.nanoTime() else 0L
          Using.resource(allocate(Shape.from(Seq(math.max(1L,(plan.workspace+1)/2))),half=true)) { workspace =>
            cudnn.get.forward(plan,input.address,weight.address,workspace.address,output.address)
          }
          if profileEnabled then
            driver.synchronize(stream)
            record(s"cudnn_conv[${g.n},${g.ci},${g.h},${g.w},${g.co},${g.kh}]",started)
          return output
        def gemm(batch: Int,group: Int,inputAddress: Long,ldb: Int,positions: Int,positionStart: Long): Unit =
          val weightOffset=group.toLong*outputChannelsPerGroup*k*2
          val outputOffset=((batch.toLong*g.co+group.toLong*outputChannelsPerGroup)*spatial+positionStart)*2
          val started=if profileEnabled then System.nanoTime() else 0L
          cublas.get.gemmHalfRowMajor(outputChannelsPerGroup,positions,k.toInt,weight.address+weightOffset,k.toInt,
            inputAddress,ldb,output.address+outputOffset,spatial.toInt)
          if profileEnabled then
            driver.synchronize(stream)
            record(s"half_conv_gemm[$outputChannelsPerGroup,$positions,$k]",started)
        // Several samples as ONE GEMM: columns [k, S*P], product [co, S*P], then the sample and channel axes are swapped into the
        // output. cuBLAS is far more efficient on the larger matrix (SD1.5 shapes: 27-30 TFLOPS per sample, about 50 on big ones)
        // and the call count drops from one per sample to one per chunk.
        val perSample=k*spatial
        val mergedCap=math.min(134217728L,(maxBytes-used)/4)
        val samplesPerChunk=if perSample>0 then math.min(g.n.toLong,mergedCap/math.max(1L,perSample+g.co.toLong*spatial)) else 0L
        if !direct && CudaSession.mergedConvolution && config.groups==1 && g.n>1 && samplesPerChunk>=2 &&
            perSample*samplesPerChunk<=Int.MaxValue && g.co.toLong*spatial*samplesPerChunk<=Int.MaxValue then
          Using.resource(allocate(Shape.from(Seq(k,samplesPerChunk*spatial)),half=true)) { columns =>
            Using.resource(allocate(Shape.from(Seq(g.co.toLong,samplesPerChunk*spatial)),half=true)) { product =>
              var start=0
              while start<g.n do
                val samples=math.min(samplesPerChunk,g.n.toLong-start).toInt
                val row=samples.toLong*spatial
                val count=k*row
                launch(count,halfKernels("h_im2col_batch"),label="h_im2col_batch",threadsPerBlock=256) { arena =>
                  Seq(input.address,columns.address).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
                    Seq(count.toInt,g.ci,g.h,g.w,g.kh,g.kw,g.oh,g.ow,config.stride._1,config.stride._2,
                      config.padding._1,config.padding._2,config.dilation._1,config.dilation._2,start,samples)
                      .map(v => arena.allocateFrom(JAVA_INT,v))
                }
                val started=if profileEnabled then System.nanoTime() else 0L
                cublas.get.gemmHalfRowMajor(g.co,row.toInt,k.toInt,weight.address,k.toInt,columns.address,row.toInt,product.address,row.toInt)
                if profileEnabled then
                  driver.synchronize(stream)
                  record(s"half_conv_gemm_merged[${g.co},$row,$k]",started)
                // [co, samples, P] -> [samples, co, P], written straight into the output
                val elements=g.co.toLong*row
                launch(elements,halfKernels("h_transpose5"),label="h_transpose5",threadsPerBlock=256) { arena =>
                  Seq(product.address,output.address+start.toLong*g.co*spatial*2).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
                    Seq(elements.toInt,g.co,1,samples,spatial.toInt).map(v => arena.allocateFrom(JAVA_INT,v))
                }
                start+=samples
            }
          }
        else if direct then
          var batch=0
          while batch<g.n do
            var group=0
            while group<config.groups do
              gemm(batch,group,input.address+(batch.toLong*g.ci+group.toLong*g.cg)*spatial*2,spatial.toInt,spatial.toInt,0L)
              group+=1
            batch+=1
        else
          val positionsPerChunk=math.min(spatial,math.min(134217728L,(maxBytes-used)/2)/k)
          require(positionsPerChunk>0,"CUDA session budget cannot hold one im2col position")
          Using.resource(allocate(Shape.from(Seq(k,positionsPerChunk)),half=true)) { columns =>
            var batch=0
            while batch<g.n do
              var group=0
              while group<config.groups do
                var positionStart=0L
                while positionStart<spatial do
                  val positions=math.min(positionsPerChunk,spatial-positionStart)
                  val count=k*positions
                  launch(count,halfKernels("h_im2col"),label="h_im2col",threadsPerBlock=256) { arena =>
                    Seq(input.address,columns.address).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
                      Seq(count.toInt,g.ci,g.h,g.w,g.cg,g.kh,g.kw,g.oh,g.ow,config.stride._1,config.stride._2,
                        config.padding._1,config.padding._2,config.dilation._1,config.dilation._2,
                        batch,group,positionStart.toInt,positions.toInt).map(v => arena.allocateFrom(JAVA_INT,v))
                  }
                  gemm(batch,group,columns.address,positions.toInt,positions.toInt,positionStart)
                  positionStart+=positions
                group+=1
              batch+=1
          }
        return output
      catch case error: Throwable =>
        try output.close() catch case cleanup: Throwable => error.addSuppressed(cleanup)
        throw error
    if fast then
      val spatial=g.oh.toLong*g.ow
      val k=g.cg.toLong*g.kh*g.kw
      // Geometry lives on the device alongside the output and im2col buffer.
      val availableElements=(maxBytes-used-g.shape.numel*4-dimensions.length.toLong*4)/4
      val columnCapacity=math.min(33554432L,availableElements)
      val positionsPerChunk=math.min(spatial,columnCapacity/k)
      require(positionsPerChunk>0,"CUDA session budget cannot hold one im2col position")
      val columnElements=k*positionsPerChunk
      require(columnElements<=Int.MaxValue,"im2col workspace exceeds Int indexing")
      val output=allocate(g.shape)
      try
        val columns=allocate(Shape.from(Seq(k,positionsPerChunk)))
        try
          withInts(dimensions) { geometry =>
            var batch=0
            while batch<g.n do
              var group=0
              while group<config.groups do
                val outputChannelsPerGroup=g.co/config.groups
                val weightOffset=group.toLong*outputChannelsPerGroup*k*4
                var positionStart=0L
                while positionStart<spatial do
                  val positions=math.min(positionsPerChunk,spatial-positionStart)
                  val count=k*positions
                  launch(count,kernels("im2col"),label=s"im2col[${g.ci},${g.h},${g.w},${g.kh},${g.kw},${g.oh},${g.ow},$positionStart,$positions]") { arena =>
                    Seq(input.address,columns.address,geometry).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
                      Seq(batch,group,positionStart.toInt,positions.toInt,count.toInt).map(v => arena.allocateFrom(JAVA_INT,v))
                  }
                  val outputOffset=((batch.toLong*g.co+group.toLong*outputChannelsPerGroup)*spatial+positionStart)*4
                  matmulConvN32Into(weight.address+weightOffset,columns.address,output.address+outputOffset,
                    outputChannelsPerGroup,positions.toInt,k.toInt,spatial.toInt,positions.toInt)
                  positionStart+=positions
                group+=1
              batch+=1
          }
        finally columns.close()
        output
      catch case error: Throwable =>
        try output.close() catch case cleanup: Throwable => error.addSuppressed(cleanup)
        throw error
    else withInts(dimensions) { geometry =>
      val kernel=kernels("conv2d")
      execute(g.shape,g.shape.numel,kernel) { (arena,result) =>
        Seq(input.address,weight.address,result.address,geometry).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
          Seq(arena.allocateFrom(JAVA_INT,g.shape.numel.toInt),arena.allocateFrom(JAVA_INT,0))
      }
    }

  /** GroupNorm with independently optional per-channel scale/bias; Float32 affine rounding. */
  def groupNorm(input: CudaTensor,groups: Int,epsilon: Double=1e-5,
      weight: Option[CudaTensor]=None,bias: Option[CudaTensor]=None): CudaTensor =
    validate(input)
    val width=GroupNormOps.width(input.shape,groups,epsilon)
    val channels=input.shape.dims(1)
    val spatial=Shape.from(input.shape.dims.drop(2)).numel
    Seq(weight,bias).flatten.foreach { value =>
      validate(value)
      require(value.shape==Shape.from(Seq(channels)),"GroupNorm affine shape mismatch")
    }
    val rows=input.shape.numel/width
    if input.half then
      Seq(weight,bias).flatten.foreach(p => require(p.half,"Float16 GroupNorm needs Float16 affine parameters"))
      return Using.resource(allocate(Shape.from(Seq(rows*2)))) { stats =>
        // One block per group: wide groups get 1024 threads, otherwise a block leaves most of its multiprocessor idle.
        val statThreads=if width>=4096 then 1024 else 256
        launch(rows,halfKernels("h_groupnorm_stats"),threadsPerBlock=statThreads,threadsPerOutput=statThreads,label="h_groupnorm_stats") { arena =>
          Seq(input.address,stats.address).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
            Seq(rows.toInt,width).map(v => arena.allocateFrom(JAVA_INT,v)) :+ arena.allocateFrom(JAVA_FLOAT,epsilon.toFloat)
        }
        hexecute(input.shape,input.shape.numel,"h_groupnorm_apply") { (arena,result) =>
          Seq(input.address,weight.fold(0L)(_.address),bias.fold(0L)(_.address),result.address,stats.address)
            .map(v => arena.allocateFrom(JAVA_LONG,v)) ++
            Seq(input.shape.numel.toInt,width,channels.toInt,spatial.toInt).map(v => arena.allocateFrom(JAVA_INT,v))
        }
      }
    if width>=4096 && rows*16<=maxBytes-used-input.shape.numel*4 then
      Using.resource(allocate(Shape.from(Seq(rows*4)))) { stats =>
        launch(rows,kernels("groupnorm_stats_par"),threadsPerBlock=256,threadsPerOutput=256,label="groupnorm_stats_par") { arena =>
          Seq(input.address,stats.address).map(v=>arena.allocateFrom(JAVA_LONG,v)) ++
            Seq(rows.toInt,width).map(v=>arena.allocateFrom(JAVA_INT,v)) :+ arena.allocateFrom(JAVA_DOUBLE,epsilon)
        }
        execute(input.shape,input.shape.numel,kernels("groupnorm_apply")) { (arena,result) =>
          Seq(input.address,weight.fold(0L)(_.address),bias.fold(0L)(_.address),result.address,stats.address)
            .map(v=>arena.allocateFrom(JAVA_LONG,v)) ++
            Seq(input.shape.numel.toInt,width,channels.toInt,spatial.toInt).map(v=>arena.allocateFrom(JAVA_INT,v))
        }
      }
    else execute(input.shape,rows,kernels("groupnorm")) { (arena,result) =>
      Seq(input.address,weight.fold(0L)(_.address),bias.fold(0L)(_.address),result.address).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
        Seq(rows.toInt,width,channels.toInt,spatial.toInt).map(v => arena.allocateFrom(JAVA_INT,v)) :+
        arena.allocateFrom(JAVA_DOUBLE,epsilon)
    }

  /** Final-axis normalization, with temporary reshape handles released on every path. */
  def layerNorm(input: CudaTensor,epsilon: Double=1e-5,
      weight: Option[CudaTensor]=None,bias: Option[CudaTensor]=None): CudaTensor =
    validate(input)
    require(input.shape.rank>=1 && input.shape.dims.last>0,"LayerNorm requires a positive final dimension")
    val width=input.shape.dims.last
    Using.resource(reshape(input,Shape.from(Seq(input.shape.numel/width,width)))) { flat =>
      Using.resource(groupNorm(flat,1,epsilon,weight,bias)) { normalized =>
        reshape(normalized,input.shape)
      }
    }

  def silu(value: CudaTensor): CudaTensor =
    validate(value)
    if value.half then return hexecute(value.shape,value.shape.numel,"h_silu") { (arena,result) =>
      Seq(value.address,result.address).map(v => arena.allocateFrom(JAVA_LONG,v)) :+ arena.allocateFrom(JAVA_INT,value.shape.numel.toInt)
    }
    execute(value.shape,value.shape.numel,kernels("silu")) { (arena,result) =>
      Seq(value.address,result.address).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
        Seq(arena.allocateFrom(JAVA_INT,value.shape.numel.toInt),arena.allocateFrom(JAVA_LONG,0L),arena.allocateFrom(JAVA_INT,0))
    }
  /** Exact (erf) GELU in Float32, accurate to about 1e-6 relative. */
  def geluErf(value: CudaTensor): CudaTensor = unary(value,"gelu_erf")
  /** CLIP QuickGELU: x * sigmoid(1.702 x). */
  def quickGelu(value: CudaTensor): CudaTensor = unary(value,"quick_gelu")
  private def unary(value: CudaTensor,kernel: String): CudaTensor =
    validate(value)
    require(value.shape.numel<=Int.MaxValue,"elementwise kernel exceeds Int indexing")
    if value.half then return hexecute(value.shape,value.shape.numel,"h_"+kernel) { (arena,result) =>
      Seq(arena.allocateFrom(JAVA_LONG,value.address),arena.allocateFrom(JAVA_LONG,result.address),
        arena.allocateFrom(JAVA_INT,value.shape.numel.toInt))
    }
    execute(value.shape,value.shape.numel,kernels(kernel)) { (arena,result) =>
      Seq(arena.allocateFrom(JAVA_LONG,value.address),arena.allocateFrom(JAVA_LONG,result.address),
        arena.allocateFrom(JAVA_INT,value.shape.numel.toInt))
    }
  /** Elementwise LeakyReLU used by RRDB image restoration models. */
  def leakyRelu(value: CudaTensor,negativeSlope: Float=0.01f): CudaTensor =
    validate(value)
    require(negativeSlope.isFinite && negativeSlope>=0f,"LeakyReLU negative slope must be finite and nonnegative")
    execute(value.shape,value.shape.numel,kernels("leaky_relu")) { (arena,result) =>
      Seq(value.address,result.address).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
        Seq(arena.allocateFrom(JAVA_INT,value.shape.numel.toInt),arena.allocateFrom(JAVA_FLOAT,negativeSlope))
    }
  def siluBackward(value: CudaTensor,gradient: CudaTensor): CudaTensor =
    validate(value);validate(gradient)
    require(value.shape==gradient.shape,"SiLU gradient shape mismatch")
    execute(value.shape,value.shape.numel,kernels("silu")) { (arena,result) =>
      Seq(value.address,result.address).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
        Seq(arena.allocateFrom(JAVA_INT,value.shape.numel.toInt),arena.allocateFrom(JAVA_LONG,gradient.address),arena.allocateFrom(JAVA_INT,1))
    }
  private def rmsBackwardDimensions(value: CudaTensor,gradient: CudaTensor,epsilon: Double): (Long,Long) =
    validate(value);validate(gradient)
    require(value.shape==gradient.shape,"RMSNorm gradient shape mismatch")
    require(epsilon.isFinite && epsilon>0,"invalid RMSNorm epsilon")
    require(value.shape.rank>0 && value.shape.dims.last>0 && value.shape.dims.last<=Int.MaxValue,"invalid RMSNorm width")
    val width=value.shape.dims.last
    (width,value.shape.numel/width)
  def rmsNormInputBackward(value: CudaTensor,weight: CudaTensor,gradient: CudaTensor,epsilon: Double = 1e-6): CudaTensor =
    val (width,rows)=rmsBackwardDimensions(value,gradient,epsilon)
    validate(weight)
    require(weight.shape==Shape.from(Seq(width)),"RMSNorm weight shape mismatch")
    execute(value.shape,rows,kernels("rms_input_backward")) { (arena,result) =>
      Seq(value.address,weight.address,gradient.address,result.address).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
        Seq(rows.toInt,width.toInt).map(v => arena.allocateFrom(JAVA_INT,v)) :+ arena.allocateFrom(JAVA_DOUBLE,epsilon)
    }
  def rmsNormWeightBackward(value: CudaTensor,gradient: CudaTensor,epsilon: Double = 1e-6): CudaTensor =
    val (width,rows)=rmsBackwardDimensions(value,gradient,epsilon)
    // Private scratch stores one Double inverse per row in two Float32-sized slots.
    val inverse=execute(Shape.from(Seq(rows,2L)),rows,kernels("rms_inverse")) { (arena,result) =>
      Seq(value.address,result.address).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
        Seq(rows.toInt,width.toInt).map(v => arena.allocateFrom(JAVA_INT,v)) :+ arena.allocateFrom(JAVA_DOUBLE,epsilon)
    }
    Using.resource(inverse) { _ =>
      execute(Shape.from(Seq(width)),width,kernels("rms_weight_backward")) { (arena,result) =>
        Seq(value.address,gradient.address,inverse.address,result.address).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
          Seq(rows.toInt,width.toInt).map(v => arena.allocateFrom(JAVA_INT,v))
      }
    }
  def rmsNorm(value: CudaTensor,weight: CudaTensor,epsilon: Double = 1e-6): CudaTensor =
    validate(value);validate(weight)
    require(epsilon.isFinite && epsilon>0,"invalid RMSNorm epsilon")
    require(value.shape.rank>0 && value.shape.dims.last>0 && weight.shape==Shape.from(Seq(value.shape.dims.last)),"invalid RMSNorm shapes")
    val width=value.shape.dims.last
    val rows=value.shape.numel/width
    execute(value.shape,rows,kernels("rmsnorm")) { (arena,result) =>
      Seq(value.address,weight.address,result.address).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
        Seq(rows.toInt,width.toInt).map(v => arena.allocateFrom(JAVA_INT,v)) :+
        arena.allocateFrom(JAVA_DOUBLE,epsilon)
    }

  private def execute(output: Shape,count: Long,kernel: MemorySegment,
      threadsPerBlock: Int=64,threadsPerOutput: Int=1,label: String="")
      (arguments: (Arena,CudaTensor) => Seq[MemorySegment]): CudaTensor =
    val result=allocate(output)
    try
      val kernelLabel=if label.nonEmpty then label else if kernel.address()==tiledFunction.address() then "gemm_tiled"
        else kernels.collectFirst { case (name,function) if function.address()==kernel.address() => name }.getOrElse("unknown")
      launch(count,kernel,threadsPerBlock,threadsPerOutput,kernelLabel)(arena => arguments(arena,result))
      result
    catch case error: Throwable =>
      try result.close() catch case cleanup: Throwable => error.addSuppressed(cleanup)
      throw error

  private def launch(count: Long,kernel: MemorySegment,threadsPerBlock: Int=64,threadsPerOutput: Int=1,label: String="unknown")
      (arguments: Arena => Seq[MemorySegment]): Unit =
      require(threadsPerBlock>0 && threadsPerOutput>0 && threadsPerBlock%threadsPerOutput==0,
        "CUDA launch dimensions must divide evenly")
      val outputsPerBlock=threadsPerBlock/threadsPerOutput
      if count > 0 then Using.resource(Arena.ofConfined()) { arena =>
        val started=if profileEnabled then System.nanoTime() else 0L
        val prepStarted=if launchStats then System.nanoTime() else 0L
        val args=arguments(arena)
        val params=arena.allocate(ADDRESS,args.size.toLong)
        args.zipWithIndex.foreach((arg,i) => params.setAtIndex(ADDRESS,i.toLong,arg))
        val blocks=((count+outputsPerBlock-1)/outputsPerBlock).toInt
        if launchStats then
          val t0=System.nanoTime()
          driver.launchKernel(kernel,blocks,threadsPerBlock,stream,params)
          launchNanos+=System.nanoTime()-t0;launchCount+=1
          launchPrepNanos+=t0-prepStarted
        else driver.launchKernel(kernel,blocks,threadsPerBlock,stream,params)
        if syncEveryLaunch then
          try driver.synchronize(stream)
          catch case error: Throwable => throw IllegalStateException(s"CUDA kernel '$label' failed during synchronization",error)
        if profileEnabled then
          val resolved=if label!="unknown" then label else kernels.collectFirst {
            case (name,function) if function.address()==kernel.address() => name
          }.getOrElse("unknown")
          record(resolved,started)
      }

  def release(tensor: CudaTensor): Unit =
    threadCheck()
    require(tensor.session eq this,"tensor belongs to another CUDA session")
    if !tensor.released && tensor.allocation.pooled then
      validate(tensor)
      tensor.released=true;live-=tensor
    else if !tensor.released then
      validate(tensor)
      if tensor.allocation.references==1 then
        val size=tensor.allocation.bytes
        tensor.allocation.arena.release(tensor.address)
        used-=size
      tensor.allocation.references-=1
      tensor.released=true;live-=tensor

  private def shutdown(): Unit =
    threadCheck()
    var failure: Throwable=null
    def attempt(body: => Unit): Unit = try body catch case error: Throwable =>
      if failure==null then failure=error else failure.addSuppressed(error)
    // External native code may have changed the current context. Rebind for all
    // reverse-order cleanup; CudaDriver subsequently restores the caller's context.
    attempt { driver.call("cuCtxSetCurrent",Seq(ADDRESS),context) }
    attempt { driver.synchronize(stream) }
    live.toVector.reverse.foreach(tensor => attempt { release(tensor) })
    attempt { trimPool() }
    attempt { CudaWeightPool.trimArenas(deviceIndex,rawFree) }
    if weightArenaUsed then weightArena.retired=true
    var shared=0L;touchedShared.forEach(a => shared+=a.bytes)
    attempt { CudaWeightPool.releaseUsers(touchedShared) }
    if launchStats && launchCount>0 then
      System.err.println(f"[scala-torch CUDA launches] $launchCount launches, ${launchNanos/1e6}%.1f ms in cuLaunchKernel, " +
        f"${launchPrepNanos/1e6}%.1f ms preparing arguments, session open ${(System.nanoTime()-opened)/1e6}%.1f ms")
    if memoryLog then
      def mb(v: Long)=v/1048576
      System.err.println(s"[scala-torch CUDA memory] session closed: peak live ${mb(peakUsed)} MiB, peak chunks ${mb(peakOwned)} MiB, kept weights used ${mb(shared)} MiB; " +
        s"process holds ${mb(held.get())} MiB (kept weights ${mb(CudaWeightPool.arenaBytes(deviceIndex)._1)} MiB in ${mb(CudaWeightPool.arenaBytes(deviceIndex)._2)} MiB of chunks, unused ${mb(CudaWeightPool.unusedBytes(deviceIndex))} MiB), driver free ${mb(driver.freeBytes())} MiB")
    if profileEnabled then
      val entries=profile.toVector
      val selected=(entries.sortBy(-_._2._2).take(400) ++ entries.filter { case (label,_) =>
        label.contains("upload") || label.startsWith("HtoD")
      }).distinctBy(_._1)
      selected.foreach { case (label,(count,nanos)) =>
        System.err.println(f"[scala-torch CUDA profile] $label%-24s $count%6d calls ${nanos/1e9}%.3f s")
      }
    if failure != null then throw failure

object CudaSession:
  /** Frees the weights kept for sessions on `deviceIndex` (no-op while a session is open there; they go on idle then). */
  def releaseSharedWeights(deviceIndex: Int): Unit = CudaContextKeeper.clearDevice(deviceIndex)
  /** Chunk size of a session's arena (`SCALATORCH_CUDA_CHUNK_MB`, default 128). */
  private[ffm] val chunkBytes: Long =
    sys.env.get("SCALATORCH_CUDA_CHUNK_MB").flatMap(_.trim.toLongOption).filter(_ > 0).getOrElse(128L)*1024L*1024L
  /** Float16 convolution of a batch as one GEMM (`SCALATORCH_CUDA_MERGED_CONV=0` keeps one GEMM per sample). */
  private[ffm] val mergedConvolution: Boolean = !sys.env.get("SCALATORCH_CUDA_MERGED_CONV").contains("0")
  /** Float16 self-attention through the fused tensor-core kernels (h_attention_fused_*): no score matrix in device memory.
    * RTX 3090: [16,4096,40] 3.08 -> 1.09 ms, [64,4096,40] 12.4 -> 3.95 ms, [20,4096,64] 3.86 -> 1.83 ms against the cuBLAS path.
    * Used when the fragment layout of the device is the expected one; `SCALATORCH_CUDA_FUSED_ATTENTION_HALF=0` keeps cuBLAS. */
  private[ffm] val fusedHalfAttention: Boolean = !sys.env.get("SCALATORCH_CUDA_FUSED_ATTENTION_HALF").contains("0")
  private[ffm] val fusedHalfAttentionTiles: Int =
    sys.env.get("SCALATORCH_CUDA_FUSED_ATTENTION_TILES").flatMap(_.trim.toIntOption).filter(_ >= 1).getOrElse(2)
  /** Fewest keys for the fused kernel (`SCALATORCH_CUDA_FUSED_ATTENTION_MIN_KEYS`). */
  private[ffm] val fusedHalfAttentionMinKeys: Int =
    sys.env.get("SCALATORCH_CUDA_FUSED_ATTENTION_MIN_KEYS").flatMap(_.trim.toIntOption).filter(_ >= 16).getOrElse(16)
  /** Bias folded into the Float16 matrix product of a linear layer (`SCALATORCH_CUDA_FUSED_LINEAR_BIAS=0` keeps the separate add). */
  private[ffm] val fusedLinearBias: Boolean = !sys.env.get("SCALATORCH_CUDA_FUSED_LINEAR_BIAS").contains("0")
  /** True when cuBLAS can be loaded (search path, `SCALATORCH_CUDNN_PATH`, a CUDA Toolkit). Float16 execution needs it. */
  def cublasLoadable: Boolean = Cublas.loadable
  /** True while device memory is known to be short (see `CudaWeightPool.tight`). */
  def memoryTight: Boolean = CudaWeightPool.tight
  /** Records that a model did not fit the device. */
  def markMemoryTight(): Unit = CudaWeightPool.markTight()
  /** `maxBytes` value meaning "92 % of the device memory that is free when the session opens". */
  val AutoBudget: Long = Long.MaxValue
  /** Parses a byte budget from configuration: a number of bytes or `auto`. */
  def parseBudget(value: String): Long =
    val v = value.trim
    if v.equalsIgnoreCase("auto") then AutoBudget else v.toLong
  /** Scratch for half-precision attention scores (bytes); smaller = less VRAM, more (still tiny) GEMM launches. */
  @volatile private[ffm] var attentionScratchBytes: Long =
    sys.env.get("SCALATORCH_CUDA_ATTENTION_SCRATCH_MB").flatMap(_.trim.toLongOption).filter(_ > 0).getOrElse(256L)*1024L*1024L
  def withDevice[A](deviceIndex: Int = 0, maxBytes: Long = 1073741824L)(body: CudaSession => A): A =
    require(maxBytes >= 0,"negative CUDA allocation budget")
    CudaDriver.withDevice(deviceIndex) { (driver,version,name) =>
      // AutoBudget: use what the device has free right now (a fixed number made big jobs fail with "allocation budget exceeded"
      // although the GPU had room, for example SDXL with a LoRA needs the weights twice).
      // Kept weights that no open session uses count as free: they are evicted when the room is needed.
      val effective=if maxBytes==AutoBudget then
        math.max(268435456L,((CudaWeightPool.governedFree(deviceIndex,driver.memoryInfo()._1)+CudaWeightPool.unusedBytes(deviceIndex))*0.92).toLong)
      else maxBytes
      body(new CudaSession(driver,deviceIndex,name,version,effective))
    }

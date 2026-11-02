package scalatorch.ffm

import scalatorch.*
import java.lang.foreign.{Arena, MemorySegment}
import java.lang.foreign.ValueLayout.{ADDRESS, JAVA_DOUBLE, JAVA_FLOAT, JAVA_INT, JAVA_LONG}
import java.nio.charset.StandardCharsets.UTF_8
import scala.collection.mutable
import scala.util.Using

private[ffm] final class CudaAllocation(val address: Long,val bytes: Long):
  var references=1L

/** Independently releasable packed Float32 handle; reshape may share an allocation. */
final class CudaTensor private[ffm] (private[ffm] val session: CudaSession,
    val shape: Shape, private[ffm] val allocation: CudaAllocation) extends AutoCloseable:
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
  private val live=mutable.LinkedHashSet.empty[CudaTensor]
  private val context=driver.pointerResult("cuCtxGetCurrent",Seq.empty)
  private val ptx=Using.resource(getClass.getResourceAsStream("/scalatorch/gemm.ptx")) { input =>
    require(input != null,"GEMM PTX resource missing")
    new String(input.readAllBytes(),UTF_8)
  }
  private val module=driver.pointerResult("cuModuleLoadData",Seq(ADDRESS),driver.arena.allocateFrom(ptx))
  driver.cleanup { driver.call("cuModuleUnload",Seq(ADDRESS),module) }
  private val function=driver.pointerResult("cuModuleGetFunction",Seq(ADDRESS,ADDRESS),module,driver.arena.allocateFrom("gemm"))
  private val opsPtx=Using.resource(getClass.getResourceAsStream("/scalatorch/transformer.ptx")) { input =>
    require(input != null,"transformer PTX resource missing")
    new String(input.readAllBytes(),UTF_8)
  }
  private val opsModule=driver.pointerResult("cuModuleLoadData",Seq(ADDRESS),driver.arena.allocateFrom(opsPtx))
  driver.cleanup { driver.call("cuModuleUnload",Seq(ADDRESS),opsModule) }
  private val kernels=Seq("geglu","broadcast_add","concat_copy","resize_nearest","groupnorm","conv2d","adamw_update","finite_status","cross_entropy_stats","cross_entropy_output","cross_entropy_gradient","binary","scale","fill","sum","expand_scalar","silu","rmsnorm","rms_input_backward","rms_inverse","rms_weight_backward","transpose","embedding","embedding_backward","rope","attention","attention_backward","attention_gradient_cast","cache_append").map { name =>
    name -> driver.pointerResult("cuModuleGetFunction",Seq(ADDRESS,ADDRESS),opsModule,driver.arena.allocateFrom(name))
  }.toMap
  private val stream=driver.pointerResult("cuStreamCreate",Seq(JAVA_INT),Int.box(1))
  driver.cleanup { driver.call("cuStreamDestroy_v2",Seq(ADDRESS),stream) }
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

  private def allocate(shape: Shape): CudaTensor =
    check()
    require(shape.numel <= Int.MaxValue,"CUDA tensor exceeds Int indexing")
    val bytes=math.max(shape.numel*4,1L)
    require(bytes <= maxBytes-used,"CUDA session allocation budget exceeded")
    val address=Using.resource(Arena.ofConfined()) { arena =>
      val result=arena.allocate(JAVA_LONG)
      driver.call("cuMemAlloc_v2",Seq(ADDRESS,JAVA_LONG),result,Long.box(bytes))
      result.get(JAVA_LONG,0)
    }
    val tensor=new CudaTensor(this,shape,new CudaAllocation(address,bytes))
    live+=tensor;used+=bytes
    tensor

  def reshape(value: CudaTensor,shape: Shape): CudaTensor =
    validate(value)
    require(value.shape.numel==shape.numel,"reshape changes element count")
    val view=new CudaTensor(this,shape,value.allocation)
    value.allocation.references+=1
    live+=view
    view

  /** Private integer staging allocation, charged to the same device byte budget. */
  private def withInts[A](values: Array[Int])(body: Long => A): A =
    val buffer=allocate(Shape.from(Seq(values.length.toLong)))
    Using.resource(buffer) { _ =>
      Using.resource(Arena.ofConfined()) { arena =>
        if values.nonEmpty then
          val host=arena.allocateFrom(JAVA_INT,values*)
          driver.copyHostToDevice(buffer.address,host,values.length.toLong*4)
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
        if bytes>0 then driver.call("cuMemcpyDtoH_v2",Seq(ADDRESS,JAVA_LONG,JAVA_LONG),host,Long.box(buffer.address),Long.box(bytes))
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
        driver.copyHostToDevice(buffer.address,host,saved.shape.numel*16)
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
        driver.copyHostToDevice(buffer.address,host,count*8)
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

  /** Noncausal self/cross-attention, including grouped query heads and unequal token counts. */
  def scaledDotProductAttention(query: CudaTensor,key: CudaTensor,value: CudaTensor,
      maxScoreElements: Long=16777216L): CudaTensor =
    validate(query);validate(key);validate(value)
    AttentionLayout.validate(query.shape,key.shape,value.shape,0,maxScoreElements)
    attentionFromStorage(query,key,value,0,key.shape.dims(2),key.shape.dims(2),maxScoreElements,causal=false)

  private def attentionFromStorage(query: CudaTensor,key: CudaTensor,value: CudaTensor,queryStart: Int,
      tokens: Long,storageTokens: Long,maxScores: Long,causal: Boolean=true): CudaTensor =
    val keyShape=Shape.from(key.shape.dims.updated(2,tokens))
    val valueShape=Shape.from(value.shape.dims.updated(2,tokens))
    val (output,scores)=AttentionLayout.validate(query.shape,keyShape,valueShape,queryStart,maxScores)
    if output.numel==0 then return allocate(output)
    require(scores*2<=Int.MaxValue,"attention Double workspace exceeds Int indexing")
    val q=query.shape.dims;val k=keyShape.dims;val v=valueShape.dims
    val rows=output.numel/v(3)
    Using.resource(allocate(Shape.from(Seq(scores*2)))) { scratch =>
      execute(output,rows,kernels("attention")) { (arena,result) =>
        Seq(query.address,key.address,value.address,result.address,scratch.address).map(x => arena.allocateFrom(JAVA_LONG,x)) ++
          Seq(rows,q(1),k(1),q(2),k(2),q(3),v(3),queryStart.toLong,storageTokens).map(x => arena.allocateFrom(JAVA_INT,x.toInt)) :+
          arena.allocateFrom(JAVA_DOUBLE,1.0/math.sqrt(q(3).toDouble)) :+
          arena.allocateFrom(JAVA_INT,if causal then 1 else 0)
      }
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
    val result=allocate(output)
    try
      if output.numel>0 then
        val suffix=Shape.from(output.dims.drop(dim+1)).numel
        val rowWidth=output.dims(dim)*suffix
        var offset=0L
        values.foreach { value =>
          val chunk=value.shape.dims(dim)*suffix
          if value.shape.numel>0 then
            launch(value.shape.numel,kernels("concat_copy")) { arena =>
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
    if output.numel==0 then return allocate(output)
    // Nonempty allocated tensors have Int-bounded dimensions and contiguous strides.
    val strides=dims.scanRight(1L)(_ * _).tail
    val metadata=(output.dims ++ swap(strides)).map(_.toInt).toArray
    withInts(metadata) { address =>
      execute(output,output.numel,kernels("transpose")) { (arena,result) =>
        Seq(value.address,result.address,address).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
          Seq(output.numel.toInt,dims.size).map(v => arena.allocateFrom(JAVA_INT,v))
      }
    }

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

  def upload(source: Tensor[Float]): CudaTensor =
    val result=allocate(source.shape)
    try
      if source.shape.numel > 0 then Using.resource(Arena.ofConfined()) { arena =>
        val host=arena.allocateFrom(JAVA_FLOAT,source.toFloatArray*)
        driver.copyHostToDevice(result.address,host,source.shape.numel*4)
      }
      result
    catch case error: Throwable =>
      try result.close() catch case cleanup: Throwable => error.addSuppressed(cleanup)
      throw error

  def download(tensor: CudaTensor): Tensor[Float] =
    validate(tensor)
    val values=Using.resource(Arena.ofConfined()) { arena =>
      val host=arena.allocate(tensor.shape.numel*4,4)
      if tensor.shape.numel > 0 then
        driver.call("cuMemcpyDtoH_v2",Seq(ADDRESS,JAVA_LONG,JAVA_LONG),host,Long.box(tensor.address),Long.box(tensor.shape.numel*4))
      host.toArray(JAVA_FLOAT)
    }
    Tensor.fromFloats(values,tensor.shape)

  def matmul(a: CudaTensor,b: CudaTensor): CudaTensor =
    validate(a);validate(b)
    require(a.shape.rank==2 && b.shape.rank==2 && a.shape.dims(1)==b.shape.dims(0),"CUDA GEMM shape mismatch")
    val m=a.shape.dims(0);val k=a.shape.dims(1);val n=b.shape.dims(1)
    require(Seq(m,n,k).forall(_ <= Int.MaxValue),"CUDA GEMM dimension exceeds Int indexing")
    val output=Shape.from(Seq(m,n))
    execute(output,output.numel,function) { (arena,result) =>
      Seq(a.address,b.address,result.address).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
        Seq(m,n,k).map(v => arena.allocateFrom(JAVA_INT,v.toInt))
    }

  private def binary(a: CudaTensor,b: CudaTensor,op: Int): CudaTensor =
    validate(a);validate(b)
    require(a.shape==b.shape,"backend arithmetic requires equal shapes")
    execute(a.shape,a.shape.numel,kernels("binary")) { (arena,result) =>
      Seq(a.address,b.address,result.address).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
        Seq(a.shape.numel.toInt,op).map(v => arena.allocateFrom(JAVA_INT,v))
    }
  def add(a: CudaTensor,b: CudaTensor): CudaTensor = binary(a,b,0)
  /** NumPy-style trailing-axis broadcasting, without materializing expanded operands. */
  def addBroadcast(a: CudaTensor,b: CudaTensor): CudaTensor =
    validate(a);validate(b)
    if a.shape==b.shape then return add(a,b)
    val rank=math.max(a.shape.rank,b.shape.rank)
    def padded(s: Shape)=Vector.fill(rank-s.rank)(1L)++s.dims
    val ad=padded(a.shape);val bd=padded(b.shape)
    val output=Shape.from(ad.zip(bd).map { (x,y) =>
      require(x==y || x==1 || y==1,"incompatible broadcast dimensions")
      if x==1 then y else x
    })
    require(output.numel<=Int.MaxValue,"broadcast output exceeds Int indexing")
    if output.numel==0 then return allocate(output)
    def strides(ds: Vector[Long])=ds.scanRight(1L)(_ * _).tail.zip(ds).map((s,d) => if d==1 then 0L else s)
    withInts((output.dims++strides(ad)++strides(bd)).map(_.toInt).toArray) { metadata =>
      execute(output,output.numel,kernels("broadcast_add")) { (arena,result) =>
        Seq(a.address,b.address,result.address,metadata).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
          Seq(output.numel.toInt,rank).map(v => arena.allocateFrom(JAVA_INT,v))
      }
    }
  /** Final-axis split: value half times Gaussian-error GELU of the gate half. */
  def geglu(input: CudaTensor): CudaTensor =
    validate(input)
    require(input.shape.rank>0 && input.shape.dims.last>0 && input.shape.dims.last%2==0 &&
      input.shape.dims.last<=Int.MaxValue,"GEGLU requires a positive even final width")
    val width=input.shape.dims.last/2
    val output=Shape.from(input.shape.dims.updated(input.shape.rank-1,width))
    execute(output,output.numel,kernels("geglu")) { (arena,result) =>
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
    execute(value.shape,value.shape.numel,kernels("scale")) { (arena,result) =>
      Seq(value.address,result.address).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
        Seq(arena.allocateFrom(JAVA_INT,value.shape.numel.toInt),arena.allocateFrom(JAVA_FLOAT,scalar))
    }
  /** NCHW/OIHW direct grouped cross-correlation. Inputs and output remain resident. */
  def conv2d(input: CudaTensor,weight: CudaTensor,config: Conv2dConfig=Conv2dConfig()): CudaTensor =
    validate(input);validate(weight)
    val g=Conv2dOps.geometry(input.shape,weight.shape,config)
    val dimensions=Array(g.ci,g.h,g.w,g.co,g.cg,g.kh,g.kw,g.oh,g.ow,
      config.stride._1,config.stride._2,config.padding._1,config.padding._2,
      config.dilation._1,config.dilation._2,config.groups)
    withInts(dimensions) { geometry =>
      execute(g.shape,g.shape.numel,kernels("conv2d")) { (arena,result) =>
        Seq(input.address,weight.address,result.address,geometry).map(v => arena.allocateFrom(JAVA_LONG,v)) :+
          arena.allocateFrom(JAVA_INT,g.shape.numel.toInt)
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
    execute(input.shape,rows,kernels("groupnorm")) { (arena,result) =>
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
    execute(value.shape,value.shape.numel,kernels("silu")) { (arena,result) =>
      Seq(value.address,result.address).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
        Seq(arena.allocateFrom(JAVA_INT,value.shape.numel.toInt),arena.allocateFrom(JAVA_LONG,0L),arena.allocateFrom(JAVA_INT,0))
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

  private def execute(output: Shape,count: Long,kernel: MemorySegment)
      (arguments: (Arena,CudaTensor) => Seq[MemorySegment]): CudaTensor =
    val result=allocate(output)
    try
      launch(count,kernel)(arena => arguments(arena,result))
      result
    catch case error: Throwable =>
      try result.close() catch case cleanup: Throwable => error.addSuppressed(cleanup)
      throw error

  private def launch(count: Long,kernel: MemorySegment)(arguments: Arena => Seq[MemorySegment]): Unit =
      if count > 0 then Using.resource(Arena.ofConfined()) { arena =>
        val args=arguments(arena)
        val params=arena.allocate(ADDRESS,args.size.toLong)
        args.zipWithIndex.foreach((arg,i) => params.setAtIndex(ADDRESS,i.toLong,arg))
        driver.call("cuLaunchKernel",Seq(ADDRESS) ++ Seq.fill(7)(JAVA_INT) ++ Seq.fill(3)(ADDRESS),
          kernel,Int.box(((count+63)/64).toInt),Int.box(1),Int.box(1),
          Int.box(64),Int.box(1),Int.box(1),Int.box(0),stream,params,MemorySegment.NULL)
        driver.call("cuStreamSynchronize",Seq(ADDRESS),stream)
      }

  def release(tensor: CudaTensor): Unit =
    threadCheck()
    require(tensor.session eq this,"tensor belongs to another CUDA session")
    if !tensor.released then
      validate(tensor)
      if tensor.allocation.references==1 then
        driver.call("cuMemFree_v2",Seq(JAVA_LONG),Long.box(tensor.address))
        used-=tensor.allocation.bytes
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
    attempt { driver.call("cuStreamSynchronize",Seq(ADDRESS),stream) }
    live.toVector.reverse.foreach(tensor => attempt { release(tensor) })
    live.foreach(_.released=true)
    live.clear();used=0;closed=true
    if failure != null then throw failure

object CudaSession:
  def withDevice[A](deviceIndex: Int = 0, maxBytes: Long = 1073741824L)(body: CudaSession => A): A =
    require(maxBytes >= 0,"negative CUDA allocation budget")
    CudaDriver.withDevice(deviceIndex) { (driver,version,name) =>
      body(new CudaSession(driver,deviceIndex,name,version,maxBytes))
    }

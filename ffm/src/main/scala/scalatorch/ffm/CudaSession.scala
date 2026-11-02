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

/** Thread-confined device storage and synchronous packed GEMM execution. */
final class CudaSession private[ffm] (driver: CudaDriver, val deviceIndex: Int,
    val deviceName: String, val driverVersion: Int, val maxBytes: Long) extends TrainingBackend[CudaTensor]:
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
  private val kernels=Seq("binary","scale","fill","sum","expand_scalar","silu","rmsnorm","rms_input_backward","rms_inverse","rms_weight_backward","transpose","embedding","rope","attention","cache_append").map { name =>
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

  def rope(value: CudaTensor,positionStart: Int = 0,theta: Double = 10000.0): CudaTensor =
    validate(value)
    val count=RotaryCoefficients.count(value.shape,positionStart,theta)
    if count==0 then return reshape(value,value.shape)
    require(count*2<=Int.MaxValue && count*8+value.shape.numel*4<=maxBytes-used,"rotary workspace budget exceeded")
    val coefficients=RotaryCoefficients.values(value.shape,positionStart,theta)
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

  private def attentionFromStorage(query: CudaTensor,key: CudaTensor,value: CudaTensor,queryStart: Int,
      tokens: Long,storageTokens: Long,maxScores: Long): CudaTensor =
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
          arena.allocateFrom(JAVA_DOUBLE,1.0/math.sqrt(q(3).toDouble))
      }
    }

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

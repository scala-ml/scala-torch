package scalatorch.ffm

import scalatorch.*
import java.lang.foreign.{Arena, MemorySegment}
import java.lang.foreign.ValueLayout.{ADDRESS, JAVA_FLOAT, JAVA_INT, JAVA_LONG}
import java.nio.charset.StandardCharsets.UTF_8
import scala.collection.mutable
import scala.util.Using

/** Owned packed Float32 device allocation; valid only inside its session scope. */
final class CudaTensor private[ffm] (private[ffm] val session: CudaSession,
    val shape: Shape, private[ffm] val address: Long, private[ffm] val bytes: Long) extends AutoCloseable:
  private[ffm] var released=false
  def toCpu: Tensor[Float] = session.download(this)
  def matmul(other: CudaTensor): CudaTensor = session.matmul(this,other)
  def close(): Unit = session.release(this)

/** Thread-confined device storage and synchronous packed GEMM execution. */
final class CudaSession private[ffm] (driver: CudaDriver, val deviceIndex: Int,
    val deviceName: String, val driverVersion: Int, val maxBytes: Long):
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
    val tensor=new CudaTensor(this,shape,address,bytes)
    live+=tensor;used+=bytes
    tensor

  def upload(source: Tensor[Float]): CudaTensor =
    val result=allocate(source.shape)
    try
      if source.shape.numel > 0 then Using.resource(Arena.ofConfined()) { arena =>
        val host=arena.allocateFrom(JAVA_FLOAT,source.toFloatArray*)
        driver.call("cuMemcpyHtoD_v2",Seq(JAVA_LONG,ADDRESS,JAVA_LONG),Long.box(result.address),host,Long.box(source.shape.numel*4))
      }
      result
    catch case error: Throwable =>
      try result.close() catch case cleanup: Throwable => error.addSuppressed(cleanup)
      throw error

  private[ffm] def download(tensor: CudaTensor): Tensor[Float] =
    validate(tensor)
    val values=Using.resource(Arena.ofConfined()) { arena =>
      val host=arena.allocate(tensor.shape.numel*4,4)
      if tensor.shape.numel > 0 then
        driver.call("cuMemcpyDtoH_v2",Seq(ADDRESS,JAVA_LONG,JAVA_LONG),host,Long.box(tensor.address),Long.box(tensor.shape.numel*4))
      host.toArray(JAVA_FLOAT)
    }
    Tensor.fromFloats(values,tensor.shape)

  private[ffm] def matmul(a: CudaTensor,b: CudaTensor): CudaTensor =
    validate(a);validate(b)
    require(a.shape.rank==2 && b.shape.rank==2 && a.shape.dims(1)==b.shape.dims(0),"CUDA GEMM shape mismatch")
    val m=a.shape.dims(0);val k=a.shape.dims(1);val n=b.shape.dims(1)
    require(Seq(m,n,k).forall(_ <= Int.MaxValue),"CUDA GEMM dimension exceeds Int indexing")
    val result=allocate(Shape.from(Seq(m,n)))
    try
      if result.shape.numel > 0 then Using.resource(Arena.ofConfined()) { arena =>
        val args=Seq(a.address,b.address,result.address).map(v => arena.allocateFrom(JAVA_LONG,v)) ++
          Seq(m,n,k).map(v => arena.allocateFrom(JAVA_INT,v.toInt))
        val params=arena.allocate(ADDRESS,6)
        args.zipWithIndex.foreach((arg,i) => params.setAtIndex(ADDRESS,i.toLong,arg))
        driver.call("cuLaunchKernel",Seq(ADDRESS) ++ Seq.fill(7)(JAVA_INT) ++ Seq.fill(3)(ADDRESS),
          function,Int.box(((result.shape.numel+63)/64).toInt),Int.box(1),Int.box(1),
          Int.box(64),Int.box(1),Int.box(1),Int.box(0),stream,params,MemorySegment.NULL)
        driver.call("cuStreamSynchronize",Seq(ADDRESS),stream)
      }
      result
    catch case error: Throwable =>
      try result.close() catch case cleanup: Throwable => error.addSuppressed(cleanup)
      throw error

  private[ffm] def release(tensor: CudaTensor): Unit =
    threadCheck()
    if !tensor.released then
      validate(tensor)
      driver.call("cuMemFree_v2",Seq(JAVA_LONG),Long.box(tensor.address))
      tensor.released=true;live-=tensor;used-=tensor.bytes

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

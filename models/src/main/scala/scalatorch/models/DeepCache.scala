package scalatorch.models

import scalatorch.ffm.{CudaSession,CudaTensor}

/** DeepCache (Ma et al., CVPR 2024) + TaylorSeer finite-difference forecasting (Liu et al., ICCV 2025, arXiv:2503.06923):
  * between full steps only the shallow blocks run and the deep features are either reused directly (`order = 0`)
  * or extrapolated via divided-difference Taylor polynomials of order 1 or 2 (`order = 1, 2`).
  */
final case class DeepCacheConfig(interval: Int,branch: Int,fullFirst: Int,order: Int=0,contraction: Double=1.0):
  require(interval>=2 && interval<=10,"DeepCache interval must be 2..10")
  require(branch>=0 && branch<=3,"DeepCache branch must be 0..3")
  require(fullFirst>=0 && fullFirst<=50,"DeepCache full_first must be 0..50")
  require(order>=0 && order<=2,"DeepCache order must be 0..2")
  require(contraction.isFinite && contraction>0.0 && contraction<=1.0,"DeepCache contraction must be in (0, 1]")

final class DeepCacheState(val config: DeepCacheConfig) extends AutoCloseable:
  private final case class SlotEntry(dims: Seq[Long],step0: Int,step1: Int,f0: CudaTensor,
      d1: Option[CudaTensor]=None,d2: Option[CudaTensor]=None):
    def close(): Unit =
      if f0!=null then try f0.close() catch case _: Throwable => ()
      d1.foreach(t => if t!=null then try t.close() catch case _: Throwable => ())
      d2.foreach(t => if t!=null then try t.close() catch case _: Throwable => ())

  private var lastTimestep=Float.NaN
  private var step= -1
  private var slot=0
  private var spatialWarmupUntil=0
  private var anchorOverride=0
  private val entries=scala.collection.mutable.HashMap.empty[Int,SlotEntry]
  var fullCalls=0
  var cachedCalls=0

  def deferUntil(minAnchorStep: Int): Unit = anchorOverride=math.max(anchorOverride,minAnchorStep)

  private def compatibleDims(entryDims: Seq[Long],inputDims: Seq[Long]): Boolean =
    entryDims==inputDims || (entryDims.size==inputDims.size && entryDims.tail==inputDims.tail &&
      entryDims.headOption.zip(inputDims.headOption).exists((eh,ih) => eh==2L*ih))

  /** True when `stepIndex` is a scheduled full-U-Net step (used to keep CFG-Cache phase-locked). */
  def isScheduledFullStep(stepIndex: Int): Boolean =
    val anchor=math.max(config.fullFirst,anchorOverride)
    stepIndex<anchor || (stepIndex-anchor)%config.interval==0 || stepIndex<spatialWarmupUntil

  /** Registers one U-Net call; returns (slot, reuse). */
  def beginCall(timestep: Float,inputDims: Seq[Long]): (Int,Boolean) =
    if !lastTimestep.isNaN && timestep>lastTimestep then
      entries.values.foreach(_.close())
      entries.clear()
      step= -1
      spatialWarmupUntil=0
    if timestep!=lastTimestep then { step+=1;slot=0;lastTimestep=timestep } else slot+=1
    if entries.get(slot).exists(e => e.dims.drop(1)!=inputDims.drop(1)) then
      entries.values.foreach(_.close())
      entries.clear()
      spatialWarmupUntil=math.max(spatialWarmupUntil,step+2)
    val full=isScheduledFullStep(step) || !entries.get(slot).exists(e => compatibleDims(e.dims,inputDims))
    if full then fullCalls+=1 else cachedCalls+=1
    (slot,!full)

  def record(slotIndex: Int,inputDims: Seq[Long],feature: CudaTensor): Unit =
    entries.put(slotIndex,SlotEntry(inputDims,step,step,feature)).foreach(_.close())

  def recordWithSession(slotIndex: Int,inputDims: Seq[Long],feature: CudaTensor,session: CudaSession): Unit =
    val minSafeStep=math.max(0,spatialWarmupUntil-2)
    val prev=entries.get(slotIndex).filter(e => compatibleDims(e.dims,inputDims) && e.step0>=minSafeStep)
    val f0Keep=session.reshape(feature,feature.shape)
    val (prevStep0,d1,d2)=
      if config.order<=0 || !feature.half then (step,None,None)
      else prev match
        case Some(p) if step>p.step0 =>
          val dt1=(step-p.step0).toFloat
          scala.util.Using.Manager { use =>
            val sliceHalf=p.dims.head==2L*inputDims.head
            val prevF0=if !sliceHalf then p.f0 else
              val b=p.f0.shape.dims.head/2L
              use(session.sliceBatch(p.f0,b,b))
            val nd1=session.taylorDiff(feature,prevF0,1.0f/dt1)
            val nd2=if config.order>=2 then
              p.d1.filter(_ => step>p.step1 && p.step0>p.step1).map { pd1 =>
                val prevD1=if !sliceHalf then pd1 else
                  val b=pd1.shape.dims.head/2L
                  use(session.sliceBatch(pd1,b,b))
                session.taylorDiff(nd1,prevD1,1.0f/(step-p.step1).toFloat)
              }
            else None
            (p.step0,Some(nd1),nd2)
          }.get
        case _ => (step,None,None)
    entries.put(slotIndex,SlotEntry(inputDims,step,prevStep0,f0Keep,d1,d2)).foreach(_.close())

  def feature(slotIndex: Int): CudaTensor = entries(slotIndex).f0

  def forecast(slotIndex: Int,session: CudaSession): CudaTensor =
    forecast(slotIndex,entries(slotIndex).dims.head,session)

  def forecast(slotIndex: Int,targetBatch: Long,session: CudaSession): CudaTensor =
    val e=entries(slotIndex)
    val tau=(step-e.step0).toFloat
    val sigma=config.contraction.toFloat
    val c1=sigma*tau
    val slicePos=e.dims.head==2L*targetBatch
    def posSlice(t: CudaTensor): CudaTensor =
      val b=t.shape.dims.head/2L
      session.sliceBatch(t,b,b)
    if config.order==0 || e.d1.isEmpty || !e.f0.half || tau<=0f then
      if slicePos then posSlice(e.f0) else session.reshape(e.f0,e.f0.shape)
    else if config.order==1 || e.d2.isEmpty then
      scala.util.Using.Manager { use =>
        val f0=if slicePos then use(posSlice(e.f0)) else e.f0
        val d1=if slicePos then use(posSlice(e.d1.get)) else e.d1.get
        session.taylorForecast1(f0,d1,c1)
      }.get
    else
      // Exact Newton divided-difference coefficient when contraction == 1.0; HiCache Hermite damping when < 1.0.
      val c2=(if config.contraction<1.0 then 0.5f*sigma*sigma else 1.0f)*tau*(step-e.step1).toFloat
      scala.util.Using.Manager { use =>
        val f0=if slicePos then use(posSlice(e.f0)) else e.f0
        val d1=if slicePos then use(posSlice(e.d1.get)) else e.d1.get
        val d2=if slicePos then use(posSlice(e.d2.get)) else e.d2.get
        session.taylorForecast2(f0,d1,d2,c1,c2)
      }.get

  def close(): Unit =
    entries.values.foreach(_.close())
    entries.clear()

object DeepCache:
  private val active=new ThreadLocal[Option[DeepCacheState]]:
    override def initialValue(): Option[DeepCacheState]=None

  def current: Option[DeepCacheState]=active.get()

  /** Installs (or clears) the state for generations started on this thread; the previous one is closed. */
  def install(config: Option[DeepCacheConfig]): Unit =
    active.get().foreach(s => try s.close() catch case _: Throwable => ())
    active.set(config.map(new DeepCacheState(_)))

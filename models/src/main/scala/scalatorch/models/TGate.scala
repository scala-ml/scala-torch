package scalatorch.models

import scalatorch.Shape
import scalatorch.ffm.CudaTensor

/** T-GATE temporal cross-attention caching (Zhang et al., TMLR 2024/2025, arXiv:2404.02747).
  * During early semantics-planning steps (`step < gateStep`), cross-attention layers evaluate normally and retain
  * their recent CFG-averaged (or unguided) output per layer and pass slot. Once `step >= gateStep`, cross-attention
  * projections and layer norms are skipped and the cached representation is broadcast into the residual stream.
  */
final case class TGateConfig(gateStep: Int=8):
  require(gateStep>=1 && gateStep<=1000,"T-GATE gate_step must be 1..1000")

final class TGateState(val config: TGateConfig) extends AutoCloseable:
  private val perSlot=scala.collection.mutable.HashMap.empty[Int,java.util.IdentityHashMap[DiffusionAttention,CudaTensor]]
  private var lastTimestep=Float.NaN
  private var stepIndex=0
  private var slot=0
  private var currentSlot: Option[Int]=None
  private var cfgPair=false
  private var spatialWarmupUntil=0
  var cachedHits=0
  var recordedEntries=0

  def currentStep: Int = stepIndex
  def isCfgPair: Boolean = cfgPair
  def activeSlot: Option[Int] = currentSlot
  def deferUntil(minStep: Int): Unit = spatialWarmupUntil=math.max(spatialWarmupUntil,minStep)

  /** Sets the 0-based diffusion step index (`0..steps-1`) and clears slot counter for the step. */
  def withStep[A](step: Int)(body: => A): A =
    val prevStep=stepIndex
    val prevSlot=slot
    val prevCurrent=currentSlot
    val prevT=lastTimestep
    stepIndex=step
    lastTimestep=Float.NaN
    slot=0
    currentSlot=None
    try body
    finally
      stepIndex=prevStep
      slot=prevSlot
      currentSlot=prevCurrent
      lastTimestep=prevT

  def withCfgPair[A](pair: Boolean)(body: => A): A =
    val prev=cfgPair
    cfgPair=pair
    try body
    finally cfgPair=prev

  def withCall[A](timestep: Float)(body: => A): A =
    val prev=currentSlot
    val s=beginCall(timestep)
    currentSlot=Some(s)
    try body
    finally currentSlot=prev

  /** Begins a U-Net forward call at `timestep`, advancing the pass slot within the same timestep. */
  def beginCall(timestep: Float): Int =
    if !lastTimestep.isNaN && timestep>lastTimestep then
      clearCache()
    if timestep!=lastTimestep then { slot=0;lastTimestep=timestep } else slot+=1
    slot

  private def mapFor(slotIndex: Int): java.util.IdentityHashMap[DiffusionAttention,CudaTensor] =
    perSlot.getOrElseUpdate(slotIndex,new java.util.IdentityHashMap[DiffusionAttention,CudaTensor]())

  private def compatibleShape(cached: Shape,target: Shape): Boolean =
    cached.rank==target.rank && cached.dims.drop(1)==target.dims.drop(1) &&
      (target.dims(0)==cached.dims(0) || target.dims(0)==2L*cached.dims(0))

  private def effectiveGateStep: Int = math.max(config.gateStep,spatialWarmupUntil)

  /** True when cached cross-attention should replace computation on this step if an entry is available. */
  def shouldReuse(slotIndex: Int,layer: DiffusionAttention): Option[CudaTensor] =
    if stepIndex<effectiveGateStep then None
    else
      val m=perSlot.getOrElse(slotIndex,null)
      if m==null then None
      else Option(m.get(layer)).filter(t => !t.isReleased).map { t => cachedHits+=1; t }

  def shouldReuse(slotIndex: Int,layer: DiffusionAttention,targetShape: Shape): Option[CudaTensor] =
    val m=perSlot.getOrElse(slotIndex,null)
    val raw=if m==null then null else m.get(layer)
    if raw!=null && !raw.isReleased && raw.shape.dims.drop(1)!=targetShape.dims.drop(1) then
      spatialWarmupUntil=math.max(spatialWarmupUntil,stepIndex+2)
      None
    else if stepIndex<effectiveGateStep || raw==null || raw.isReleased || !compatibleShape(raw.shape,targetShape) then None
    else { cachedHits+=1; Some(raw) }

  /** True when the current step should capture (or refresh) the cross-attention cache for `layer`. */
  def shouldRecord(slotIndex: Int,layer: DiffusionAttention,dcInterval: Option[Int],targetShape: Option[Shape]=None): Boolean =
    val m=perSlot.getOrElse(slotIndex,null)
    val existing=if m==null then null else m.get(layer)
    if existing!=null && !existing.isReleased && targetShape.exists(s => existing.shape.dims.drop(1)!=s.dims.drop(1)) then
      spatialWarmupUntil=math.max(spatialWarmupUntil,stepIndex+2)
    val missing=existing==null || existing.isReleased || targetShape.exists(s => !compatibleShape(existing.shape,s))
    val gate=effectiveGateStep
    if stepIndex>=gate then missing
    else
      val window=dcInterval.getOrElse(1)
      stepIndex==gate-1 || (window>1 && stepIndex>=gate-window)

  def record(slotIndex: Int,layer: DiffusionAttention,value: CudaTensor): Unit =
    val m=mapFor(slotIndex)
    val prev=m.put(layer,value)
    if prev!=null && (prev ne value) then prev.close()
    recordedEntries+=1

  private def clearCache(): Unit =
    perSlot.values.foreach { m =>
      val it=m.values().iterator()
      while it.hasNext do
        val t=it.next()
        if t!=null && !t.isReleased then try t.close() catch case _: Throwable => ()
    }
    perSlot.clear()
    spatialWarmupUntil=0

  def close(): Unit = clearCache()

object TGate:
  private val active=new ThreadLocal[Option[TGateState]]:
    override def initialValue(): Option[TGateState]=None

  def current: Option[TGateState]=active.get()

  def withConfig[A](config: Option[TGateConfig])(body: => A): A =
    val prev=active.get()
    val state=config.map(new TGateState(_))
    active.set(state)
    try body
    finally
      state.foreach(s => try s.close() catch case _: Throwable => ())
      active.set(prev)

  def install(config: Option[TGateConfig]): Unit =
    active.get().foreach(s => try s.close() catch case _: Throwable => ())
    active.set(config.map(new TGateState(_)))

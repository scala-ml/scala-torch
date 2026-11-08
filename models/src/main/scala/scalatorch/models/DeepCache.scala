package scalatorch.models

import scalatorch.ffm.CudaTensor

/** DeepCache for the CUDA U-Net (Ma et al., CVPR 2024): between full steps only the shallow blocks run and the deep
  * features recorded by the last full step are reused. Written from the paper's method; semantics follow NUI's hook
  * `deepcache {interval, branch, full_first}`.
  *
  * A "step" is detected from the timestep value, so one step may contain several U-Net calls (conditional, unconditional);
  * every call slot keeps its own recorded feature.
  */
final case class DeepCacheConfig(interval: Int,branch: Int,fullFirst: Int):
  require(interval>=2 && interval<=10,"DeepCache interval must be 2..10")
  require(branch>=0 && branch<=3,"DeepCache branch must be 0..3")
  require(fullFirst>=0 && fullFirst<=50,"DeepCache full_first must be 0..50")

final class DeepCacheState(val config: DeepCacheConfig) extends AutoCloseable:
  private var lastTimestep=Float.NaN
  private var step= -1
  private var slot=0
  private val features=scala.collection.mutable.HashMap.empty[Int,(Seq[Long],CudaTensor)]
  var fullCalls=0
  var cachedCalls=0

  /** Registers one U-Net call; returns (slot, reuse). */
  def beginCall(timestep: Float,inputDims: Seq[Long]): (Int,Boolean) =
    if timestep!=lastTimestep then { step+=1;slot=0;lastTimestep=timestep } else slot+=1
    val full=step<config.fullFirst || (step-config.fullFirst)%config.interval==0 ||
      !features.get(slot).exists(_._1==inputDims)
    if full then fullCalls+=1 else cachedCalls+=1
    (slot,!full)

  def record(slotIndex: Int,inputDims: Seq[Long],feature: CudaTensor): Unit =
    features.put(slotIndex,(inputDims,feature)).foreach(old => if old._2!=null then old._2.close())

  def feature(slotIndex: Int): CudaTensor = features(slotIndex)._2

  def close(): Unit =
    features.values.foreach(v => if v._2!=null then v._2.close())
    features.clear()

object DeepCache:
  private val active=new ThreadLocal[Option[DeepCacheState]]:
    override def initialValue(): Option[DeepCacheState]=None

  def current: Option[DeepCacheState]=active.get()

  /** Installs (or clears) the state for generations started on this thread; the previous one is closed. */
  def install(config: Option[DeepCacheConfig]): Unit =
    active.get().foreach(s => try s.close() catch case _: Throwable => ())
    active.set(config.map(new DeepCacheState(_)))

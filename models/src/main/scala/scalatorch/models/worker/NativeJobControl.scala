package scalatorch.models.worker

final case class NativeProtection(enabled: Boolean,revision: Long)
final case class NativeInterrupt(accepted: Boolean,blocked: Boolean,protection: NativeProtection,
    affected: Boolean=false,tooLate: Boolean=false,activeId: Option[String]=None)

/** Atomic admission state shared by a reader thread and a single generation worker.
  * This owns cancellation/publication decisions, not model work or protocol output.
  */
final class NativeJobControl(maxQueued: Int=128):
  require(maxQueued>0,"queue capacity must be positive")
  final class Job private[NativeJobControl] (val id: String,val owner: Option[String]):
    @volatile private[NativeJobControl] var stopped=false
    private[NativeJobControl] var publishing=false
    def cancelled: Boolean=stopped

  private val queued=scala.collection.mutable.LinkedHashMap.empty[String,Job]
  private var active: Option[Job]=None
  private var protection=NativeProtection(false,0)
  private var closed=false
  def protectionState: NativeProtection=synchronized(protection)
  def setProtection(enabled: Boolean): NativeProtection=synchronized {
    if protection.enabled!=enabled then
      protection=NativeProtection(enabled,Math.addExact(protection.revision,1L))
    protection
  }
  def admit(id: String,owner: Option[String]): Job=synchronized {
    require(id.nonEmpty && id.length<=256,"invalid request id")
    require(owner.forall(v => v.nonEmpty && v.length<=256),"invalid owner id")
    if closed then throw IllegalStateException("worker is shutting down")
    require(!queued.contains(id) && !active.exists(_.id==id),"duplicate request id")
    require(queued.size<maxQueued,"worker queue is full")
    val job=new Job(id,owner)
    queued(id)=job
    job
  }
  /** Cancelled queued jobs are returned too, so the worker can emit their terminal error. */
  def startNext(): Option[Job]=synchronized {
    if active.nonEmpty then throw IllegalStateException("a job is already active")
    active=queued.headOption.map { (id,job) => queued.remove(id);job }
    active
  }
  def finish(job: Job): Unit=synchronized {
    require(active.contains(job),"job is not active")
    active=None
  }
  /** Must precede committing outputs; cancellation and publication cannot both win. */
  def beginPublication(job: Job): Boolean=synchronized {
    require(active.contains(job) && !job.publishing,"job cannot begin publication")
    if job.cancelled || closed then false
    else { job.publishing=true;true }
  }
  def restartAuthorization: NativeInterrupt=synchronized {
    NativeInterrupt(!protection.enabled,protection.enabled,protection)
  }
  def interrupt(target: Option[String],owner: Option[String]): NativeInterrupt=synchronized {
    require(owner.forall(v => v.nonEmpty && v.length<=256),"invalid owner id")
    val decision=restartAuthorization
    def matches(job: Job)=target.forall(_==job.id) && owner.forall(v => job.owner.contains(v))
    if decision.blocked then decision
    else active.filter(matches) match
      case Some(job) if job.publishing => decision.copy(accepted=false,tooLate=true)
      case Some(job) =>
        job.stopped=true
        decision.copy(affected=true,activeId=Some(job.id))
      case None =>
        queued.valuesIterator.find(matches) match
          case Some(job) => job.stopped=true;decision.copy(affected=true)
          case None => decision
  }
  /** EOF/shutdown cancels running work and returns the discarded pending jobs. */
  def shutdown(): Vector[Job]=synchronized {
    closed=true
    active.foreach(_.stopped=true)
    val dropped=queued.values.toVector
    dropped.foreach(_.stopped=true)
    queued.clear()
    dropped
  }

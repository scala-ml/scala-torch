package scalatorch.models.worker

import java.io.{OutputStream,Reader}
import java.util.concurrent.CancellationException
import scala.util.control.NonFatal

trait NativeWorkerBackend:
  def supported: Set[String]
  def features: Vector[String]=Vector.empty
  def device: String="cpu"
  def usesCuda: Boolean=false
  /** Called on the reader thread; implementations must only read published snapshots. */
  def systemStatus: Map[String,NativeJson]=NativeSystemStatus.snapshot()
  def execute(request: NativeRequest,context: NativeWorkContext): Map[String,NativeJson]

final class NativeWorkContext private[worker] (val cancelled: () => Boolean,
    publish: () => Boolean,val event: (String,Map[String,NativeJson]) => Unit):
  def checkCancelled(): Unit=
    if cancelled() then throw new CancellationException("request interrupted")
  def beginPublication(): Unit=
    if !publish() then throw new CancellationException("request interrupted before publication")

/** Reader-side controls remain responsive while one backend operation executes. */
final class NativeWorker(out: OutputStream,backend: NativeWorkerBackend) extends AutoCloseable:
  import NativeJson.*
  private val emitter=new NativeEmitter(out)
  private val control=new NativeJobControl
  private val gate=new Object
  private val pending=scala.collection.mutable.Map.empty[String,NativeRequest]
  @volatile private var closed=false
  private def protection(p: NativeProtection)=Map("enabled"->Bool(p.enabled),"revision"->Number(BigDecimal(p.revision)))
  private def decision(d: NativeInterrupt)=Map("accepted"->Bool(d.accepted),"blocked"->Bool(d.blocked),
    "affected"->Bool(d.affected),"too_late"->Bool(d.tooLate),"protection"->Obj(protection(d.protection)),
    "active_id"->d.activeId.map(Str.apply).getOrElse(Null))
  private val hello=Map("protocol"->Number(BigDecimal(1)),"engine_version"->Str("scala-torch-0.1.0"),
    "python"->Str(""),"cuda"->Bool(backend.usesCuda),"device"->Str(backend.device),"dry_run"->Bool(false),
    "features"->Arr(backend.features.map(Str.apply)))
  emitter.emit(hello++Map("id"->Null,"type"->Str("hello")))
  private val worker=new Thread(() => work(),"scala-torch-native-worker")
  worker.setDaemon(true)
  worker.start()

  private def work(): Unit=
    var running=true
    while running do
      val next=gate.synchronized {
        while pending.isEmpty && !closed do gate.wait()
        if closed then None
        else control.startNext().map(job => (job,pending.remove(job.id).get))
      }
      next match
        case None => running=false
        case Some((job,request)) =>
          var publishing=false
          val context=new NativeWorkContext(() => job.cancelled,() => {
            if publishing then true
            else { publishing=control.beginPublication(job);publishing }
          },(name,fields) => emitter.event(job.id,name,fields))
          try
            context.checkCancelled()
            val result=backend.execute(request,context)
            context.beginPublication()
            emitter.result(job.id,result)
          catch
            case _: CancellationException => emitter.error(Some(job.id),"interrupted","request interrupted")
            case e: NativeBackendError => emitter.error(Some(job.id),e.code,e.getMessage)
            case e: IllegalArgumentException => emitter.error(Some(job.id),"invalid_params",Option(e.getMessage).getOrElse("invalid parameters"))
            case NonFatal(e) => emitter.error(Some(job.id),"internal",Option(e.getMessage).getOrElse(e.getClass.getSimpleName))
          finally control.finish(job)

  /** Returns false after shutdown. Malformed input emits an error and permits recovery. */
  def handle(line: String): Boolean=
    if closed then return false
    if line.trim.isEmpty then return true
    var id: Option[String]=None
    try
      val request=NativeRequest.parse(line)
      id=Some(request.id)
      def optional(key: String): Option[String]=request.params.get(key) match
        case None | Some(Null) => None
        case Some(Str(s)) => Some(s)
        case _ => throw IllegalArgumentException(s"invalid $key")
      request.kind match
        case "ping" => emitter.result(request.id,hello)
        case "system_status" =>
          require(request.params.isEmpty,"system_status takes no parameters")
          emitter.result(request.id,backend.systemStatus)
        case "interruption_protection" =>
          val state=request.params.get("enabled") match
            case None => control.protectionState
            case Some(Bool(value)) => control.setProtection(value)
            case _ => throw IllegalArgumentException("enabled must be boolean")
          emitter.result(request.id,protection(state))
        case "restart_authorization" => emitter.result(request.id,decision(control.restartAuthorization))
        case "interrupt" =>
          val owner=optional("owner_id")
          require(!request.params.contains("owner_id") || owner.nonEmpty,"owner_id must be a string")
          emitter.result(request.id,decision(control.interrupt(optional("target_id"),owner)))
        case "shutdown" =>
          stop()
          emitter.result(request.id,Map.empty)
        case kind if backend.supported.contains(kind) => gate.synchronized {
          control.admit(request.id,request.owner)
          pending(request.id)=request
          gate.notifyAll()
        }
        case _ => emitter.error(id,"unsupported",s"unsupported request: ${request.kind}")
    catch case e: IllegalArgumentException => emitter.error(id,"invalid_params",Option(e.getMessage).getOrElse("invalid request"))
    !closed

  /** Bounded character framing; consumes the remainder of oversized lines before recovery. */
  def serve(input: Reader): Unit=
    try
      val line=new java.lang.StringBuilder()
      var oversized=false
      var ch=input.read()
      while ch>=0 && !closed do
        if ch=='\n' then
          if oversized then emitter.error(None,"invalid_params","request exceeds size limit")
          else handle(line.toString)
          line.setLength(0);oversized=false
        else if line.length()<NativeJson.maxChars then line.append(ch.toChar)
        else oversized=true
        if !closed then ch=input.read()
      if !closed && (line.length()>0 || oversized) then
        if oversized then emitter.error(None,"invalid_params","request exceeds size limit")
        else handle(line.toString)
    finally close()
  private def stop(): Unit=gate.synchronized {
    if !closed then
      closed=true
      control.shutdown()
      pending.clear()
      gate.notifyAll()
  }
  def close(): Unit=
    stop()
    if Thread.currentThread()!=worker then worker.join(5000)

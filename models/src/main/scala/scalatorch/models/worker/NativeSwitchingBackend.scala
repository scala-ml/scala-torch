package scalatorch.models.worker

import java.nio.file.Path
import NativeJson.*

/** Requests run serially; status readers see only the last published backend. */
final class NativeSwitchingBackend(initial: Path,makeBackend: Path => NativeWorkerBackend) extends NativeWorkerBackend:
  private val initialPath=initial.toRealPath()
  @volatile private var current=makeBackend(initialPath)
  override def supported: Set[String]=current.supported
  override def features: Vector[String]=current.features
  override def device: String=current.device
  override def usesCuda: Boolean=current.usesCuda
  override def systemStatus: Map[String,NativeJson]=current.systemStatus

  def execute(request: NativeRequest,context: NativeWorkContext): Map[String,NativeJson] =
    if request.kind=="list_models" then
      require(request.params.keySet==Set("dirs"),"list_models requires dirs")
      val dirs=request.params("dirs") match
        case Arr(values) => values.map {
          case Str(path) => path
          case _ => throw IllegalArgumentException("dirs must contain strings")
        }
        case _ => throw IllegalArgumentException("dirs must be an array")
      Map("models"->Arr(NativeModelCatalog.scan(dirs,()=>context.checkCancelled())))
    else if request.kind!="load_model" then current.execute(request,context)
    else
      val path=request.params.get("path") match
        case None | Some(Null) => initialPath
        case Some(Str(value)) if value.nonEmpty => Path.of(value).toRealPath()
        case _ => throw IllegalArgumentException("path must be a model path")
      val candidate=makeBackend(path)
      val result=candidate.execute(request,context)
      context.beginPublication()
      current=candidate
      result

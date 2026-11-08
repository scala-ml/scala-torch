package scalatorch.models

import scalatorch.*
import scalatorch.autograd.Variable
import scalatorch.nn.*
import scalatorch.ffm.{CudaSession,CudaTensor}
import scala.util.Using

/** Runs CLIP text transformer layers on CUDA. Embedding lookup, EOS pooling and the projection stay on the CPU
  * (they are tiny). Enabled with `SCALATORCH_CLIP_DEVICE=cuda[:index]`, or by `ClipCuda.enable` from the worker.
  */
object ClipCuda:
  @volatile private var deviceIndex: Option[Int]=sys.env.get("SCALATORCH_CLIP_DEVICE").flatMap {
    case "cuda" => Some(0)
    case v if v.startsWith("cuda:") => v.drop(5).toIntOption
    case _ => None
  }
  def enable(index: Int): Unit = deviceIndex=Some(index)
  def disable(): Unit = deviceIndex=None
  def enabled: Boolean = deviceIndex.isDefined

  /** Output of the layer stack: hidden state after `selectedDepth` layers, final-norm of the full stack. */
  final case class States(selected: Tensor[Float],normalized: Tensor[Float])

  /** None when CUDA is off or unusable, so callers fall back to the CPU path. */
  def run(model: ClipTextModel,embedded: Tensor[Float],selectedDepth: Int): Option[States] =
    deviceIndex.flatMap { index =>
      try
        val stack=ClipStack(model.layers,model.finalNorm)
        val weightBytes=Module.parameters(stack).map(_.data.shape.numel*4L).sum
        val budget=weightBytes+(embedded.shape.numel*4L*48)+(768L<<20)
        Some(CudaSession.withDevice(index,maxBytes=budget) { session =>
          Using.resource(CudaDiffusionRuntime.upload(session,stack,includeAdapters=false)) { runtime =>
            runtime.clipStates(stack,session.upload(embedded),selectedDepth,model.maxScoreElements)
          }
        })
      catch case error: Throwable =>
        System.err.println(s"[scala-torch] CLIP CUDA path failed, using CPU: $error")
        None
    }

/** Transformer stack without embeddings, so only the layers are uploaded. */
private[models] final case class ClipStack(layers: Vector[ClipTextLayer],finalNorm: LayerNorm) derives Module

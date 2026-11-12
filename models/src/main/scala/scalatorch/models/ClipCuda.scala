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

  // Results of recent runs. Generating again with the same prompts is the common case, and encoding them took about 0.2 s of a
  // 1.7 s SD1.5 image (a session per prompt). An entry belongs to one layer stack (held weakly, so an unloaded model is not kept
  // alive) and to the weight tensors it had; it is used only when the token embeddings are equal value by value.
  private final class Entry(val depth: Int,val half: Boolean,val dims: Vector[Long],val values: Array[Float],val states: States)
  private final class Owner(val layers: java.lang.ref.WeakReference[AnyRef],val weights: Long,
      val entries: java.util.ArrayDeque[Entry]=new java.util.ArrayDeque[Entry]())
  private val owners=new java.util.ArrayDeque[Owner]()
  private val cacheEnabled= !sys.env.get("SCALATORCH_CLIP_CACHE").contains("0")
  private def weightIdentity(stack: ClipStack): Long =
    Module.parameters(stack).foldLeft(17L)((acc,p) => acc*31+System.identityHashCode(p.data))
  /** The LoRA factors that apply to the weights of this stack right now, with their strengths. */
  private def stackAdapters(stack: ClipStack): Vector[LinearAdapterScope.Applied] =
    Module.parameters(stack).flatMap(LinearAdapterScope.forWeight)
  private def cached(model: ClipTextModel,stack: ClipStack,embedded: Tensor[Float],depth: Int,half: Boolean)(compute: => States): States =
    if !cacheEnabled then return compute
    val values=embedded.toFloatArray
    val dims=embedded.shape.dims.toVector
    // A text-encoder LoRA changes the result: its factor tensors and strengths are part of the identity. (A LoRA file that is
    // read again gives new tensors and so a new identity; the encoding is then computed again on the GPU.)
    val identity=stackAdapters(stack).foldLeft(weightIdentity(stack)) { case (acc,(adapter,strength)) =>
      ((acc*31+System.identityHashCode(adapter.a.data))*31+System.identityHashCode(adapter.b.data))*31+java.lang.Float.floatToIntBits(strength)
    }
    owners.synchronized {
      owners.removeIf(o => o.layers.get()==null)
      val it=owners.iterator()
      while it.hasNext do
        val owner=it.next()
        if (owner.layers.get() eq model.layers) && owner.weights==identity then
          val entries=owner.entries.iterator()
          while entries.hasNext do
            val e=entries.next()
            if e.depth==depth && e.half==half && e.dims==dims && java.util.Arrays.equals(e.values,values) then return e.states
    }
    val states=compute
    owners.synchronized {
      owners.removeIf(o => (o.layers.get() eq model.layers) && o.weights!=identity)
      var owner: Owner=null
      val it=owners.iterator()
      while it.hasNext do { val o=it.next();if o.layers.get() eq model.layers then owner=o }
      if owner==null then
        owner=new Owner(new java.lang.ref.WeakReference[AnyRef](model.layers),identity)
        owners.addFirst(owner)
        while owners.size>4 do owners.removeLast()
      owner.entries.addFirst(new Entry(depth,half,dims,values,states))
      while owner.entries.size>16 do owner.entries.removeLast()
    }
    states

  /** None when CUDA is off or unusable, so callers fall back to the CPU path. */
  def run(model: ClipTextModel,embedded: Tensor[Float],selectedDepth: Int): Option[States] =
    deviceIndex.flatMap { index =>
      try
        val stack=ClipStack(model.layers,model.finalNorm)
        val weightBytes=Module.parameters(stack).map(_.data.shape.numel*4L).sum
        val budget=weightBytes+(embedded.shape.numel*4L*48)+(768L<<20)
        val half=CudaDiffusionRuntime.fp16Default
        Some(cached(model,stack,embedded,selectedDepth,half) {
          CudaSession.withDevice(index,maxBytes=budget) { session =>
            // Only the LoRA factors of the text encoder are uploaded. Without them the linear layers of a prompt with a LoRA
            // failed ("key not found") and the encoder fell back to the CPU: about 20 s per SDXL prompt.
            val factors=stackAdapters(stack).flatMap((adapter,_) => Seq(adapter.a,adapter.b)).distinct
            Using.resource(CudaDiffusionRuntime.upload(session,stack,includeAdapters=false,half=half,extraParameters=factors)) { runtime =>
              runtime.clipStates(stack,session.upload(embedded),selectedDepth,model.maxScoreElements)
            }
          }
        })
      catch case error: Throwable =>
        System.err.println(s"[scala-torch] CLIP CUDA path failed, using CPU: $error")
        None
    }

/** Transformer stack without embeddings, so only the layers are uploaded. */
private[models] final case class ClipStack(layers: Vector[ClipTextLayer],finalNorm: LayerNorm) derives Module

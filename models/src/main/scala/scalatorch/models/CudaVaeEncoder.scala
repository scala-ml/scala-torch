package scalatorch.models

import scalatorch.ffm.{CudaSession,CudaTensor}

/** FP32 encoder snapshot returning raw mean/log-variance moments.
  * Sampling and diffusion scaling belong to the caller. Session must outlive encoder.
  */
final class CudaVaeEncoder private (session: CudaSession,runtime: CudaDiffusionRuntime,model: VaeEncoder,
    cancelled: () => Boolean,offloadWeights: Boolean) extends AutoCloseable:
  private var fallback: Option[CudaDiffusionRuntime]=None
  /** Runs in Float16; non-finite moments are recomputed in Float32. */
  def apply(input: CudaTensor): CudaTensor=
    val out=runtime.encode(input,model)
    if !runtime.half || session.allFinite(out) then out
    else
      out.close()
      System.err.println("[scala-torch] Float16 VAE encode produced non-finite values, retrying in Float32")
      val full=fallback.getOrElse {
        val created=CudaDiffusionRuntime.upload(session,model,cancelled,offloadWeights,half=false)
        fallback=Some(created)
        created
      }
      full.encode(input,model)
  def close(): Unit=
    try fallback.foreach(_.close())
    finally runtime.close()

object CudaVaeEncoder:
  def upload(session: CudaSession,model: VaeEncoder,cancelled: () => Boolean=() => false,
      offloadWeights: Boolean=false): CudaVaeEncoder =
    new CudaVaeEncoder(session,CudaDiffusionRuntime.upload(session,model,cancelled,offloadWeights,half=CudaDiffusionRuntime.fp16Default),model,cancelled,offloadWeights)

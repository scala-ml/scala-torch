package scalatorch.models

import scalatorch.ffm.{CudaSession,CudaTensor}

/** FP32 decoder snapshot. Input is unscaled latents; output is raw RGB,
  * with no pipeline scaling or pixel clamp. The session must outlive the decoder.
  */
final class CudaVaeDecoder private (session: CudaSession,runtime: CudaDiffusionRuntime,model: VaeDecoder,
    cancelled: () => Boolean,offloadWeights: Boolean) extends AutoCloseable:
  private var fallback: Option[CudaDiffusionRuntime]=None
  /** Runs in Float16; non-finite output (some SDXL VAEs overflow) is recomputed in Float32. */
  def apply(input: CudaTensor): CudaTensor=
    val out=runtime.decode(input,model)
    if !runtime.half || session.allFinite(out) then out
    else
      out.close()
      System.err.println("[scala-torch] Float16 VAE decode produced non-finite values, retrying in Float32")
      val full=fallback.getOrElse {
        val created=CudaDiffusionRuntime.upload(session,model,cancelled,offloadWeights,half=false)
        fallback=Some(created)
        created
      }
      full.decode(input,model)
  def close(): Unit=
    try fallback.foreach(_.close())
    finally runtime.close()

object CudaVaeDecoder:
  /** offloadWeights keeps immutable CPU snapshots and scopes GPU weight storage
    * to each operation. Defaults to retaining all weights on the GPU.
    */
  def upload(session: CudaSession,model: VaeDecoder,cancelled: () => Boolean=() => false,offloadWeights: Boolean=false): CudaVaeDecoder =
    new CudaVaeDecoder(session,CudaDiffusionRuntime.upload(session,model,cancelled,offloadWeights,half=CudaDiffusionRuntime.fp16Default),model,cancelled,offloadWeights)

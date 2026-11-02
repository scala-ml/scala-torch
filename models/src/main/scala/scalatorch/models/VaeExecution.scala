package scalatorch.models

import scalatorch.*
import scalatorch.autograd.{Autograd,Variable}
import scalatorch.ffm.CudaSession
import scala.util.Using
import java.util.concurrent.CancellationException

/** Explicit encoder/decoder placement. CUDA resources are scoped to each call on its calling thread. */
sealed trait VaeExecution:
  def offloadWeights: Boolean=false
  def device: String
  def encode(model: VaeEncoder,input: Tensor[Float],cancelled: () => Boolean): Tensor[Float]
  def decode(model: VaeDecoder,input: Tensor[Float],cancelled: () => Boolean): Tensor[Float]

object VaeExecution:
  private[models] def offloadFromEnvironment(environment: Map[String,String],key: String): Boolean =
    environment.getOrElse(key,"false") match
      case "false" => false
      case "true" => true
      case _ => throw IllegalArgumentException(s"$key must be true or false")

  private[models] def checkCancelled(cancelled: () => Boolean): Unit =
    if cancelled() || Thread.currentThread().isInterrupted then throw new CancellationException("diffusion inference cancelled")

  case object Cpu extends VaeExecution:
    val device="cpu"
    def encode(model: VaeEncoder,input: Tensor[Float],cancelled: () => Boolean): Tensor[Float] =
      checkCancelled(cancelled)
      val result=Autograd.noGrad(model(Variable(input))).data
      checkCancelled(cancelled)
      result
    def decode(model: VaeDecoder,input: Tensor[Float],cancelled: () => Boolean): Tensor[Float] =
      checkCancelled(cancelled)
      val result=Autograd.noGrad(model(Variable(input))).data
      checkCancelled(cancelled)
      result

  final case class Cuda(deviceIndex: Int=0,maxBytes: Long=1073741824L,override val offloadWeights: Boolean=false) extends VaeExecution:
    require(deviceIndex>=0 && maxBytes>0,"invalid CUDA VAE device or memory budget")
    val device=s"cuda:$deviceIndex"
    def encode(model: VaeEncoder,input: Tensor[Float],cancelled: () => Boolean): Tensor[Float] =
      checkCancelled(cancelled)
      model.preflight(input.shape)
      CudaSession.withDevice(deviceIndex,maxBytes) { session =>
        Using.Manager { use =>
          val encoder=use(CudaVaeEncoder.upload(session,model,cancelled,offloadWeights))
          checkCancelled(cancelled)
          val image=use(session.upload(input))
          val output=use(encoder(image))
          checkCancelled(cancelled)
          val result=output.toCpu
          checkCancelled(cancelled)
          result
        }.get
      }
    def decode(model: VaeDecoder,input: Tensor[Float],cancelled: () => Boolean): Tensor[Float] =
      checkCancelled(cancelled)
      model.preflight(input.shape)
      CudaSession.withDevice(deviceIndex,maxBytes) { session =>
        Using.Manager { use =>
          val decoder=use(CudaVaeDecoder.upload(session,model,cancelled,offloadWeights))
          checkCancelled(cancelled)
          val latent=use(session.upload(input))
          val output=use(decoder(latent))
          checkCancelled(cancelled)
          val result=output.toCpu
          checkCancelled(cancelled)
          result
        }.get
      }

  def fromEnvironment(environment: Map[String,String]): VaeExecution =
    val device=environment.getOrElse("SDFX_SCALATORCH_VAE_DEVICE","cpu")
    def cuda(index: Int)=Cuda(index,environment.get("SDFX_SCALATORCH_VAE_MAX_BYTES").fold(1073741824L)(_.toLong),
      VaeExecution.offloadFromEnvironment(environment,"SDFX_SCALATORCH_VAE_OFFLOAD_WEIGHTS"))
    device match
      case "cpu" => Cpu
      case "cuda" => cuda(0)
      case value if value.matches("cuda:[0-9]+") => cuda(value.drop(5).toInt)
      case _ => throw IllegalArgumentException("VAE device must be cpu, cuda or cuda:<index>")

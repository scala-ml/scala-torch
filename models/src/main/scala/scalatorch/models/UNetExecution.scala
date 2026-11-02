package scalatorch.models

import scalatorch.*
import scalatorch.autograd.{Autograd,Variable}
import scalatorch.ffm.CudaSession
import scala.util.Using

private[models] trait UNetPredictor:
  def apply(sample: Tensor[Float],timesteps: Tensor[Float],guidance: Float): Tensor[Float]

/** Placement of U-Net prediction. CUDA weights/contexts live for one denoising loop;
  * CPU schedulers exchange only latent inputs and guided predictions each step.
  */
sealed trait UNetExecution:
  def offloadWeights: Boolean=false
  def device: String
  private[models] def withPredictor[A](model: ConditionalUNet2d,conditional: Tensor[Float],
      unconditional: Option[Tensor[Float]],cancelled: () => Boolean,
      positiveAdded: Option[DiffusionAddedConditioning]=None,negativeAdded: Option[DiffusionAddedConditioning]=None)(body: UNetPredictor => A): A

object UNetExecution:
  case object Cpu extends UNetExecution:
    val device="cpu"
    private[models] def withPredictor[A](model: ConditionalUNet2d,conditional: Tensor[Float],
        unconditional: Option[Tensor[Float]],cancelled: () => Boolean,
        positiveAdded: Option[DiffusionAddedConditioning]=None,negativeAdded: Option[DiffusionAddedConditioning]=None)(body: UNetPredictor => A): A =
      VaeExecution.checkCancelled(cancelled)
      require(unconditional.isDefined || negativeAdded.isEmpty,"negative addition requires negative context")
      body(new UNetPredictor:
        def apply(sample: Tensor[Float],timesteps: Tensor[Float],guidance: Float): Tensor[Float] = Autograd.noGrad {
          VaeExecution.checkCancelled(cancelled)
          require(guidance.isFinite && guidance>=0,"invalid guidance scale")
          val positive=model(Variable(sample),timesteps,Variable(conditional),positiveAdded)
          VaeExecution.checkCancelled(cancelled)
          val result=unconditional.fold(positive) { context =>
            val negative=model(Variable(sample),timesteps,Variable(context),negativeAdded)
            VaeExecution.checkCancelled(cancelled)
            ClassifierFreeGuidance.combine(negative,positive,guidance)
          }
          result.data
        }
      )

  final case class Cuda(deviceIndex: Int=0,maxBytes: Long=1073741824L,override val offloadWeights: Boolean=false) extends UNetExecution:
    require(deviceIndex>=0 && maxBytes>0,"invalid CUDA U-Net device or memory budget")
    val device=s"cuda:$deviceIndex"
    private[models] def withPredictor[A](model: ConditionalUNet2d,conditional: Tensor[Float],
        unconditional: Option[Tensor[Float]],cancelled: () => Boolean,
        positiveAdded: Option[DiffusionAddedConditioning]=None,negativeAdded: Option[DiffusionAddedConditioning]=None)(body: UNetPredictor => A): A =
      VaeExecution.checkCancelled(cancelled)
      require(conditional.shape.rank==3 && unconditional.forall(_.shape.rank==3),"U-Net contexts must be rank three")
      require(unconditional.isDefined || negativeAdded.isEmpty,"negative addition requires negative context")
      val positiveFeatures=model.addedFeatures(conditional.shape.dims.head,positiveAdded)
      val negativeFeatures=unconditional.flatMap(c => model.addedFeatures(c.shape.dims.head,negativeAdded))
      CudaSession.withDevice(deviceIndex,maxBytes) { session =>
        Using.Manager { use =>
          val unet=use(CudaUNet.upload(session,model,cancelled,offloadWeights))
          val positiveContext=use(session.upload(conditional))
          val negativeContext=unconditional.map(x => use(session.upload(x)))
          val positiveAddition=positiveFeatures.map(x => use(session.upload(x)))
          val negativeAddition=negativeFeatures.map(x => use(session.upload(x)))
          body(new UNetPredictor:
            def apply(sample: Tensor[Float],timesteps: Tensor[Float],guidance: Float): Tensor[Float] =
              VaeExecution.checkCancelled(cancelled)
              require(guidance.isFinite && guidance>=0,"invalid guidance scale")
              Using.Manager { temporary =>
                val input=temporary(session.upload(sample))
                val positive=temporary(unet.applyPrepared(input,timesteps,positiveContext,positiveAddition))
                VaeExecution.checkCancelled(cancelled)
                val predicted=negativeContext.fold(positive) { context =>
                  val negative=temporary(unet.applyPrepared(input,timesteps,context,negativeAddition))
                  VaeExecution.checkCancelled(cancelled)
                  // Preserve CPU subtraction and CFG operation ordering and rounding.
                  val negated=temporary(session.scale(negative,-1f))
                  val difference=temporary(session.add(positive,negated));negated.close();positive.close()
                  val scaled=temporary(session.scale(difference,guidance));difference.close()
                  val result=temporary(session.add(negative,scaled));negative.close();scaled.close()
                  result
                }
                input.close()
                VaeExecution.checkCancelled(cancelled)
                val result=predicted.toCpu
                VaeExecution.checkCancelled(cancelled)
                result
              }.get
          )
        }.get
      }

  def fromEnvironment(environment: Map[String,String]): UNetExecution =
    val device=environment.getOrElse("SDFX_SCALATORCH_UNET_DEVICE","cpu")
    def cuda(index: Int)=Cuda(index,environment.get("SDFX_SCALATORCH_UNET_MAX_BYTES").fold(1073741824L)(_.toLong),
      VaeExecution.offloadFromEnvironment(environment,"SDFX_SCALATORCH_UNET_OFFLOAD_WEIGHTS"))
    device match
      case "cpu" => Cpu
      case "cuda" => cuda(0)
      case value if value.matches("cuda:[0-9]+") => cuda(value.drop(5).toInt)
      case _ => throw IllegalArgumentException("U-Net device must be cpu, cuda or cuda:<index>")

package scalatorch.models

import scalatorch.*
import scalatorch.ffm.{CudaSession,CudaTensor}
import scalatorch.nn.*
import scala.util.Using

/** Inference-only snapshot of a diffusion residual block in one CUDA session.
  * Parameters upload once; all intermediate activations remain on the device.
  * The session must outlive this object and all returned tensors.
  */
final class CudaDiffusionResnet private (session: CudaSession,model: DiffusionResnet2d,
    weights: Map[Parameter,CudaTensor]) extends AutoCloseable:
  private var closed=false

  def apply(input: CudaTensor,timeEmbedding: Option[CudaTensor]=None): CudaTensor =
    if closed then throw IllegalStateException("CUDA residual block is closed")
    val shape=session.shape(input)
    require(shape.rank==4 && shape.dims(1)==model.norm1.channels,"residual input must be NCHW with matching channels")
    require(timeEmbedding.isDefined==model.timeEmbProj.isDefined,"timestep embedding presence mismatch")
    timeEmbedding.foreach { t =>
      require(session.shape(t)==Shape.from(Seq(shape.dims.head,weights(model.timeEmbProj.get.weight).shape.dims(1))),
        "timestep embedding batch/width mismatch")
    }
    Using.Manager { use =>
      def norm(x: CudaTensor,n: GroupNorm): CudaTensor =
        use(session.groupNorm(x,n.groups,n.epsilon,n.weight.map(weights),n.bias.map(weights)))
      def conv(x: CudaTensor,c: Conv2d): CudaTensor =
        val out=use(session.conv2d(x,weights(c.weight),c.config))
        c.bias.fold(out) { p =>
          val bias=use(session.reshape(weights(p),Shape.from(Seq(1L,weights(p).shape.numel,1L,1L))))
          val result=use(session.addBroadcast(out,bias))
          bias.close();out.close()
          result
        }
      def project(x: CudaTensor,p: Parameter): CudaTensor =
        val transposed=use(session.transpose(weights(p),0,1))
        val out=use(session.matmul(x,transposed))
        transposed.close()
        out
      def linear(x: CudaTensor,l: Linear): CudaTensor =
        val base=project(x,l.weight)
        val adapted=l.lora.fold(base) { a =>
          val first=project(x,a.a);val second=project(first,a.b)
          first.close()
          val scaled=use(session.scale(second,(a.alpha/weights(a.a).shape.dims.head).toFloat))
          second.close()
          val sum=use(session.add(base,scaled))
          base.close();scaled.close()
          sum
        }
        l.bias.fold(adapted) { p =>
          val out=use(session.addBroadcast(adapted,weights(p)))
          adapted.close()
          out
        }
      val normalized1=norm(input,model.norm1)
      val activated1=use(session.silu(normalized1));normalized1.close()
      val hidden=conv(activated1,model.conv1);activated1.close()
      val conditioned=model.timeEmbProj.fold(hidden) { projection =>
        val activatedTime=use(session.silu(timeEmbedding.get))
        val projected=linear(activatedTime,projection);activatedTime.close()
        val expanded=use(session.reshape(projected,Shape.from(Seq(shape.dims.head,model.norm2.channels.toLong,1L,1L))))
        projected.close()
        val out=use(session.addBroadcast(hidden,expanded))
        expanded.close();hidden.close()
        out
      }
      val normalized2=norm(conditioned,model.norm2);conditioned.close()
      val activated2=use(session.silu(normalized2));normalized2.close()
      val residual=conv(activated2,model.conv2);activated2.close()
      val shortcut=model.convShortcut.fold(input)(conv(input,_))
      val sum=use(session.add(shortcut,residual))
      residual.close()
      if model.convShortcut.isDefined then shortcut.close()
      val result=use(session.scale(sum,(1.0/model.outputScale).toFloat));sum.close()
      // A separately retained handle escapes the temporary scope.
      session.reshape(result,result.shape)
    }.get

  def close(): Unit =
    if !closed then
      session.allocatedBytes // Check thread/context before altering ownership.
      weights.values.foreach(_.close())
      closed=true

object CudaDiffusionResnet:
  def upload(session: CudaSession,model: DiffusionResnet2d): CudaDiffusionResnet =
    val uploaded=scala.collection.mutable.LinkedHashMap.empty[Parameter,CudaTensor]
    try
      Module.parameters(model).foreach(p => uploaded(p)=session.upload(p.data))
      new CudaDiffusionResnet(session,model,uploaded.toMap)
    catch case error: Throwable =>
      uploaded.values.toVector.reverse.foreach { t =>
        try t.close() catch case cleanup: Throwable => error.addSuppressed(cleanup)
      }
      throw error

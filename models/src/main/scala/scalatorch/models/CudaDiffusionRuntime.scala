package scalatorch.models

import scalatorch.*
import scalatorch.ffm.{CudaSession,CudaTensor}
import scalatorch.nn.*
import scala.util.Using

/** Shared inference executor for a snapshot of a diffusion parameter tree. */
private[models] final class CudaDiffusionRuntime private (session: CudaSession,
    weights: Map[Parameter,CudaTensor]) extends AutoCloseable:
  private var closed=false
  private def check(): Unit =
    if closed then throw IllegalStateException("CUDA diffusion runtime is closed")
    session.allocatedBytes
  private def scope(body: Using.Manager => CudaTensor): CudaTensor =
    check()
    Using.Manager { use =>
      val result=body(use)
      session.reshape(result,result.shape)
    }.get

  def norm(x: CudaTensor,n: GroupNorm): CudaTensor =
    check()
    require(session.shape(x).rank>=2 && x.shape.dims(1)==n.channels,"GroupNorm input channel mismatch")
    session.groupNorm(x,n.groups,n.epsilon,n.weight.map(weights),n.bias.map(weights))

  def conv(x: CudaTensor,c: Conv2d): CudaTensor = scope { use =>
    val out=use(session.conv2d(x,weights(c.weight),c.config))
    c.bias.fold(out) { p =>
      val bias=use(session.reshape(weights(p),Shape.from(Seq(1L,weights(p).shape.numel,1L,1L))))
      val result=use(session.addBroadcast(out,bias))
      bias.close();out.close()
      result
    }
  }

  private def project(x: CudaTensor,p: Parameter): CudaTensor = scope { use =>
    val transposed=use(session.transpose(weights(p),0,1))
    val out=use(session.matmul(x,transposed))
    transposed.close()
    out
  }

  def linear(x: CudaTensor,l: Linear): CudaTensor = scope { use =>
    val base=use(project(x,l.weight))
    val adapted=l.lora.fold(base) { a =>
      val first=use(project(x,a.a));val second=use(project(first,a.b))
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
  }

  def resnet(input: CudaTensor,model: DiffusionResnet2d,timeEmbedding: Option[CudaTensor]): CudaTensor =
    check()
    val shape=session.shape(input)
    require(shape.rank==4 && shape.dims(1)==model.norm1.channels,"residual input must be NCHW with matching channels")
    require(timeEmbedding.isDefined==model.timeEmbProj.isDefined,"timestep embedding presence mismatch")
    timeEmbedding.foreach { t =>
      require(session.shape(t)==Shape.from(Seq(shape.dims.head,weights(model.timeEmbProj.get.weight).shape.dims(1))),
        "timestep embedding batch/width mismatch")
    }
    scope { use =>
      val normalized1=use(norm(input,model.norm1))
      val activated1=use(session.silu(normalized1));normalized1.close()
      val hidden=use(conv(activated1,model.conv1));activated1.close()
      val conditioned=model.timeEmbProj.fold(hidden) { projection =>
        val activatedTime=use(session.silu(timeEmbedding.get))
        val projected=use(linear(activatedTime,projection));activatedTime.close()
        val expanded=use(session.reshape(projected,Shape.from(Seq(shape.dims.head,model.norm2.channels.toLong,1L,1L))))
        projected.close()
        val out=use(session.addBroadcast(hidden,expanded))
        expanded.close();hidden.close()
        out
      }
      val normalized2=use(norm(conditioned,model.norm2));conditioned.close()
      val activated2=use(session.silu(normalized2));normalized2.close()
      val residual=use(conv(activated2,model.conv2));activated2.close()
      val shortcut=model.convShortcut.fold(input)(c => use(conv(input,c)))
      val sum=use(session.add(shortcut,residual));residual.close()
      if model.convShortcut.isDefined then shortcut.close()
      val result=use(session.scale(sum,(1.0/model.outputScale).toFloat));sum.close()
      result
    }

  def attention(input: CudaTensor,model: DiffusionAttention,context: Option[CudaTensor]): CudaTensor =
    check()
    val source=context.getOrElse(input)
    val x=session.shape(input);val c=session.shape(source)
    val inner=weights(model.toQ.weight).shape.dims.head
    val queryWidth=weights(model.toQ.weight).shape.dims(1)
    val contextWidth=weights(model.toK.weight).shape.dims(1)
    require(x.rank==3 && x.dims(2)==queryWidth,"query must have matching (batch,tokens,channels)")
    require(c.rank==3 && c.dims(0)==x.dims(0) && c.dims(2)==contextWidth,"context batch/channel mismatch")
    val batch=x.dims(0);val queries=x.dims(1);val keys=c.dims(1)
    AttentionLayout.validate(Shape.from(Seq(batch,model.heads.toLong,queries,inner/model.heads)),
      Shape.from(Seq(batch,model.heads.toLong,keys,inner/model.heads)),
      Shape.from(Seq(batch,model.heads.toLong,keys,inner/model.heads)),0,model.maxScoreElements)
    scope { use =>
      def heads(value: CudaTensor,l: Linear,tokens: Long): CudaTensor =
        val flat=use(session.reshape(value,Shape.from(Seq(batch*tokens,value.shape.dims(2)))))
        val projected=use(linear(flat,l));flat.close()
        val view=use(session.reshape(projected,Shape.from(Seq(batch,tokens,model.heads.toLong,inner/model.heads))))
        projected.close()
        val out=use(session.transpose(view,1,2));view.close()
        out
      val q=heads(input,model.toQ,queries);val k=heads(source,model.toK,keys);val v=heads(source,model.toV,keys)
      val attended=use(session.scaledDotProductAttention(q,k,v,model.maxScoreElements))
      q.close();k.close();v.close()
      val packed=use(session.transpose(attended,1,2));attended.close()
      val flat=use(session.reshape(packed,Shape.from(Seq(batch*queries,inner))));packed.close()
      val projected=use(linear(flat,model.toOut));flat.close()
      val result=use(session.reshape(projected,Shape.from(Seq(batch,queries,queryWidth))))
      projected.close()
      result
    }

  def vaeAttention(input: CudaTensor,model: VaeAttention2d): CudaTensor =
    check()
    val shape=session.shape(input)
    model.preflight(shape)
    scope { use =>
      val normalized=use(norm(input,model.norm))
      val view=use(session.reshape(normalized,Shape.from(Seq(shape.dims(0),shape.dims(1),shape.dims(2)*shape.dims(3)))))
      normalized.close()
      val tokens=use(session.transpose(view,1,2));view.close()
      val attended=use(attention(tokens,model.attention,None));tokens.close()
      val packed=use(session.transpose(attended,1,2));attended.close()
      val image=use(session.reshape(packed,shape));packed.close()
      val out=use(session.add(input,image));image.close()
      out
    }

  def decode(input: CudaTensor,model: VaeDecoder): CudaTensor =
    check()
    model.preflight(session.shape(input))
    scope { use =>
      var hidden=use(session.reshape(input,input.shape))
      def next(value: CudaTensor): Unit =
        val owned=use(value)
        hidden.close()
        hidden=owned
      model.postQuantConv.foreach(c => next(conv(hidden,c)))
      next(conv(hidden,model.convIn))
      next(resnet(hidden,model.midBlock.first,None))
      model.midBlock.attention.foreach(a => next(vaeAttention(hidden,a)))
      next(resnet(hidden,model.midBlock.second,None))
      model.upBlocks.foreach { block =>
        block.layers.foreach(r => next(resnet(hidden,r,None)))
        block.upsample.foreach { up =>
          val shape=hidden.shape
          next(session.resizeNearest2d(hidden,(shape.dims(2)*2).toInt,(shape.dims(3)*2).toInt))
          next(conv(hidden,up.conv))
        }
      }
      next(norm(hidden,model.normOut))
      next(session.silu(hidden))
      next(conv(hidden,model.convOut))
      hidden
    }

  def close(): Unit =
    if !closed then
      session.allocatedBytes
      weights.values.foreach(_.close())
      closed=true

private[models] object CudaDiffusionRuntime:
  def upload[A: Module](session: CudaSession,model: A): CudaDiffusionRuntime =
    val uploaded=scala.collection.mutable.LinkedHashMap.empty[Parameter,CudaTensor]
    try
      Module.parameters(model).foreach(p => uploaded(p)=session.upload(p.data))
      new CudaDiffusionRuntime(session,uploaded.toMap)
    catch case error: Throwable =>
      uploaded.values.toVector.reverse.foreach { t =>
        try t.close() catch case cleanup: Throwable => error.addSuppressed(cleanup)
      }
      throw error

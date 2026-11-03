package scalatorch.models

import scalatorch.*
import scalatorch.ffm.{CudaSession,CudaTensor}
import scalatorch.nn.*
import scala.util.Using

/** Shared inference executor for a snapshot of a diffusion parameter tree. */
private[models] final class CudaDiffusionRuntime private (session: CudaSession,
    weights: Map[Parameter,CudaTensor],private var host: Map[Parameter,Tensor[Float]],cancelled: () => Boolean) extends AutoCloseable:
  private var closed=false
  private def check(): Unit =
    if closed then throw IllegalStateException("CUDA diffusion runtime is closed")
    session.allocatedBytes
    VaeExecution.checkCancelled(cancelled)
  private def scope(body: Using.Manager => CudaTensor): CudaTensor =
    check()
    Using.Manager { use =>
      val result=body(use)
      check()
      session.reshape(result,result.shape)
    }.get

  private def weightShape(p: Parameter): Shape =
    host.get(p).fold(weights(p).shape)(_.shape)

  private def weight(p: Parameter,use: Using.Manager): CudaTensor =
    check()
    host.get(p).fold(weights.get(p).fold(use(session.upload(p.data)))(identity))(t => use(session.upload(t)))

  def norm(x: CudaTensor,n: GroupNorm): CudaTensor = scope { use =>
    check()
    require(session.shape(x).rank>=2 && x.shape.dims(1)==n.channels,"GroupNorm input channel mismatch")
    use(session.groupNorm(x,n.groups,n.epsilon,n.weight.map(p => weight(p,use)),n.bias.map(p => weight(p,use))))
  }

  def conv(x: CudaTensor,c: Conv2d): CudaTensor = scope { use =>
    val out=use(session.conv2d(x,weight(c.weight,use),c.config))
    c.bias.fold(out) { p =>
      val bias=use(session.reshape(weight(p,use),Shape.from(Seq(1L,weightShape(p).numel,1L,1L))))
      val result=use(session.addBroadcast(out,bias))
      bias.close();out.close()
      result
    }
  }

  private def project(x: CudaTensor,p: Parameter): CudaTensor = scope { use =>
    val transposed=use(session.transpose(weight(p,use),0,1))
    val out=use(session.matmul(x,transposed))
    transposed.close()
    out
  }

  def linear(x: CudaTensor,l: Linear): CudaTensor = scope { use =>
    val base=use(project(x,l.weight))
    val attached=l.lora.fold(base) { a =>
      val first=use(project(x,a.a));val second=use(project(first,a.b))
      first.close()
      val scaled=use(session.scale(second,(a.alpha/weightShape(a.a).dims.head).toFloat))
      second.close()
      val sum=use(session.add(base,scaled))
      base.close();scaled.close()
      sum
    }
    val adapted=LinearAdapterScope.forWeight(l.weight).foldLeft(attached) { case (sum,(a,multiplier)) =>
      val first=use(project(x,a.a));val second=use(project(first,a.b))
      first.close()
      val scaled=use(session.scale(second,multiplier*(a.alpha/weightShape(a.a).dims.head).toFloat))
      second.close()
      val out=use(session.add(sum,scaled))
      sum.close();scaled.close()
      out
    }
    l.bias.fold(adapted) { p =>
      val out=use(session.addBroadcast(adapted,weight(p,use)))
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
      require(session.shape(t)==Shape.from(Seq(shape.dims.head,weightShape(model.timeEmbProj.get.weight).dims(1))),
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
    val inner=weightShape(model.toQ.weight).dims.head
    val queryWidth=weightShape(model.toQ.weight).dims(1)
    val contextWidth=weightShape(model.toK.weight).dims(1)
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

  def layerNorm(input: CudaTensor,model: LayerNorm): CudaTensor = scope { use =>
    check()
    val shape=session.shape(input)
    require(shape.rank>0 && shape.dims.last==model.width,"LayerNorm final width mismatch")
    use(session.layerNorm(input,model.epsilon,model.weight.map(p => weight(p,use)),model.bias.map(p => weight(p,use))))
  }

  def feedForward(input: CudaTensor,model: DiffusionFeedForward): CudaTensor =
    check()
    val shape=session.shape(input)
    require(shape.rank>0 && shape.dims.last==model.inputWidth,"GEGLU input width mismatch")
    scope { use =>
      val flat=use(session.reshape(input,Shape.from(Seq(shape.numel/model.inputWidth,model.inputWidth))))
      val projected=use(linear(flat,model.projection));flat.close()
      val gated=use(session.geglu(projected));projected.close()
      val output=use(linear(gated,model.output));gated.close()
      val result=use(session.reshape(output,Shape.from(shape.dims.updated(shape.rank-1,model.outputWidth))))
      output.close()
      result
    }

  def transformer(input: CudaTensor,model: DiffusionTransformerBlock,context: CudaTensor): CudaTensor =
    check()
    val shape=session.shape(input);val contextShape=session.shape(context)
    require(shape.rank==3 && shape.dims(2)==model.norm1.width,"Transformer input must be matching token features")
    require(contextShape.rank==3 && contextShape.dims(0)==shape.dims(0) &&
      contextShape.dims(2)==weightShape(model.attn2.toK.weight).dims(1),"Transformer context batch/width mismatch")
    scope { use =>
      val norm1=use(layerNorm(input,model.norm1))
      val attn1=use(attention(norm1,model.attn1,None));norm1.close()
      val self=use(session.add(input,attn1));attn1.close()
      val norm2=use(layerNorm(self,model.norm2))
      val attn2=use(attention(norm2,model.attn2,Some(context)));norm2.close()
      val cross=use(session.add(self,attn2));self.close();attn2.close()
      val norm3=use(layerNorm(cross,model.norm3))
      val ff=use(feedForward(norm3,model.feedForward));norm3.close()
      val result=use(session.add(cross,ff));cross.close();ff.close()
      result
    }

  def spatialTransformer(input: CudaTensor,model: DiffusionSpatialTransformer,context: CudaTensor): CudaTensor =
    check()
    val shape=session.shape(input);val contextShape=session.shape(context)
    require(shape.rank==4 && shape.dims(1)==model.norm.channels,"spatial Transformer requires matching NCHW input")
    require(contextShape.rank==3 && contextShape.dims(0)==shape.dims(0) &&
      contextShape.dims(2)==weightShape(model.blocks.head.attn2.toK.weight).dims(1),"spatial Transformer context batch/width mismatch")
    val inner=weightShape(model.projIn.weight).dims.head
    val tokens=Shape.from(shape.dims.drop(2)).numel
    scope { use =>
      val normalized=use(norm(input,model.norm))
      val projected=use(conv(normalized,model.projIn));normalized.close()
      val view=use(session.reshape(projected,Shape.from(Seq(shape.dims(0),inner,tokens))));projected.close()
      var hidden=use(session.transpose(view,1,2));view.close()
      model.blocks.foreach { block =>
        val next=use(transformer(hidden,block,context))
        hidden.close();hidden=next
      }
      val packed=use(session.transpose(hidden,1,2));hidden.close()
      val image=use(session.reshape(packed,Shape.from(shape.dims.updated(1,inner))));packed.close()
      val out=use(conv(image,model.projOut));image.close()
      val result=use(session.add(input,out));out.close()
      result
    }

  private def downsample(input: CudaTensor,model: DiffusionDownsample2d): CudaTensor =
    if model.conv.config.padding==(1,1) then conv(input,model.conv)
    else scope { use =>
      val shape=session.shape(input)
      val column=use(session.fill(Shape.from(shape.dims.updated(3,1L)),0f))
      val wider=use(session.cat(Seq(input,column),3));column.close()
      val row=use(session.fill(Shape.from(wider.shape.dims.updated(2,1L)),0f))
      val padded=use(session.cat(Seq(wider,row),2));wider.close();row.close()
      val result=use(conv(padded,model.conv));padded.close()
      result
    }

  def unet(input: CudaTensor,model: ConditionalUNet2d,timesteps: Tensor[Float],context: CudaTensor,
      added: Option[CudaTensor]=None): CudaTensor =
    check()
    require(added.isDefined==model.additionEmbedding.isDefined,"U-Net additional conditioning presence mismatch")
    val features=model.timestepFeatures(session.shape(input),timesteps,session.shape(context))
    added.foreach { a =>
      require(session.shape(a)==Shape.from(Seq(session.shape(input).dims.head,
        weightShape(model.additionEmbedding.get.linear1.weight).dims(1))),"U-Net added feature shape mismatch")
    }
    // Small deterministic sinusoidal conditioning constants are prepared on CPU.
    // Learned timestep projections and every image/context operation run on CUDA.
    scope { use =>
      val f=use(session.upload(features))
      val first=use(linear(f,model.timeEmbedding.linear1));f.close()
      val activated=use(session.silu(first));first.close()
      val baseTime=use(linear(activated,model.timeEmbedding.linear2));activated.close()
      val time=model.additionEmbedding.fold(baseTime) { embedding =>
        val projected=use(linear(added.get,embedding.linear1))
        val activatedAddition=use(session.silu(projected));projected.close()
        val embedded=use(linear(activatedAddition,embedding.linear2));activatedAddition.close()
        val result=use(session.add(baseTime,embedded));baseTime.close();embedded.close()
        result
      }
      var hidden=use(conv(input,model.convIn))
      val skips=scala.collection.mutable.ArrayBuffer.empty[CudaTensor]
      def saveSkip(): Unit=skips+=use(session.reshape(hidden,hidden.shape))
      def next(value: CudaTensor): Unit =
        val owned=use(value)
        hidden.close();hidden=owned
        check()
      def layer(block: DiffusionUNetLayer): Unit =
        next(resnet(hidden,block.resnet,Some(time)))
        block.attention.foreach(a => next(spatialTransformer(hidden,a,context)))
      saveSkip()
      model.downBlocks.foreach { block =>
        block.layers.foreach { l => layer(l);saveSkip() }
        block.downsample.foreach { d => next(downsample(hidden,d));saveSkip() }
      }
      next(resnet(hidden,model.midBlock.first,Some(time)))
      next(spatialTransformer(hidden,model.midBlock.attention,context))
      next(resnet(hidden,model.midBlock.second,Some(time)))
      model.upBlocks.foreach { block =>
        block.layers.foreach { l =>
          val skip=skips.remove(skips.size-1)
          next(session.cat(Seq(hidden,skip),1));skip.close()
          layer(l)
        }
        block.upsample.foreach { up =>
          val target=skips.last.shape
          next(session.resizeNearest2d(hidden,target.dims(2).toInt,target.dims(3).toInt))
          next(conv(hidden,up.conv))
        }
      }
      require(skips.isEmpty,"U-Net leaves unused skip tensors")
      time.close()
      next(norm(hidden,model.normOut))
      next(session.silu(hidden))
      next(conv(hidden,model.convOut))
      hidden
    }

  def encode(input: CudaTensor,model: VaeEncoder): CudaTensor =
    check()
    model.preflight(session.shape(input))
    scope { use =>
      var hidden=use(conv(input,model.convIn))
      def next(value: CudaTensor): Unit =
        val owned=use(value)
        hidden.close();hidden=owned
        check()
      model.downBlocks.foreach { block =>
        block.layers.foreach(r => next(resnet(hidden,r,None)))
        block.downsample.foreach(d => next(downsample(hidden,d)))
      }
      next(resnet(hidden,model.midBlock.first,None))
      model.midBlock.attention.foreach(a => next(vaeAttention(hidden,a)))
      next(resnet(hidden,model.midBlock.second,None))
      next(norm(hidden,model.normOut))
      next(session.silu(hidden))
      next(conv(hidden,model.convOut))
      model.quantConv.foreach(c => next(conv(hidden,c)))
      hidden
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
        check()
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
      host=Map.empty
      closed=true

private[models] object CudaDiffusionRuntime:
  def upload[A: Module](session: CudaSession,model: A,cancelled: () => Boolean=() => false,offloadWeights: Boolean=false): CudaDiffusionRuntime =
    session.allocatedBytes
    val host=scala.collection.mutable.LinkedHashMap.empty[Parameter,Tensor[Float]]
    val uploaded=scala.collection.mutable.LinkedHashMap.empty[Parameter,CudaTensor]
    try
      (Module.parameters(model) ++ LinearAdapterScope.parameters).distinct.foreach { p =>
        VaeExecution.checkCancelled(cancelled)
        if offloadWeights then host(p)=p.data else uploaded(p)=session.upload(p.data)
      }
      VaeExecution.checkCancelled(cancelled)
      new CudaDiffusionRuntime(session,uploaded.toMap,host.toMap,cancelled)
    catch case error: Throwable =>
      uploaded.values.toVector.reverse.foreach { t =>
        try t.close() catch case cleanup: Throwable => error.addSuppressed(cleanup)
      }
      throw error

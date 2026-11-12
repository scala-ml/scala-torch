package scalatorch.models

import scalatorch.*
import scalatorch.ffm.{CudaSession,CudaTensor}
import scalatorch.nn.*
import scala.util.Using

/** Shared inference executor for a snapshot of a diffusion parameter tree. */
private[models] final class CudaDiffusionRuntime private (session: CudaSession,
    weights: Map[Parameter,CudaTensor],private var host: Map[Parameter,Tensor[Float]],cancelled: () => Boolean,val half: Boolean=false) extends AutoCloseable:
  private var closed=false
  private def up(t: Tensor[Float]): CudaTensor = if half then session.uploadHalf(t) else session.upload(t)
  private def fillZero(shape: Shape,value: Float): CudaTensor = if half then session.fillHalf(shape,value) else session.fill(shape,value)
  private val offloadedWeights=scala.collection.mutable.LinkedHashMap.empty[Parameter,CudaTensor]
  private var offloadedWeightBytes=0L
  private val unit=if half then 2L else 4L
  private var offloadedWeightLimit=
    if host.isEmpty then 0L else CudaDiffusionRuntime.offloadedWeightLimit(session.maxBytes,session.availableBytes)
  // Offloaded weights handed out inside the running operation: its kernels read them, so they must not be evicted before it ends.
  private val pinned=java.util.Collections.newSetFromMap(new java.util.IdentityHashMap[Parameter,java.lang.Boolean]())
  private var scopeDepth=0
  /** Closes the kept offloaded weight that was added last and is not in use by the running operation. */
  private def evictNewest(): Boolean =
    offloadedWeights.keysIterator.toVector.reverseIterator.find(p => !pinned.contains(p)) match
      case None => false
      case Some(parameter) =>
        val tensor=offloadedWeights.remove(parameter).get
        offloadedWeightBytes-=math.max(tensor.shape.numel*unit,1L)
        offloadedWeightEvictions+=1
        tensor.close()
        true

  private val profileEnabled=sys.env.get("SCALATORCH_CUDA_PROFILE").contains("1")
  private var offloadedWeightCacheHits=0L
  private var offloadedWeightCacheHitBytes=0L
  private var offloadedWeightUploads=0L
  private var offloadedWeightUploadBytes=0L
  private var offloadedWeightTransientUploads=0L
  private var offloadedWeightEvictions=0L
  private def check(): Unit =
    if closed then throw IllegalStateException("CUDA diffusion runtime is closed")
    session.allocatedBytes
    try VaeExecution.checkCancelled(cancelled)
    catch case error: Throwable =>
      try clearOffloadedWeights() catch case cleanup: Throwable => error.addSuppressed(cleanup)
      throw error
  private def scope(body: Using.Manager => CudaTensor): CudaTensor =
    scopeDepth+=1
    try
      check()
      Using.Manager { use =>
        val result=body(use)
        check()
        session.reshape(result,result.shape)
      }.get
    catch case error: Throwable =>
      try clearOffloadedWeights() catch case cleanup: Throwable => error.addSuppressed(cleanup)
      throw error
    finally
      scopeDepth-=1
      if scopeDepth==0 then pinned.clear()

  private def weightShape(p: Parameter): Shape =
    host.get(p).fold(weights(p).shape)(_.shape)

  private def clearOffloadedWeights(): Unit =
    offloadedWeights.values.toVector.foreach(_.close())
    offloadedWeights.clear()
    offloadedWeightBytes=0L

  private def weight(p: Parameter,use: Using.Manager): CudaTensor =
    check()
    host.get(p) match
      case None => weights.get(p).fold(use(up(p.data)))(identity)
      case Some(snapshot) =>
        offloadedWeights.get(p) match
          case Some(cached) =>
            offloadedWeightCacheHits+=1
            offloadedWeightCacheHitBytes+=math.max(cached.shape.numel*unit,1L)
            pinned.add(p)
            cached
          case None =>
            val bytes=math.max(snapshot.shape.numel*unit,1L)
            val retain=bytes<=offloadedWeightLimit && bytes<=offloadedWeightLimit-offloadedWeightBytes
            while bytes>session.maxBytes-session.allocatedBytes && evictNewest() do ()
            val uploaded=if half then session.uploadHalfBits(snapshot.shape,HalfWeightCache.bits(p,snapshot)) else up(snapshot)
            offloadedWeightUploads+=1
            offloadedWeightUploadBytes+=bytes
            if retain then
              offloadedWeights(p)=uploaded
              pinned.add(p)
              offloadedWeightBytes+=math.max(uploaded.shape.numel*unit,1L)
              uploaded
            else
              offloadedWeightTransientUploads+=1
              use(uploaded)

  def norm(x: CudaTensor,n: GroupNorm): CudaTensor = scope { use =>
    check()
    require(session.shape(x).rank>=2 && x.shape.dims(1)==n.channels,"GroupNorm input channel mismatch")
    use(session.groupNorm(x,n.groups,n.epsilon,n.weight.map(p => weight(p,use)),n.bias.map(p => weight(p,use))))
  }

  def conv(x: CudaTensor,c: Conv2d): CudaTensor = scope { use =>
    val (effX,effConfig)=CircularConvPadding.current match
      case Some(tiling) if c.config.padding._1>0 || c.config.padding._2>0 =>
        (use(up(CircularConvPadding.pad(x.toCpu,c.config.padding._1,c.config.padding._2,tiling.tileX,tiling.tileY))),
          c.config.copy(padding=(0,0)))
      case _ => (x,c.config)
    val base=use(session.conv2dFast(effX,weight(c.weight,use),effConfig))
    val out=LinearAdapterScope.forWeight(c.weight).foldLeft(base) { case (sum,(a,multiplier)) =>
      require(a.a.data.shape.rank==4 && effConfig.groups==1 && a.b.data.shape.dims.drop(2)==Vector(1L,1L),
        "convolution adapter requires ungrouped down kernel and 1x1 up kernel")
      val down=use(session.conv2dFast(effX,weight(a.a,use),effConfig))
      val up=use(session.conv2dFast(down,weight(a.b,use),Conv2dConfig()))
      val scaled=use(session.scale(up,multiplier*(a.alpha/weightShape(a.a).dims.head).toFloat))
      val combined=use(session.add(sum,scaled))
      down.close();up.close();scaled.close();sum.close()
      combined
    }
    c.bias.fold(out) { p =>
      val bias=use(session.reshape(weight(p,use),Shape.from(Seq(1L,weightShape(p).numel,1L,1L))))
      val result=use(session.addBroadcast(out,bias))
      bias.close();out.close()
      result
    }
  }

  private def project(x: CudaTensor,p: Parameter): CudaTensor =
    if session.hasCublas && weightShape(p).rank==2 && session.shape(x).rank==2 then scope { use =>
      use(session.matmulTransposedB(x,weight(p,use)))
    } else projectGeneric(x,p)

  private def projectGeneric(x: CudaTensor,p: Parameter): CudaTensor = scope { use =>
    val transposed=use(session.transpose(weight(p,use),0,1))
    val out=use(session.matmul(x,transposed))
    transposed.close()
    out
  }

  // LoRA merged into a private copy of the Float16 weight: one GEMM per step instead of five kernels. Rebuilt whenever
  // the adapter set or a multiplier changes (per-step LoRA schedules), and disabled with SCALATORCH_LORA_MERGE=0.
  private val mergedWeights=scala.collection.mutable.HashMap.empty[Parameter,(Vector[(AnyRef,Float)],CudaTensor)]
  private val mergeEnabled= !sys.env.get("SCALATORCH_LORA_MERGE").contains("0")

  private def mergedWeight(l: Linear,adapters: Vector[(LowRankAdapter,Float)]): CudaTensor =
    val signature=adapters.map((a,m)=>(a:AnyRef,m))
    mergedWeights.get(l.weight) match
      case Some((sig,w)) if sig.size==signature.size && sig.zip(signature).forall((x,y)=>(x._1 eq y._1) && x._2==y._2) => w
      case old =>
        old.foreach(_._2.close())
        mergedWeights.remove(l.weight)
        val base=weights(l.weight)
        var acc=session.scale(base,1f)
        try
          adapters.foreach { (a,multiplier) =>
            val rank=weightShape(a.a).dims.head
            val at=session.transpose(weights(a.a),0,1)
            val delta=try session.matmulTransposedB(weights(a.b),at) finally at.close()
            val scaled=try session.scale(delta,multiplier*(a.alpha/rank).toFloat) finally delta.close()
            val sum=try session.add(acc,scaled) finally scaled.close()
            acc.close();acc=sum
          }
        catch case e: Throwable => { acc.close();throw e }
        mergedWeights(l.weight)=(signature,acc)
        acc

  def linear(x: CudaTensor,l: Linear): CudaTensor =
    val adapters=l.lora.toVector.map(a=>(a,1f))++LinearAdapterScope.forWeight(l.weight).toVector
    if mergeEnabled && adapters.nonEmpty && host.isEmpty && session.hasCublas && session.shape(x).rank==2 &&
        weightShape(l.weight).rank==2 && weights.contains(l.weight) && adapters.forall((a,_)=>weights.contains(a.a) && weights.contains(a.b)) then
      scope { use =>
        val merged=mergedWeight(l,adapters)
        val fused=l.bias.flatMap(p => session.linearWithBias(x,merged,weight(p,use)))
        if fused.nonEmpty then use(fused.get)
        else
         val out=use(session.matmulTransposedB(x,merged))
         l.bias.fold(out) { p =>
          val withBias=use(session.addBroadcast(out,weight(p,use)))
          out.close()
          withBias
         }
      }
    else linearPlain(x,l)

  private def linearPlain(x: CudaTensor,l: Linear): CudaTensor = scope { use =>
    // no adapter: product and bias in one pass over the output
    val direct=
      if l.lora.isEmpty && l.bias.nonEmpty && session.hasCublas && weightShape(l.weight).rank==2 && session.shape(x).rank==2 &&
          LinearAdapterScope.forWeight(l.weight).isEmpty then session.linearWithBias(x,weight(l.weight,use),weight(l.bias.get,use))
      else None
    if direct.nonEmpty then use(direct.get) else linearRest(x,l,use)
  }

  private def linearRest(x: CudaTensor,l: Linear,use: Using.Manager): CudaTensor =
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
      Shape.from(Seq(batch,model.heads.toLong,keys,inner/model.heads)),0,model.maxScoreElements,!input.half)
    if context.isEmpty && Pag.isCurrentLayerPerturbed then
      return scope { use =>
        val flat=use(session.reshape(input,Shape.from(Seq(batch*queries,queryWidth))))
        val v=use(linear(flat,model.toV));flat.close()
        val projected=use(linear(v,model.toOut));v.close()
        val result=use(session.reshape(projected,Shape.from(Seq(batch,queries,queryWidth))))
        projected.close()
        result
      }
    // Plain Float16 self-attention: the fused kernels read the heads where the projections put them, without transposed copies.
    val special=(context.isEmpty && Pag.currentSection=="mid" && (Sag.currentStore.isDefined || Incantations.currentSeg.isDefined)) ||
      (context.isDefined && NegPip.currentSigns.isDefined)
    if input.half && !special && session.canAttendTokensMajor(queries,keys,inner/model.heads) then
      return scope { use =>
        def project(value: CudaTensor,l: Linear,tokens: Long): CudaTensor =
          val flat=use(session.reshape(value,Shape.from(Seq(batch*tokens,value.shape.dims(2)))))
          val projected=use(linear(flat,l));flat.close()
          val view=use(session.reshape(projected,Shape.from(Seq(batch,tokens,inner))));projected.close()
          view
        val q=project(input,model.toQ,queries);val k=project(source,model.toK,keys);val v=project(source,model.toV,keys)
        val attended=use(session.attentionTokensMajor(q,k,v,model.heads))
        q.close();k.close();v.close()
        val flat=use(session.reshape(attended,Shape.from(Seq(batch*queries,inner))));attended.close()
        val projected=use(linear(flat,model.toOut));flat.close()
        val result=use(session.reshape(projected,Shape.from(Seq(batch,queries,queryWidth))))
        projected.close()
        result
      }
    scope { use =>
      def heads(value: CudaTensor,l: Linear,tokens: Long): CudaTensor =
        val flat=use(session.reshape(value,Shape.from(Seq(batch*tokens,value.shape.dims(2)))))
        val projected=use(linear(flat,l));flat.close()
        val view=use(session.reshape(projected,Shape.from(Seq(batch,tokens,model.heads.toLong,inner/model.heads))))
        projected.close()
        val out=use(session.transpose(view,1,2));view.close()
        out
      val qRaw=heads(input,model.toQ,queries);val k=heads(source,model.toK,keys)
      val q=if context.isEmpty && Pag.currentSection=="mid" then
        if Sag.currentStore.isDefined then Sag.recordFromQK(qRaw.toCpu,k.toCpu)
        Incantations.currentSeg match
          case Some((sigma,ih,iw)) =>
            val blurred=use(up(Incantations.segBlurQuery(qRaw.toCpu,sigma,ih,iw)))
            qRaw.close();blurred
          case None => qRaw
      else qRaw
      val vRaw=heads(source,model.toV,keys)
      val v=if context.isDefined then NegPip.currentSigns match
        case Some(signs) =>
          val signed=use(up(NegPip.applySigns(vRaw.toCpu,signs)))
          vRaw.close();signed
        case None => vRaw
      else vRaw
      // Larger query slices reduce synchronous kernel launches on full-size
      // diffusion images. CudaSession still caps scratch to its available
      // allocation budget, so low-memory sessions retain automatic slicing.
      val attended=use(session.scaledDotProductAttention(q,k,v,model.maxScoreElements,maxWorkspaceBytes=64L*1024*1024))
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
      val gated=use(session.geglu(projected,fast=true));projected.close()
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
      val attn1=HyperTile.currentUnscoped.flatMap(_.split(shape.dims(1).toInt)) match
        case Some((nh,nw,h,w)) =>
          val b=shape.dims(0).toInt
          val tiled=use(up(HyperTile.tileTokens(norm1.toCpu,nh,nw,h,w)))
          val tiledOut=use(attention(tiled,model.attn1,None))
          use(up(HyperTile.untileTokens(tiledOut.toCpu,b,nh,nw,h,w)))
        case None =>
          val tdCtx=for
            td <- TokenDownsampling.current
            (sh,sw) <- TokenDownsampling.currentSpatialShape
            factor = td.factorForSection(Pag.currentSection)
            if factor > 1.0 && sh.toLong * sw == shape.dims(1)
          yield use(up(TokenDownsampling.resampleTokens(norm1.toCpu,sh,sw,factor)))
          val aaOut=if tdCtx.isEmpty then
            for
              aa <- AgentAttention.current
              (sh,sw) <- TokenDownsampling.currentSpatialShape
              tokens = shape.dims(1).toInt
              if sh * sw == tokens && AgentAttention.downsampleOf(aa.latentH,aa.latentW,tokens) <= aa.settings.maxDownsample
              r = (tokens * aa.settings.ratio).toInt
              if r > 0 && sh >= aa.settings.sy && sw >= aa.settings.sx
            yield
              val norm1Cpu=norm1.toCpu
              val b=shape.dims(0).toInt
              val seedOpt=if b%2==0 then Some(aa.seed) else None
              val plan=AgentAttention.planMerge(norm1Cpu,sh,sw,aa.settings.sx,aa.settings.sy,r,(tokens*aa.settings.agentRatio).toInt,seedOpt,aa.settings.useFp32)
              val (features,agentsOpt)=AgentAttention.merge(plan,norm1Cpu)
              val mergedOut=agentsOpt match
                case Some(ag) if ag.shape.dims(1)*2 < features.shape.dims(1) =>
                  AgentAttention.applyAgentAttention(model.attn1,features,ag)
                case _ =>
                  val fGpu=use(up(features))
                  val mGpu=use(attention(fGpu,model.attn1,None))
                  mGpu.toCpu
              use(up(AgentAttention.unmerge(plan,mergedOut)))
          else None
          val refOrLlliteOut=if aaOut.isEmpty && tdCtx.isEmpty && (ReferenceOnly.current.isDefined || ControlLLLite.current.isDefined) then
            Some(use(up(model.attn1(scalatorch.autograd.Variable(norm1.toCpu,requiresGrad=false),None).data)))
          else None
          val out=aaOut.orElse(refOrLlliteOut).getOrElse(use(attention(norm1,model.attn1,tdCtx)))
          tdCtx.foreach(_.close())
          out
      norm1.close()
      val self=use(session.add(input,attn1));attn1.close()
      val norm2=use(layerNorm(self,model.norm2))
      val attn2=if IpAdapter.current.isDefined || ControlLLLite.current.isDefined then
        val res=model.attn2(scalatorch.autograd.Variable(norm2.toCpu,requiresGrad=false),Some(scalatorch.autograd.Variable(context.toCpu,requiresGrad=false))).data
        norm2.close()
        use(up(res))
      else
        val res=use(attention(norm2,model.attn2,Some(context)));norm2.close();res
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
      TokenDownsampling.withSpatialShape(shape.dims(2).toInt,shape.dims(3).toInt) { model.blocks.foreach { block =>
        val next=use(transformer(hidden,block,context))
        hidden.close();hidden=next
      } }
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
      val column=use(fillZero(Shape.from(shape.dims.updated(3,1L)),0f))
      val wider=use(session.cat(Seq(input,column),3));column.close()
      val row=use(fillZero(Shape.from(wider.shape.dims.updated(2,1L)),0f))
      val padded=use(session.cat(Seq(wider,row),2));wider.close();row.close()
      val result=use(conv(padded,model.conv));padded.close()
      result
    }

  /** Computes one ControlNet pass in this CUDA session and returns caller-owned residual tensors. */
  def controlNet(sample: CudaTensor,model: ConditionalControlNet2d,timesteps: Tensor[Float],
      context: CudaTensor,image: CudaTensor,scales: Vector[Float],added: Option[CudaTensor]=None): CudaControlNetResiduals =
    check()
    require(added.isDefined==model.additionEmbedding.isDefined,"ControlNet additional conditioning presence mismatch")
    val shape=session.shape(sample)
    val contextShape=session.shape(context)
    val imageShape=session.shape(image)
    val c=model.config.unet
    require(shape.rank==4 && shape.dims(1)==c.inChannels,"ControlNet input must be four-channel NCHW")
    require(contextShape.rank==3 && contextShape.dims.head==shape.dims.head && contextShape.dims(2)==c.contextWidth,
      "ControlNet text context batch/width mismatch")
    require(imageShape==Shape.from(Seq(shape.dims.head,model.config.conditioningChannels,shape.dims(2)*8,shape.dims(3)*8)),
      "ControlNet image must match the latent dimensions at eight times scale")
    require(scales.size==model.downProjections.size+1 && scales.forall(_.isFinite),"ControlNet scale count/value mismatch")
    require(timesteps.shape.rank==1 && (timesteps.shape.dims.head==1 || timesteps.shape.dims.head==shape.dims.head),
      "ControlNet timestep batch mismatch")
    val times=if timesteps.shape.dims.head==shape.dims.head then timesteps
      else timesteps.broadcastTo(Shape.from(Seq(shape.dims.head)))
    val features=DiffusionTimesteps.sinusoidal(times,c.channels.head,c.flipSinToCos,c.frequencyShift,
      phaseScale=1.0,maxPeriod=10000.0,maxElements=model.maxElements)
    val outputs=scala.collection.mutable.ArrayBuffer.empty[CudaTensor]
    var outputMiddle: Option[CudaTensor]=None
    try
      val result=Using.Manager { use =>
        val embedded=use(up(features))
        val projected=use(linear(embedded,model.timeEmbedding.linear1));embedded.close()
        val activated=use(session.silu(projected));projected.close()
        val baseTime=use(linear(activated,model.timeEmbedding.linear2));activated.close()
        val time=model.additionEmbedding.fold(baseTime) { embedding =>
          val addProjected=use(linear(added.get,embedding.linear1))
          val addActivated=use(session.silu(addProjected));addProjected.close()
          val addEmbedded=use(linear(addActivated,embedding.linear2));addActivated.close()
          val sum=use(session.add(baseTime,addEmbedded));baseTime.close();addEmbedded.close()
          sum
        }
        var hidden=use(conv(sample,model.convIn))
        def next(value: CudaTensor): Unit =
          val owned=use(value)
          hidden.close();hidden=owned
          check()
        def layer(value: DiffusionUNetLayer): Unit =
          next(resnet(hidden,value.resnet,Some(time)))
          value.attention.foreach(a=>next(spatialTransformer(hidden,a,context)))
        val condIn=use(conv(image,model.conditionEmbedding.convIn))
        val condAct=use(session.silu(condIn));condIn.close()
        var cond=condAct
        model.conditionEmbedding.blocks.foreach { block =>
          val convolved=use(conv(cond,block));cond.close()
          cond=use(session.silu(convolved));convolved.close()
        }
        val condition=use(conv(cond,model.conditionEmbedding.convOut));cond.close()
        next(session.add(hidden,condition));condition.close()
        outputs+=session.reshape(hidden,hidden.shape)
        model.downBlocks.foreach { block =>
          block.layers.foreach { unit => layer(unit);outputs+=session.reshape(hidden,hidden.shape) }
          block.downsample.foreach { down => next(downsample(hidden,down));outputs+=session.reshape(hidden,hidden.shape) }
        }
        next(resnet(hidden,model.midBlock.first,Some(time)))
        next(Pag.withSection("mid")(spatialTransformer(hidden,model.midBlock.attention,context)))
        next(resnet(hidden,model.midBlock.second,Some(time)))
        require(outputs.size==model.downProjections.size,"ControlNet CUDA skip count mismatch")
        val down=outputs.indices.map { index =>
          val projected=use(conv(outputs(index),model.downProjections(index)))
          val scaled=use(session.scale(projected,scales(index)))
          val detached=session.reshape(scaled,scaled.shape)
          outputs(index).close()
          detached
        }.toVector
        outputs.clear()
        val projectedMiddle=use(conv(hidden,model.middleProjection))
        val scaledMiddle=use(session.scale(projectedMiddle,scales.last))
        outputMiddle=Some(session.reshape(scaledMiddle,scaledMiddle.shape))
        down
      }.get
      val middle=outputMiddle.getOrElse(throw IllegalStateException("ControlNet CUDA middle residual missing"))
      check()
      CudaControlNetResiduals(result,middle)
    catch
      case error: Throwable =>
        outputs.foreach(t=>try t.close() catch case cleanup: Throwable=>error.addSuppressed(cleanup))
        outputMiddle.foreach(t=>try t.close() catch case cleanup: Throwable=>error.addSuppressed(cleanup))
        throw error

  /** Float32 in, Float32 out; runs in Float16 inside when this runtime was uploaded with half=true. */
  def unet(input: CudaTensor,model: ConditionalUNet2d,timesteps: Tensor[Float],context: CudaTensor,
      added: Option[CudaTensor]=None,control: Option[CudaControlNetResiduals]=None): CudaTensor =
    if !half then unetCore(input,model,timesteps,context,added,control)
    else scope { use =>
      def h(t: CudaTensor): CudaTensor = if t.half then t else use(session.castToHalf(t))
      val residuals=control.map(c => CudaControlNetResiduals(c.down.map(h),h(c.middle)))
      val out=use(unetCore(h(input),model,timesteps,h(context),added.map(h),residuals))
      use(session.castToFloat(out))
    }

  private def unetCore(input: CudaTensor,model: ConditionalUNet2d,timesteps: Tensor[Float],context: CudaTensor,
      added: Option[CudaTensor],control: Option[CudaControlNetResiduals]): CudaTensor =
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
      val f=use(up(features))
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
      FooocusInpaint.current.foreach { fi =>
        val updated=use(up(FooocusInpaint.applyConvInResidual(input.toCpu,hidden.toCpu,fi)))
        hidden.close();hidden=updated
      }
      val skips=scala.collection.mutable.ArrayBuffer.empty[CudaTensor]
      var skipIndex=0
      // DeepCache: reuse the deep features recorded by the last full call (see DeepCache.scala).
      val dcState=if control.isEmpty then DeepCache.current else None
      val dcDims=session.shape(input).dims.toSeq
      val (dcSlot,dcReuse)=dcState.fold((0,false))(st => st.beginCall(timesteps.toFloatArray(0),dcDims))
      val dcBranch=dcState.fold(0)(_.config.branch)
      require(dcState.isEmpty || dcBranch<=model.downBlocks.size-2,"DeepCache branch is too deep for this U-Net")
      val dcUpKeep=model.upBlocks.size-1-dcBranch
      val dcNeededSkips=model.upBlocks.drop(math.max(dcUpKeep,0)).map(_.layers.size).sum
      control.foreach(c=>require(c.down.size==1+model.downBlocks.map(_.layers.size).sum+model.downBlocks.dropRight(1).size,
        "ControlNet down residual count mismatch"))
      def saveSkip(): Unit=
        val base=use(session.reshape(hidden,hidden.shape))
        val value=control.map { residuals =>
          val residual=residuals.down(skipIndex)
          require(session.shape(residual)==base.shape,"ControlNet down residual shape mismatch")
          val sum=use(session.add(base,residual));sum
        }.getOrElse(base)
        skips+=value
        skipIndex+=1
      def next(value: CudaTensor): Unit =
        val owned=use(value)
        hidden.close();hidden=owned
        check()
      def layer(block: DiffusionUNetLayer): Unit =
        next(resnet(hidden,block.resnet,Some(time)))
        block.attention.foreach(a => next(spatialTransformer(hidden,a,context)))
      saveSkip()
      val kohya=KohyaHrFix.current
      val kohyaLoc=kohya.flatMap { case (cfg,_) => KohyaHrFix.blockLocation(cfg.blockNumber,model.downBlocks.map(_.layers.size)) }
      def shrinkHiddenIfActive(): Unit =
        val (cfg,t)=kohya.get
        if KohyaHrFix.isActive(cfg,t) && cfg.downscaleFactor!=1.0 then
          next(up(KohyaHrFix.shrinkIfActive(hidden.toCpu,cfg,t)))
      model.downBlocks.zipWithIndex.foreach { (block,blockIdx) => if !(dcReuse && blockIdx>dcBranch) then Pag.withSection(s"down.$blockIdx") {
        block.layers.zipWithIndex.foreach { (l,layerIdx) =>
          layer(l)
          if kohyaLoc.contains((blockIdx,layerIdx)) then
            val (cfg,_)=kohya.get
            if !cfg.downscaleAfterSkip then shrinkHiddenIfActive()
            saveSkip()
            if cfg.downscaleAfterSkip then shrinkHiddenIfActive()
          else saveSkip()
        }
        block.downsample.foreach { d =>
          next(downsample(hidden,d))
          if kohyaLoc.contains((blockIdx,block.layers.size)) then
            val (cfg,_)=kohya.get
            if !cfg.downscaleAfterSkip then shrinkHiddenIfActive()
            saveSkip()
            if cfg.downscaleAfterSkip then shrinkHiddenIfActive()
          else saveSkip()
        } }
      }
      control.foreach(c=>require(skipIndex==c.down.size,
        s"ControlNet down residual count mismatch: expected $skipIndex, got ${c.down.size}"))
      if !dcReuse then
        next(resnet(hidden,model.midBlock.first,Some(time)))
        next(Pag.withSection("mid")(spatialTransformer(hidden,model.midBlock.attention,context)))
        next(resnet(hidden,model.midBlock.second,Some(time)))
        control.foreach { residuals =>
          val residual=residuals.middle
          require(session.shape(residual)==hidden.shape,"ControlNet middle residual shape mismatch")
          next(session.add(hidden,residual))
        }
      val freeU=FreeU.current
      model.upBlocks.zipWithIndex.foreach { (block,stageIndex) => if !(dcReuse && stageIndex<dcUpKeep) then Pag.withSection(s"up.$stageIndex") {
        dcState.foreach { st =>
          if stageIndex==dcUpKeep then
            if dcReuse then
              while skips.size>dcNeededSkips do skips.remove(skips.size-1).close()
              next(session.scale(st.feature(dcSlot),1f))
            else st.record(dcSlot,dcDims,session.scale(hidden,1f))
        }
        kohya.foreach { case (cfg,_) =>
          val startIdx=skips.size-block.layers.size
          val (_,maxH,maxW)=(startIdx until skips.size).map { k =>
            val d=skips(k).shape.dims
            (d(2)*d(3),d(2).toInt,d(3).toInt)
          }.maxBy(_._1)
          for k <- startIdx until skips.size do
            val d=skips(k).shape.dims
            if d(2).toInt!=maxH || d(3).toInt!=maxW then
              val old=skips(k)
              skips(k)=use(up(KohyaHrFix.interpolate2d(old.toCpu,maxH,maxW,cfg.upscaleMethod)))
              old.close()
          val hd=hidden.shape.dims
          if hd(2).toInt!=maxH || hd(3).toInt!=maxW then
            next(up(KohyaHrFix.interpolate2d(hidden.toCpu,maxH,maxW,cfg.upscaleMethod)))
        }
        block.layers.foreach { l =>
          val skip=skips.remove(skips.size-1)
          freeU match
            case Some(cfg) if stageIndex==0 || stageIndex==1 =>
              val (scaledHidden,filteredSkip)=FreeU(cfg,stageIndex,hidden.toCpu,skip.toCpu)
              skip.close()
              val devHidden=use(up(scaledHidden))
              val devSkip=use(up(filteredSkip))
              next(session.cat(Seq(devHidden,devSkip),1))
              devHidden.close();devSkip.close()
            case _ =>
              next(session.cat(Seq(hidden,skip),1));skip.close()
          layer(l)
        }
        block.upsample.foreach { up =>
          val target=skips.last.shape
          next(session.resizeNearest2d(hidden,target.dims(2).toInt,target.dims(3).toInt))
          next(conv(hidden,up.conv))
        }
      } }
      require(skips.isEmpty,"U-Net leaves unused skip tensors")
      time.close()
      val preOutCpu=Llul.current.filter(ctx => ctx.isActive && ctx.cfg.applyTo.contains("out")).flatMap(ctx => Llul.cutRegion(hidden.toCpu,ctx.cfg,ctx.fx,ctx.fy).map(part => (ctx,part)))
      next(norm(hidden,model.normOut))
      next(session.silu(hidden))
      next(conv(hidden,model.convOut))
      preOutCpu.foreach { case (ctx,part) =>
        val pGpu=use(up(part))
        val pNorm=use(norm(pGpu,model.normOut))
        val pAct=use(session.silu(pNorm))
        val pConv=use(conv(pAct,model.convOut))
        next(up(Llul.mixRegion(hidden.toCpu,pConv.toCpu,ctx.cfg,ctx.fx,ctx.fy)))
      }
      hidden
    }

  /** Float32 in and out; Float16 inside when this runtime was uploaded with half=true. */
  def encode(input: CudaTensor,model: VaeEncoder): CudaTensor =
    if !half then encodeCore(input,model)
    else scope { use =>
      val out=use(encodeCore(if input.half then input else use(session.castToHalf(input)),model))
      use(session.castToFloat(out))
    }

  private def encodeCore(input: CudaTensor,model: VaeEncoder): CudaTensor =
    check()
    model.preflight(session.shape(input))
    scope { use =>
      var hidden=use(conv(input,model.convIn))
      FooocusInpaint.current.foreach { fi =>
        val updated=use(up(FooocusInpaint.applyConvInResidual(input.toCpu,hidden.toCpu,fi)))
        hidden.close();hidden=updated
      }
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

  /** Float32 in and out; Float16 inside when this runtime was uploaded with half=true. */
  def decode(input: CudaTensor,model: VaeDecoder): CudaTensor =
    if !half then decodeCore(input,model)
    else scope { use =>
      val out=use(decodeCore(if input.half then input else use(session.castToHalf(input)),model))
      use(session.castToFloat(out))
    }

  private def decodeCore(input: CudaTensor,model: VaeDecoder): CudaTensor =
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

  private def clipActivation(x: CudaTensor,activation: ClipActivation): CudaTensor = activation match
    case ClipActivation.QuickGelu => session.quickGelu(x)
    case ClipActivation.Gelu => session.geluErf(x)
    case ClipActivation.GeluTanh => throw UnsupportedOperationException("CLIP tanh GELU has no CUDA kernel")

  /** One pre-norm causal CLIP layer on a (batch,tokens,width) activation; the caller keeps ownership of `input`. */
  def clipLayer(input: CudaTensor,layer: ClipTextLayer,maxScoreElements: Long): CudaTensor = scope { use =>
    val shape=session.shape(input)
    val b=shape.dims(0);val t=shape.dims(1);val width=layer.width.toLong
    val rows=Shape.from(Seq(b*t,width))
    val headDim=width/layer.heads
    val normalized=use(layerNorm(input,layer.norm1))
    val flat=use(session.reshape(normalized,rows));normalized.close()
    def heads(l: Linear): CudaTensor =
      val projected=use(linear(flat,l))
      val view=use(session.reshape(projected,Shape.from(Seq(b,t,layer.heads.toLong,headDim))));projected.close()
      val out=use(session.transpose(view,1,2));view.close()
      out
    val q=heads(layer.qProj);val k=heads(layer.kProj);val v=heads(layer.vProj)
    flat.close()
    val attended=use(session.causalAttention(q,k,v,0,maxScoreElements))
    q.close();k.close();v.close()
    val packed=use(session.transpose(attended,1,2));attended.close()
    val packedFlat=use(session.reshape(packed,rows));packed.close()
    val projected=use(linear(packedFlat,layer.outProj));packedFlat.close()
    val projectedView=use(session.reshape(projected,shape));projected.close()
    val hidden=use(session.add(input,projectedView));projectedView.close()
    val norm2=use(layerNorm(hidden,layer.norm2))
    val flat2=use(session.reshape(norm2,rows));norm2.close()
    val up=use(linear(flat2,layer.fc1));flat2.close()
    val activated=use(clipActivation(up,layer.activation));up.close()
    val down=use(linear(activated,layer.fc2));activated.close()
    val downView=use(session.reshape(down,shape));down.close()
    val result=use(session.add(hidden,downView))
    hidden.close();downView.close()
    result
  }

  /** Runs the whole stack; consumes `embedded`. selected is the state after `selectedDepth` layers. */
  def clipStates(model: ClipStack,embedded: CudaTensor,selectedDepth: Int,maxScoreElements: Long): ClipCuda.States =
    var hidden=if half && !embedded.half then
      val converted=session.castToHalf(embedded)
      embedded.close()
      converted
    else embedded
    var selected: Option[Tensor[Float]]=if selectedDepth==0 then Some(hidden.toCpu) else None
    try
      model.layers.zipWithIndex.foreach { (layer,index) =>
        val next=clipLayer(hidden,layer,maxScoreElements)
        hidden.close();hidden=next
        if index+1==selectedDepth then selected=Some(hidden.toCpu)
      }
      val normalized=layerNorm(hidden,model.finalNorm)
      try ClipCuda.States(selected.getOrElse(hidden.toCpu),normalized.toCpu)
      finally normalized.close()
    finally hidden.close()

  def close(): Unit =
    if !closed then
      session.allocatedBytes
      if profileEnabled && (offloadedWeightUploads>0 || offloadedWeightCacheHits>0) then
        System.err.println(f"[scala-torch CUDA profile] offload cache hits=$offloadedWeightCacheHits " +
          f"hit=${offloadedWeightCacheHitBytes/1048576.0}%.1f MiB uploads=$offloadedWeightUploads " +
          f"uploaded=${offloadedWeightUploadBytes/1048576.0}%.1f MiB retained=${offloadedWeightBytes/1048576.0}%.1f MiB " +
          f"transient=$offloadedWeightTransientUploads evictions=$offloadedWeightEvictions")
      weights.values.foreach(_.close())
      mergedWeights.values.foreach(_._2.close());mergedWeights.clear()
      clearOffloadedWeights()
      host=Map.empty
      closed=true

private[models] object CudaDiffusionRuntime:
  private val MaximumOffloadedWeightBytes=512L*1024L*1024L
  /** Precision chosen for the loaded model (`load_model` dtype). None = follow the environment default. */
  @volatile private var precisionOverride: Option[Boolean]=None
  private[models] def environmentFp16: Boolean = !sys.env.get("SCALATORCH_CUDA_FP16").contains("0")
  private[models] def setHalfPrecision(value: Option[Boolean]): Unit = precisionOverride=value
  /** Float16 U-Net / VAE / CLIP (weights, activations, tensor-core GEMM): the dtype of the loaded model (`load_model`), else
    * Float16 unless `SCALATORCH_CUDA_FP16=0`. CPU execution is always Float32. */
  private[models] def fp16Default: Boolean = precisionOverride.getOrElse(environmentFp16) && halfPossible
  /** Float16 needs cuBLAS. Without it the request used to fail in the middle ("broadcast add needs one element type"); now the
    * model runs in Float32 and says once what is missing. */
  private lazy val halfPossible: Boolean =
    val found=CudaSession.cublasLoadable
    if !found then System.err.println("[scala-torch] cuBLAS (cublas64_12.dll) was not found, so Float16 is not available: running in " +
      "Float32, which is slower and needs more memory. Install the CUDA Toolkit, or point SCALATORCH_CUDNN_PATH (Settings > scala-torch " +
      "cuDNN folder) at a folder that holds cuBLAS, for example a PyTorch torch\\lib folder.")
    found
  private[models] def offloadedWeightLimit(sessionMaxBytes: Long): Long =
    require(sessionMaxBytes>=0,"negative CUDA session budget")
    math.min(sessionMaxBytes/4,MaximumOffloadedWeightBytes)
  /** Room for activations that weights must leave free: SDXL at 1024 x 1024 needs about 0.75 GiB of chunks. */
  private[models] val ActivationReserve=1280L*1024L*1024L
  /** Offloaded weights that stay on the device for the whole run: as many as fit beside a fixed reserve for the activations. The rest is
    * sent again on every use, so a model that is larger than the free memory still runs, at the cost of the transfers.
    * (Giving up kept weights when the activations needed the room kept more of them, but changed the images: not used.) */
  private[models] def offloadedWeightLimit(sessionMaxBytes: Long,availableBytes: Long): Long =
    math.max(offloadedWeightLimit(sessionMaxBytes),math.min(sessionMaxBytes,availableBytes)-ActivationReserve)
  /** True when the weights of `model` do not fit the device beside the activations: they are then offloaded (see above). */
  private[models] def shouldOffload[A: Module](session: CudaSession,model: A,half: Boolean,label: String): Boolean =
    val parameters=Module.parameters(model).distinct
    val unit=if half then 2L else 4L
    val total=parameters.iterator.map(p => math.max(p.data.shape.numel*unit,1L)).sum
    val missing=total-session.sharedInUseBytes(parameters,half)
    val available=session.availableBytes
    val offload=missing>0 && missing+ActivationReserve>available
    if offload then
      CudaSession.markMemoryTight()
      System.err.println(s"[scala-torch] $label weights (${total/1048576} MiB) do not fit the free device memory (${available/1048576} MiB): " +
        s"keeping up to ${math.max(0L,offloadedWeightLimit(session.maxBytes,available))/1048576} MiB on the device and sending the rest on every use")
    offload

  def upload[A: Module](session: CudaSession,model: A,cancelled: () => Boolean=() => false,offloadWeights: Boolean=false,
      includeAdapters: Boolean=true,half: Boolean=false): CudaDiffusionRuntime =
    session.allocatedBytes
    val uploadStart=System.nanoTime()
    val host=scala.collection.mutable.LinkedHashMap.empty[Parameter,Tensor[Float]]
    val uploaded=scala.collection.mutable.LinkedHashMap.empty[Parameter,CudaTensor]
    try
      val base=Module.parameters(model).distinct
      val baseSet=base.toSet
      // Model weights stay on the device between sessions (shared); LoRA adapters differ per request and are uploaded privately.
      base.foreach { p =>
        VaeExecution.checkCancelled(cancelled)
        if offloadWeights then host(p)=p.data else uploaded(p)=session.sharedUpload(p,p.data,half,if half then Some(() => HalfWeightCache.bits(p,p.data)) else None)
      }
      (if includeAdapters then LinearAdapterScope.parameters.filterNot(baseSet.contains) else Vector.empty).distinct.foreach { p =>
        VaeExecution.checkCancelled(cancelled)
        if offloadWeights then host(p)=p.data else uploaded(p)=(if half then session.uploadHalf(p.data) else session.upload(p.data))
      }
      VaeExecution.checkCancelled(cancelled)
      // A Float16 upload stages every tensor as Float32 on the device first; those blocks are of no use to the sampling that follows.
      if !offloadWeights then session.trimFreed()
      if sys.env.get("SCALATORCH_LOAD_PROFILE").contains("1") then
        System.err.println(f"[scala-torch load] CUDA weight upload (${model.getClass.getSimpleName}, half=$half): ${(System.nanoTime()-uploadStart)/1e9}%.2f s, ${uploaded.size} tensors")
      new CudaDiffusionRuntime(session,uploaded.toMap,host.toMap,cancelled,half)
    catch case error: Throwable =>
      uploaded.values.toVector.reverse.foreach { t =>
        try t.close() catch case cleanup: Throwable => error.addSuppressed(cleanup)
      }
      throw error

/** Float16 copies of offloaded weights in native memory, converted once per parameter and reused by every later upload (a low-VRAM run uploads
  * the weights on every use). Entries disappear with their parameter; the memory is GC-managed (`Arena.ofAuto`). */
private[models] object HalfWeightCache:
  private val cache=new java.util.WeakHashMap[Parameter,java.lang.foreign.MemorySegment]()
  private val arena=java.lang.foreign.Arena.ofAuto()
  def bits(p: Parameter,source: Tensor[Float]): java.lang.foreign.MemorySegment=
    cache.synchronized(cache.get(p)) match
      case null =>
        val values=source.toFloatArray
        val segment=arena.allocate(math.max(values.length.toLong*2,2L),2)
        val shorts=segment.asByteBuffer().order(java.nio.ByteOrder.nativeOrder()).asShortBuffer()
        if values.length<65536 then
          var i=0;while i<values.length do { shorts.put(i,java.lang.Float.floatToFloat16(values(i)));i+=1 }
        else
          val block=32768
          java.util.stream.IntStream.range(0,(values.length+block-1)/block).parallel().forEach { (b: Int) =>
            val start=b*block;val end=math.min(values.length,start+block)
            var i=start;while i<end do { shorts.put(i,java.lang.Float.floatToFloat16(values(i)));i+=1 }
          }
        cache.synchronized { cache.put(p,segment) }
        segment
      case found => found

package scalatorch.models

import scalatorch.*
import scalatorch.nn.*
import scala.collection.mutable
import scala.util.Using

/** Owns backend handles and supports explicitly transferring a result to its caller. */
private[models] final class BackendScope[T](backend: FloatBackend[T]) extends AutoCloseable:
  private val owned=mutable.ArrayBuffer.empty[T]
  def keep(value: T): T = { owned+=value;value }
  def take(value: T): T =
    val index=owned.lastIndexOf(value)
    require(index>=0,"backend handle is not owned by this scope")
    owned.remove(index);value
  def release(value: T): Unit = backend.release(take(value))
  def close(): Unit =
    var failure: Throwable=null
    owned.reverseIterator.foreach { value =>
      try backend.release(value) catch case error: Throwable =>
        if failure==null then failure=error else failure.addSuppressed(error)
    }
    owned.clear()
    if failure!=null then throw failure

/** Prepared, thread-confined inference over resident weights. No autograd. */
final class BackendLlama[T] private (private[models] val model: LlamaModel,private[models] val backend: FloatBackend[T],
    weights: Map[(Parameter,Boolean),T],resources: BackendScope[T]) extends AutoCloseable:
  private val owner=Thread.currentThread()
  private val snapshot=Module.parameters(model).map(p => p -> p.value)
  private var closed=false
  private val caches=mutable.LinkedHashSet.empty[BackendLlamaCache[T]]
  private val c=model.config
  private[models] def checkThread(): Unit =
    if Thread.currentThread() ne owner then throw IllegalStateException("prepared Llama used from another thread")
  private[models] def check(): Unit =
    checkThread()
    if closed then throw IllegalStateException("prepared Llama is closed")
    require(snapshot.forall((p,v) => p.value eq v),"model weights changed; prepare again")
  private def weight(p: Parameter,transposed: Boolean = false): T = weights((p,transposed))
  private def linear(input: T,layer: Linear): T = backend.matmul(input,weight(layer.weight,true))
  private def norm(input: T,layer: RMSNorm): T = backend.rmsNorm(input,weight(layer.weight),layer.epsilon)

  private def attention(input: T,layer: LlamaAttention,b: Long,t: Long,position: Int,cache: Option[AttentionCache[T]]): T =
    Using.resource(new BackendScope(backend)) { scope =>
      def keep(value: T) = scope.keep(value)
      val flat=keep(backend.reshape(input,Shape.from(Seq(b*t,c.hiddenSize.toLong))))
      def heads(projection: Linear,count: Int): T =
        val projected=keep(linear(flat,projection))
        val split=keep(backend.reshape(projected,Shape.from(Seq(b,t,count.toLong,c.headDim.toLong))))
        keep(backend.transpose(split,1,2))
      val q=keep(backend.rope(heads(layer.qProj,c.numAttentionHeads),positionStart=position,theta=c.ropeTheta))
      val k=keep(backend.rope(heads(layer.kProj,c.numKeyValueHeads),positionStart=position,theta=c.ropeTheta))
      val v=heads(layer.vProj,c.numKeyValueHeads)
      val attended=keep(cache.fold(backend.causalAttention(q,k,v,maxScoreElements=model.limits.maxScoreElements))(
        _.attend(q,k,v,position,model.limits.maxScoreElements)))
      val joined=keep(backend.transpose(attended,1,2))
      val matrix=keep(backend.reshape(joined,Shape.from(Seq(b*t,c.hiddenSize.toLong))))
      val projected=keep(linear(matrix,layer.oProj))
      scope.take(keep(backend.reshape(projected,Shape.from(Seq(b,t,c.hiddenSize.toLong)))))
    }
  private def mlp(input: T,layer: LlamaMLP,b: Long,t: Long): T =
    Using.resource(new BackendScope(backend)) { scope =>
      def keep(value: T) = scope.keep(value)
      val flat=keep(backend.reshape(input,Shape.from(Seq(b*t,c.hiddenSize.toLong))))
      val gate=keep(linear(flat,layer.gateProj))
      val activated=keep(backend.silu(gate))
      val up=keep(linear(flat,layer.upProj))
      val product=keep(backend.multiply(activated,up))
      val down=keep(linear(product,layer.downProj))
      scope.take(keep(backend.reshape(down,Shape.from(Seq(b,t,c.hiddenSize.toLong)))))
    }
  private def block(input: T,layer: LlamaBlock,b: Long,t: Long,position: Int,cache: Option[AttentionCache[T]]): T =
    Using.resource(new BackendScope(backend)) { scope =>
      def keep(value: T) = scope.keep(value)
      val normalized=keep(norm(input,layer.inputNorm))
      val attended=keep(attention(normalized,layer.attention,b,t,position,cache))
      val hidden=keep(backend.add(input,attended))
      val post=keep(norm(hidden,layer.postNorm))
      val feedForward=keep(mlp(post,layer.mlp,b,t))
      scope.take(keep(backend.add(hidden,feedForward)))
    }

  /** Caller owns the returned backend tensor. All intermediate activations are released. */
  def forward(tokens: Tensor[Long]): T = run(tokens,0,None)
  private def run(tokens: Tensor[Long],position: Int,buffers: Option[Vector[AttentionCache[T]]]): T =
    check()
    require(tokens.shape.rank==2,"Llama tokens must have shape (batch,tokens)")
    val b=tokens.shape.dims(0);val t=tokens.shape.dims(1)
    require(position>=0 && position.toLong+t<=c.maxPositions,"Llama context limit exceeded")
    val output=Shape.from(Seq(b,t,c.vocabSize.toLong))
    require(output.numel<=model.limits.maxLogitsElements && output.numel<=Int.MaxValue,"Llama logits allocation limit exceeded")
    Using.resource(new BackendScope(backend)) { scope =>
      var hidden=scope.keep(backend.embedding(weight(model.embedding.weight),tokens))
      model.layers.zipWithIndex.foreach { (layer,index) =>
        val next=scope.keep(block(hidden,layer,b,t,position,buffers.map(_(index))))
        scope.release(hidden);hidden=next
      }
      val normalized=scope.keep(norm(hidden,model.norm))
      val flat=scope.keep(backend.reshape(normalized,Shape.from(Seq(b*t,c.hiddenSize.toLong))))
      val logits=scope.keep(linear(flat,model.lmHead))
      scope.take(scope.keep(backend.reshape(logits,output)))
    }
  def logits(tokens: Tensor[Long]): Tensor[Float] =
    download(forward(tokens))
  private def download(result: T): Tensor[Float] =
    Using.resource(new AutoCloseable { def close(): Unit = backend.release(result) }) { _ => backend.download(result) }
  private[models] def cachedLogits(tokens: Tensor[Long],position: Int,buffers: Vector[AttentionCache[T]]): Tensor[Float] =
    download(run(tokens,position,Some(buffers)))
  def newCache(batchSize: Int = 1,capacity: Int = c.maxPositions,maxBytes: Long = 268435456L): BackendLlamaCache[T] =
    check()
    val cache=new BackendLlamaCache(this,batchSize,capacity,maxBytes)
    caches+=cache
    cache
  private[models] def cacheClosed(cache: BackendLlamaCache[T]): Unit = { caches-=cache;() }

  /** Cached decode by default; full-prefix mode remains available for comparisons. */
  def generateIds(prompt: Seq[Long],maxNewTokens: Int,sampling: SamplingConfig = SamplingConfig(),useCache: Boolean = true): Vector[Long] =
    check()
    require(prompt.nonEmpty && maxNewTokens>=0 && prompt.length.toLong+maxNewTokens<=c.maxPositions,"invalid generation length")
    require(prompt.forall(id => id>=0 && id<c.vocabSize),"generation token outside vocabulary")
    if maxNewTokens==0 then return prompt.toVector
    Using.Manager { use =>
      val cache=if useCache then Some(use(newCache(capacity=prompt.length+maxNewTokens))) else None
      val sampler=new TokenSampler(sampling)
      var ids=prompt.toVector;var chunk=ids;var generated=0;var stopped=false
      while generated<maxNewTokens && !stopped do
        val current=if useCache then chunk else ids
        val input=Tensor.fromLongs(current.toArray,Shape.from(Seq(1L,current.size.toLong)))
        val next=sampler.sample(cache.fold(logits(input))(_.decode(input)).toFloatArray.takeRight(c.vocabSize))
        ids=ids :+ next;chunk=Vector(next);generated+=1;stopped=c.eosTokenIds.contains(next)
      ids
    }.get
  def close(): Unit =
    checkThread()
    if !closed then
      closed=true
      Using.Manager { use =>
        use(resources)
        caches.toVector.foreach(cache => use(cache))
        ()
      }.get

object BackendLlama:
  def prepare[T](model: LlamaModel,backend: FloatBackend[T]): BackendLlama[T] =
    require(model.layers.size==model.config.numLayers && model.layers.size<=model.limits.maxLayers,"Llama layer count/limit mismatch")
    val projections=model.layers.flatMap { l => Vector(l.attention.qProj,l.attention.kProj,l.attention.vProj,l.attention.oProj,
      l.mlp.gateProj,l.mlp.upProj,l.mlp.downProj) } :+ model.lmHead
    require(projections.forall(p => p.bias.isEmpty && p.lora.isEmpty),"prepared Llama currently requires bias-free projections without adapters")
    require(Module.parameters(model).map(p => BigInt(p.data.shape.numel)).sum<=model.limits.maxParameters,"Llama parameter limit exceeded")
    val resources=new BackendScope(backend)
    val weights=mutable.Map.empty[(Parameter,Boolean),T]
    def upload(p: Parameter,transposed: Boolean = false): Unit =
      weights.getOrElseUpdate((p,transposed),resources.keep(backend.upload(if transposed then p.data.transpose(0,1) else p.data)))
      ()
    try
      upload(model.embedding.weight);upload(model.norm.weight)
      model.layers.foreach(l => { upload(l.inputNorm.weight);upload(l.postNorm.weight) })
      projections.foreach(p => upload(p.weight,true))
      new BackendLlama(model,backend,weights.toMap,resources)
    catch case error: Throwable =>
      try resources.close() catch case cleanup: Throwable => error.addSuppressed(cleanup)
      throw error

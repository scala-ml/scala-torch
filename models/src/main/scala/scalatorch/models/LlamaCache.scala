package scalatorch.models

import scalatorch.*
import scalatorch.autograd.{Autograd,Variable}
import scalatorch.nn.{Module,Parameter}

/** Model-bound CPU inference session. Mutable K/V arrays never escape through its public API. */
final class LlamaCache private[models] (model: LlamaModel, val batchSize: Int, val capacity: Int, maxBytes: Long)
    extends AutoCloseable:
  private val config = model.config
  require(batchSize>0 && capacity>0 && capacity<=config.maxPositions && maxBytes>=0, "invalid cache batch/capacity/budget")
  require(model.layers.size==config.numLayers,"model layer count mismatch")
  private val perArray = BigInt(batchSize)*config.numKeyValueHeads*capacity*config.headDim
  private val bytes = perArray*2*config.numLayers*4
  require(perArray<=Int.MaxValue && bytes<=maxBytes,"KV cache allocation budget exceeded")
  private var buffers = Vector.fill(config.numLayers)(new LlamaCache.Layer(batchSize,config.numKeyValueHeads,capacity,config.headDim))
  private var saved = snapshot()
  private var used = 0
  private var closed = false
  private def snapshot(): Vector[(Parameter,Variable)] = Module.parameters(model).map(p => p -> p.value)
  private def checkOpen(): Unit = if closed then throw IllegalStateException("Llama cache is closed")
  private def checkWeights(): Unit = require(saved.forall((p,v) => p.value eq v),"model weights changed; reset the cache before decoding")
  def position: Int = synchronized { checkOpen(); used }
  def allocatedBytes: Long = synchronized { if closed then 0L else bytes.toLong }

  def decode(tokens: Tensor[Long]): Tensor[Float] = synchronized {
    checkOpen(); checkWeights()
    require(tokens.shape.rank==2 && tokens.shape.dims(0)==batchSize && tokens.shape.dims(1)>0,"cache tokens must match batch with positive chunk length")
    val count = tokens.shape.dims(1)
    val end = used.toLong+count
    require(end<=capacity,"KV cache capacity exceeded")
    val outputShape = Shape.from(Seq(batchSize.toLong,count,config.vocabSize.toLong))
    require(outputShape.numel<=model.limits.maxLogitsElements && outputShape.numel<=Int.MaxValue,"cached logits allocation limit exceeded")
    val scores = BigInt(batchSize)*config.numAttentionHeads*count*end
    require(scores<=model.limits.maxScoreElements && scores<=Int.MaxValue,"cached attention workspace limit exceeded")
    val result = Autograd.noGrad {
      var hidden = model.embedding(tokens)
      model.layers.zip(buffers).foreach { (layer,cache) =>
        hidden = layer.cached(hidden,config,model.limits.maxScoreElements,used,cache)
      }
      val normalized = model.norm(hidden).reshape(Shape.from(Seq(batchSize.toLong*count,config.hiddenSize.toLong)))
      model.lmHead(normalized).reshape(outputShape).data
    }
    checkWeights()
    used = end.toInt // publish only after the entire chunk succeeds
    result
  }

  def reset(): Unit = synchronized {
    checkOpen()
    used=0
    saved=snapshot() // old tails are invisible and overwritten by the next append
  }
  override def close(): Unit = synchronized {
    if !closed then
      buffers=Vector.empty
      saved=Vector.empty
      closed=true
  }

object LlamaCache:
  private[models] final class Layer(batch: Int, heads: Int, capacity: Int, width: Int):
    private val size = (batch.toLong*heads*capacity*width).toInt // checked by owner before allocation
    private val keys = new Storage(new Array[Float](size),DType.Float32)
    private val values = new Storage(new Array[Float](size),DType.Float32)
    private val strides = Vector(heads.toLong*capacity*width,capacity.toLong*width,width.toLong,1L)
    def append(key: Tensor[Float], value: Tensor[Float], position: Int): (Tensor[Float],Tensor[Float]) =
      require(key.shape.rank==4 && key.shape.dims(0)==batch && key.shape.dims(1)==heads && key.shape.dims(3)==width && value.shape==key.shape,"KV append shape mismatch")
      val count = key.shape.dims(2).toInt
      require(position>=0 && position.toLong+count<=capacity,"KV append exceeds capacity")
      val k = key.toFloatArray; val v = value.toFloatArray
      for b <- 0 until batch; h <- 0 until heads do
        val from = (b*heads+h)*count*width
        val to = ((b*heads+h)*capacity+position)*width
        System.arraycopy(k,from,keys.data,to,count*width)
        System.arraycopy(v,from,values.data,to,count*width)
      val shape = Shape.from(Seq(batch.toLong,heads.toLong,position.toLong+count,width.toLong))
      (new Tensor(keys,Layout.checked(shape,strides,0L,size)),new Tensor(values,Layout.checked(shape,strides,0L,size)))

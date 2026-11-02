package scalatorch.models

import scalatorch.*
import scala.collection.mutable
import scala.util.Using

/** Model-bound cache; only a fully successful decoded chunk advances position. */
final class BackendLlamaCache[T] private[models] (prepared: BackendLlama[T],val batchSize: Int,
    val capacity: Int,maxBytes: Long) extends AutoCloseable:
  private val c=prepared.model.config
  require(batchSize>0 && capacity>0 && capacity<=c.maxPositions && maxBytes>=0,"invalid model cache dimensions/budget")
  private val perArray=BigInt(batchSize)*c.numKeyValueHeads*capacity*c.headDim
  require(perArray<=Int.MaxValue && perArray*2*c.numLayers*4<=maxBytes,"model cache allocation budget exceeded")
  private var used=0
  private var closed=false
  private val buffers =
    val acquired=mutable.ArrayBuffer.empty[AttentionCache[T]]
    try
      for _ <- 0 until c.numLayers do
        acquired+=prepared.backend.attentionCache(batchSize,c.numKeyValueHeads,capacity,c.headDim,c.headDim,maxBytes)
      acquired.toVector
    catch case error: Throwable =>
      acquired.reverseIterator.foreach { cache =>
        try cache.close() catch case cleanup: Throwable => error.addSuppressed(cleanup)
      }
      throw error
  private def checkOpen(): Unit =
    prepared.checkThread()
    if closed then throw IllegalStateException("model cache is closed")
  def position: Int = { checkOpen();used }
  def decode(tokens: Tensor[Long]): Tensor[Float] =
    checkOpen();prepared.check()
    require(tokens.shape.rank==2 && tokens.shape.dims(0)==batchSize && tokens.shape.dims(1)>0,"cache tokens must match batch with positive chunk length")
    val count=tokens.shape.dims(1);val end=used.toLong+count
    require(end<=capacity,"model cache capacity exceeded")
    val scores=BigInt(batchSize)*c.numAttentionHeads*count*end
    require(scores<=prepared.model.limits.maxScoreElements && scores<=Int.MaxValue,"cached score workspace limit exceeded")
    val result=prepared.cachedLogits(tokens,used,buffers)
    prepared.check()
    used=end.toInt
    result
  def reset(): Unit =
    checkOpen();prepared.check();used=0
  def close(): Unit =
    prepared.checkThread()
    if !closed then
      closed=true
      try Using.Manager { use => buffers.foreach(cache => use(cache));() }.get
      finally prepared.cacheClosed(this)

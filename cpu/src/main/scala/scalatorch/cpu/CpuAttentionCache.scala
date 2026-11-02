package scalatorch.cpu

import scalatorch.*

private[cpu] final class CpuAttentionCache(layout: CacheLayout) extends AttentionCache[Tensor[Float]]:
  private val owner=Thread.currentThread()
  private var closed=false
  private var written=0
  private var keys=Option(new Storage(new Array[Float](layout.keyShape(layout.capacity).numel.toInt),DType.Float32))
  private var values=Option(new Storage(new Array[Float](layout.valueShape(layout.capacity).numel.toInt),DType.Float32))
  private def checkThread(): Unit =
    if Thread.currentThread() ne owner then throw IllegalStateException("attention cache used from another thread")
  def attend(q: Tensor[Float],k: Tensor[Float],v: Tensor[Float],position: Int,maxScoreElements: Long = 16777216L): Tensor[Float] =
    checkThread()
    if closed then throw IllegalStateException("attention cache is closed")
    val (_,_,end)=layout.validate(q.shape,k.shape,v.shape,position,written,maxScoreElements)
    val chunk=k.shape.dims(2).toInt
    def append(input: Tensor[Float],storage: Storage[Float],width: Int): Tensor[Float] =
      val source=input.toFloatArray
      for row <- 0 until layout.batch*layout.heads do
        System.arraycopy(source,row*chunk*width,storage.data,(row*layout.capacity+position)*width,chunk*width)
      val shape=Shape.from(Seq(layout.batch.toLong,layout.heads.toLong,end.toLong,width.toLong))
      val strides=Vector(layout.heads.toLong*layout.capacity*width,layout.capacity.toLong*width,width.toLong,1L)
      new Tensor(storage,Layout.checked(shape,strides,0L,storage.length))
    val key=append(k,keys.get,layout.keyWidth)
    val value=append(v,values.get,layout.valueWidth)
    val result=CpuAttention.causal(q,key,value,position,maxScoreElements)
    written=end
    result
  def close(): Unit =
    checkThread()
    keys=None;values=None;closed=true

package scalatorch.cpu

import scalatorch.*

/** Immutable CPU Float32 inference backend; release is a no-op for GC-owned storage. */
object CpuFloatBackend extends TrainingBackend[Tensor[Float]]:
  def crossEntropy(logits: Tensor[Float],targets: Tensor[Long],reduction: LossReduction = LossReduction.Mean,ignoreIndex: Long = -100L): Tensor[Float] =
    CpuCrossEntropy.forward(logits,targets,reduction,ignoreIndex)
  def crossEntropyBackward(logits: Tensor[Float],targets: Tensor[Long],gradient: Tensor[Float],
      reduction: LossReduction = LossReduction.Mean,ignoreIndex: Long = -100L): Tensor[Float] =
    CpuCrossEntropy.backward(logits,targets,gradient,reduction,ignoreIndex)
  def causalAttentionBackward(query: Tensor[Float],key: Tensor[Float],value: Tensor[Float],gradient: Tensor[Float],
      queryStart: Int = 0,maxScoreElements: Long = 16777216L): (Tensor[Float],Tensor[Float],Tensor[Float]) =
    CpuAttentionBackward(query,key,value,gradient,queryStart,maxScoreElements)
  def embeddingBackward(tableShape: Shape,indices: Tensor[Long],gradient: Tensor[Float]): Tensor[Float] =
    CpuBackward.embedding(tableShape,indices,gradient)
  def ropeBackward(gradient: Tensor[Float],positionStart: Int = 0,theta: Double = 10000.0): Tensor[Float] =
    CpuAttention.rope(gradient,positionStart,theta,inverse=true)
  def siluBackward(value: Tensor[Float],gradient: Tensor[Float]): Tensor[Float] = CpuBackward.silu(value,gradient)
  def rmsNormInputBackward(value: Tensor[Float],weight: Tensor[Float],gradient: Tensor[Float],epsilon: Double = 1e-6): Tensor[Float] =
    CpuBackward.rmsInput(value,weight,gradient,epsilon)
  def rmsNormWeightBackward(value: Tensor[Float],gradient: Tensor[Float],epsilon: Double = 1e-6): Tensor[Float] =
    CpuBackward.rmsWeight(value,gradient,epsilon)
  def fill(shape: Shape,value: Float): Tensor[Float] =
    require(shape.numel<=Int.MaxValue,"fill exceeds Int indexing")
    Tensor.fromFloats(Array.fill(shape.numel.toInt)(value),shape)
  def sum(value: Tensor[Float]): Tensor[Float] =
    Tensor.fromFloats(Array(value.toFloatArray.foldLeft(0.0)((a,b) => a+b).toFloat),Shape.from(Seq.empty))
  def expandScalar(value: Tensor[Float],shape: Shape): Tensor[Float] =
    require(value.shape.rank==0,"expandScalar requires rank zero")
    fill(shape,value.toFloatArray.head)
  def attentionCache(batchSize: Int,heads: Int,capacity: Int,keyWidth: Int,valueWidth: Int,
      maxBytes: Long = 268435456L): AttentionCache[Tensor[Float]] =
    new CpuAttentionCache(CacheLayout(batchSize,heads,capacity,keyWidth,valueWidth,maxBytes))
  def rope(value: Tensor[Float],positionStart: Int = 0,theta: Double = 10000.0): Tensor[Float] =
    CpuAttention.rope(value,positionStart,theta)
  def causalAttention(query: Tensor[Float],key: Tensor[Float],value: Tensor[Float],queryStart: Int = 0,
      maxScoreElements: Long = 16777216L): Tensor[Float] = CpuAttention.causal(query,key,value,queryStart,maxScoreElements)
  def upload(source: Tensor[Float]): Tensor[Float] = source
  def download(value: Tensor[Float]): Tensor[Float] = value
  def shape(value: Tensor[Float]): Shape = value.shape
  def release(value: Tensor[Float]): Unit = ()
  def reshape(value: Tensor[Float], shape: Shape): Tensor[Float] = value.contiguous.reshape(shape)
  def transpose(value: Tensor[Float], dim0: Int, dim1: Int): Tensor[Float] = value.transpose(dim0,dim1)
  def embedding(table: Tensor[Float], indices: Tensor[Long]): Tensor[Float] =
    require(table.shape.rank==2 && table.shape.dims(1)>0,"embedding requires a matrix with positive width")
    val ids=indices.toLongArray
    require(ids.forall(id => id>=0 && id<table.shape.dims(0)),"embedding token ID out of bounds")
    val output=Shape.from(indices.shape.dims :+ table.shape.dims(1))
    require(output.numel<=Int.MaxValue,"embedding output exceeds Int indexing")
    val source=table.toFloatArray
    val width=table.shape.dims(1).toInt
    val values=new Array[Float](output.numel.toInt)
    ids.indices.foreach(i => System.arraycopy(source,ids(i).toInt*width,values,i*width,width))
    Tensor.fromFloats(values,output)
  def matmul(a: Tensor[Float], b: Tensor[Float]): Tensor[Float] = NumscalaBackend.matmul(a,b)
  def add(a: Tensor[Float], b: Tensor[Float]): Tensor[Float] =
    require(a.shape==b.shape,"backend add requires equal shapes")
    Ops.add(a,b)
  def multiply(a: Tensor[Float], b: Tensor[Float]): Tensor[Float] =
    require(a.shape==b.shape,"backend multiply requires equal shapes")
    val x=a.toFloatArray;val y=b.toFloatArray
    Tensor.fromFloats(x.indices.map(i => x(i)*y(i)).toArray,a.shape)
  def scale(value: Tensor[Float], scalar: Float): Tensor[Float] =
    Tensor.fromFloats(value.toFloatArray.map(_ * scalar),value.shape)
  def silu(value: Tensor[Float]): Tensor[Float] =
    val out=value.toFloatArray.map { x =>
      val v=x.toDouble
      val sigmoid=if v>=0 then 1.0/(1.0+math.exp(-v)) else
        val e=math.exp(v);e/(1.0+e)
      (if v==Double.NegativeInfinity then -0.0 else v*sigmoid).toFloat
    }
    Tensor.fromFloats(out,value.shape)
  def rmsNorm(value: Tensor[Float], weight: Tensor[Float], epsilon: Double = 1e-6): Tensor[Float] =
    require(epsilon.isFinite && epsilon>0,"invalid RMSNorm epsilon")
    require(value.shape.rank>0 && value.shape.dims.last>0 && weight.shape==Shape.from(Seq(value.shape.dims.last)),"invalid RMSNorm shapes")
    val x=value.toFloatArray;val w=weight.toFloatArray;val width=w.length
    val out=new Array[Float](x.length)
    var start=0
    while start<x.length do
      var squares=0.0;var j=0
      while j<width do
        squares+=x(start+j).toDouble*x(start+j);j+=1
      val inverse=1.0/math.sqrt(squares/width+epsilon)
      j=0
      while j<width do
        out(start+j)=(x(start+j).toDouble*inverse*w(j)).toFloat;j+=1
      start+=width
    Tensor.fromFloats(out,value.shape)

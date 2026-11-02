package scalatorch.cpu

import scalatorch.*

/** Immutable CPU Float32 inference backend; release is a no-op for GC-owned storage. */
object CpuFloatBackend extends FloatBackend[Tensor[Float]]:
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

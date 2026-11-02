package scalatorch.cpu

import scalatorch.*

/** Immutable CPU Float32 inference backend; release is a no-op for GC-owned storage. */
object CpuFloatBackend extends FloatBackend[Tensor[Float]]:
  def upload(source: Tensor[Float]): Tensor[Float] = source
  def download(value: Tensor[Float]): Tensor[Float] = value
  def shape(value: Tensor[Float]): Shape = value.shape
  def release(value: Tensor[Float]): Unit = ()
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

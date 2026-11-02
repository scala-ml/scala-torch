package scalatorch.cpu

import scalatorch.*

private[cpu] object CpuBackward:
  def embedding(table: Shape,indices: Tensor[Long],gradient: Tensor[Float]): Tensor[Float] =
    val ids=EmbeddingGradient.indices(table,indices,gradient.shape)
    val out=new Array[Float](table.numel.toInt)
    if ids.nonEmpty then
      val width=table.dims(1).toInt;val g=gradient.toFloatArray
      for i <- ids.indices do
        var j=0
        while j<width do
          out(ids(i)*width+j)+=g(i*width+j);j+=1
    Tensor.fromFloats(out,table)
  def silu(value: Tensor[Float],gradient: Tensor[Float]): Tensor[Float] =
    require(value.shape==gradient.shape,"SiLU gradient shape mismatch")
    val x=value.toFloatArray;val g=gradient.toFloatArray
    val out=x.indices.map { i =>
      val v=x(i).toDouble
      val s=if v>=0 then 1.0/(1.0+math.exp(-v)) else
        val e=math.exp(v);e/(1.0+e)
      val derivative=if s==0 then 0.0 else if s==1 then 1.0 else s*(1.0+v*(1.0-s))
      (if derivative==0 then 0.0 else g(i).toDouble*derivative).toFloat
    }.toArray
    Tensor.fromFloats(out,value.shape)
  private def dimensions(value: Tensor[Float],gradient: Tensor[Float],epsilon: Double): (Int,Int) =
    require(value.shape==gradient.shape,"RMSNorm gradient shape mismatch")
    require(epsilon.isFinite && epsilon>0,"invalid RMSNorm epsilon")
    require(value.shape.rank>0 && value.shape.dims.last>0 && value.shape.dims.last<=Int.MaxValue,"RMSNorm requires positive Int-indexed final width")
    val width=value.shape.dims.last.toInt
    (width,value.shape.numel.toInt/width)
  private def inverse(x: Array[Float],start: Int,width: Int,epsilon: Double): Double =
    var squares=0.0;var j=0
    while j<width do
      squares+=x(start+j).toDouble*x(start+j);j+=1
    1.0/math.sqrt(squares/width+epsilon)
  def rmsInput(value: Tensor[Float],weight: Tensor[Float],gradient: Tensor[Float],epsilon: Double): Tensor[Float] =
    val (width,rows)=dimensions(value,gradient,epsilon)
    require(weight.shape==Shape.from(Seq(width.toLong)),"RMSNorm weight shape mismatch")
    val x=value.toFloatArray;val w=weight.toFloatArray;val g=gradient.toFloatArray
    val out=new Array[Float](x.length)
    for row <- 0 until rows do
      val offset=row*width;val inv=inverse(x,offset,width,epsilon)
      var dot=0.0;var j=0
      while j<width do
        dot+=g(offset+j).toDouble*w(j)*x(offset+j)*inv;j+=1
      j=0
      while j<width do
        out(offset+j)=(inv*(g(offset+j).toDouble*w(j)-x(offset+j).toDouble*inv*dot/width)).toFloat;j+=1
    Tensor.fromFloats(out,value.shape)
  def rmsWeight(value: Tensor[Float],gradient: Tensor[Float],epsilon: Double): Tensor[Float] =
    val (width,rows)=dimensions(value,gradient,epsilon)
    val x=value.toFloatArray;val g=gradient.toFloatArray;val out=new Array[Double](width)
    for row <- 0 until rows do
      val offset=row*width;val inv=inverse(x,offset,width,epsilon)
      var j=0
      while j<width do
        out(j)+=g(offset+j).toDouble*x(offset+j)*inv;j+=1
    Tensor.fromFloats(out.map(_.toFloat),Shape.from(Seq(width.toLong)))

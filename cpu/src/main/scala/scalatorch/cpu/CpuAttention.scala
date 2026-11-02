package scalatorch.cpu

import scalatorch.*

private[cpu] object CpuAttention:
  def rope(value: Tensor[Float],start: Int,theta: Double,inverse: Boolean = false): Tensor[Float] =
    val coefficients=RotaryCoefficients.values(value.shape,start,theta)
    if value.shape.numel==0 then return value
    val width=value.shape.dims(3).toInt;val half=width/2;val tokens=value.shape.dims(2).toInt
    val input=value.toFloatArray;val out=new Array[Float](input.length)
    for row <- 0 until input.length/width; d <- 0 until half do
      val offset=(row%tokens)*width+2*d
      val c=coefficients(offset);val s=coefficients(offset+1)*(if inverse then -1.0 else 1.0)
      val a=input(row*width+d).toDouble;val b=input(row*width+half+d).toDouble
      out(row*width+d)=(a*c-b*s).toFloat
      out(row*width+half+d)=(b*c+a*s).toFloat
    Tensor.fromFloats(out,value.shape)

  def causal(q: Tensor[Float],k: Tensor[Float],v: Tensor[Float],start: Int,maxScores: Long): Tensor[Float] =
    val (shape,scores)=AttentionLayout.validate(q.shape,k.shape,v.shape,start,maxScores)
    val out=new Array[Float](shape.numel.toInt)
    if out.isEmpty then return Tensor.fromFloats(out,shape)
    val Seq(batch,heads,tokens,width)=q.shape.dims.map(_.toInt): @unchecked
    val kvHeads=k.shape.dims(1).toInt;val keys=k.shape.dims(2).toInt;val vw=v.shape.dims(3).toInt
    val query=q.toFloatArray;val key=k.toFloatArray;val value=v.toFloatArray
    // One row of scratch suffices for CPU inference, but the shared logical score limit still applies.
    val probabilities=new Array[Double](keys);val scale=1.0/math.sqrt(width.toDouble)
    for b <- 0 until batch; h <- 0 until heads; t <- 0 until tokens do
      val row=(b*heads+h)*tokens+t
      val kv=(b*kvHeads+h/(heads/kvHeads))*keys
      val visible=math.min(keys.toLong,start.toLong+t+1).toInt
      var maximum=Double.NegativeInfinity;var j=0
      while j<visible do
        var dot=0.0;var d=0
        while d<width do
          dot+=query(row*width+d).toDouble*key((kv+j)*width+d);d+=1
        probabilities(j)=dot*scale;maximum=math.max(maximum,probabilities(j));j+=1
      var denominator=0.0;j=0
      while j<visible do
        probabilities(j)=math.exp(probabilities(j)-maximum);denominator+=probabilities(j);j+=1
      j=0
      while j<visible do
        probabilities(j)/=denominator;j+=1
      var d=0
      while d<vw do
        var sum=0.0;j=0
        while j<visible do
          sum+=probabilities(j)*value((kv+j)*vw+d);j+=1
        out(row*vw+d)=sum.toFloat;d+=1
    Tensor.fromFloats(out,shape)

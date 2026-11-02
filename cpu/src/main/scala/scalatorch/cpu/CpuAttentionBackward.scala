package scalatorch.cpu

import scalatorch.*

private[cpu] object CpuAttentionBackward:
  def apply(query: Tensor[Float],key: Tensor[Float],value: Tensor[Float],gradient: Tensor[Float],
      start: Int,maxScores: Long): (Tensor[Float],Tensor[Float],Tensor[Float]) =
    val (output,_)=AttentionLayout.validate(query.shape,key.shape,value.shape,start,maxScores)
    require(gradient.shape==output,"attention gradient shape mismatch")
    val dq=new Array[Double](query.shape.numel.toInt)
    val dk=new Array[Double](key.shape.numel.toInt)
    val dv=new Array[Double](value.shape.numel.toInt)
    if query.shape.numel>0 then
      val Seq(batch,heads,tokens,width)=query.shape.dims.map(_.toInt): @unchecked
      val kvHeads=key.shape.dims(1).toInt;val keys=key.shape.dims(2).toInt;val vw=value.shape.dims(3).toInt
      val q=query.toFloatArray;val k=key.toFloatArray;val v=value.toFloatArray;val g=gradient.toFloatArray
      val probabilities=new Array[Double](keys);val dp=new Array[Double](keys)
      val group=heads/kvHeads;val scale=1.0/math.sqrt(width.toDouble)
      for b <- 0 until batch; h <- 0 until heads; t <- 0 until tokens do
        val row=(b*heads+h)*tokens+t;val base=(b*kvHeads+h/group)*keys
        val count=math.min(keys.toLong,start.toLong+t+1).toInt
        var maximum=Double.NegativeInfinity;var j=0
        while j<count do
          var dot=0.0;var d=0
          while d<width do
            dot+=q(row*width+d).toDouble*k((base+j)*width+d);d+=1
          probabilities(j)=dot*scale;maximum=math.max(maximum,probabilities(j));j+=1
        var denominator=0.0;j=0
        while j<count do
          probabilities(j)=math.exp(probabilities(j)-maximum);denominator+=probabilities(j);j+=1
        var weighted=0.0;j=0
        while j<count do
          probabilities(j)/=denominator
          var dot=0.0;var d=0
          while d<vw do
            dot+=g(row*vw+d).toDouble*v((base+j)*vw+d);d+=1
          dp(j)=dot;weighted+=probabilities(j)*dot;j+=1
        j=0
        while j<count do
          val dz=probabilities(j)*(dp(j)-weighted)*scale
          var d=0
          while d<width do
            dq(row*width+d)+=dz*k((base+j)*width+d)
            dk((base+j)*width+d)+=dz*q(row*width+d);d+=1
          d=0
          while d<vw do
            dv((base+j)*vw+d)+=probabilities(j)*g(row*vw+d);d+=1
          j+=1
    (Tensor.fromFloats(dq.map(_.toFloat),query.shape),Tensor.fromFloats(dk.map(_.toFloat),key.shape),
      Tensor.fromFloats(dv.map(_.toFloat),value.shape))

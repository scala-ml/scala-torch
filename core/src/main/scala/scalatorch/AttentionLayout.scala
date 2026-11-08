package scalatorch

private[scalatorch] object AttentionLayout:
  def validate(q: Shape,k: Shape,v: Shape,start: Int,maxScores: Long,int32Scores: Boolean=true): (Shape,Long) =
    require(q.rank==4 && k.rank==4 && v.rank==4,"attention expects (batch,heads,tokens,width)")
    require(start>=0 && maxScores>=0,"invalid attention offset or workspace limit")
    require(q.dims(0)==k.dims(0) && k.dims(0)==v.dims(0),"attention batch mismatch")
    require(q.dims(1)>0 && k.dims(1)>0 && k.dims(1)==v.dims(1) && q.dims(1)%k.dims(1)==0,"invalid grouped attention heads")
    require(q.dims(3)>0 && q.dims(3)==k.dims(3) && v.dims(3)>0 && k.dims(2)==v.dims(2),"attention width/token mismatch")
    val output=Shape.from(Seq(q.dims(0),q.dims(1),q.dims(2),v.dims(3)))
    require(output.numel<=Int.MaxValue,"attention output exceeds Int indexing")
    if q.numel==0 then return (output,0L)
    require(k.dims(2)>0,"nonempty queries require keys")
    val scores=Shape.from(Seq(q.dims(0),q.dims(1),q.dims(2),k.dims(2))).numel
    require(scores<=maxScores && (!int32Scores || scores<=Int.MaxValue),"attention score workspace limit exceeded")
    (output,scores)

private[scalatorch] object RotaryCoefficients:
  def count(shape: Shape,start: Int,theta: Double): Long =
    require(shape.rank==4 && shape.dims(3)>0 && shape.dims(3)%2==0,"RoPE requires rank 4 and positive even width")
    require(start>=0 && theta.isFinite && theta>0,"invalid rotary position or theta")
    if shape.numel==0 then 0L else shape.dims(2)*shape.dims(3)
  def values(shape: Shape,start: Int,theta: Double): Array[Double] =
    val size=count(shape,start,theta)
    require(size<=Int.MaxValue,"rotary coefficient limit exceeded")
    val result=new Array[Double](size.toInt)
    if size>0 then
      val width=shape.dims(3).toInt;val half=width/2
      for t <- 0 until shape.dims(2).toInt; d <- 0 until half do
        val angle=(start.toLong+t).toDouble*math.pow(theta,-2.0*d/width)
        require(angle.isFinite,"rotary angle overflow")
        result(t*width+2*d)=math.cos(angle)
        result(t*width+2*d+1)=math.sin(angle)
    result

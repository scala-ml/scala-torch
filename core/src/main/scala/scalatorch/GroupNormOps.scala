package scalatorch

/** CPU (N,C,...) normalization with per-sample grouped channels and biased variance. */
object GroupNormOps:
  private def width(input: Tensor[Float],groups: Int,epsilon: Double): Int =
    require(input.device==Device.Cpu && input.dtype==DType.Float32 && input.shape.rank>=2,
      "GroupNorm requires CPU Float32 (N,C,...) input")
    width(input.shape,groups,epsilon)
  private[scalatorch] def width(shape: Shape,groups: Int,epsilon: Double): Int =
    require(shape.rank>=2,"GroupNorm requires (N,C,...) input")
    require(epsilon.isFinite && epsilon>0,"GroupNorm epsilon must be finite and positive")
    val dims=shape.dims
    require(dims.forall(_<=Int.MaxValue) && shape.numel<=Int.MaxValue,"GroupNorm exceeds Int indexing")
    require(dims.tail.forall(_>0),"GroupNorm channels/spatial dimensions must be positive")
    require(groups>0 && dims(1)%groups==0,"GroupNorm channels must divide into groups")
    val groupShape=Shape.from((dims(1)/groups)+:dims.drop(2))
    require(groupShape.numel<=Int.MaxValue,"GroupNorm group exceeds Int indexing")
    groupShape.numel.toInt

  private def stats(x: Array[Float],start: Int,count: Int,epsilon: Double): (Double,Double) =
    var mean=0.0
    var i=0
    while i<count do
      mean+=x(start+i).toDouble
      i+=1
    mean/=count
    var variance=0.0
    i=0
    while i<count do
      val centered=x(start+i).toDouble-mean
      variance+=centered*centered
      i+=1
    (mean,1.0/math.sqrt(variance/count+epsilon))

  def forward(input: Tensor[Float],groups: Int,epsilon: Double = 1e-5): Tensor[Float] =
    val count=width(input,groups,epsilon)
    val x=input.toFloatArray
    val out=new Array[Float](x.length)
    var start=0
    while start<x.length do
      val (mean,inv)=stats(x,start,count,epsilon)
      var i=0
      while i<count do
        out(start+i)=((x(start+i).toDouble-mean)*inv).toFloat
        i+=1
      start+=count
    Tensor.fromFloats(out,input.shape)

  private[scalatorch] def backward(input: Tensor[Float],upstream: Tensor[Float],groups: Int,epsilon: Double): Tensor[Float] =
    val count=width(input,groups,epsilon)
    require(upstream.shape==input.shape && upstream.device==Device.Cpu && upstream.dtype==DType.Float32,
      "GroupNorm upstream shape/type mismatch")
    val x=input.toFloatArray;val seed=upstream.toFloatArray
    val out=new Array[Float](x.length)
    var start=0
    while start<x.length do
      val (mean,inv)=stats(x,start,count,epsilon)
      var sum=0.0;var dot=0.0
      var i=0
      while i<count do
        sum+=seed(start+i)
        dot+=seed(start+i).toDouble*((x(start+i).toDouble-mean)*inv)
        i+=1
      i=0
      while i<count do
        val normalized=(x(start+i).toDouble-mean)*inv
        out(start+i)=(inv*(seed(start+i).toDouble-sum/count-normalized*(dot/count))).toFloat
        i+=1
      start+=count
    Tensor.fromFloats(out,input.shape)

package scalatorch

enum NearestMode:
  case Floor, Exact

/** CPU NCHW nearest resize to an explicit output size. */
object Resize2d:
  private def outputShape(input: Tensor[Float],height: Int,width: Int): Shape =
    require(input.device==Device.Cpu && input.dtype==DType.Float32 && input.shape.rank==4,
      "nearest resize requires CPU Float32 NCHW")
    outputShape(input.shape,height,width)

  private[scalatorch] def outputShape(input: Shape,height: Int,width: Int): Shape =
    require(input.rank==4,"nearest resize requires NCHW")
    require(height>0 && width>0,"resize output dimensions must be positive")
    val dims=input.dims
    require(dims.tail.forall(_>0),"resize channels/spatial dimensions must be positive")
    require(dims.forall(_<=Int.MaxValue) && input.numel<=Int.MaxValue,"resize input exceeds Int indexing")
    val shape=Shape.from(Seq(dims(0),dims(1),height.toLong,width.toLong))
    require(shape.numel<=Int.MaxValue,"resize output exceeds Int indexing")
    shape

  private def coordinate(index: Int,inputSize: Int,outputSize: Int,mode: NearestMode): Int =
    mode match
      case NearestMode.Floor => (index.toLong*inputSize/outputSize).toInt
      case NearestMode.Exact => ((2L*index+1)*inputSize/(2L*outputSize)).toInt

  private def indices(input: Shape,output: Shape,mode: NearestMode)(consume: (Int,Int) => Unit): Unit =
    val ih=input.dims(2).toInt;val iw=input.dims(3).toInt
    val oh=output.dims(2).toInt;val ow=output.dims(3).toInt
    var i=0
    while i<output.numel do
      val y=coordinate(i/ow%oh,ih,oh,mode)
      val x=coordinate(i%ow,iw,ow,mode)
      consume((i/ow/oh*ih+y)*iw+x,i)
      i+=1

  def nearest(input: Tensor[Float],height: Int,width: Int,mode: NearestMode = NearestMode.Floor): Tensor[Float] =
    val shape=outputShape(input,height,width)
    val x=input.toFloatArray
    val out=new Array[Float](shape.numel.toInt)
    indices(input.shape,shape,mode)((source,target) => out(target)=x(source))
    Tensor.fromFloats(out,shape)

  private[scalatorch] def backward(input: Tensor[Float],upstream: Tensor[Float],height: Int,width: Int,
      mode: NearestMode): Tensor[Float] =
    val shape=outputShape(input,height,width)
    require(upstream.shape==shape && upstream.device==Device.Cpu && upstream.dtype==DType.Float32,
      "resize upstream shape/type mismatch")
    val seed=upstream.toFloatArray
    val out=new Array[Double](input.shape.numel.toInt)
    indices(input.shape,shape,mode)((target,source) => out(target)+=seed(source))
    Tensor.fromFloats(out.map(_.toFloat),input.shape)

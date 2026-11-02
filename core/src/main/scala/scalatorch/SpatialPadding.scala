package scalatorch

enum PaddingMode:
  case Zero, Circular

/** Nonnegative spatial padding, with independent horizontal and vertical modes. */
final case class Padding2d(left: Int,right: Int,top: Int,bottom: Int,
    horizontal: PaddingMode = PaddingMode.Zero,vertical: PaddingMode = PaddingMode.Zero):
  require(Seq(left,right,top,bottom).forall(_>=0),"padding extents must be nonnegative")

/** CPU NCHW padding and its scatter-add pullback. */
object SpatialPadding:
  private def outputShape(input: Tensor[Float],p: Padding2d): Shape =
    require(input.device==Device.Cpu && input.dtype==DType.Float32 && input.shape.rank==4,
      "padding requires CPU Float32 NCHW input")
    val dims=input.shape.dims
    require(dims.forall(_<=Int.MaxValue) && input.shape.numel<=Int.MaxValue,"padding input exceeds Int indexing")
    require(dims.tail.forall(_>0),"padding requires positive channels and spatial dimensions")
    val h=dims(2);val w=dims(3)
    if p.horizontal==PaddingMode.Circular then require(p.left<=w && p.right<=w,"horizontal padding wraps more than once")
    if p.vertical==PaddingMode.Circular then require(p.top<=h && p.bottom<=h,"vertical padding wraps more than once")
    val oh=h+p.top.toLong+p.bottom;val ow=w+p.left.toLong+p.right
    require(oh<=Int.MaxValue && ow<=Int.MaxValue,"padding output dimension exceeds Int indexing")
    val out=Shape.from(Seq(dims(0),dims(1),oh,ow))
    require(out.numel<=Int.MaxValue,"padding output exceeds Int indexing")
    out

  private def indices(input: Shape,output: Shape,p: Padding2d)(consume: (Int,Int) => Unit): Unit =
    val h=input.dims(2).toInt;val w=input.dims(3).toInt
    val oh=output.dims(2).toInt;val ow=output.dims(3).toInt
    var i=0
    while i<output.numel do
      val y=i/ow%oh-p.top;val x=i%ow-p.left
      val visibleY=y>=0 && y<h
      val visibleX=x>=0 && x<w
      if (visibleY || p.vertical==PaddingMode.Circular) && (visibleX || p.horizontal==PaddingMode.Circular) then
        val iy=if visibleY then y else Math.floorMod(y,h)
        val ix=if visibleX then x else Math.floorMod(x,w)
        consume((i/ow/oh*h+iy)*w+ix,i)
      i+=1

  def forward(input: Tensor[Float],padding: Padding2d): Tensor[Float] =
    val shape=outputShape(input,padding)
    val source=input.toFloatArray
    val out=new Array[Float](shape.numel.toInt)
    indices(input.shape,shape,padding)((sourceIndex,target) => out(target)=source(sourceIndex))
    Tensor.fromFloats(out,shape)

  private[scalatorch] def backward(input: Tensor[Float],upstream: Tensor[Float],padding: Padding2d): Tensor[Float] =
    val shape=outputShape(input,padding)
    require(upstream.shape==shape && upstream.device==Device.Cpu && upstream.dtype==DType.Float32,
      "padding upstream shape/type mismatch")
    val seed=upstream.toFloatArray
    val out=new Array[Double](input.shape.numel.toInt)
    indices(input.shape,shape,padding)((target,source) => out(target)+=seed(source))
    Tensor.fromFloats(out.map(_.toFloat),input.shape)

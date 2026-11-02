package scalatorch

/** Symmetric zero padding, NCHW input and OIHW grouped cross-correlation. */
final case class Conv2dConfig(stride: (Int,Int) = (1,1),padding: (Int,Int) = (0,0),
    dilation: (Int,Int) = (1,1),groups: Int = 1):
  require(stride._1>0 && stride._2>0,"convolution stride must be positive")
  require(padding._1>=0 && padding._2>=0,"convolution padding must be nonnegative")
  require(dilation._1>0 && dilation._2>0,"convolution dilation must be positive")
  require(groups>0,"convolution groups must be positive")

/** Direct deterministic CPU reference; Double accumulation with Float32 outputs. */
object Conv2dOps:
  private final case class Geometry(n: Int,ci: Int,h: Int,w: Int,co: Int,cg: Int,kh: Int,kw: Int,
      oh: Int,ow: Int,shape: Shape)
  private def geometry(x: Tensor[Float],weight: Tensor[Float],c: Conv2dConfig): Geometry =
    for value <- Seq(x,weight) do
      require(value.device==Device.Cpu && value.dtype==DType.Float32,"convolution requires CPU Float32")
      require(value.shape.rank==4,"convolution requires rank four")
      require(value.shape.dims.forall(_<=Int.MaxValue) && value.shape.numel<=Int.MaxValue,"convolution exceeds Int indexing")
    val Vector(n,ci,h,w)=x.shape.dims.map(_.toInt): @unchecked
    val Vector(co,cg,kh,kw)=weight.shape.dims.map(_.toInt): @unchecked
    require(ci>0 && h>0 && w>0 && co>0 && cg>0 && kh>0 && kw>0,"convolution dimensions must be positive except batch")
    require(ci%c.groups==0 && co%c.groups==0 && ci/c.groups==cg,"convolution group/channel mismatch")
    def extent(size: Int,kernel: Int,pad: Int,dilation: Int,stride: Int): Int =
      val span=size.toLong+2L*pad-dilation.toLong*(kernel-1)-1
      require(span>=0,"convolution kernel exceeds padded input")
      val out=span/stride+1
      require(out<=Int.MaxValue,"convolution output dimension exceeds Int indexing")
      out.toInt
    val oh=extent(h,kh,c.padding._1,c.dilation._1,c.stride._1)
    val ow=extent(w,kw,c.padding._2,c.dilation._2,c.stride._2)
    val shape=Shape.from(Seq(n.toLong,co.toLong,oh.toLong,ow.toLong))
    require(shape.numel<=Int.MaxValue,"convolution output exceeds Int indexing")
    Geometry(n,ci,h,w,co,cg,kh,kw,oh,ow,shape)

  // Each callback receives the logical input, weight and output index of one visible product.
  private def products(g: Geometry,c: Conv2dConfig)(consume: (Int,Int,Int) => Unit): Unit =
    var out=0
    while out<g.shape.numel do
      val ox=out%g.ow
      val oy=(out/g.ow)%g.oh
      val oc=(out/g.ow/g.oh)%g.co
      val batch=out/g.ow/g.oh/g.co
      val first=(oc/(g.co/c.groups))*g.cg
      var channel=0
      while channel<g.cg do
        var ky=0
        while ky<g.kh do
          val iy=oy.toLong*c.stride._1-c.padding._1+ky.toLong*c.dilation._1
          var kx=0
          while kx<g.kw do
            val ix=ox.toLong*c.stride._2-c.padding._2+kx.toLong*c.dilation._2
            if iy>=0 && iy<g.h && ix>=0 && ix<g.w then
              val input=(((batch*g.ci+first+channel)*g.h+iy.toInt)*g.w+ix.toInt)
              val weight=((oc*g.cg+channel)*g.kh+ky)*g.kw+kx
              consume(input,weight,out)
            kx+=1
          ky+=1
        channel+=1
      out+=1

  def forward(input: Tensor[Float],weight: Tensor[Float],config: Conv2dConfig = Conv2dConfig()): Tensor[Float] =
    val g=geometry(input,weight,config)
    val x=input.toFloatArray;val w=weight.toFloatArray
    val out=new Array[Double](g.shape.numel.toInt)
    products(g,config)((i,k,o) => out(o)+=x(i).toDouble*w(k))
    Tensor.fromFloats(out.map(_.toFloat),g.shape)

  /** One pullback at a time avoids allocating gradients for frozen parents. */
  private[scalatorch] def backward(input: Tensor[Float],weight: Tensor[Float],upstream: Tensor[Float],
      config: Conv2dConfig,inputGradient: Boolean): Tensor[Float] =
    val g=geometry(input,weight,config)
    require(upstream.shape==g.shape && upstream.device==Device.Cpu && upstream.dtype==DType.Float32,
      "convolution upstream shape/type mismatch")
    val source=if inputGradient then weight.toFloatArray else input.toFloatArray
    val seed=upstream.toFloatArray
    val shape=if inputGradient then input.shape else weight.shape
    val out=new Array[Double](shape.numel.toInt)
    products(g,config) { (i,k,o) =>
      if inputGradient then out(i)+=seed(o).toDouble*source(k)
      else out(k)+=seed(o).toDouble*source(i)
    }
    Tensor.fromFloats(out.map(_.toFloat),shape)

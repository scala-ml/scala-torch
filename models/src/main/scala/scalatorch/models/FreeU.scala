package scalatorch.models

import scalatorch.{Shape,Tensor}

/** FreeU v2 parameters (`b1`, `b2` backbone amplification and `s1`, `s2` skip low-frequency attenuation). */
final case class FreeUConfig(b1: Float,b2: Float,s1: Float,s2: Float):
  require(b1.isFinite && b2.isFinite && s1.isFinite && s2.isFinite,"FreeU scaling factors must be finite")

/** Diffusers-compatible FreeU v2 implementation and request-local U-Net scope. */
object FreeU:
  private val active = new ThreadLocal[Option[FreeUConfig]]:
    override def initialValue(): Option[FreeUConfig] = None

  def current: Option[FreeUConfig] = active.get()

  def withConfig[A](config: Option[FreeUConfig])(body: => A): A =
    val previous=active.get()
    active.set(config)
    try body
    finally active.set(previous)

  def sd1Defaults: FreeUConfig = FreeUConfig(b1=1.5f,b2=1.6f,s1=0.9f,s2=0.2f)
  def sdxlDefaults: FreeUConfig = FreeUConfig(b1=1.3f,b2=1.4f,s1=0.9f,s2=0.2f)
  def defaults(sdxl: Boolean): FreeUConfig = if sdxl then sdxlDefaults else sd1Defaults

  /** Multiplies the first `C / 2` channels of an NCHW tensor by `scale`. */
  def scaleBackbone(hidden: Tensor[Float],scale: Float): Tensor[Float] =
    require(hidden.shape.rank==4,"FreeU backbone tensor must be NCHW")
    if scale==1f then return hidden
    val dims=hidden.shape.dims
    val n=dims(0).toInt
    val c=dims(1).toInt
    val half=c/2
    if half==0 then return hidden
    val plane=dims(2).toInt*dims(3).toInt
    val halfElems=half*plane
    val sampleElems=c*plane
    val out=hidden.toFloatArray
    var b=0
    while b<n do
      val base=b*sampleElems
      var i=0
      while i<halfElems do
        out(base+i)*=scale
        i+=1
      b+=1
    Tensor.fromFloats(out,hidden.shape)

  /** Exact port of `diffusers.utils.torch_utils.fourier_filter(x, threshold, scale)`.
    *
    * In shifted 2D frequency coordinates, the mask sets rows `[H/2 - threshold : H/2 + threshold]`
    * and cols `[W/2 - threshold : W/2 + threshold]` (with Python slice clamping) to `scale` and all
    * other frequencies to `1`. By linearity:
    *   `IFFT(FFT(x) * (1 + (scale - 1) * M)) = x + (scale - 1) * IFFT(FFT(x) * M)`
    * Since `M` selects at most `(2 * threshold) x (2 * threshold)` frequencies (4 modes for `threshold = 1`),
    * evaluating those modes directly in `O(K * H * W)` is exact to float precision and avoids a full 2D FFT.
    */
  def fourierFilter(skip: Tensor[Float],threshold: Int,scale: Float): Tensor[Float] =
    require(skip.shape.rank==4 && threshold>=0,"FreeU skip tensor must be NCHW with non-negative threshold")
    if scale==1f || threshold==0 then return skip
    val dims=skip.shape.dims
    val n=dims(0).toInt
    val c=dims(1).toInt
    val h=dims(2).toInt
    val w=dims(3).toInt
    def pySlice(start: Int,end: Int,size: Int): Range =
      val s=if start<0 then math.max(start+size,0) else math.min(start,size)
      val e=if end<0 then math.max(end+size,0) else math.min(end,size)
      s until e
    val shiftedRows=pySlice(h/2-threshold,h/2+threshold,h)
    val shiftedCols=pySlice(w/2-threshold,w/2+threshold,w)
    if shiftedRows.isEmpty || shiftedCols.isEmpty then return skip
    // Map shifted index s to unshifted DFT frequency k = (s - size/2) mod size (matching ifftshift).
    val kyList=shiftedRows.map(s => ((s - h/2)%h + h)%h).distinct.toArray
    val kxList=shiftedCols.map(s => ((s - w/2)%w + w)%w).distinct.toArray
    val cosY=Array.tabulate(kyList.length,h)(( idx,y ) => math.cos(2.0*math.Pi*kyList(idx)*y/h))
    val sinY=Array.tabulate(kyList.length,h)(( idx,y ) => math.sin(2.0*math.Pi*kyList(idx)*y/h))
    val cosX=Array.tabulate(kxList.length,w)(( idx,x ) => math.cos(2.0*math.Pi*kxList(idx)*x/w))
    val sinX=Array.tabulate(kxList.length,w)(( idx,x ) => math.sin(2.0*math.Pi*kxList(idx)*x/w))
    val factor=(scale.toDouble - 1.0)/(h.toDouble*w.toDouble)
    val in=skip.toFloatArray
    val out=new Array[Float](in.length)
    val planeSize=h*w
    val totalPlanes=n*c
    val rowRe=new Array[Double](w)
    val rowIm=new Array[Double](w)
    val modeRe=Array.ofDim[Double](kyList.length,kxList.length)
    val modeIm=Array.ofDim[Double](kyList.length,kxList.length)
    val synRe=new Array[Double](w)
    var p=0
    while p<totalPlanes do
      val base=p*planeSize
      var iy=0
      while iy<kyList.length do
        val cy=cosY(iy);val sy=sinY(iy)
        java.util.Arrays.fill(rowRe,0.0)
        java.util.Arrays.fill(rowIm,0.0)
        var y=0
        while y<h do
          val cVal=cy(y);val sVal=sy(y)
          val rowOff=base+y*w
          var x=0
          while x<w do
            val v=in(rowOff+x).toDouble
            rowRe(x)+=v*cVal
            rowIm(x)-=v*sVal
            x+=1
          y+=1
        var ix=0
        while ix<kxList.length do
          val cx=cosX(ix);val sx=sinX(ix)
          var sr=0.0;var si=0.0
          var x=0
          while x<w do
            val r=rowRe(x);val im=rowIm(x)
            val cVal=cx(x);val sVal=sx(x)
            sr+=r*cVal+im*sVal
            si+=im*cVal-r*sVal
            x+=1
          modeRe(iy)(ix)=sr
          modeIm(iy)(ix)=si
          ix+=1
        iy+=1
      var y=0
      while y<h do
        java.util.Arrays.fill(synRe,0.0)
        var iy2=0
        while iy2<kyList.length do
          val cVal=cosY(iy2)(y);val sVal=sinY(iy2)(y)
          var ix2=0
          while ix2<kxList.length do
            val mr=modeRe(iy2)(ix2);val mi=modeIm(iy2)(ix2)
            // Real part of (mr + i*mi) * (cosY + i*sinY) * (cosX + i*sinX)
            val yr=mr*cVal - mi*sVal
            val yi=mr*sVal + mi*cVal
            val cx=cosX(ix2);val sx=sinX(ix2)
            var x=0
            while x<w do
              synRe(x)+=yr*cx(x) - yi*sx(x)
              x+=1
            ix2+=1
          iy2+=1
        val rowOff=base+y*w
        var x=0
        while x<w do
          out(rowOff+x)=(in(rowOff+x).toDouble + factor*synRe(x)).toFloat
          x+=1
        y+=1
      p+=1
    Tensor.fromFloats(out,skip.shape)

  def apply(config: FreeUConfig,stageIndex: Int,hidden: Tensor[Float],skip: Tensor[Float]): (Tensor[Float],Tensor[Float]) =
    stageIndex match
      case 0 => (scaleBackbone(hidden,config.b1),fourierFilter(skip,1,config.s1))
      case 1 => (scaleBackbone(hidden,config.b2),fourierFilter(skip,1,config.s2))
      case _ => (hidden,skip)

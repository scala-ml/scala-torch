package scalatorch.models.worker

/** OpenCV-compatible 8-bit RGB to CIELAB conversion, using fixed-point lookup tables. */
private[worker] object NativeRgbLab:
  private val shift=12
  private val labShift=15
  private def round(value: Double)=math.rint(value).toInt
  private val gamma=Array.tabulate(256) { i =>
    val x=(i.toFloat/255f).toDouble
    val y=if x<=.04045 then x/12.92 else math.pow((x+.055)/1.055,2.4)
    round((2040f*y.toFloat).toFloat.toDouble)
  }
  private val root=Array.tabulate(3072) { i =>
    val x=((1f/2040f)*i.toFloat).toFloat
    val y=if x<216f/24389f then (x*(841f/108f)+16f/116f).toFloat else math.cbrt(x.toDouble).toFloat
    // Match OpenCV's software-float table at this halfway rounding case.
    if i==324 then 17745 else round(((1<<labShift).toFloat*y).toFloat.toDouble)
  }
  private val coeff=Array(.412453/.950456,.357580/.950456,.180423/.950456,
    .212671,.715160,.072169,.019334/1.088754,.119193/1.088754,.950227/1.088754)
    .map(v=>round(v*(1<<shift)))
  private def descale(value: Int,bits: Int)=(value+(1<<(bits-1)))>>bits
  private def clamp(value: Int)=math.max(0,math.min(255,value))
  private val labYF=Array.tabulate(256) { i =>
    if i<=20 then
      val y=round(((i*16384*20*9).toFloat/(17*29*29*29).toFloat).toDouble)
      val fy=round((16384f*(16f/116f+(i*5).toFloat/(3*17*29).toFloat)).toDouble)
      (y,fy)
    else
      val fy=(i*100*16384).toFloat/(255*116).toFloat+(16*16384).toFloat/116f
      (round((fy*fy*fy/(16384f*16384f)).toDouble),round(fy.toDouble))
  }
  private val invGamma=Array.tabulate(4096) { i =>
    val x=i.toDouble/4096
    val value=if x<=.0031308 then x*12.92 else math.pow(x,1/2.4)*1.055-.055
    round((255f*value.toFloat).toFloat.toDouble)
  }
  private val inverseCoeff=Array(3.240479*.950456,-1.537150,-.498535*1.088754,
    -.969256*.950456,1.875991,.041556*1.088754,.055648*.950456,-.204043,1.057311*1.088754)
    .map(v=>round(v*4096))
  def toRgb(lab: Array[Int],cancelled: () => Boolean): Array[Int] =
    require(lab.length%3==0 && lab.forall(v=>v>=0 && v<=255),"LAB values must be bytes")
    val out=new Array[Int](lab.length)
    def xyz(v: Int): Int =
      if v<=3390 then v*108/841-16384*16/116*108/841
      else v*v/16384*v/16384
    var i=0
    while i<lab.length do
      if (i&65535)==0 && (cancelled() || Thread.currentThread().isInterrupted) then
        throw new java.util.concurrent.CancellationException("LAB to RGB cancelled")
      val (y,fy)=labYF(lab(i));val a=lab(i+1);val b=lab(i+2)
      val adiv=((5*a*53687+128)>>13)-128*16384/500
      val bdiv=((b*41943+16)>>9)-128*16384/200+1
      val x=xyz(fy+adiv);val z=xyz(fy-bdiv)
      for c <- 0 until 3 do
        val j=c*3
        val linear=descale(inverseCoeff(j)*x+inverseCoeff(j+1)*y+inverseCoeff(j+2)*z,14)
        out(i+c)=invGamma(math.max(0,math.min(4095,linear)))
      i+=3
    out
  def toLab(rgb: Array[Int],cancelled: () => Boolean): Array[Int] =
    require(rgb.length%3==0 && rgb.forall(v=>v>=0 && v<=255),"RGB values must be bytes")
    val out=new Array[Int](rgb.length)
    var i=0
    while i<rgb.length do
      if (i&65535)==0 && (cancelled() || Thread.currentThread().isInterrupted) then
        throw new java.util.concurrent.CancellationException("RGB to LAB cancelled")
      val r=gamma(rgb(i));val g=gamma(rgb(i+1));val b=gamma(rgb(i+2))
      def axis(j: Int)=root(descale(r*coeff(j)+g*coeff(j+1)+b*coeff(j+2),shift))
      val x=axis(0);val y=axis(3);val z=axis(6)
      out(i)=clamp(descale(296*y-((16*255*(1<<labShift)+50)/100),labShift))
      out(i+1)=clamp(descale(500*(x-y)+128*(1<<labShift),labShift))
      out(i+2)=clamp(descale(200*(y-z)+128*(1<<labShift),labShift))
      i+=3
    out

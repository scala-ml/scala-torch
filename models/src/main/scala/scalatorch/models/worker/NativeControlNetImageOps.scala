package scalatorch.models.worker

import java.awt.image.BufferedImage
import java.util.concurrent.CancellationException

/** Bounded RGB resampling and blur operators used by NUI's Pillow ControlNet preprocessors. */
private[worker] object NativeControlNetImageOps:
  private final case class Filter(start: Int,weights: Array[Double])
  private def cubic(x0: Double,a: Double): Double=
    val x=math.abs(x0)
    if x<1 then ((a+2)*x-(a+3))*x*x+1
    else if x<2 then ((a*x-5*a)*x+8*a)*x-4*a
    else 0.0

  private def filters(input: Int,output: Int,a: Double= -0.5): Array[Filter]=
    val scale=input.toDouble/output
    val filterScale=math.max(1.0,scale)
    val support=2*filterScale
    Array.tabulate(output) { position =>
      val center=(position+0.5)*scale
      val start=math.max(0,(center-support+0.5).toInt)
      val end=math.min(input,(center+support+0.5).toInt)
      val weights=Array.tabulate(end-start)(i=>cubic((start+i-center+0.5)/filterScale,a))
      val total=weights.sum
      Filter(start,weights.map(_/total))
    }

  private def channel(rgb: Int,shift: Int): Int=(rgb>>>shift)&255
  private def pack(red: Double,green: Double,blue: Double): Int=
    def byte(value: Double)=math.max(0,math.min(255,math.floor(value+0.5).toInt))
    0xff000000.toInt|(byte(red)<<16)|(byte(green)<<8)|byte(blue)

  private def resizePass(input: Array[Int],sourceWidth: Int,sourceHeight: Int,outputWidth: Int,outputHeight: Int,
      horizontal: Boolean,weights: Array[Filter],cancelled: () => Boolean): Array[Int]=
    val output=new Array[Int](Math.multiplyExact(outputWidth,outputHeight))
    for y <- 0 until outputHeight do
      if cancelled() || Thread.currentThread().isInterrupted then throw CancellationException("ControlNet resize cancelled")
      for x <- 0 until outputWidth do
        val filter=weights(if horizontal then x else y)
        var r=0.0;var g=0.0;var b=0.0;var tap=0
        while tap<filter.weights.length do
          val sx=if horizontal then filter.start+tap else x
          val sy=if horizontal then y else filter.start+tap
          val pixel=input(sy*sourceWidth+sx);val weight=filter.weights(tap)
          r+=channel(pixel,16)*weight;g+=channel(pixel,8)*weight;b+=channel(pixel,0)*weight
          tap+=1
        output(y*outputWidth+x)=pack(r,g,b)
    output

  def bicubic(image: BufferedImage,width: Int,height: Int,cancelled: () => Boolean): BufferedImage=
    cubicResize(image,width,height,-0.5,cancelled)

  private def cubicResize(image: BufferedImage,width: Int,height: Int,a: Double,cancelled: () => Boolean): BufferedImage=
    require(width>0 && height>0 && width.toLong*height<=16777216L,"ControlNet Bicubic resize exceeds pixel limit")
    require(image.getWidth.toLong*image.getHeight<=16777216L,"ControlNet Bicubic input exceeds pixel limit")
    val oldWidth=image.getWidth;val oldHeight=image.getHeight
    if oldWidth==width && oldHeight==height then image
    else
      val input=new Array[Int](oldWidth*oldHeight)
      for y <- 0 until oldHeight do
        if cancelled() || Thread.currentThread().isInterrupted then throw CancellationException("ControlNet resize cancelled")
        for x <- 0 until oldWidth do input(y*oldWidth+x)=image.getRGB(x,y)
      val horizontal=resizePass(input,oldWidth,oldHeight,width,oldHeight,true,filters(oldWidth,width,a),cancelled)
      val resized=resizePass(horizontal,width,oldHeight,width,height,false,filters(oldHeight,height,a),cancelled)
      val output=new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB)
      for y <- 0 until height; x <- 0 until width do output.setRGB(x,y,resized(y*width+x))
      output

  private[worker] def cvResize(image: BufferedImage,width: Int,height: Int,interpolation: String,
      cancelled: () => Boolean): BufferedImage=
    require(width>0 && height>0 && width.toLong*height<=16777216L,"ControlNet source resize exceeds pixel limit")
    val sw=image.getWidth;val sh=image.getHeight
    require(sw.toLong*sh<=16777216L,"ControlNet source resize input exceeds pixel limit")
    interpolation match
      case "cubic" => cvCubic(image,width,height,cancelled)
      case "nearest" =>
        val output=new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB)
        for y <- 0 until height do
          if cancelled() || Thread.currentThread().isInterrupted then throw CancellationException("ControlNet source resize cancelled")
          val sy=math.min(sh-1,(y.toLong*sh/height).toInt)
          for x <- 0 until width do
            val sx=math.min(sw-1,(x.toLong*sw/width).toInt)
            output.setRGB(x,y,image.getRGB(sx,sy))
        output
      case "area" =>
        val output=new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB)
        val xScale=sw.toDouble/width;val yScale=sh.toDouble/height
        for y <- 0 until height do
          if cancelled() || Thread.currentThread().isInterrupted then throw CancellationException("ControlNet area resize cancelled")
          val y0=y*yScale;val y1=(y+1)*yScale
          val sy0=math.floor(y0).toInt;val sy1=math.min(sh,math.ceil(y1).toInt)
          for x <- 0 until width do
            val x0=x*xScale;val x1=(x+1)*xScale
            val sx0=math.floor(x0).toInt;val sx1=math.min(sw,math.ceil(x1).toInt)
            var r=0.0;var g=0.0;var b=0.0;var total=0.0
            var sy=sy0
            while sy<sy1 do
              val wy=math.max(0.0,math.min(y1,sy+1.0)-math.max(y0,sy.toDouble))
              var sx=sx0
              while sx<sx1 do
                val wx=math.max(0.0,math.min(x1,sx+1.0)-math.max(x0,sx.toDouble))
                val weight=wx*wy;val pixel=image.getRGB(sx,sy)
                r+=channel(pixel,16)*weight;g+=channel(pixel,8)*weight;b+=channel(pixel,0)*weight;total+=weight
                sx+=1
              sy+=1
            output.setRGB(x,y,pack(r/total,g/total,b/total))
        output
      case _ => throw IllegalArgumentException("unknown ControlNet resize interpolation")

  /** OpenCV-compatible cubic interpolation with clamped borders and one final rounding. */
  private def cvCubic(image: BufferedImage,width: Int,height: Int,cancelled: () => Boolean): BufferedImage =
    val sw=image.getWidth;val sh=image.getHeight
    val small=sw<4 || sh<4
    def coefficients(input: Int,output: Int): Vector[(Int,Array[Double])] =
      val scale=input.toDouble/output
      Vector.tabulate(output) { i =>
        val exact=(i+.5)*scale-.5
        val position=if small then exact.toFloat.toDouble else exact
        val base=math.floor(position).toInt
        val fraction=position-base
        val weights=if !small then Array.tabulate(4)(j => cubic(j-1-fraction,-.75)) else
          val x=fraction.toFloat;val a= -0.75f;val t=x+1f;val u=1f-x
          val c0=((a*t-5f*a)*t+8f*a)*t-4f*a
          val c1=((a+2f)*x-(a+3f))*x*x+1f
          val c2=((a+2f)*u-(a+3f))*u*u+1f
          val c3=1f-c0-c1-c2
          Array(c0,c1,c2,c3).map(c => math.rint((c*2048f).toDouble)/2048.0)
        (base-1,weights)
      }
    val xs=coefficients(sw,width);val ys=coefficients(sh,height)
    val out=new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB)
    for y <- 0 until height do
      if cancelled() || Thread.currentThread().isInterrupted then throw CancellationException("ControlNet cubic resize cancelled")
      val (baseY,wy)=ys(y)
      for x <- 0 until width do
        val (baseX,wx)=xs(x)
        var rgb=0
        for shift <- Seq(16,8,0) do
          var sum=0.0
          for j <- 0 until 4; i <- 0 until 4 do
            val sx=math.max(0,math.min(sw-1,baseX+i));val sy=math.max(0,math.min(sh-1,baseY+j))
            sum+=channel(image.getRGB(sx,sy),shift).toDouble*wx(i)*wy(j)
          val value=math.max(0,math.min(255,math.floor(sum+.5).toInt))
          rgb|=value<<shift
        out.setRGB(x,y,rgb)
    out

  private def rotated(pattern: Array[Int],turns: Int): Array[Int]=
    var current=pattern
    var turn=0
    while turn<turns do
      val next=new Array[Int](9)
      for y <- 0 until 3; x <- 0 until 3 do next(y*3+x)=current(x*3+(2-y))
      current=next;turn+=1
    current

  private def thin(gray: Array[Int],width: Int,height: Int,prune: Boolean,cancelled: () => Boolean): Unit=
    val bases=Vector(Array(-1,-1,-1,0,1,0,1,1,1),Array(0,-1,-1,1,1,-1,0,1,0))
    val pruning=Vector(Array(-1,-1,-1,-1,1,-1,0,0,-1),Array(-1,-1,-1,-1,1,-1,-1,0,0))
    def erase(pattern: Array[Int]): Boolean=
      val matches=new Array[Int](gray.length);var count=0
      for y <- 0 until height do
        if cancelled() || Thread.currentThread().isInterrupted then throw CancellationException("ControlNet edge thinning cancelled")
        for x <- 0 until width do
          val center=y*width+x
          if gray(center)>127 then
            var matchesPattern=true;var ky=0
            while ky<3 && matchesPattern do
              var kx=0
              while kx<3 && matchesPattern do
                val expected=pattern(ky*3+kx)
                if expected>=0 then
                  val nx=x+kx-1;val ny=y+ky-1
                  val value=if nx<0 || nx>=width || ny<0 || ny>=height then 0 else gray(ny*width+nx)
                  if (expected==1 && value<=127) || (expected==0 && value>127) then matchesPattern=false
                kx+=1
              ky+=1
            if matchesPattern then {matches(count)=center;count+=1}
      var i=0
      while i<count do {gray(matches(i))=0;i+=1}
      count>0
    var iteration=0;var changed=true
    while iteration<32 && changed do
      changed=false
      for turns <- 0 until 4; pattern <- bases do changed=erase(rotated(pattern,turns)) || changed
      iteration+=1
    if prune then for turns <- 0 until 4; pattern <- pruning do erase(rotated(pattern,turns))

  private def edgeNms(gray: Array[Int],width: Int,height: Int): Unit=
    val kernels=Vector(Vector(3,4,5),Vector(1,4,7),Vector(0,4,8),Vector(2,4,6))
    val output=new Array[Int](gray.length)
    for y <- 0 until height; x <- 0 until width do
      val index=y*width+x;val value=gray(index)
      if kernels.exists(_.forall { offset =>
          val dx=offset%3-1;val dy=offset/3-1;val nx=x+dx;val ny=y+dy
          nx<0 || nx>=width || ny<0 || ny>=height || gray(ny*width+nx)<=value
        }) then output(index)=value
    System.arraycopy(output,0,gray,0,gray.length)

  private def otsu(gray: Array[Int]): Unit=
    val histogram=new Array[Long](256);var totalSum=0L
    gray.foreach { value => histogram(value)+=1;totalSum+=value }
    val total=gray.length.toLong
    var backgroundWeight=0L;var backgroundSum=0L;var best=0.0;var threshold=0
    var t=0
    while t<256 do
      backgroundWeight+=histogram(t);backgroundSum+=t*histogram(t)
      val foregroundWeight=total-backgroundWeight
      if backgroundWeight>0 && foregroundWeight>0 then
        val difference=backgroundSum.toDouble/backgroundWeight-(totalSum-backgroundSum).toDouble/foregroundWeight
        val variance=backgroundWeight.toDouble*foregroundWeight*difference*difference
        if variance>best then {best=variance;threshold=t}
      t+=1
    var i=0
    while i<gray.length do {gray(i)=if gray(i)>threshold then 255 else 0;i+=1}

  private def binaryResize(image: BufferedImage,width: Int,height: Int,cancelled: () => Boolean): (Boolean,Boolean)=
    val sw=image.getWidth;val sh=image.getHeight
    val seen=scala.collection.mutable.HashSet.empty[Int]
    var min=255;var max=0
    val original=new Array[Int](sw*sh)
    var y=0
    while y<sh do
      if cancelled() || Thread.currentThread().isInterrupted then throw CancellationException("ControlNet resize analysis cancelled")
      var x=0
      while x<sw do
        val pixel=image.getRGB(x,y)&0xffffff;original(y*sw+x)=pixel
        min=math.min(min,math.min(channel(pixel,16),math.min(channel(pixel,8),channel(pixel,0))))
        max=math.max(max,math.max(channel(pixel,16),math.max(channel(pixel,8),channel(pixel,0))))
        if seen.size<=200 then seen+=pixel
        x+=1
      y+=1
    val binary=seen.size==2 && min<16 && max>240
    if !binary then (false,false)
    else
      val eroded=new Array[Int](original.length)
      for y <- 0 until sh do
        if cancelled() || Thread.currentThread().isInterrupted then throw CancellationException("ControlNet morphology cancelled")
        for x <- 0 until sw do
          var red=255;var green=255;var blue=255
          for dy <- -1 to 1; dx <- -1 to 1 do
            val px=x+dx;val py=y+dy
            val pixel=if px<0 || px>=sw || py<0 || py>=sh then 0xffffff else original(py*sw+px)
            red=math.min(red,channel(pixel,16));green=math.min(green,channel(pixel,8));blue=math.min(blue,channel(pixel,0))
          eroded(y*sw+x)=(red<<16)|(green<<8)|blue
      var lost=0L;var foreground=0L
      for y <- 0 until sh do
        if cancelled() || Thread.currentThread().isInterrupted then throw CancellationException("ControlNet morphology cancelled")
        for x <- 0 until sw do
          var red=0;var green=0;var blue=0
          for dy <- -1 to 1; dx <- -1 to 1 do
            val px=x+dx;val py=y+dy
            val pixel=if px<0 || px>=sw || py<0 || py>=sh then 0 else eroded(py*sw+px)
            red=math.max(red,channel(pixel,16));green=math.max(green,channel(pixel,8));blue=math.max(blue,channel(pixel,0))
          val originalPixel=original(y*sw+x)
          if red<channel(originalPixel,16) then lost+=1
          if green<channel(originalPixel,8) then lost+=1
          if blue<channel(originalPixel,0) then lost+=1
          if channel(originalPixel,16)>127 then foreground+=1
          if channel(originalPixel,8)>127 then foreground+=1
          if channel(originalPixel,0)>127 then foreground+=1
      (true,lost*2>foreground)

  private def medianBorder(image: BufferedImage): Int=
    val width=image.getWidth;val height=image.getHeight;val count=2*width+2*height
    val red=new Array[Int](count);val green=new Array[Int](count);val blue=new Array[Int](count)
    var i=0
    def add(pixel: Int): Unit=
      red(i)=channel(pixel,16);green(i)=channel(pixel,8);blue(i)=channel(pixel,0);i+=1
    for x <- 0 until width do add(image.getRGB(x,0))
    for x <- 0 until width do add(image.getRGB(x,height-1))
    for y <- 0 until height do add(image.getRGB(0,y))
    for y <- 0 until height do add(image.getRGB(width-1,y))
    def median(values: Array[Int]): Int=
      java.util.Arrays.sort(values)
      if values.length%2==1 then values(values.length/2) else (values(values.length/2-1)+values(values.length/2))/2
    0xff000000.toInt|(median(red)<<16)|(median(green)<<8)|median(blue)

  def sourceCompatibleResize(image: BufferedImage,width: Int,height: Int,mode: Int,
      cancelled: () => Boolean): BufferedImage=
    require(width>0 && height>0 && mode>=0 && mode<=2,"invalid source-compatible ControlNet resize")
    require(width.toLong*height<=16777216L && image.getWidth.toLong*image.getHeight<=16777216L,
      "source-compatible ControlNet resize exceeds pixel limit")
    val sw=image.getWidth;val sh=image.getHeight
    // Python's _resize_map returns a copy before color analysis when dimensions already match.
    if sw==width && sh==height then return image
    val scale=mode match
      case 0 => 0.0
      case 1 => math.max(height.toDouble/sh,width.toDouble/sw)
      case _ => math.min(height.toDouble/sh,width.toDouble/sw)
    val nw=if mode==0 then width else math.max(1,math.rint(sw*scale).toInt)
    val nh=if mode==0 then height else math.max(1,math.rint(sh*scale).toInt)
    val (binary,onePixel)=binaryResize(image,nw,nh,cancelled)
    val area=sw.toLong*sh;val target=nw.toLong*nh
    val interpolation=if binaryResizeIsColor(image) then "nearest" else if target<area then "area" else "cubic"
    val resized=cvResize(image,nw,nh,interpolation,cancelled)
    if binary then
      val gray=new Array[Int](nw*nh)
      for y <- 0 until nh; x <- 0 until nw do
        val pixel=resized.getRGB(x,y);gray(y*nw+x)=((channel(pixel,16)+channel(pixel,8)+channel(pixel,0))/3).toInt
      if onePixel then edgeNms(gray,nw,nh)
      otsu(gray)
      if onePixel then thin(gray,nw,nh,width.toLong*height>area,cancelled)
      for y <- 0 until nh; x <- 0 until nw do
        val value=gray(y*nw+x);resized.setRGB(x,y,0xff000000.toInt|(value<<16)|(value<<8)|value)
    if mode==0 then resized
    else if mode==1 then
      val left=math.max(0,(nw-width)/2);val top=math.max(0,(nh-height)/2)
      val output=new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB)
      for y <- 0 until height; x <- 0 until width do output.setRGB(x,y,resized.getRGB(x+left,y+top))
      output
    else
      val output=new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB)
      val fill=medianBorder(image)
      for y <- 0 until height; x <- 0 until width do output.setRGB(x,y,fill)
      val left=math.max(0,(width-nw)/2);val top=math.max(0,(height-nh)/2)
      for y <- 0 until nh; x <- 0 until nw do
        if x+left<width && y+top<height then output.setRGB(x+left,y+top,resized.getRGB(x,y))
      output

  private def binaryResizeIsColor(image: BufferedImage): Boolean=
    val colors=scala.collection.mutable.HashSet.empty[Int]
    var y=0
    while y<image.getHeight && colors.size<=200 do
      var x=0
      while x<image.getWidth && colors.size<=200 do {colors+=image.getRGB(x,y)&0xffffff;x+=1}
      y+=1
    colors.size>2 && colors.size<200

  def gaussian(image: BufferedImage,sigma: Double,cancelled: () => Boolean): BufferedImage=
    require(sigma.isFinite && sigma>=0 && sigma<=32,"ControlNet Gaussian sigma must be in 0..32")
    require(image.getWidth.toLong*image.getHeight<=16777216L,"ControlNet Gaussian input exceeds pixel limit")
    if sigma==0 then image
    else
      // Pillow approximates a Gaussian using three extended box filters per axis.
      val s=sigma.toFloat
      val sigma2=(s*s/3f).toFloat
      val length=math.sqrt(12.0*sigma2+1.0).toFloat
      val whole=math.floor((length-1.0)/2.0).toFloat
      val numerator=((2f*whole+1f)*(whole*(whole+1f)-3f*sigma2)).toFloat
      val fraction=(numerator/(6f*(sigma2-(whole+1f)*(whole+1f)))).toFloat
      val box=(whole+fraction).toFloat
      val radius=box.toInt
      val weight=((1<<24).toFloat/(box*2f+1f)).toLong
      val farWeight=((1L<<24)-(radius*2+1)*weight)/2
      val width=image.getWidth;val height=image.getHeight
      var data=image.getRGB(0,0,width,height,null,0,width)
      def pass(horizontal: Boolean): Unit =
        val output=new Array[Int](data.length)
        val lines=if horizontal then height else width
        val size=if horizontal then width else height
        def at(line: Int,index: Int): Int =
          val i=math.max(0,math.min(size-1,index))
          if horizontal then data(line*width+i) else data(i*width+line)
        for line <- 0 until lines do
          if cancelled() || Thread.currentThread().isInterrupted then throw CancellationException("ControlNet Gaussian blur cancelled")
          var sums=Array.fill(3)(0L)
          for i <- -radius to radius; c <- 0 to 2 do sums(c)+=(at(line,i) >>> (16-8*c))&255
          for position <- 0 until size do
            val left=at(line,position-radius-1);val right=at(line,position+radius+1)
            var rgb=0
            for c <- 0 to 2 do
              val shift=16-8*c
              val outside=((left>>>shift)&255)+((right>>>shift)&255)
              val value=((sums(c)*weight+outside*farWeight+(1L<<23))>>24).toInt
              rgb|=value<<shift
              sums(c)+=((at(line,position+radius+1)>>>shift)&255)-((at(line,position-radius)>>>shift)&255)
            output(if horizontal then line*width+position else position*width+line)=rgb
        data=output
      for _ <- 0 until 3 do pass(true)
      for _ <- 0 until 3 do pass(false)
      val output=new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB)
      output.setRGB(0,0,width,height,data,0,width)
      output

package scalatorch.models.worker

/** Opaque luminosity blend, following blendmodes' MIT-licensed set-luminance operation. */
private[worker] object NativeLuminosity:
  def blend(background: Array[Int],foreground: Array[Int],cancelled: () => Boolean): Array[Int] =
    require(background.length==foreground.length && background.length%3==0,"blend images must have matching RGB triples")
    require(background.forall(v=>v>=0 && v<=255) && foreground.forall(v=>v>=0 && v<=255),"blend pixels must be bytes")
    val output=new Array[Int](background.length)
    def lum(r: Double,g: Double,b: Double)=r*.299+g*.587+b*.114
    var i=0
    while i<background.length do
      if (i&65535)==0 && (cancelled() || Thread.currentThread().isInterrupted) then
        throw new java.util.concurrent.CancellationException("luminosity blend cancelled")
      val r=background(i)/255.0;val g=background(i+1)/255.0;val b=background(i+2)/255.0
      val target=lum(foreground(i)/255.0,foreground(i+1)/255.0,foreground(i+2)/255.0)
      val delta=target-lum(r,g,b)
      val colors=Array(r+delta,g+delta,b+delta)
      val l=lum(colors(0),colors(1),colors(2))
      val min=colors.min;val max=colors.max
      if min<0 then for c <- 0 until 3 do colors(c)=l+(colors(c)-l)*l/(l-min)
      if max>1 then for c <- 0 until 3 do colors(c)=l+(colors(c)-l)*(1-l)/(max-l)
      for c <- 0 until 3 do output(i+c)=math.max(0,math.min(255,math.rint(colors(c)*255).toInt))
      i+=3
    output

package scalatorch.models

import java.awt.image.BufferedImage

/** RGB preprocessing with Lanczos3 and NUI stretch/crop/fill geometry. */
object DiffusionImageResize:
  // Pillow's default bicubic zero-width/height box extrapolates the edge:
  // normalized cubic weights at distances .5 and 1.5 are 9/8 and -1/8.
  private def edge(first: Int,second: Int): Int =
    Seq(16,8,0).foldLeft(0) { (rgb,shift) =>
      val value=((((first >>> shift)&255)*9-((second >>> shift)&255)+4) >> 3)
      rgb | (math.max(0,math.min(255,value)) << shift)
    }
  private val precision=1 << 22
  private case class Filter(start: Int,weights: Array[Int])
  private def filters(input: Int,output: Int): Array[Filter] =
    val scale=input.toDouble/output
    val filterScale=math.max(1.0,scale)
    val support=3*filterScale
    Array.tabulate(output) { position =>
      val center=(position+.5)*scale
      val start=math.max(0,(center-support+.5).toInt)
      val end=math.min(input,(center+support+.5).toInt)
      def lanczos(x: Double): Double =
        if x==0 then 1.0
        else if math.abs(x)>=3 then 0.0
        else
          val p=math.Pi*x
          math.sin(p)/p * (math.sin(p/3)/(p/3))
      val weights=Array.tabulate(end-start)(i => lanczos((start+i-center+.5)/filterScale))
      val total=weights.sum
      Filter(start,weights.map { w =>
        val scaled=w/total*precision
        (scaled+(if scaled<0 then -.5 else .5)).toInt
      })
    }
  private def resize(input: BufferedImage,width: Int,height: Int,maxPixels: Long): BufferedImage =
    val verticalFirst=input.getHeight>height && input.getHeight.toLong-height>2L*(input.getWidth.toLong-width)
    val intermediate=if verticalFirst then input.getWidth.toLong*height else width.toLong*input.getHeight
    require(intermediate<=maxPixels,"resize intermediate exceeds pixel limit")
    def pass(source: BufferedImage,size: Int,horizontal: Boolean): BufferedImage =
      val old=if horizontal then source.getWidth else source.getHeight
      if old==size then source
      else
        val out=new BufferedImage(if horizontal then size else source.getWidth,
          if horizontal then source.getHeight else size,BufferedImage.TYPE_INT_RGB)
        val coefficients=filters(old,size)
        for y <- 0 until out.getHeight; x <- 0 until out.getWidth do
          val filter=coefficients(if horizontal then x else y)
          var rgb=0
          for shift <- Seq(16,8,0) do
            var sum=precision/2L
            var i=0
            while i<filter.weights.length do
              val pixel=source.getRGB(if horizontal then filter.start+i else x,if horizontal then y else filter.start+i)
              sum+=((pixel >>> shift)&255).toLong*filter.weights(i)
              i+=1
            val channel=math.max(0L,math.min(255L,sum >> 22)).toInt
            rgb|=channel << shift
          out.setRGB(x,y,rgb)
        out
    if verticalFirst then pass(pass(input,height,false),width,true)
    else pass(pass(input,width,true),height,false)

  def apply(input: BufferedImage,width: Int,height: Int,mode: Int,maxPixels: Long=16777216L): BufferedImage =
    require(width>0 && height>0 && mode>=0 && mode<=2 && maxPixels>=0,"invalid image resize settings")
    require(width.toLong*height<=maxPixels && input.getWidth.toLong*input.getHeight<=maxPixels,"resize exceeds pixel limit")
    val sw=input.getWidth;val sh=input.getHeight
    if mode==0 || (sw==width && sh==height) then resize(input,width,height,maxPixels)
    else
      val wider=width.toLong*sh>height.toLong*sw
      val narrower=width.toLong*sh<height.toLong*sw
      val rw=if (mode==1 && wider) || (mode==2 && narrower) then width.toLong else sw.toLong*height/sh
      val rh=if (mode==1 && !wider) || (mode==2 && !narrower) then height.toLong else sh.toLong*width/sw
      require(rw<=Int.MaxValue && rh<=Int.MaxValue && math.max(1L,rw)*math.max(1L,rh)<=maxPixels,"resize intermediate exceeds pixel limit")
      val resized=resize(input,math.max(1L,rw).toInt,math.max(1L,rh).toInt,maxPixels)
      val dx=width/2-rw.toInt/2;val dy=height/2-rh.toInt/2
      val output=new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB)
      for y <- 0 until resized.getHeight; x <- 0 until resized.getWidth do
        if x+dx>=0 && x+dx<width && y+dy>=0 && y+dy<height then output.setRGB(x+dx,y+dy,resized.getRGB(x,y))
      if mode==2 && narrower && dy>0 then
        for y <- 0 until dy; x <- 0 until width do
          output.setRGB(x,y,edge(resized.getRGB(x,0),resized.getRGB(x,math.min(1,resized.getHeight-1))))
          if y+dy+rh<height then output.setRGB(x,(y+dy+rh).toInt,
            edge(resized.getRGB(x,resized.getHeight-1),resized.getRGB(x,math.max(0,resized.getHeight-2))))
      else if mode==2 && wider && dx>0 then
        for x <- 0 until dx; y <- 0 until height do
          output.setRGB(x,y,edge(resized.getRGB(0,y),resized.getRGB(math.min(1,resized.getWidth-1),y)))
          if x+dx+rw<width then output.setRGB((x+dx+rw).toInt,y,
            edge(resized.getRGB(resized.getWidth-1,y),resized.getRGB(math.max(0,resized.getWidth-2),y)))
      output

package scalatorch.models.worker

import scalatorch.{Shape,Tensor}

/** Geometry, schedules and CPU interpolation for NUI's built-in latent Hires upscalers. */
private[worker] object NativeHires:
  final case class Settings(scale: Double,upscaler: String,secondPassSteps: Int,resizeX: Int,resizeY: Int,
      denoisingStrength: Double,sampler: Option[String]=None,scheduler: Option[String]=None,
      prompt: String="",negativePrompt: String="",cfgScale: Option[Float]=None,upscalerPath: Option[String]=None)
  val LatentModes=Vector("Latent","Latent (antialiased)","Latent (bicubic)",
    "Latent (bicubic antialiased)","Latent (nearest)","Latent (nearest-exact)")

  def parse(value: Option[NativeJson],upscalerPaths: Map[String,String]=Map.empty): Option[Settings]=value match
    case None | Some(NativeJson.Null) => None
    case Some(NativeJson.Obj(fields)) =>
      val enabled=fields.get("enabled") match
        case None | Some(NativeJson.Bool(false)) => false
        case Some(NativeJson.Bool(true)) => true
        case _ => throw IllegalArgumentException("hires.enabled must be boolean")
      if !enabled then None
      else
        val allowed=Set("enabled","scale","upscaler","second_pass_steps","resize_x","resize_y",
          "denoising_strength","sampler_name","scheduler","prompt","negative_prompt","cfg_scale")
        val unknown=fields.keySet--allowed
        if unknown.nonEmpty then throw new NativeBackendError("unsupported",s"unsupported hires options: ${unknown.toVector.sorted.mkString(", ")}")
        def text(key: String): String=fields.get(key) match
          case None => ""
          case Some(NativeJson.Str(value)) => value
          case _ => throw IllegalArgumentException(s"hires.$key must be a string")
        def optionalText(key: String): Option[String]=fields.get(key) match
          case None | Some(NativeJson.Null) | Some(NativeJson.Str("")) => None
          case Some(NativeJson.Str(value)) => Some(value)
          case _ => throw IllegalArgumentException(s"hires.$key must be a string or null")
        def number(key: String,default: BigDecimal): BigDecimal=fields.get(key) match
          case None => default
          case Some(NativeJson.Number(result)) => result
          case _ => throw IllegalArgumentException(s"hires.$key must be numeric")
        def integer(key: String,default: Int): Int=
          val result=number(key,BigDecimal(default))
          require(result.isValidInt,s"hires.$key must be an integer")
          result.toInt
        val scale=number("scale",BigDecimal(2)).toDouble
        require(scale.isFinite && scale>0 && scale<=16,"hires.scale must be in (0,16]")
        val upscaler=fields.get("upscaler") match
          case None => "Latent"
          case Some(NativeJson.Str(result)) => result
          case _ => throw IllegalArgumentException("hires.upscaler must be a string")
        val selectedModel=NativeStandaloneUpscale.canonicalUpscaler(upscaler).nonEmpty
        val upscalerPath=if selectedModel then upscalerPaths.get(upscaler).orElse(upscalerPaths.get("R-ESRGAN 4x+")) else None
        val builtInPixelUpscalers=Set("Lanczos","Nearest","None","LDSR","Waifu2x","waifu2x","waifu2x_anime_art_rgb","waifu2x_photo")
        if !LatentModes.contains(upscaler) && !builtInPixelUpscalers(upscaler) && (!selectedModel || upscalerPath.isEmpty) then
          throw new NativeBackendError("unsupported",s"native Hires does not support upscaler '$upscaler'")
        val secondPassSteps=integer("second_pass_steps",0)
        val resizeX=integer("resize_x",0)
        val resizeY=integer("resize_y",0)
        require(secondPassSteps>=0 && secondPassSteps<=1000,"hires.second_pass_steps must be in 0..1000")
        require(resizeX>=0 && resizeX<=16384 && resizeY>=0 && resizeY<=16384,"hires resize dimensions must be in 0..16384")
        val denoisingStrength=number("denoising_strength",BigDecimal("0.55")).toDouble
        require(denoisingStrength.isFinite && denoisingStrength>=0 && denoisingStrength<=1,
          "hires.denoising_strength must be in 0..1")
        val cfg=fields.get("cfg_scale") match
          case None | Some(NativeJson.Null) => None
          case Some(NativeJson.Number(value)) =>
            val converted=value.toFloat
            require(converted.isFinite,"hires.cfg_scale must be finite")
            Some(converted)
          case _ => throw IllegalArgumentException("hires.cfg_scale must be numeric or null")
        Some(Settings(scale,if selectedModel then "R-ESRGAN 4x+" else upscaler,secondPassSteps,resizeX,resizeY,denoisingStrength,
          optionalText("sampler_name"),optionalText("scheduler"),text("prompt"),text("negative_prompt"),cfg,upscalerPath))
    case _ => throw IllegalArgumentException("hires must be an object")

  private def floor8(value: Int): Int=math.max(64,value/8*8)
  private def checkedDimension(value: Long): Int=
    require(value>0 && value<=Int.MaxValue,"hires target dimension is out of range")
    floor8(value.toInt)

  def targetSize(width: Int,height: Int,scale: Double,resizeX: Int,resizeY: Int): (Int,Int)=
    require(width>0 && height>0 && scale.isFinite && scale>0 && resizeX>=0 && resizeY>=0,
      "invalid hires size settings")
    val (rawW,rawH)=(resizeX,resizeY) match
      case (0,0) => ((width*scale).toLong,(height*scale).toLong)
      case (x,0) => (x.toLong,x.toLong*height/width)
      case (0,y) => (y.toLong*width/height,y.toLong)
      case (x,y) => (x.toLong,y.toLong)
    (checkedDimension(rawW),checkedDimension(rawH))

  /** A1111 upscales to cover a two-sided target, then center crops in latent units. */
  def upscaleSize(width: Int,height: Int,targetWidth: Int,targetHeight: Int,cover: Boolean): (Int,Int)=
    require(width>0 && height>0 && targetWidth>0 && targetHeight>0,"invalid hires upscale geometry")
    if !cover then (targetWidth,targetHeight)
    else if width.toLong*targetHeight<height.toLong*targetWidth then
      (targetWidth,math.max(targetHeight,floor8((targetWidth.toLong*height/width).min(Int.MaxValue).toInt)))
    else
      (math.max(targetWidth,floor8((targetHeight.toLong*width/height).min(Int.MaxValue).toInt)),targetHeight)

  /** Returns integer latent offsets `(left, top)` matching NUI's A1111-compatible center crop. */
  def centerCrop(width: Int,height: Int,targetWidth: Int,targetHeight: Int): (Int,Int)=
    require(width>=targetWidth && height>=targetHeight && width>0 && height>0 && targetWidth>0 && targetHeight>0,
      "invalid hires latent crop")
    ((width-targetWidth)/2,(height-targetHeight)/2)

  def crop(input: Tensor[Float],width: Int,height: Int,left: Int,top: Int): Tensor[Float]=
    val dims=input.shape.dims
    require(dims.size==4 && width>0 && height>0 && left>=0 && top>=0 &&
      left.toLong+width<=dims(3) && top.toLong+height<=dims(2),"invalid hires latent crop bounds")
    val batches=dims(0).toInt;val channels=dims(1).toInt;val sourceH=dims(2).toInt;val sourceW=dims(3).toInt
    val source=input.toFloatArray
    val result=new Array[Float](batches*channels*width*height)
    val sourcePlane=sourceW*sourceH;val targetPlane=width*height
    for b <- 0 until batches; c <- 0 until channels; y <- 0 until height; x <- 0 until width do
      result((b*channels+c)*targetPlane+y*width+x)=source((b*channels+c)*sourcePlane+(top+y)*sourceW+left+x)
    Tensor.fromFloats(result,Shape.from(Seq(batches.toLong,channels.toLong,height.toLong,width.toLong)))

  /** `(schedule length, effective strength, active steps)` from `sdfx.params.hires_steps`. */
  def steps(requested: Int,strength: Double): (Int,Double,Int)=
    val count=math.max(1,requested)
    require(strength.isFinite,"hires denoising strength must be finite")
    val value=math.max(0.0,math.min(1.0,strength))
    if value<=0 then (count,1.0/count,1)
    else
      val schedule=math.max(count,math.ceil(count/math.min(value,0.999)-1e-9).toInt)
      (schedule,value,math.min((schedule*value).toInt,schedule))

  private def cubic(x: Double,antialias: Boolean): Double=
    val a=if antialias then -0.5 else -0.75
    val v=math.abs(x)
    if v<1 then (a+2)*v*v*v-(a+3)*v*v+1
    else if v<2 then a*v*v*v-5*a*v*v+8*a*v-4*a
    else 0.0

  private def linear(x: Double): Double=math.max(0.0,1.0-math.abs(x))
  private def sinc(x: Double): Double=if x==0 then 1.0 else math.sin(math.Pi*x)/(math.Pi*x)
  private def lanczos(x: Double): Double=if math.abs(x)<3.0 then sinc(x)*sinc(x/3.0) else 0.0

  private def axisWeights(input: Int,output: Int,antialias: Boolean,cubicMode: Boolean,
      lanczosMode: Boolean=false): Array[Array[(Int,Double)]]=
    val scale=input.toDouble/output
    val filterScale=if antialias then math.max(1.0,scale) else 1.0
    val support=(if lanczosMode then 3.0 else if cubicMode then 2.0 else 1.0)*filterScale
    Array.tabulate(output) { position =>
      val center=(position+0.5)*scale
      val start=if antialias then math.max(0,(center-support+0.5).toInt) else math.ceil(center-support-0.5).toInt
      val end=if antialias then math.min(input,math.ceil(center+support+0.5).toInt)
        else math.floor(center+support-0.5).toInt+1
      val weights=(start until end).map { rawIndex =>
        val distance=(rawIndex+0.5-center)/filterScale
        (if antialias then rawIndex else rawIndex.max(0).min(input-1)) ->
          (if lanczosMode then lanczos(distance) else if cubicMode then cubic(distance,antialias) else linear(distance))
      }.filter(_._2!=0.0)
      val total=weights.map(_._2).sum
      require(total!=0.0 && total.isFinite,"latent interpolation produced empty filter")
      weights.map((index,weight) => index -> (weight/total)).toArray
    }

  private def resizeFiltered(input: Array[Float],batches: Int,channels: Int,inH: Int,inW: Int,
      outH: Int,outW: Int,antialias: Boolean,cubicMode: Boolean,lanczosMode: Boolean=false): Array[Float]=
    val horizontal=axisWeights(inW,outW,antialias,cubicMode,lanczosMode)
    val vertical=axisWeights(inH,outH,antialias,cubicMode,lanczosMode)
    val planeIn=inH*inW
    val planeMid=inH*outW
    val planeOut=outH*outW
    val mid=new Array[Float](batches*channels*planeMid)
    for b <- 0 until batches; c <- 0 until channels; y <- 0 until inH; x <- 0 until outW do
      val weights=horizontal(x)
      var value=0.0
      var i=0
      while i<weights.length do
        val (sourceX,weight)=weights(i)
        value+=input((b*channels+c)*planeIn+y*inW+sourceX)*weight
        i+=1
      mid((b*channels+c)*planeMid+y*outW+x)=value.toFloat
    val output=new Array[Float](batches*channels*planeOut)
    for b <- 0 until batches; c <- 0 until channels; y <- 0 until outH; x <- 0 until outW do
      val weights=vertical(y)
      var value=0.0
      var i=0
      while i<weights.length do
        val (sourceY,weight)=weights(i)
        value+=mid((b*channels+c)*planeMid+sourceY*outW+x)*weight
        i+=1
      output((b*channels+c)*planeOut+y*outW+x)=value.toFloat
    output

  /** Resizes one CPU Float32 NCHW tensor to a latent width and height. */
  def resize(input: Tensor[Float],width: Int,height: Int,mode: String): Tensor[Float]=
    val dims=input.shape.dims
    require(input.device==scalatorch.Device.Cpu && input.dtype==scalatorch.DType.Float32 && dims.size==4,
      "latent hires input must be a CPU Float32 NCHW tensor")
    require(width>0 && height>0 && width.toLong*height<=16777216L,"latent hires output dimensions exceed limit")
    val batches=dims(0).toInt;val channels=dims(1).toInt
    val inH=dims(2).toInt;val inW=dims(3).toInt
    require(batches>0 && channels>0 && inH>0 && inW>0 && batches.toLong*channels*inH*inW<=Int.MaxValue,
      "latent hires input shape exceeds limit")
    if !LatentModes.contains(mode) && mode!="Lanczos" && mode!="area" then throw new NativeBackendError("unsupported",s"unsupported native hires upscaler: $mode")
    val values=input.toFloatArray
    require(values.forall(_.isFinite),"latent hires input contains nonfinite values")
    if width==inW && height==inH then return input
    val output=mode match
      case "area" =>
        val result=new Array[Float](batches*channels*height*width)
        for b <- 0 until batches; c <- 0 until channels; y <- 0 until height; x <- 0 until width do
          val y0=(y.toLong*inH/height).toInt
          val y1=(((y.toLong+1)*inH+height-1)/height).toInt
          val x0=(x.toLong*inW/width).toInt
          val x1=(((x.toLong+1)*inW+width-1)/width).toInt
          var sum=0f
          for sy <- y0 until y1; sx <- x0 until x1 do sum+=values((b*channels+c)*inH*inW+sy*inW+sx)
          result((b*channels+c)*height*width+y*width+x)=sum/((y1-y0)*(x1-x0))
        result
      case "Latent (nearest)" | "Latent (nearest-exact)" =>
        val exact=mode=="Latent (nearest-exact)"
        val result=new Array[Float](batches*channels*height*width)
        val planeIn=inH*inW;val planeOut=height*width
        for b <- 0 until batches; c <- 0 until channels; y <- 0 until height; x <- 0 until width do
          val sourceX=if exact then math.floor((x+0.5)*inW.toDouble/width).toInt else x*inW/width
          val sourceY=if exact then math.floor((y+0.5)*inH.toDouble/height).toInt else y*inH/height
          result((b*channels+c)*planeOut+y*width+x)=values((b*channels+c)*planeIn+sourceY.min(inH-1)*inW+sourceX.min(inW-1))
        result
      case "Latent (bicubic)" => resizeFiltered(values,batches,channels,inH,inW,height,width,false,true)
      case "Latent (bicubic antialiased)" => resizeFiltered(values,batches,channels,inH,inW,height,width,true,true)
      case "Latent (antialiased)" => resizeFiltered(values,batches,channels,inH,inW,height,width,true,false)
      case "Latent" => resizeFiltered(values,batches,channels,inH,inW,height,width,false,false)
      case "Lanczos" => resizeFiltered(values,batches,channels,inH,inW,height,width,true,false,true)
    Tensor.fromFloats(output,Shape.from(Seq(batches.toLong,channels.toLong,height.toLong,width.toLong)))

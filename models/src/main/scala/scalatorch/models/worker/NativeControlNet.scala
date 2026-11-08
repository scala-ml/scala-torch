package scalatorch.models.worker

import scalatorch.{Shape,Tensor}
import scalatorch.models.{ConditionalControlNet2d,DiffusionControlNetRequest,DiffusionImageInput,DiffusionImageRequest,DiffusionImageResize}
import java.awt.image.BufferedImage
import java.nio.file.Path

/** NUI ControlNet request adapter. All model and image handling stays in the native worker. */
private[worker] object NativeControlNet:
  import NativeJson.*

  private def str(fields: Map[String,NativeJson],key: String,default: String): String=fields.get(key) match
    case None | Some(Null) => default
    case Some(Str(value)) => value
    case _ => throw IllegalArgumentException(s"ControlNet $key must be a string")

  private def number(fields: Map[String,NativeJson],key: String,default: Double): Double=fields.get(key) match
    case None | Some(Null) => default
    case Some(Number(value)) => value.toDouble
    case _ => throw IllegalArgumentException(s"ControlNet $key must be numeric")

  private def imageTensor(image: BufferedImage,invert: Boolean,cancelled: () => Boolean): Tensor[Float]=
    val width=image.getWidth;val height=image.getHeight
    require(width>0 && height>0 && width.toLong*height<=16777216L,"ControlNet image exceeds pixel limit")
    val plane=width*height
    val values=new Array[Float](plane*3)
    for y <- 0 until height do
      if cancelled() || Thread.currentThread().isInterrupted then
        throw new java.util.concurrent.CancellationException("ControlNet image conversion cancelled")
      for x <- 0 until width do
        val pixel=image.getRGB(x,y);val offset=y*width+x
        val red=((pixel>>>16)&255)/255f
        val green=((pixel>>>8)&255)/255f
        val blue=(pixel&255)/255f
        values(offset)=if invert then 1f-red else red
        values(plane+offset)=if invert then 1f-green else green
        values(2*plane+offset)=if invert then 1f-blue else blue
    Tensor.fromFloats(values,Shape.from(Seq(1L,3L,height.toLong,width.toLong)))

  def resizeRequest(request: DiffusionImageRequest,width: Int,height: Int,cancelled: () => Boolean): DiffusionImageRequest=
    def resized(control: DiffusionControlNetRequest): DiffusionControlNetRequest=
      val source=control.preprocessedImage.getOrElse(control.image)
      val dims=source.shape.dims
      require(dims.size==4 && dims.take(2)==Vector(1L,3L) && source.device==scalatorch.Device.Cpu,
        "ControlNet image must be a CPU RGB batch")
      val sw=dims(3).toInt;val sh=dims(2).toInt
      val values=source.toFloatArray
      val image=new BufferedImage(sw,sh,BufferedImage.TYPE_INT_RGB)
      val plane=sw*sh
      for y <- 0 until sh do
        if cancelled() || Thread.currentThread().isInterrupted then
          throw new java.util.concurrent.CancellationException("ControlNet Hires image conversion cancelled")
        for x <- 0 until sw do
          val i=y*sw+x
          def byte(v: Float)=math.max(0,math.min(255,math.floor(v*255.0+0.5).toInt))
          image.setRGB(x,y,0xff000000.toInt|(byte(values(i))<<16)|(byte(values(plane+i))<<8)|byte(values(2*plane+i)))
      val map=if control.sourceCompatibleResize then
        NativeControlNetImageOps.sourceCompatibleResize(image,width,height,control.resizeMode,cancelled)
        else DiffusionImageResize(image,width,height,control.resizeMode)
      control.copy(image=imageTensor(map,false,cancelled),hiresSoftInjection=control.hiresSoftEligible)
    request.copy(width=width,height=height,controlNet=None,
      controlNets=request.controlNet.toVector.concat(request.controlNets).map(resized))

  def hasEnabled(value: Option[NativeJson]): Boolean=value.exists {
    case Arr(units) => units.exists {
      case Obj(fields) => fields.get("enabled").contains(Bool(true))
      case _ => false
    }
    case _ => false
  }

  private def canny(image: BufferedImage,low: Int,high: Int,cancelled: () => Boolean): BufferedImage=
    val width=image.getWidth;val height=image.getHeight
    val (lowThreshold,highThreshold)=if low<=high then (low,high) else (high,low)
    val count=Math.multiplyExact(width,height)
    val magnitude=new Array[Int](count)
    val dxs=new Array[Int](count);val dys=new Array[Int](count)
    def rgb(x: Int,y: Int): Int=image.getRGB(math.max(0,math.min(width-1,x)),math.max(0,math.min(height-1,y)))
    for y <- 0 until height do
      if cancelled() || Thread.currentThread().isInterrupted then
        throw new java.util.concurrent.CancellationException("ControlNet Canny preprocessing cancelled")
      for x <- 0 until width do
        val i=y*width+x
        val p0=rgb(x-1,y-1);val p1=rgb(x,y-1);val p2=rgb(x+1,y-1)
        val p3=rgb(x-1,y);val p4=rgb(x+1,y)
        val p5=rgb(x-1,y+1);val p6=rgb(x,y+1);val p7=rgb(x+1,y+1)
        var strongest=0;var selectedX=0;var selectedY=0
        var channel=0
        while channel<3 do
          val shift=16-channel*8
          val a=(p0>>>shift)&255;val b=(p1>>>shift)&255;val c=(p2>>>shift)&255
          val d=(p3>>>shift)&255;val f=(p4>>>shift)&255
          val g=(p5>>>shift)&255;val h=(p6>>>shift)&255;val j=(p7>>>shift)&255
          val gx= -a+c-2*d+2*f-g+j
          val gy= -a-2*b-c+g+2*h+j
          val strength=math.abs(gx)+math.abs(gy)
          if strength>strongest then
            strongest=strength
            selectedX=gx
            selectedY=gy
          channel+=1
        dxs(i)=selectedX;dys(i)=selectedY;magnitude(i)=strongest
    val suppressed=new Array[Int](count)
    def magnitudeAt(x: Int,y: Int): Int=if x<0 || x>=width || y<0 || y>=height then 0 else magnitude(y*width+x)
    for y <- 0 until height do
      if cancelled() || Thread.currentThread().isInterrupted then
        throw new java.util.concurrent.CancellationException("ControlNet Canny preprocessing cancelled")
      for x <- 0 until width do
        val i=y*width+x;val gx=dxs(i);val gy=dys(i)
        val horizontal=math.abs(gy)*1000<=math.abs(gx)*414
        val vertical=math.abs(gx)*1000<=math.abs(gy)*414
        val (before,after)=
          if horizontal then (magnitudeAt(x-1,y),magnitudeAt(x+1,y))
          else if vertical then (magnitudeAt(x,y-1),magnitudeAt(x,y+1))
          else if (gx<0)==(gy<0) then (magnitudeAt(x-1,y-1),magnitudeAt(x+1,y+1))
          else (magnitudeAt(x+1,y-1),magnitudeAt(x-1,y+1))
        val value=magnitude(i)
        if value>before && (if horizontal || vertical then value>=after else value>after) then suppressed(i)=value
    val edges=new Array[Boolean](count)
    val queue=new java.util.ArrayDeque[Int]()
    var i=0
    while i<count do
      if suppressed(i)>highThreshold then
        edges(i)=true
        queue.addLast(i)
      i+=1
    var visits=0
    while !queue.isEmpty do
      if (visits&8191)==0 && (cancelled() || Thread.currentThread().isInterrupted) then
        throw new java.util.concurrent.CancellationException("ControlNet Canny preprocessing cancelled")
      visits+=1
      val current=queue.removeFirst();val x=current%width;val y=current/width
      var dy= -1
      while dy<=1 do
        var dx= -1
        while dx<=1 do
          val nx=x+dx;val ny=y+dy
          if nx>=0 && nx<width && ny>=0 && ny<height then
            val next=ny*width+nx
            if !edges(next) && suppressed(next)>lowThreshold then
              edges(next)=true
              queue.addLast(next)
          dx+=1
        dy+=1
    val output=new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB)
    i=0
    while i<count do
      output.setRGB(i%width,i/width,if edges(i) then 0xffffffff.toInt else 0xff000000.toInt)
      i+=1
    output

  private def sourceCanny(image: BufferedImage,resolution: Int,low: Int,high: Int,
      cancelled: () => Boolean): BufferedImage=
    require(resolution>0,"ControlNet Canny resolution must be positive")
    val scale=resolution.toDouble/math.min(image.getWidth,image.getHeight)
    val width=math.max(1,math.rint(image.getWidth*scale).toInt)
    val height=math.max(1,math.rint(image.getHeight*scale).toInt)
    val interpolation=if scale>1.0 then "cubic" else "area"
    val resized=NativeControlNetImageOps.cvResize(image,width,height,interpolation,cancelled)
    val paddedWidth=((width+63)/64)*64
    val paddedHeight=((height+63)/64)*64
    require(paddedWidth.toLong*paddedHeight<=16777216L,"ControlNet Canny padded image exceeds pixel limit")
    if paddedWidth==width && paddedHeight==height then canny(resized,low,high,cancelled)
    else
      val padded=new BufferedImage(paddedWidth,paddedHeight,BufferedImage.TYPE_INT_RGB)
      for y <- 0 until paddedHeight do
        if cancelled() || Thread.currentThread().isInterrupted then
          throw new java.util.concurrent.CancellationException("ControlNet Canny padding cancelled")
        val sy=math.min(y,height-1)
        for x <- 0 until paddedWidth do padded.setRGB(x,y,resized.getRGB(math.min(x,width-1),sy))
      val edges=canny(padded,low,high,cancelled)
      val cropped=new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB)
      for y <- 0 until height; x <- 0 until width do cropped.setRGB(x,y,edges.getRGB(x,y))
      cropped

  private def pixelPerfectResolution(image: BufferedImage,width: Int,height: Int,resize: Int): Int=
    val vertical=height.toDouble/image.getHeight
    val horizontal=width.toDouble/image.getWidth
    val scale=if resize==2 then math.min(vertical,horizontal) else math.max(vertical,horizontal)
    math.rint(scale*math.min(image.getWidth,image.getHeight)).toInt

  private def recolor(image: BufferedImage,luminance: Boolean,gamma: Double,cancelled: () => Boolean): BufferedImage=
    require(gamma.isFinite,"ControlNet recolor gamma must be finite")
    val output=new BufferedImage(image.getWidth,image.getHeight,BufferedImage.TYPE_INT_RGB)
    for y <- 0 until image.getHeight do
      if cancelled() || Thread.currentThread().isInterrupted then
        throw new java.util.concurrent.CancellationException("ControlNet recolor preprocessing cancelled")
      for x <- 0 until image.getWidth do
        val pixel=image.getRGB(x,y)
        val red=((pixel>>>16)&255)/255f
        val green=((pixel>>>8)&255)/255f
        val blue=(pixel&255)/255f
        val intensity=if luminance then 0.2126f*red+0.7152f*green+0.0722f*blue else math.max(red,math.max(green,blue))
        val corrected=math.pow(intensity.toDouble,gamma)*255.0
        val level=if corrected.isNaN then 0 else math.max(0,math.min(255,corrected.toInt))
        output.setRGB(x,y,0xff000000.toInt|(level<<16)|(level<<8)|level)
    output

  /** Preprocess without loading ControlNet weights, with NUI standalone resolution semantics. */
  def preprocessImage(name: String,source: BufferedImage,args: Map[String,NativeJson],standalone: Boolean,
      resolution: Int,cancelled: () => Boolean): BufferedImage =
    def numeric(key: String,default: Double): Double=args.get(key) match
      case None => default
      case Some(Number(value)) if value.toDouble.isFinite => value.toDouble
      case _ => throw IllegalArgumentException(s"ControlNet $key must be finite and numeric")
    name match
      case "none" | "inpaint" => source
      case "invert" =>
        val output=new BufferedImage(source.getWidth,source.getHeight,BufferedImage.TYPE_INT_RGB)
        for y <- 0 until source.getHeight do
          if cancelled() then throw new java.util.concurrent.CancellationException("preprocess cancelled")
          for x <- 0 until source.getWidth do output.setRGB(x,y,0xff000000.toInt | (~source.getRGB(x,y)&0xffffff))
        output
      case "canny" =>
        val low=numeric("low_threshold",100).toInt;val high=numeric("high_threshold",200).toInt
        if standalone then sourceCanny(source,resolution,if low>=1 && low<=255 then low else 100,
          if high>=1 && high<=255 then high else 200,cancelled)
        else canny(source,low,high,cancelled)
      case "recolor_luminance" | "recolor_intensity" =>
        recolor(source,name=="recolor_luminance",numeric("gamma_correction",1.0),cancelled)
      case "tile_resample" =>
        val down=numeric("down_sampling_rate",1.0)
        if down<=1 then source else
          val small=NativeControlNetImageOps.bicubic(source,math.max(1,(source.getWidth/down).toInt),
            math.max(1,(source.getHeight/down).toInt),cancelled)
          NativeControlNetImageOps.bicubic(small,source.getWidth,source.getHeight,cancelled)
      case "tile_colorfix" => NativeControlNetImageOps.gaussian(source,math.max(1,numeric("variation",8).toInt).toDouble,cancelled)
      case "blur_gaussian" => NativeControlNetImageOps.gaussian(source,numeric("sigma",9.0),cancelled)
      case "depth" | "depth_marigold" => NativeDepthAnything.preprocess(source,cancelled)
      case "normal_marigold" | "normal_bae" =>
        val w=source.getWidth;val h=source.getHeight
        val out=new BufferedImage(w,h,BufferedImage.TYPE_INT_RGB)
        def lumAt(x: Int,y: Int): Double =
          val cx=math.max(0,math.min(w-1,x));val cy=math.max(0,math.min(h-1,y))
          val c=source.getRGB(cx,cy)
          (0.299*((c>>>16)&255)+0.587*((c>>>8)&255)+0.114*(c&255))/255.0
        for y <- 0 until h do
          if cancelled() then throw new java.util.concurrent.CancellationException("preprocess cancelled")
          for x <- 0 until w do
            val gx=(lumAt(x+1,y)-lumAt(x-1,y))*2.0
            val gy=(lumAt(x,y+1)-lumAt(x,y-1))*2.0
            val gz=1.0
            val norm=math.sqrt(gx*gx+gy*gy+gz*gz)
            val nx=(-gx/norm*0.5+0.5)*255.0
            val ny=(-gy/norm*0.5+0.5)*255.0
            val nz=(gz/norm*0.5+0.5)*255.0
            val r=math.max(0,math.min(255,math.round(nx).toInt))
            val g=math.max(0,math.min(255,math.round(ny).toInt))
            val b=math.max(0,math.min(255,math.round(nz).toInt))
            out.setRGB(x,y,0xff000000.toInt|(r<<16)|(g<<8)|b)
        out
      case "softedge_hed" =>
        val safe=args.get("safe") match { case Some(Bool(true)) => true; case _ => false }
        NativeHed.preprocess(source,resolution,safe,cancelled)
      case "lineart" => NativeLineart.preprocess(source,resolution,cancelled)
      case "lineart_anime" => NativeLineartAnime.preprocess(source,cancelled)
      case "mlsd" =>
        val thrV=args.get("thr_v") match { case Some(Number(v)) if v>=0 => v.toFloat; case None => 0.1f; case _ => throw IllegalArgumentException("ControlNet mlsd thr_v must be a non-negative number") }
        val thrD=args.get("thr_d") match { case Some(Number(v)) if v>=0 => v.toFloat; case None => 0.1f; case _ => throw IllegalArgumentException("ControlNet mlsd thr_d must be a non-negative number") }
        NativeMlsd.preprocess(source,resolution,thrV,thrD,cancelled)
      case "openpose" => NativeOpenPose.preprocess(source,resolution,cancelled)
      case other => throw new NativeBackendError("unsupported",s"preprocessor '$other' is not available")

  def prepare(value: Option[NativeJson],modelPaths: Map[String,String],fallbackImage: Option[String],
      width: Int,height: Int,cancelled: () => Boolean,
      loadModel: Path => ConditionalControlNet2d): Vector[DiffusionControlNetRequest]=
    val units=value match
      case None | Some(Null) => Vector.empty
      case Some(Arr(values)) => values
      case _ => throw IllegalArgumentException("controlnet must be an array")
    val active=units.zipWithIndex.collect {
      case (Obj(fields),index) if fields.get("enabled").contains(Bool(true)) => index->fields
      case (other,index) if other!=Null && !other.isInstanceOf[Obj] =>
        throw IllegalArgumentException(s"controlnet[$index] must be an object")
    }
    require(active.size<=16,"native ControlNet accepts at most 16 enabled units")
    if active.isEmpty then Vector.empty
    else if active.size>1 then active.flatMap { case (_,fields) =>
      prepare(Some(Arr(Vector(Obj(fields)))),modelPaths,fallbackImage,width,height,cancelled,loadModel)
    }
    else
      val (index,fields)=active.head
      val allowed=Set("enabled","model","image","preprocessor","weight","guidance_start","guidance_end",
        "preprocessor_args","resize_mode","source_compatible_resize","control_mode","processor_res","pixel_perfect")
      val unknown=fields.keySet--allowed
      if unknown.nonEmpty then throw new NativeBackendError("unsupported",s"unsupported ControlNet fields: ${unknown.toVector.sorted.mkString(", ")}")
      val modelName=str(fields,"model","")
      require(modelName.nonEmpty && modelName!="None",s"controlnet[$index].model is required")
      val modelPath=modelPaths.getOrElse(modelName,modelName)
      val path=Path.of(modelPath)
      val filename=modelPath.toLowerCase(java.util.Locale.ROOT)
      if !java.nio.file.Files.isDirectory(path) && !(java.nio.file.Files.isRegularFile(path) &&
          Vector(".safetensors",".pt",".pth",".ckpt",".bin").exists(filename.endsWith)) then
        throw new NativeBackendError("unsupported","native ControlNet requires a Diffusers directory or supported SD1 checkpoint")
      val preprocessor=str(fields,"preprocessor","none").trim.toLowerCase(java.util.Locale.ROOT)
      if !Set("","none","inpaint","invert","canny","recolor_luminance","recolor_intensity",
          "tile_resample","tile_colorfix","blur_gaussian","depth","depth_marigold","normal_marigold",
          "openpose","lineart","lineart_anime","softedge_hed","mlsd","normal_bae")(preprocessor) then
        throw new NativeBackendError("unsupported",s"native ControlNet preprocessor '$preprocessor' is not implemented")
      val args=fields.get("preprocessor_args") match
        case None | Some(Null) => Map.empty[String,NativeJson]
        case Some(Obj(values)) => values
        case _ => throw IllegalArgumentException("ControlNet preprocessor_args must be an object")
      val thresholds=if preprocessor=="canny" then
        def threshold(key: String,default: Int): Int=args.get(key) match
          case None => default
          case Some(Number(value)) if value.isValidInt && value.toInt>=0 && value.toInt<=255 => value.toInt
          case _ => throw IllegalArgumentException(s"ControlNet Canny $key must be an integer in 0..255")
        val low=threshold("low_threshold",100);val high=threshold("high_threshold",200)
        Some(low->high)
      else
        None
      val resize=fields.get("resize_mode") match
        case None | Some(Null) => 1
        case Some(Number(value)) if value.isValidInt => value.toInt
        case _ => throw IllegalArgumentException("ControlNet resize_mode must be an integer")
      require(resize>=0 && resize<=2,"ControlNet resize_mode must be 0, 1 or 2")
      val sourceCompatibleResize=fields.get("source_compatible_resize") match
        case None | Some(Null) => false
        case Some(Bool(value)) => value
        case _ => throw IllegalArgumentException("ControlNet source_compatible_resize must be a boolean")
      val imageName=str(fields,"image","") match
        case "" => fallbackImage.getOrElse(throw new NativeBackendError("unsupported","ControlNet requires an image for txt2img"))
        case path => path
      val source=DiffusionImageInput.readRgb(Path.of(imageName))
      val preprocessed=preprocessor match
        case "invert" =>
          val output=new BufferedImage(source.getWidth,source.getHeight,BufferedImage.TYPE_INT_RGB)
          for y <- 0 until source.getHeight; x <- 0 until source.getWidth do
            val pixel=source.getRGB(x,y)
            output.setRGB(x,y,0xff000000.toInt|((255-((pixel>>>16)&255))<<16)|
              ((255-((pixel>>>8)&255))<<8)|(255-(pixel&255)))
          output
        case "canny" =>
          val (low,high)=thresholds.get
          val processorRes=fields.get("processor_res") match
            case None | Some(Null) => -1
            case Some(Number(value)) if value.isValidInt => value.toInt
            case _ => throw IllegalArgumentException("ControlNet processor_res must be an integer")
          val pixelPerfect=fields.get("pixel_perfect").contains(Bool(true))
          if pixelPerfect || processorRes!= -1 || sourceCompatibleResize then
            val resolution=if pixelPerfect then pixelPerfectResolution(source,width,height,resize)
              else if processorRes>=64 && processorRes<=2048 then processorRes else 512
            sourceCanny(source,resolution,if low>=1 then low else 100,if high>=1 then high else 200,cancelled)
          else canny(source,low,high,cancelled)
        case "recolor_luminance" | "recolor_intensity" =>
          val gamma=args.get("gamma_correction") match
            case None => 1.0
            case Some(Number(value)) => value.toDouble
            case _ => throw IllegalArgumentException("ControlNet gamma_correction must be numeric")
          recolor(source,preprocessor=="recolor_luminance",gamma,cancelled)
        case "tile_resample" =>
          val down=args.get("down_sampling_rate") match
            case None => 1.0
            case Some(Number(value)) => value.toDouble
            case _ => throw IllegalArgumentException("ControlNet down_sampling_rate must be numeric")
          require(down.isFinite,"ControlNet down_sampling_rate must be finite")
          if down<=1 then source
          else
            val smallWidth=math.max(1,(source.getWidth/down).toInt)
            val smallHeight=math.max(1,(source.getHeight/down).toInt)
            val small=NativeControlNetImageOps.bicubic(source,smallWidth,smallHeight,cancelled)
            NativeControlNetImageOps.bicubic(small,source.getWidth,source.getHeight,cancelled)
        case "tile_colorfix" =>
          val variation=args.get("variation") match
            case None => 8
            case Some(Number(value)) if value.isValidInt => value.toInt
            case _ => throw IllegalArgumentException("ControlNet variation must be an integer")
          NativeControlNetImageOps.gaussian(source,math.max(1,variation).toDouble,cancelled)
        case "blur_gaussian" =>
          val sigma=args.get("sigma") match
            case None => 9.0
            case Some(Number(value)) => value.toDouble
            case _ => throw IllegalArgumentException("ControlNet sigma must be numeric")
          NativeControlNetImageOps.gaussian(source,sigma,cancelled)
        case "depth" | "depth_marigold" | "normal_marigold" | "normal_bae" | "lineart" | "lineart_anime" | "softedge_hed" | "mlsd" | "openpose" =>
          preprocessImage(preprocessor,source,args,standalone=false,resolution=512,cancelled)
        case _ => source
      val resized=if sourceCompatibleResize then
        NativeControlNetImageOps.sourceCompatibleResize(preprocessed,width,height,resize,cancelled)
        else DiffusionImageResize(preprocessed,width,height,resize)
      val sourceImage=imageTensor(preprocessed,false,cancelled)
      val image=imageTensor(resized,false,cancelled)
      val model=loadModel(Path.of(modelPath))
      Vector(DiffusionControlNetRequest(model,image,number(fields,"weight",1.0),
        number(fields,"guidance_start",0.0),number(fields,"guidance_end",1.0),str(fields,"control_mode","Balanced"),
        resize,sourceCompatibleResize,Some(sourceImage),hiresSoftEligible=sourceCompatibleResize && preprocessor.contains("canny")))

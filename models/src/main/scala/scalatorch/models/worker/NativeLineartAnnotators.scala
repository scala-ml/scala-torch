package scalatorch.models.worker

import java.awt.image.BufferedImage
import java.nio.file.{Files,Path}
import scalatorch.models.{AnimeLineartUnet,ControlNetHed,LineartGenerator,MlsdLarge,U2Net}

/** Weight lookup shared by the annotators: SDFX_SCALATORCH_ANNOTATOR_DIR, then models/annotators in the working directory and
  * each of its parents. There is no fallback when the weights are missing. */
object NativeAnnotatorFiles:
  def candidates(fileName: String,env: Map[String,String]=sys.env,cwd: Path=Path.of("").toAbsolutePath): Vector[Path]=
    val fromEnv=env.get("SDFX_SCALATORCH_ANNOTATOR_DIR").filter(_.nonEmpty).toVector.map(d=>Path.of(d).resolve(fileName))
    fromEnv++Iterator.iterate(cwd)(_.getParent).takeWhile(_!=null).map(_.resolve("models").resolve("annotators").resolve(fileName)).toVector

  def locate(fileName: String,label: String,env: Map[String,String]=sys.env,cwd: Path=Path.of("").toAbsolutePath): Path=
    val options=candidates(fileName,env,cwd)
    options.find(Files.isRegularFile(_)).getOrElse(
      throw IllegalStateException(s"$label weights not found (expected $fileName); set SDFX_SCALATORCH_ANNOTATOR_DIR or place them under models/annotators. Searched: ${options.take(4).mkString(", ")}"))

  /** RGB CHW samples of an image, scaled by `scale` and offset by `offset` (`v * scale + offset`). */
  def rgbChw(image: BufferedImage,scale: Float,offset: Float): Array[Float]=
    val w=image.getWidth;val h=image.getHeight;val out=new Array[Float](3*w*h)
    for y <- 0 until h; x <- 0 until w do
      val c=image.getRGB(x,y);val i=y*w+x
      out(i)=((c>>>16)&255)*scale+offset;out(w*h+i)=((c>>>8)&255)*scale+offset;out(2*w*h+i)=(c&255)*scale+offset
    out

  /** Grey 3-channel image from 0..255 values (truncated like `astype(uint8)`). */
  def grey(values: Array[Float],w: Int,h: Int,invert: Boolean): BufferedImage=
    val image=new BufferedImage(w,h,BufferedImage.TYPE_INT_RGB)
    val pixels=new Array[Int](w*h)
    for i <- pixels.indices do
      val v0=math.max(0,math.min(255,values(i).toInt));val v=if invert then 255-v0 else v0
      pixels(i)=0xff000000.toInt|(v<<16)|(v<<8)|v
    image.setRGB(0,0,w,h,pixels,0,w);image

  /** Scales the source so its shorter side is `resolution` (0 keeps the size), then snaps both sides to a multiple of `multiple`. */
  def workingSize(width: Int,height: Int,resolution: Int,multiple: Int,roundUp: Boolean): (Int,Int)=
    val shorter=math.min(width,height)
    val s=if resolution>0 then resolution.toDouble/shorter else 1.0
    def snap(v: Int)=
      val scaled=v*s
      val m=if roundUp then math.ceil(scaled/multiple) else math.rint(scaled/multiple)
      math.max(multiple,m.toInt*multiple)
    (snap(width),snap(height))

  def resize(image: BufferedImage,w: Int,h: Int,cancelled: () => Boolean): BufferedImage=
    if image.getWidth==w && image.getHeight==h then image else NativeControlNetImageOps.cvResize(image,w,h,"cubic",cancelled)

/** The `softedge_hed` ControlNet preprocessor: the trained HED network (`ControlNetHED.pth`) on the CPU. */
object NativeHed:
  val FileName="ControlNetHED.pth"
  private var cached: Option[(Path,ControlNetHed)]=None
  def locate(env: Map[String,String]=sys.env,cwd: Path=Path.of("").toAbsolutePath): Path=NativeAnnotatorFiles.locate(FileName,"HED",env,cwd)
  private def model(file: Path,cancelled: () => Boolean): ControlNetHed=synchronized {
    cached match
      case Some((p,m)) if p==file => m
      case _ => val m=ControlNetHed.load(file,cancelled);cached=Some((file,m));m
  }

  /** Sigmoid edge map (0..1) at the image size, before 8-bit quantisation. */
  def edges(image: BufferedImage,weights: Path,cancelled: () => Boolean): Array[Float]=
    model(weights,cancelled).edges(NativeAnnotatorFiles.rgbChw(image,1f,0f),image.getHeight,image.getWidth,cancelled)

  /** Edge map as a 3-channel grey image (white edges on black); `safe` applies ControlNet's safe quantisation. */
  def preprocess(source: BufferedImage,resolution: Int,safe: Boolean,cancelled: () => Boolean,env: Map[String,String]=sys.env): BufferedImage=
    val weights=locate(env)
    val (w,h)=NativeAnnotatorFiles.workingSize(source.getWidth,source.getHeight,resolution,8,roundUp=false)
    val working=NativeAnnotatorFiles.resize(source,w,h,cancelled)
    val e=edges(working,weights,cancelled)
    val values=e.map(v=>(if safe then ControlNetHed.safeStep(v) else v)*255f)
    NativeAnnotatorFiles.resize(NativeAnnotatorFiles.grey(values,w,h,invert=false),source.getWidth,source.getHeight,cancelled)

/** The `lineart` ControlNet preprocessor: the trained line-art generator (`sk_model.pth`) on the CPU. */
object NativeLineart:
  val FileName="sk_model.pth"
  private var cached: Option[(Path,LineartGenerator)]=None
  def locate(env: Map[String,String]=sys.env,cwd: Path=Path.of("").toAbsolutePath): Path=NativeAnnotatorFiles.locate(FileName,"lineart",env,cwd)
  private def model(file: Path,cancelled: () => Boolean): LineartGenerator=synchronized {
    cached match
      case Some((p,m)) if p==file => m
      case _ => val m=LineartGenerator.load(file,cancelled);cached=Some((file,m));m
  }

  /** Raw generator output (0..1, dark lines on white) for a working image whose sides are divisible by 4. */
  def raw(image: BufferedImage,weights: Path,cancelled: () => Boolean): Array[Float]=
    model(weights,cancelled).infer(NativeAnnotatorFiles.rgbChw(image,1f/255f,0f),image.getHeight,image.getWidth,cancelled)

  /** White lines on black, the form ControlNet's lineart models expect. */
  def preprocess(source: BufferedImage,resolution: Int,cancelled: () => Boolean,env: Map[String,String]=sys.env): BufferedImage=
    val weights=locate(env)
    val (w,h)=NativeAnnotatorFiles.workingSize(source.getWidth,source.getHeight,resolution,4,roundUp=false)
    val working=NativeAnnotatorFiles.resize(source,w,h,cancelled)
    val out=raw(working,weights,cancelled)
    NativeAnnotatorFiles.resize(NativeAnnotatorFiles.grey(out.map(_*255f),w,h,invert=true),source.getWidth,source.getHeight,cancelled)

/** The `lineart_anime` ControlNet preprocessor: the trained anime line-art U-Net (`netG.pth`) on the CPU. */
object NativeLineartAnime:
  val FileName="netG.pth"
  private var cached: Option[(Path,AnimeLineartUnet)]=None
  def locate(env: Map[String,String]=sys.env,cwd: Path=Path.of("").toAbsolutePath): Path=NativeAnnotatorFiles.locate(FileName,"anime lineart",env,cwd)
  private def model(file: Path,cancelled: () => Boolean): AnimeLineartUnet=synchronized {
    cached match
      case Some((p,m)) if p==file => m
      case _ => val m=AnimeLineartUnet.load(file,cancelled);cached=Some((file,m));m
  }

  /** Raw generator output (-1..1) for a working image whose sides are divisible by 256. */
  def raw(image: BufferedImage,weights: Path,cancelled: () => Boolean): Array[Float]=
    model(weights,cancelled).infer(NativeAnnotatorFiles.rgbChw(image,1f/127.5f,-1f),image.getHeight,image.getWidth,cancelled)

  /** The source is resized (bicubic) up to multiples of 256, translated, mapped to 0..255 and resized back; white lines on black. */
  def preprocess(source: BufferedImage,cancelled: () => Boolean,env: Map[String,String]=sys.env): BufferedImage=
    val weights=locate(env)
    val w=256*math.ceil(source.getWidth/256.0).toInt;val h=256*math.ceil(source.getHeight/256.0).toInt
    val working=NativeAnnotatorFiles.resize(source,w,h,cancelled)
    val out=raw(working,weights,cancelled)
    NativeAnnotatorFiles.resize(NativeAnnotatorFiles.grey(out.map(_*127.5f+127.5f),w,h,invert=true),source.getWidth,source.getHeight,cancelled)

/** The `mlsd` ControlNet preprocessor: the M-LSD large line-segment detector (`mlsd_large_512_fp32.pth`) on the CPU. */
object NativeMlsd:
  val FileName="mlsd_large_512_fp32.pth"
  val InputSize=512
  private var cached: Option[(Path,MlsdLarge)]=None
  def locate(env: Map[String,String]=sys.env,cwd: Path=Path.of("").toAbsolutePath): Path=NativeAnnotatorFiles.locate(FileName,"M-LSD",env,cwd)
  private def model(file: Path,cancelled: () => Boolean): MlsdLarge=synchronized {
    cached match
      case Some((p,m)) if p==file => m
      case _ => val m=MlsdLarge.load(file,cancelled);cached=Some((file,m));m
  }

  /** Segments (best first) in the coordinates of `image`: the image is area-resized to 512x512, the network runs on it and the
    * decoded 512-space segments are scaled back. */
  def segments(image: BufferedImage,weights: Path,thrV: Float,thrD: Float,cancelled: () => Boolean): Vector[(Float,MlsdLarge.Segment)]=
    val net=NativeControlNetImageOps.cvResize(image,InputSize,InputSize,"area",cancelled)
    val (tp,h,w)=model(weights,cancelled).tpMap(NativeAnnotatorFiles.rgbChw(net,1f,0f),InputSize,InputSize,cancelled)
    val sx=image.getWidth.toFloat/InputSize;val sy=image.getHeight.toFloat/InputSize
    MlsdLarge.decode(tp,h,w,thrV,thrD).map { (s,l) => (s,(l._1*sx,l._2*sy,l._3*sx,l._4*sy)) }

  /** White 1-pixel lines on black (integer endpoints, truncated like `int()`), at the working size. */
  def render(segs: Seq[MlsdLarge.Segment],w: Int,h: Int): BufferedImage=
    val pix=new Array[Boolean](w*h)
    for (x0,y0,x1,y1) <- segs do MlsdLarge.drawLine(pix,w,h,x0.toInt,y0.toInt,x1.toInt,y1.toInt)
    val image=new BufferedImage(w,h,BufferedImage.TYPE_INT_RGB)
    image.setRGB(0,0,w,h,Array.tabulate(w*h)(i=>if pix(i) then 0xffffffff else 0xff000000.toInt),0,w);image

  /** ControlNet defaults: `thr_v` 0.1, `thr_d` 0.1; the source is resized to `resolution` (multiples of 64) first and the
    * line map is resized back to the source size. */
  def preprocess(source: BufferedImage,resolution: Int,thrV: Float,thrD: Float,cancelled: () => Boolean,env: Map[String,String]=sys.env): BufferedImage=
    val weights=locate(env)
    val (w,h)=NativeAnnotatorFiles.workingSize(source.getWidth,source.getHeight,resolution,64,roundUp=false)
    val working=NativeAnnotatorFiles.resize(source,w,h,cancelled)
    val lines=render(segments(working,weights,thrV,thrD,cancelled).map(_._2),w,h)
    NativeAnnotatorFiles.resize(lines,source.getWidth,source.getHeight,cancelled)

/** The `remove_background` image tool: U2-Net (full, rembg's `u2net` weights converted to `u2net/u2net.safetensors` by
  * scripts/convert_u2net_onnx_to_safetensors.py) on the CPU. Documented rembg preprocessing: RGB, resize to 320x320, divide by the image
  * maximum, ImageNet mean/std; the fused map d0 (sigmoid already applied) is min-max normalised to 0..1 and resized back to the source
  * size to be the alpha matte. U2-Net code is Apache-2.0 (Qin et al. 2020); no upstream code is copied. */
object NativeU2Net:
  val FileName="u2net/u2net.safetensors"
  val InputSize=320
  private var cached: Option[(Path,U2Net)]=None
  def locate(env: Map[String,String]=sys.env,cwd: Path=Path.of("").toAbsolutePath): Path=NativeAnnotatorFiles.locate(FileName,"U2-Net (u2net)",env,cwd)
  private def model(file: Path): U2Net=synchronized {
    cached match
      case Some((p,m)) if p==file => m
      case _ => val m=U2Net.load(file);cached=Some((file,m));m
  }

  /** Normalised 3x320x320 network input for `image`. */
  def input(image: BufferedImage,cancelled: () => Boolean): Array[Float]=
    val small=NativeControlNetImageOps.cvResize(image,InputSize,InputSize,"area",cancelled)
    val chw=NativeAnnotatorFiles.rgbChw(small,1f,0f)
    var mx=0f;var i=0
    while i<chw.length do { if chw(i)>mx then mx=chw(i);i+=1 }
    val inv=1f/math.max(mx,1e-6f);val n=InputSize*InputSize
    val mean=Array(0.485f,0.456f,0.406f);val std=Array(0.229f,0.224f,0.225f)
    i=0
    while i<chw.length do { val c=i/n;chw(i)=(chw(i)*inv-mean(c))/std(c);i+=1 }
    chw

  /** d0 and the side maps (sigmoid) at 320x320. */
  def maps(image: BufferedImage,weights: Path,cancelled: () => Boolean): Vector[Array[Float]]=
    model(weights).infer(input(image,cancelled),InputSize,InputSize)

  /** Min-max normalised d0 resized (bilinear) to `w` x `h`, 0..1. */
  def matte(d0: Array[Float],w: Int,h: Int): Array[Float]=
    var lo=Float.PositiveInfinity;var hi=Float.NegativeInfinity
    for v <- d0 do { if v<lo then lo=v;if v>hi then hi=v }
    val range=math.max(hi-lo,1e-12f)
    U2Net.bilinear(d0.map(v=>(v-lo)/range),1,InputSize,InputSize,h,w).map(v=>math.max(0f,math.min(1f,v)))

  /** Alpha matte (0..255, row-major) at the source size. */
  def alpha(image: BufferedImage,cancelled: () => Boolean=()=>false,env: Map[String,String]=sys.env): Array[Int]=
    val weights=locate(env)
    matte(maps(image,weights,cancelled).head,image.getWidth,image.getHeight).map(v=>math.round(v*255f))

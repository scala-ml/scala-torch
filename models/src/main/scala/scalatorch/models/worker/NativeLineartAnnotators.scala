package scalatorch.models.worker

import java.awt.image.BufferedImage
import java.nio.file.{Files,Path}
import scalatorch.models.{AnimeLineartUnet,ControlNetHed,LineartGenerator}

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

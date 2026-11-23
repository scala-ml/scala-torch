package scalatorch.models.worker

import java.awt.image.BufferedImage
import java.nio.file.Path
import scalatorch.models.{DiffusionImageResize,Waifu2x}

/** The `waifu2x` image tool / Waifu2x upscaler on the JVM: nagadomi's `upconv_7` (see [[scalatorch.models.Waifu2x]]), styles Anime ("art") and
  * Photo, denoise level 0..3, repeated 2x passes up to the requested scale and a final Lanczos resample to the exact target size (the same
  * scheme as the NUI Python engine, python/sdfx/waifu2x.py). The Python engine's CarnV2 style (GPL-3.0 weights) is not implemented natively.
  * Weights live under `waifu2x/` in the annotator folder (see NativeAnnotatorFiles); missing weights fail loudly. */
object NativeWaifu2x:
  private var cached: Option[(Path,Waifu2x)]=None
  private val MaxPixels=16777216L

  def locate(style: String,noise: Int,env: Map[String,String]=sys.env,cwd: Path=Path.of("").toAbsolutePath): Path=
    NativeAnnotatorFiles.locate(Waifu2x.fileName(style,noise),s"waifu2x ($style, noise $noise)",env,cwd)

  private def model(file: Path): Waifu2x=synchronized {
    cached match
      case Some((p,m)) if p==file => m
      case _ => val m=Waifu2x.load(file);cached=Some((file,m));m
  }

  /** Maps NUI style names to the model folder name ("Anime" -> art). */
  def styleKey(style: String): String=style.toLowerCase(java.util.Locale.ROOT) match
    case "anime" | "art" => "art"
    case "photo" => "photo"
    case "carnv2" => throw new NativeBackendError("unsupported","waifu2x style CarnV2 is not implemented in the native backend (GPL-3.0 CARN weights); use Anime or Photo")
    case other => throw new NativeBackendError("invalid_params",s"invalid waifu2x style '$other'")

  /** CHW 0..1 samples of an image. */
  def toChw(image: BufferedImage): Array[Float]=NativeAnnotatorFiles.rgbChw(image,1f/255f,0f)

  /** Clamp to 0..1 and round to 8 bits. */
  def toImage(chw: Array[Float],w: Int,h: Int): BufferedImage=
    val image=new BufferedImage(w,h,BufferedImage.TYPE_INT_RGB);val px=new Array[Int](w*h);val n=w*h
    def q(v: Float)=math.max(0,math.min(255,math.round(v*255f)))
    var i=0
    while i<n do { px(i)=0xff000000.toInt|(q(chw(i))<<16)|(q(chw(n+i))<<8)|q(chw(2*n+i));i+=1 }
    image.setRGB(0,0,w,h,px,0,w);image

  /** Number of native 2x passes for a target (at least one, so scale 1 is a pure denoise). */
  def passes(srcW: Int,srcH: Int,width: Int,height: Int): Int=
    val ratio=math.max(width.toDouble/srcW,height.toDouble/srcH)
    math.max(1,math.ceil(math.log(ratio)/math.log(2.0)-1e-9).toInt)

  def upscale(source: BufferedImage,width: Int,height: Int,style: String,noise: Int,cancelled: () => Boolean,tile: Int=256,
      progress: (Int,Int)=>Unit=(_,_)=>(),env: Map[String,String]=sys.env): BufferedImage=
    val key=styleKey(style)
    require(noise>=0 && noise<=3,"waifu2x noise must be 0..3")
    val n=passes(source.getWidth,source.getHeight,width,height)
    if n>4 then throw new NativeBackendError("invalid_params","waifu2x target may not exceed 16x the input size")
    val finalPixels=source.getWidth.toLong*source.getHeight<<(2*n)
    if finalPixels>MaxPixels then throw new NativeBackendError("invalid_params",s"waifu2x intermediate size exceeds the pixel limit ($MaxPixels)")
    val net=model(locate(key,noise,env))
    var cur=toChw(source);var h=source.getHeight;var w=source.getWidth;var p=0
    while p<n do
      val base=p
      cur=net.upscale2x(cur,h,w,tile,cancelled,(d,t)=>progress(base*1000+d*1000/t,n*1000))
      h*=2;w*=2;p+=1
      // intermediate passes are re-quantised to 8 bits like any image hand-off; the last pass is quantised once below
      if p<n then cur=toChw(toImage(cur,w,h))
    val full=toImage(cur,w,h)
    if w==width && h==height then full else DiffusionImageResize(full,width,height,0)

package scalatorch.models.worker

import java.awt.image.BufferedImage
import java.nio.file.Path
import scalatorch.models.{DepthAnythingV2,NormalBae}

/** The `normal_bae` ControlNet preprocessor: the trained NormalBAE network (`scannet.pt`, lllyasviel/Annotators) on the CPU, with no
  * fallback when the weights are missing.
  *
  * Documented ControlNet-annotator behaviour: RGB; shorter side scaled to the processor resolution (default 512) and both sides rounded
  * to multiples of 64 (area interpolation when shrinking); pixels / 255, ImageNet mean/std; the network's final-resolution prediction
  * is mapped `((n + 1) * 0.5)` clipped to 0..1, times 255, truncated to uint8, with the normal's x, y, z written to R, G, B (no channel
  * flip); the map is resized (linear) to the output size, which here is the source size like the other annotators. */
object NativeNormalBae:
  val FileName=NormalBae.FileName
  private val Mean=Array(0.485f,0.456f,0.406f);private val Std=Array(0.229f,0.224f,0.225f)
  private var cached: Option[(Path,NormalBae)]=None
  def locate(env: Map[String,String]=sys.env,cwd: Path=Path.of("").toAbsolutePath): Path=NativeAnnotatorFiles.locate(FileName,"NormalBAE (normal_bae)",env,cwd)
  private def model(file: Path,cancelled: () => Boolean): NormalBae=synchronized {
    cached match
      case Some((p,m)) if p==file => m
      case _ => val m=NormalBae.load(file,cancelled);cached=Some((file,m));m
  }

  /** ImageNet-normalised CHW input of `image`. */
  def input(image: BufferedImage): Array[Float]=
    val chw=NativeAnnotatorFiles.rgbChw(image,1f/255f,0f);val n=image.getWidth*image.getHeight
    var i=0
    while i<chw.length do { val c=i/n;chw(i)=(chw(i)-Mean(c))/Std(c);i+=1 }
    chw

  /** Raw 4 x h x w network output (unit normals and kappa) for a working image whose sides are multiples of 32. */
  def raw(image: BufferedImage,weights: Path,cancelled: () => Boolean): Array[Float]=
    model(weights,cancelled).infer(input(image),image.getHeight,image.getWidth,cancelled)

  /** Normal map (RGB = x, y, z) at the working size. */
  def normalMap(image: BufferedImage,weights: Path,cancelled: () => Boolean): BufferedImage=
    val w=image.getWidth;val h=image.getHeight
    val out=new BufferedImage(w,h,BufferedImage.TYPE_INT_RGB)
    out.setRGB(0,0,w,h,NormalBae.toRgb8(raw(image,weights,cancelled),h,w),0,w);out

  /** Bilinear (half-pixel, like OpenCV INTER_LINEAR) resize of an RGB image. */
  def linearResize(image: BufferedImage,w: Int,h: Int): BufferedImage=
    if image.getWidth==w && image.getHeight==h then image else
      val sw=image.getWidth;val sh=image.getHeight;val chw=NativeAnnotatorFiles.rgbChw(image,1f,0f)
      val r=DepthAnythingV2.bilinear(chw,3,sh,sw,h,w,false);val n=w*h
      val out=new BufferedImage(w,h,BufferedImage.TYPE_INT_RGB)
      def q(v: Float)=math.max(0,math.min(255,math.rint(v).toInt))
      out.setRGB(0,0,w,h,Array.tabulate(n)(i=>0xff000000.toInt|(q(r(i))<<16)|(q(r(n+i))<<8)|q(r(2*n+i))),0,w);out

  def preprocess(source: BufferedImage,resolution: Int,cancelled: () => Boolean,env: Map[String,String]=sys.env): BufferedImage=
    val weights=locate(env)
    val (w,h)=NativeAnnotatorFiles.workingSize(source.getWidth,source.getHeight,resolution,64,roundUp=false)
    val working=if source.getWidth==w && source.getHeight==h then source else
      NativeControlNetImageOps.cvResize(source,w,h,if w*h<=source.getWidth*source.getHeight then "area" else "cubic",cancelled)
    linearResize(normalMap(working,weights,cancelled),source.getWidth,source.getHeight)

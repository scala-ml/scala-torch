package scalatorch.models.worker

import java.awt.image.BufferedImage
import java.nio.file.{Files,Path}
import scalatorch.models.DepthAnythingV2

/** The `depth` ControlNet preprocessor: Depth Anything V2 Small on the CPU, no fallback when the weights are missing. */
object NativeDepthAnything:
  val DirectoryName="depth-anything-v2-small"
  private val Mean=Array(0.485f,0.456f,0.406f);private val Std=Array(0.229f,0.224f,0.225f)
  private var cached: Option[(Path,DepthAnythingV2)]=None

  /** Candidate weight locations: SDFX_SCALATORCH_ANNOTATOR_DIR (the model directory or its parent), then
    * models/annotators/depth-anything-v2-small in the working directory and each of its parents. */
  def candidates(env: Map[String,String]=sys.env,cwd: Path=Path.of("").toAbsolutePath): Vector[Path]=
    val fromEnv=env.get("SDFX_SCALATORCH_ANNOTATOR_DIR").filter(_.nonEmpty).toVector.flatMap { d =>
      val p=Path.of(d);Vector(p.resolve(DirectoryName),p)
    }
    val parents=Iterator.iterate(cwd)(_.getParent).takeWhile(_!=null).map(_.resolve("models").resolve("annotators").resolve(DirectoryName)).toVector
    fromEnv++parents

  def locate(env: Map[String,String]=sys.env,cwd: Path=Path.of("").toAbsolutePath): Path=
    val options=candidates(env,cwd)
    options.find(p=>Files.isRegularFile(p.resolve("model.safetensors"))).getOrElse(
      throw IllegalStateException(s"Depth Anything V2 weights not found (expected $DirectoryName/model.safetensors); set SDFX_SCALATORCH_ANNOTATOR_DIR or place them under models/annotators. Searched: ${options.take(4).mkString(", ")}"))

  private def model(dir: Path): DepthAnythingV2=synchronized {
    cached match
      case Some((p,m)) if p==dir => m
      case _ => val m=DepthAnythingV2.load(dir);cached=Some((dir,m));m
  }

  /** HF DPT sizing: keep the aspect ratio, scale towards 518 on the side nearer to 1, round each side to a multiple of 14. */
  def inputSize(width: Int,height: Int): (Int,Int)=
    val sw=518.0/width;val sh=518.0/height
    val s=if math.abs(1-sw)<math.abs(1-sh) then sw else sh
    def snap(v: Int)=math.max(14,math.round(s*v/14.0).toInt*14)
    (snap(width),snap(height))

  def normalise(image: BufferedImage): Array[Float]=
    val w=image.getWidth;val h=image.getHeight;val out=new Array[Float](3*w*h)
    for y <- 0 until h; x <- 0 until w do
      val c=image.getRGB(x,y)
      out(y*w+x)=(((c>>>16)&255)/255f-Mean(0))/Std(0)
      out(w*h+y*w+x)=(((c>>>8)&255)/255f-Mean(1))/Std(1)
      out(2*w*h+y*w+x)=((c&255)/255f-Mean(2))/Std(2)
    out

  /** Raw relative inverse depth at the source size (bilinear, as the HF post-processor). */
  def depth(source: BufferedImage,weights: Path,cancelled: () => Boolean): (Array[Float],Int,Int)=
    val (nw,nh)=inputSize(source.getWidth,source.getHeight)
    val scaled=NativeControlNetImageOps.bicubic(source,nw,nh,cancelled)
    val out=model(weights).infer(normalise(scaled),nh,nw,cancelled).depth
    (DepthAnythingV2.bilinear(out,1,nh,nw,source.getHeight,source.getWidth,false),source.getWidth,source.getHeight)

  /** Min-max normalised depth as a grey RGB image: near is bright (Depth Anything already predicts inverse depth). */
  def preprocess(source: BufferedImage,cancelled: () => Boolean,env: Map[String,String]=sys.env): BufferedImage=
    val (d,w,h)=depth(source,locate(env),cancelled)
    val lo=d.min;val hi=d.max;val range=if hi>lo then hi-lo else 1f
    val image=new BufferedImage(w,h,BufferedImage.TYPE_INT_RGB)
    for y <- 0 until h; x <- 0 until w do
      val v=math.max(0,math.min(255,math.round((d(y*w+x)-lo)/range*255f)))
      image.setRGB(x,y,0xff000000.toInt|(v<<16)|(v<<8)|v)
    image

package scalatorch.models.worker

import java.awt.image.BufferedImage
import java.nio.file.Path
import scalatorch.models.{MarigoldNormals,NormalBae,UNetExecution,VaeExecution}

/** The `normal_marigold` ControlNet preprocessor: real Marigold surface-normal estimation (`prs-eth/marigold-normals-v1-1`) on the
  * worker's own U-Net / VAE / CLIP components, with no fallback when the weights are missing.
  *
  * Execution follows the worker's environment exactly like the diffusion backend: `SDFX_SCALATORCH_UNET_DEVICE`/`_MAX_BYTES` and
  * `SDFX_SCALATORCH_VAE_DEVICE`/`_MAX_BYTES` (CPU is correct but very slow for an SD2-size U-Net; use `cuda`), FP16 U-Net on CUDA
  * unless `SCALATORCH_CUDA_FP16=0`.
  *
  * Output convention: identical to `normal_bae` (`NormalBae.toRgb8`): x, y, z of the unit normal written to R, G, B as
  * `trunc(clip((n + 1) / 2, 0, 1) * 255)`. Marigold's own frame is x right, y up, z towards the viewer (documented by its
  * `visualize_normals`). The NormalBAE map that ControlNet's normal model is used with has the same y (up: ground is green) and
  * z (towards the viewer: blue) but the opposite x (a surface facing left is red). Measured on three generated images the per-pixel
  * correlation of the x channels of the two preprocessors is -0.65, -0.22 and -0.86 for the raw Marigold frame, and y, z are +0.68/+0.63,
  * +0.46/+0.44, +0.93/+0.72; so x is negated here, which makes `normal_marigold` and `normal_bae` interchangeable.
  *
  * The result is deterministic: the initial noise is `java.util.Random(seed).nextGaussian()` (default seed 0), not a PyTorch stream. */
object NativeMarigoldNormals:
  val Unet="unet/diffusion_pytorch_model.safetensors"
  private val Relative=s"${MarigoldNormals.ModelDirectory}/$Unet"
  private var cached: Option[(Path,java.lang.ref.SoftReference[MarigoldNormals])]=None

  /** The model directory (the one holding `unet/`, `vae/`, `text_encoder/`, `tokenizer/`), from SDFX_SCALATORCH_ANNOTATOR_DIR or
    * models/annotators in the working directory and its parents. */
  def locate(env: Map[String,String]=sys.env,cwd: Path=Path.of("").toAbsolutePath): Path=
    NativeAnnotatorFiles.locate(Relative,"Marigold normals (normal_marigold)",env,cwd).getParent.getParent

  /** Same execution settings as the diffusion backend. Without an explicit `SDFX_SCALATORCH_*_MAX_BYTES` a CUDA annotator session gets
    * up to three quarters of the free device memory (at most 8 GiB, instead of the backend's 2 GiB): the SD2-size U-Net weights alone
    * are 1.7 GB in FP16 and 3.5 GB in FP32, and the session is released as soon as the estimate is done. */
  def executions(env: Map[String,String]): (UNetExecution,VaeExecution,Long,Long)=
    def sized(device: String,budget: String,current: Map[String,String])=
      val index=current.get(device).flatMap {
        case "cuda" => Some(0)
        case v if v.matches("cuda:[0-9]+") => Some(v.drop(5).toInt)
        case _ => None
      }
      if current.contains(budget) || index.isEmpty then current
      else current.updated(budget,math.min(8L<<30,scalatorch.ffm.CudaProbe.memoryInfo(index.get).freeBytes/4*3).toString)
    val merged=sized("SDFX_SCALATORCH_VAE_DEVICE","SDFX_SCALATORCH_VAE_MAX_BYTES",
      sized("SDFX_SCALATORCH_UNET_DEVICE","SDFX_SCALATORCH_UNET_MAX_BYTES",env))
    val vae=VaeExecution.fromEnvironment(merged);val unet=UNetExecution.fromEnvironment(merged)
    val configured=merged.get("SDFX_SCALATORCH_MAX_ATTENTION_SCORE_ELEMENTS")
    val attention=NativeWorkerMain.attentionScoreLimit(configured,unet.device!="cpu")
    (unet,vae,attention,NativeWorkerMain.vaeAttentionLimit(attention,configured.nonEmpty,vae.device!="cpu"))

  /** Loaded components, kept softly so a memory-constrained worker can reclaim the 3.5 GB of CPU weights. */
  def model(root: Path,env: Map[String,String]): MarigoldNormals=synchronized {
    cached.flatMap { case (p,ref) => if p==root then Option(ref.get) else None }.getOrElse {
      val (unet,vae,attention,vaeAttention)=executions(env)
      val loaded=MarigoldNormals.load(root,unet,vae,attention,vaeAttention)
      cached=Some((root,new java.lang.ref.SoftReference(loaded)));loaded
    }
  }

  def gaussian(seed: Long,count: Int): Array[Float]=
    val random=new java.util.Random(seed)
    Array.fill(count)(random.nextGaussian().toFloat)

  /** Unit normals (x, y, z planes) at the source size, in the NormalBAE / ControlNet frame (x negated relative to Marigold). */
  def normals(source: BufferedImage,root: Path,steps: Int,resolution: Int,seed: Long,cancelled: () => Boolean,
      env: Map[String,String]=sys.env): Array[Float]=
    val rgb=NativeAnnotatorFiles.rgbChw(source,1f/255f,0f)
    val m=model(root,env)
    val prepared=MarigoldNormals.prepare(rgb,source.getHeight,source.getWidth,resolution)
    val latent=m.encode(prepared,cancelled)
    val target=m.denoise(latent,scalatorch.Tensor.fromFloats(gaussian(seed,latent.shape.numel.toInt),latent.shape),steps,cancelled)
    val n=source.getHeight*source.getWidth
    val result=MarigoldNormals.finish(m.decode(target,cancelled),prepared,source.getHeight,source.getWidth)
    var i=0
    while i<n do { result(i)= -result(i);i+=1 } // Marigold x (right) -> NormalBAE/ControlNet x (see above)
    result

  def preprocess(source: BufferedImage,cancelled: () => Boolean,steps: Int=MarigoldNormals.DefaultSteps,
      resolution: Int=MarigoldNormals.DefaultResolution,seed: Long=0L,env: Map[String,String]=sys.env): BufferedImage=
    require(steps>=1 && steps<=50,"Marigold denoising steps must be in 1..50")
    require(resolution==0 || (resolution>=64 && resolution<=2048),"Marigold processing resolution must be 0 or in 64..2048")
    val root=locate(env)
    val w=source.getWidth;val h=source.getHeight
    val out=new BufferedImage(w,h,BufferedImage.TYPE_INT_RGB)
    out.setRGB(0,0,w,h,NormalBae.toRgb8(normals(source,root,steps,resolution,seed,cancelled,env),h,w),0,w);out

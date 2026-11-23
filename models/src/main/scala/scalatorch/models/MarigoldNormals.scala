package scalatorch.models

import java.nio.file.Path
import scalatorch.*
import scalatorch.autograd.{Autograd,Variable}

/** Marigold surface-normal estimation (Ke et al.; `prs-eth/marigold-normals-v1-1`), re-implemented from the documented algorithm and
  * built from this repository's verified diffusion components: the Stable Diffusion 2 U-Net (here with 8 input channels), the
  * AutoencoderKL and the CLIP text encoder. There is no Python, and no other network.
  *
  * Algorithm (single ensemble member, the `MarigoldNormalsPipeline` defaults of the v1-1 model):
  *  1. the image is scaled so its longer side is the processing resolution (default 768; bilinear, antialiased when shrinking) and
  *     padded to a multiple of 8 by replicating the border; values are mapped to [-1, 1];
  *  2. the VAE encoder's posterior mean times the scaling factor (0.18215) is the image latent;
  *  3. the conditioning is the text encoder's final hidden state for the empty prompt, unpadded: two tokens (BOS, EOS), not 77;
  *  4. the target latent starts as Gaussian noise; at every DDIM step (trailing spacing, v-prediction, zero-terminal-SNR betas,
  *     eta 0, no clipping) the U-Net input is the image latent and the target latent concatenated along channels (image first);
  *  5. the final target latent / 0.18215 is decoded, clipped to [-1, 1], L2-normalised per pixel, un-padded, bilinearly resized to
  *     the input size and normalised again. Channels are x (right), y (up), z (towards the viewer).
  *
  * Licence: the Marigold code is Apache-2.0; the `marigold-normals-v1-1` weights are released under OpenRAIL++-M (the Stable
  * Diffusion 2 licence the model was fine-tuned under; see the model card). */
final class MarigoldNormals(unet: ConditionalUNet2d,vae: VaeModel,textEmbedding: Tensor[Float],
    unetExecution: UNetExecution,vaeExecution: VaeExecution,scalingFactor: Float=MarigoldNormals.ScalingFactor):
  import MarigoldNormals.*
  require(unet.convIn.weight.data.shape.dims(1)==2*vae.config.latentChannels,"Marigold U-Net must take image and target latents")
  require(textEmbedding.shape.rank==3 && textEmbedding.shape.dims(0)==1,"Marigold text embedding must be 1 x tokens x width")

  /** Scaled posterior mean (1 x 4 x h/8 x w/8) of a prepared image. */
  def encode(prepared: Prepared,cancelled: () => Boolean=()=>false): Tensor[Float] =
    val image=Tensor.fromFloats(prepared.data,Shape.from(Seq(1L,3L,prepared.height.toLong,prepared.width.toLong)))
    val moments=vaeExecution.encode(vae.encoder,image,cancelled)
    val dims=moments.shape.dims
    require(dims.length==4 && dims(1)==2*vae.config.latentChannels,"unexpected VAE moments shape")
    val c=vae.config.latentChannels;val plane=(dims(2)*dims(3)).toInt;val all=moments.toFloatArray
    val mean=java.util.Arrays.copyOfRange(all,0,c*plane)
    var i=0
    while i<mean.length do { mean(i)*=scalingFactor;i+=1 }
    Tensor.fromFloats(mean,Shape.from(Seq(1L,c.toLong,dims(2),dims(3))))

  /** The DDIM loop; `noise` is the initial target latent. Returns the final target latent. */
  def denoise(imageLatent: Tensor[Float],noise: Tensor[Float],steps: Int,cancelled: () => Boolean=()=>false): Tensor[Float] =
    require(imageLatent.shape==noise.shape,"image latent and noise shapes differ")
    val schedule=Schedule(steps)
    val image=imageLatent.toFloatArray
    var target=noise.toFloatArray
    val d=noise.shape.dims
    val shape8=Shape.from(Seq(1L,2L*d(1),d(2),d(3)))
    unetExecution.withPredictor(unet,textEmbedding,None,cancelled) { predict =>
      for index <- schedule.timesteps.indices do
        VaeExecution.checkCancelled(cancelled)
        val input=new Array[Float](image.length+target.length)
        System.arraycopy(image,0,input,0,image.length);System.arraycopy(target,0,input,image.length,target.length)
        val t=Tensor.fromFloats(Array(schedule.timesteps(index).toFloat),Shape.from(Seq(1L)))
        val v=Autograd.noGrad(predict(Tensor.fromFloats(input,shape8),t,1f)).toFloatArray
        require(v.length==target.length,"unexpected Marigold U-Net output shape")
        target=schedule.step(index,v,target)
    }
    Tensor.fromFloats(target,noise.shape)

  /** Decoded, clipped and per-pixel normalised normals, 3 x padded height x padded width (still padded). */
  def decode(latent: Tensor[Float],cancelled: () => Boolean=()=>false): Array[Float] =
    val scaled=latent.toFloatArray.map(_/scalingFactor)
    val out=vaeExecution.decode(vae.decoder,Tensor.fromFloats(scaled,latent.shape),cancelled)
    val dims=out.shape.dims
    require(dims.length==4 && dims(0)==1 && dims(1)==3,"unexpected VAE output shape")
    val data=out.toFloatArray
    var i=0
    while i<data.length do { data(i)=math.max(-1f,math.min(1f,data(i)));i+=1 }
    normalise(data,(dims(2)*dims(3)).toInt)
    data

  /** Full estimate: unit normals (x, y, z planes) at the input size `originalHeight` x `originalWidth`. */
  def estimate(rgb: Array[Float],originalHeight: Int,originalWidth: Int,noise: Array[Float],steps: Int,
      processingResolution: Int,cancelled: () => Boolean=()=>false): Array[Float] =
    val prepared=prepare(rgb,originalHeight,originalWidth,processingResolution)
    val latent=encode(prepared,cancelled)
    require(noise.length==latent.shape.numel.toInt,s"noise must have ${latent.shape.numel} values")
    val target=denoise(latent,Tensor.fromFloats(noise,latent.shape),steps,cancelled)
    finish(decode(target,cancelled),prepared,originalHeight,originalWidth)

object MarigoldNormals:
  val ScalingFactor=0.18215f
  val ModelDirectory="marigold-normals-v1-1"
  val DefaultSteps=4
  val DefaultResolution=768

  /** A model image ready for the VAE: CHW in [-1, 1], padded height/width (multiples of 8) and the padding that was added. */
  final case class Prepared(data: Array[Float],height: Int,width: Int,padHeight: Int,padWidth: Int):
    def innerHeight: Int=height-padHeight
    def innerWidth: Int=width-padWidth

  /** `rgb` is CHW in [0, 1]. The longer side becomes `maxEdge` (0 keeps the size), then each side is padded up to a multiple of 8. */
  def prepare(rgb: Array[Float],height: Int,width: Int,maxEdge: Int): Prepared =
    require(rgb.length==3*height*width && height>0 && width>0,"invalid Marigold input image")
    val (h,w)=if maxEdge>0 then
      val longer=math.max(height,width)
      (math.max(1,(height.toLong*maxEdge/longer).toInt),math.max(1,(width.toLong*maxEdge/longer).toInt))
    else (height,width)
    val scaled=if h==height && w==width then rgb.clone() else resizeBilinear(rgb,3,height,width,h,w,antialias=true)
    var i=0
    while i<scaled.length do { scaled(i)=scaled(i)*2f-1f;i+=1 }
    val ph=(-h%8+8)%8;val pw=(-w%8+8)%8
    val H=h+ph;val W=w+pw
    val out=new Array[Float](3*H*W)
    for c <- 0 until 3; y <- 0 until H; x <- 0 until W do
      out(c*H*W+y*W+x)=scaled(c*h*w+math.min(y,h-1)*w+math.min(x,w-1)) // replicate padding to the bottom/right
    Prepared(out,H,W,ph,pw)

  /** Un-pads, resizes (bilinear, no antialiasing, as the pipeline's default output resampling) and renormalises. */
  def finish(decoded: Array[Float],prepared: Prepared,height: Int,width: Int): Array[Float] =
    val H=prepared.height;val W=prepared.width;val h=prepared.innerHeight;val w=prepared.innerWidth
    val inner=if h==H && w==W then decoded else
      val o=new Array[Float](3*h*w)
      for c <- 0 until 3; y <- 0 until h do System.arraycopy(decoded,c*H*W+y*W,o,c*h*w+y*w,w)
      o
    val resized=if h==height && w==width then inner else resizeBilinear(inner,3,h,w,height,width,antialias=false)
    val out=resized.clone();normalise(out,height*width);out

  /** In place L2 normalisation of the three planes with the pipeline's 1e-6 floor on the norm. */
  private[models] def normalise(chw: Array[Float],plane: Int): Unit =
    var i=0
    while i<plane do
      val x=chw(i);val y=chw(plane+i);val z=chw(2*plane+i)
      val n=math.max(math.sqrt((x*x+y*y+z*z).toDouble).toFloat,1e-6f)
      chw(i)=x/n;chw(plane+i)=y/n;chw(2*plane+i)=z/n
      i+=1

  /** The model's DDIM schedule: `scaled_linear` betas (0.00085..0.012, 1000 steps) rescaled to zero terminal SNR, `trailing` timestep
    * spacing, v-prediction, `set_alpha_to_one=false` (the step after the last one uses alpha_bar[0]), eta 0, no clipping.
    * `DdimScheduler` is not used because its zero-SNR rescaling clamps alpha_bar[999] to 1e-7 while the model needs it to be exactly 0. */
  final class Schedule private (val timesteps: Vector[Int],alphaBar: Vector[Double],previousAlphaBar: Vector[Double]):
    /** One step for flattened tensors: `v` is the U-Net output, `sample` the current target latent. */
    def step(index: Int,v: Array[Float],sample: Array[Float]): Array[Float] =
      val a=alphaBar(index);val p=previousAlphaBar(index)
      val sa=math.sqrt(a);val sb=math.sqrt(1-a);val sp=math.sqrt(p);val sq=math.sqrt(1-p)
      val out=new Array[Float](sample.length)
      var i=0
      while i<out.length do
        val x=sample(i).toDouble;val m=v(i).toDouble
        val original=sa*x-sb*m
        val epsilon=sa*m+sb*x
        out(i)=(sp*original+sq*epsilon).toFloat
        i+=1
      out
  object Schedule:
    val TrainSteps=1000
    def apply(steps: Int): Schedule =
      require(steps>0 && steps<=TrainSteps,"Marigold steps must be in 1..1000")
      val start=math.sqrt(0.00085);val end=math.sqrt(0.012)
      val betas=Vector.tabulate(TrainSteps)(i=>{val b=start+(end-start)*i/(TrainSteps-1);b*b})
      var cumulative=1.0
      val sqrtBars=betas.map { b => cumulative*=1-b;math.sqrt(cumulative) }
      val s0=sqrtBars.head;val sT=sqrtBars.last
      val bars=sqrtBars.map { s => val r=(s-sT)*s0/(s0-sT);r*r } // exact zero at the last step
      val ratio=TrainSteps.toDouble/steps
      val times=Vector.tabulate(steps)(i=>math.rint(TrainSteps-i*ratio).toInt-1)
      val stride=TrainSteps/steps
      new Schedule(times,times.map(bars),times.map(t=>if t-stride>=0 then bars(t-stride) else bars(0)))

  /** Resizes `c` planes with a half-pixel bilinear kernel; with `antialias` the kernel widens to the shrink factor (triangle filter,
    * as `torch.nn.functional.interpolate(..., mode="bilinear", antialias=True)` and PIL). Without it this is the plain two-tap
    * bilinear used by `align_corners=False`, clamped at the border. */
  def resizeBilinear(x: Array[Float],c: Int,h: Int,w: Int,oh: Int,ow: Int,antialias: Boolean): Array[Float] =
    def taps(inSize: Int,outSize: Int): (Array[Int],Array[Array[Float]]) =
      val scale=inSize.toDouble/outSize
      val filterScale=if antialias then math.max(scale,1.0) else 1.0
      val support=filterScale
      val starts=new Array[Int](outSize);val weights=new Array[Array[Float]](outSize)
      for o <- 0 until outSize do
        val center=scale*(o+0.5)
        if antialias then
          val lo=math.max((center-support+0.5).toInt,0);val hi=math.min((center+support+0.5).toInt,inSize)
          val n=math.max(hi-lo,1)
          val ws=Array.tabulate(n)(k=>{val d=(lo+k-center+0.5)/filterScale;math.max(0.0,1.0-math.abs(d))})
          val total=ws.sum
          starts(o)=lo;weights(o)=ws.map(v=>(if total>0 then v/total else 1.0/n).toFloat)
        else
          val src=math.max(center-0.5,0.0);val f=math.min(src.toInt,inSize-1);val l=(src-f).toFloat
          val f1=math.min(f+1,inSize-1)
          starts(o)=f
          weights(o)=if f1==f then Array(1f) else Array(1f-l,l)
      (starts,weights)
    val (ys,yw)=taps(h,oh);val (xs,xw)=taps(w,ow)
    val out=new Array[Float](c*oh*ow)
    val tmp=new Array[Float](h*ow)
    for ch <- 0 until c do
      val base=ch*h*w
      for y <- 0 until h; ox <- 0 until ow do
        var acc=0.0;val ws=xw(ox);val s=xs(ox)
        var k=0
        while k<ws.length do { acc+=x(base+y*w+s+k)*ws(k);k+=1 }
        tmp(y*ow+ox)=acc.toFloat
      for oy <- 0 until oh; ox <- 0 until ow do
        var acc=0.0;val ws=yw(oy);val s=ys(oy)
        var k=0
        while k<ws.length do { acc+=tmp((s+k)*ow+ox)*ws(k);k+=1 }
        out(ch*oh*ow+oy*ow+ox)=acc.toFloat
    out

  /** Tokens for the empty prompt without padding (BOS, EOS), as `tokenizer("", padding="do_not_pad")`. */
  def emptyPromptTokens(tokenizer: ClipFastTokenizer): Tensor[Long] =
    val ids=tokenizer.encode("",addSpecialTokens=true)
    require(ids.length==2,"the empty prompt must tokenise to BOS and EOS")
    Tensor.fromLongs(ids,Shape.from(Seq(1L,2L)))

  /** `text_encoder(ids)[0]`: the final-norm hidden state of the (23 layer) SD2 text encoder for the empty prompt, 1 x 2 x width. */
  def emptyTextEmbedding(tokenizer: ClipFastTokenizer,encoder: ClipTextModel): Tensor[Float] =
    Autograd.noGrad(encoder(emptyPromptTokens(tokenizer))).data

  /** Loads the components from a Diffusers directory (`unet/`, `vae/`, `text_encoder/`, `tokenizer/`). */
  def load(root: Path,unetExecution: UNetExecution,vaeExecution: VaeExecution,maxAttention: Long,maxVaeAttention: Long,
      textEmbedding: Option[Tensor[Float]]=None): MarigoldNormals =
    val embedding=textEmbedding.getOrElse {
      val tokenizer=ClipFastTokenizer.loadDirectory(root.resolve("tokenizer"))
      emptyTextEmbedding(tokenizer,PretrainedClip.load(root.resolve("text_encoder"),maxScoreElements=maxAttention))
    }
    val vae=PretrainedVae.load(root.resolve("vae"),maxScoreElements=maxVaeAttention)
    val unet=PretrainedUNet.load(root.resolve("unet"),maxScoreElements=maxAttention)
    new MarigoldNormals(unet,vae,embedding,unetExecution,vaeExecution,vae.config.scalingFactor.toFloat)

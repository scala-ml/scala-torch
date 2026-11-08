package scalatorch.models

import java.nio.file.Path

/** The ControlNet `lineart` generator (`sk_model.pth`): a ResNet image-to-image generator with a reflection-padded 7x7 stem
  * (64), two stride-2 3x3 downsamples (128, 256), three residual blocks (reflection pad, 3x3 conv, instance norm, ReLU,
  * reflection pad, 3x3 conv, instance norm, plus the skip), two transposed-convolution upsamples (128, 64) and a 7x7 head with a
  * sigmoid. Input is RGB 0..1 (sides divisible by 4); the output is 0..1 with dark lines on a white background. Plain CPU Float32.
  *
  * Licence note: the architecture follows the generator of "Learning to generate line drawings that convey geometry and
  * semantics" (Chan et al., CVPR 2022, informative-drawings, MIT) that lllyasviel/Annotators ships as `sk_model.pth`. The
  * weights are not bundled with this repository. Written from the checkpoint keys, not copied from any repository. */
final class LineartGenerator private (weights: Map[String,Array[Float]]):
  import AnnotatorOps.*
  private def w(n: String)=weights(n)

  def infer(rgb: Array[Float],height: Int,width: Int,cancelled: () => Boolean=()=>false): Array[Float]=
    require(height>=8 && width>=8 && height%4==0 && width%4==0 && rgb.length==3*height*width,"lineart input must be CHW RGB with sides divisible by 4")
    require(height.toLong*width<=4194304L,"lineart input is too large")
    def check(): Unit=if cancelled() || Thread.currentThread().isInterrupted then throw new java.util.concurrent.CancellationException("lineart cancelled")
    var h=height;var wd=width
    var x=reflectPad(rgb,3,h,wd,3)
    x=relu(instanceNorm(DepthAnythingV2.conv2d(x,3,h+6,wd+6,w("model0.1.weight"),Some(w("model0.1.bias")),64,7,1,0),64,h,wd))
    def down(prefix: String,ci: Int,co: Int): Unit=
      val y=DepthAnythingV2.conv2d(x,ci,h,wd,w(s"$prefix.weight"),Some(w(s"$prefix.bias")),co,3,2,1)
      h/=2;wd/=2;x=relu(instanceNorm(y,co,h,wd));check()
    down("model1.0",64,128);down("model1.3",128,256)
    for i <- 0 until 3 do
      var y=reflectPad(x,256,h,wd,1)
      y=relu(instanceNorm(DepthAnythingV2.conv2d(y,256,h+2,wd+2,w(s"model2.$i.conv_block.1.weight"),Some(w(s"model2.$i.conv_block.1.bias")),256,3,1,0),256,h,wd))
      y=reflectPad(y,256,h,wd,1)
      y=instanceNorm(DepthAnythingV2.conv2d(y,256,h+2,wd+2,w(s"model2.$i.conv_block.5.weight"),Some(w(s"model2.$i.conv_block.5.bias")),256,3,1,0),256,h,wd)
      var j=0
      while j<y.length do { y(j)+=x(j);j+=1 }
      x=y;check()
    def up(prefix: String,ci: Int,co: Int): Unit=
      val y=convTranspose(x,ci,h,wd,w(s"$prefix.weight"),w(s"$prefix.bias"),co,3,2,1,1)
      h*=2;wd*=2;x=relu(instanceNorm(y,co,h,wd));check()
    up("model3.0",256,128);up("model3.3",128,64)
    x=reflectPad(x,64,h,wd,3)
    val out=DepthAnythingV2.conv2d(x,64,h+6,wd+6,w("model4.1.weight"),Some(w("model4.1.bias")),1,7,1,0)
    var i=0
    while i<out.length do { out(i)=sigmoid(out(i));i+=1 }
    out

object LineartGenerator:
  def load(file: Path,cancelled: () => Boolean=()=>false): LineartGenerator=
    new LineartGenerator(AnnotatorOps.loadStrict(file,expectedShapes,"lineart",cancelled))

  private def expectedShapes: Map[String,Seq[Int]]=
    val m=scala.collection.mutable.LinkedHashMap.empty[String,Seq[Int]]
    def cv(n: String,co: Int,ci: Int,k: Int): Unit={ m(s"$n.weight")=Seq(co,ci,k,k);m(s"$n.bias")=Seq(co) }
    def cvt(n: String,ci: Int,co: Int,k: Int): Unit={ m(s"$n.weight")=Seq(ci,co,k,k);m(s"$n.bias")=Seq(co) }
    cv("model0.1",64,3,7);cv("model1.0",128,64,3);cv("model1.3",256,128,3)
    for i <- 0 until 3 do { cv(s"model2.$i.conv_block.1",256,256,3);cv(s"model2.$i.conv_block.5",256,256,3) }
    cvt("model3.0",256,128,3);cvt("model3.3",128,64,3);cv("model4.1",1,64,7)
    m.toMap

/** The ControlNet `lineart_anime` generator (`netG.pth`): a pix2pix-style U-Net (eight stride-2 4x4 convolutions with skip
  * connections, LeakyReLU(0.2) / ReLU, non-affine instance normalisation, transposed-convolution decoder, Tanh output) mapping
  * RGB -1..1 to one channel -1..1. Input sides must be divisible by 256. Plain CPU Float32.
  *
  * Licence note: netG.pth is the anime line-art extraction model (Anime2Sketch family, MIT code) shipped in
  * lllyasviel/Annotators; that repo states no separate weight licence. The weights are not bundled with this repository.
  * Written from the checkpoint keys (the standard pix2pix U-Net skip-block layout), not copied from any repository. */
final class AnimeLineartUnet private (weights: Map[String,Array[Float]]):
  import AnnotatorOps.*
  import AnimeLineartUnet.*

  def infer(rgb: Array[Float],height: Int,width: Int,cancelled: () => Boolean=()=>false): Array[Float]=
    require(height>=256 && width>=256 && height%256==0 && width%256==0 && rgb.length==3*height*width,"anime lineart input must be CHW RGB with sides divisible by 256")
    require(height.toLong*width<=4194304L,"anime lineart input is too large")
    def check(): Unit=if cancelled() || Thread.currentThread().isInterrupted then throw new java.util.concurrent.CancellationException("lineart_anime cancelled")
    val first=DepthAnythingV2.conv2d(rgb,3,height,width,weights(s"$Root.0.weight"),Some(weights(s"$Root.0.bias")),Channels(0),4,2,1)
    val h0=height/2;val w0=width/2
    val inner=block(0,s"$Root.1.model",first,h0,w0,check)
    val y=convTranspose(relu(inner),2*Channels(0),h0,w0,weights(s"$Root.3.weight"),weights(s"$Root.3.bias"),1,4,2,1,0)
    var i=0
    while i<y.length do { y(i)=math.tanh(y(i)).toFloat;i+=1 }
    y

  /** One U-Net level; returns `cat(input, decoded)`. `x` has `Channels(k)` channels at h*w. */
  private def block(k: Int,prefix: String,x: Array[Float],h: Int,w: Int,check: () => Unit): Array[Float]=
    check()
    val innermost= k==Channels.length-2
    val ci=Channels(k);val co=Channels(k+1)
    val down=DepthAnythingV2.conv2d(leakyRelu(x,0.2f),ci,h,w,weights(s"$prefix.1.weight"),Some(weights(s"$prefix.1.bias")),co,4,2,1)
    val oh=h/2;val ow=w/2
    val decoded=
      if innermost then
        instanceNorm(convTranspose(relu(down),co,oh,ow,weights(s"$prefix.3.weight"),weights(s"$prefix.3.bias"),ci,4,2,1,0),ci,h,w)
      else
        val sub=block(k+1,s"$prefix.3.model",instanceNorm(down,co,oh,ow),oh,ow,check)
        instanceNorm(convTranspose(relu(sub),2*co,oh,ow,weights(s"$prefix.5.weight"),weights(s"$prefix.5.bias"),ci,4,2,1,0),ci,h,w)
    concat(x,decoded)

object AnimeLineartUnet:
  private val Root="module.model.model"
  /** Channels after each stride-2 convolution; the first entry is the outermost convolution. */
  private val Channels=Vector(64,128,256,512,512,512,512,512)

  def load(file: Path,cancelled: () => Boolean=()=>false): AnimeLineartUnet=
    new AnimeLineartUnet(AnnotatorOps.loadStrict(file,expectedShapes,"anime lineart",cancelled))

  private def expectedShapes: Map[String,Seq[Int]]=
    val m=scala.collection.mutable.LinkedHashMap.empty[String,Seq[Int]]
    def cv(n: String,co: Int,ci: Int): Unit={ m(s"$n.weight")=Seq(co,ci,4,4);m(s"$n.bias")=Seq(co) }
    def cvt(n: String,ci: Int,co: Int): Unit={ m(s"$n.weight")=Seq(ci,co,4,4);m(s"$n.bias")=Seq(co) }
    cv(s"$Root.0",Channels(0),3);cvt(s"$Root.3",2*Channels(0),1)
    var prefix=s"$Root.1.model"
    for k <- 0 until Channels.length-1 do
      cv(s"$prefix.1",Channels(k+1),Channels(k))
      if k==Channels.length-2 then cvt(s"$prefix.3",Channels(k+1),Channels(k))
      else { cvt(s"$prefix.5",2*Channels(k+1),Channels(k));prefix=s"$prefix.3.model" }
    m.toMap

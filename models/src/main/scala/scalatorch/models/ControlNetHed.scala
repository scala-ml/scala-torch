package scalatorch.models

import java.nio.file.Path

/** The soft-edge HED network as used by ControlNet (`ControlNetHED.pth`): a VGG-like trunk of five blocks (2, 2, 3, 3, 3 3x3
  * convolutions with ReLU, 2x2 max-pooling between blocks), each with a 1x1 side projection to one channel. The input is RGB
  * 0..255 minus the stored per-channel `norm`; the five side maps are resized bilinearly to the input size, averaged and passed
  * through a sigmoid. Plain CPU Float32.
  *
  * Licence note: the ControlNet HED weights are a re-training of the HED detector (Xie & Tu, "Holistically-Nested Edge
  * Detection", ICCV 2015) distributed through lllyasviel/Annotators; the network definition used by ControlNet is Apache-2.0
  * and the original HED model is for academic research use. The weights are not bundled with this repository (kept in the
  * NUI `models/annotators` folder). Written from the checkpoint keys and the documented algorithm, not copied. */
final class ControlNetHed private (weights: Map[String,Array[Float]]):
  /** The five side outputs (before resizing) as (data, height, width). */
  def sideOutputs(rgb: Array[Float],height: Int,width: Int,cancelled: () => Boolean=()=>false): Vector[(Array[Float],Int,Int)]=
    require(height>=16 && width>=16 && rgb.length==3*height*width,"HED input must be CHW RGB of at least 16x16")
    require(height.toLong*width<=16777216L,"HED input is too large")
    val norm=weights("norm");val n=height*width
    var x=new Array[Float](rgb.length)
    for c <- 0 until 3; i <- 0 until n do x(c*n+i)=rgb(c*n+i)-norm(c)
    var h=height;var w=width;var c=3
    val sides=Vector.newBuilder[(Array[Float],Int,Int)]
    for ((count,co),index) <- ControlNetHed.Blocks.zipWithIndex do
      val blockIndex=index+1
      if cancelled() || Thread.currentThread().isInterrupted then throw new java.util.concurrent.CancellationException("hed cancelled")
      if blockIndex>1 then
        val oh=h/2;val ow=w/2;val y=new Array[Float](c*oh*ow);val xin=x;val hh=h;val ww=w
        DepthAnythingV2.parallel(c) { ch =>
          var oy=0
          while oy<oh do
            var ox=0
            while ox<ow do
              val b=ch*hh*ww+2*oy*ww+2*ox
              y(ch*oh*ow+oy*ow+ox)=math.max(math.max(xin(b),xin(b+1)),math.max(xin(b+ww),xin(b+ww+1)));ox+=1
            oy+=1
        }
        x=y;h=oh;w=ow
      for i <- 0 until count do
        val name=s"block$blockIndex.convs.$i"
        x=AnnotatorOps.relu(DepthAnythingV2.conv2d(x,c,h,w,weights(s"$name.weight"),Some(weights(s"$name.bias")),co,3,1,1));c=co
      val proj=s"block$blockIndex.projection"
      sides+=((DepthAnythingV2.conv2d(x,c,h,w,weights(s"$proj.weight"),Some(weights(s"$proj.bias")),1,1,1,0),h,w))
    sides.result()

  /** Sigmoid edge map (values 0..1, h*w) at the input size: sigmoid of the mean of the bilinearly resized side outputs. */
  def edges(rgb: Array[Float],height: Int,width: Int,cancelled: () => Boolean=()=>false): Array[Float]=
    val sum=new Array[Float](height*width)
    val sides=sideOutputs(rgb,height,width,cancelled)
    for (d,sh,sw) <- sides do
      val r=DepthAnythingV2.bilinear(d,1,sh,sw,height,width,false)
      var i=0
      while i<sum.length do { sum(i)+=r(i);i+=1 }
    var i=0
    while i<sum.length do { sum(i)=AnnotatorOps.sigmoid(sum(i)/sides.size);i+=1 }
    sum

object ControlNetHed:
  /** (convolutions, channels) of the five blocks. */
  val Blocks=Vector((2,64),(2,128),(3,256),(3,512),(3,512))

  def load(file: Path,cancelled: () => Boolean=()=>false): ControlNetHed=
    new ControlNetHed(AnnotatorOps.loadStrict(file,expectedShapes,"HED",cancelled))

  private def expectedShapes: Map[String,Seq[Int]]=
    val m=scala.collection.mutable.LinkedHashMap.empty[String,Seq[Int]]
    m("norm")=Seq(1,3,1,1)
    var ci=3
    for ((count,co),i) <- Blocks.zipWithIndex do
      for j <- 0 until count do
        m(s"block${i+1}.convs.$j.weight")=Seq(co,if j==0 then ci else co,3,3);m(s"block${i+1}.convs.$j.bias")=Seq(co)
      m(s"block${i+1}.projection.weight")=Seq(1,co,1,1);m(s"block${i+1}.projection.bias")=Seq(1)
      ci=co
    m.toMap

  /** ControlNet's "safe" quantisation of the edge map: `trunc(x * 3) / 2`. */
  def safeStep(v: Float): Float=(v*3f).toInt.toFloat/2f

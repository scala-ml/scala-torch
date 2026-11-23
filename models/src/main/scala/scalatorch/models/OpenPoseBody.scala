package scalatorch.models

import scalatorch.io.{LegacyPyTorchStateDict,PyTorchArchiveLimits,PyTorchTensorArchive}
import java.nio.file.{Files,Path}
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/** A body keypoint candidate in the detection image: pixel position, smoothed heat-map score and a global id. */
final case class OpenPosePeak(x: Int,y: Int,score: Float,id: Int)

/** Detected people: `people(n)` has 20 entries, 18 peak ids (-1 when missing), the total score and the part count. */
final case class OpenPoseDetection(peaks: Vector[Vector[OpenPosePeak]],people: Vector[Array[Double]],width: Int,height: Int):
  def peak(id: Int): OpenPosePeak=peaks.iterator.flatten.find(_.id==id).get

/** Raw network output at stride 8: `paf` is 38*gh*gw (limb x/y fields), `heat` is 19*gh*gw (18 keypoints + background). */
final case class OpenPoseOutputs(paf: Array[Float],heat: Array[Float],gridHeight: Int,gridWidth: Int)

/** The OpenPose COCO body-pose network (VGG-19 front-end followed by six PAF/heat-map refinement stages), plain CPU Float32
  * inference, plus the documented peak finding and part-affinity-field association.
  *
  * Licence note: the weights (`body_pose_model.pth`) come from CMU OpenPose and are available for academic / non-commercial
  * use only (see https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE). This implementation is written
  * from the published algorithm (Cao et al., "Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields") and does
  * not copy code from any repository. */
final class OpenPoseBody private (weights: Map[String,Array[Float]]):
  import OpenPoseBody.*

  private def conv(x: Array[Float],ci: Int,h: Int,w: Int,name: String,relu: Boolean): Array[Float]=
    val bias=weights(s"$name.bias");val k=kernelSizes(name)
    val y=DepthAnythingV2.conv2d(x,ci,h,w,weights(s"$name.weight"),Some(bias),bias.length,k,1,k/2)
    if relu then
      var i=0
      while i<y.length do { if y(i)<0f then y(i)=0f;i+=1 }
    y

  private def pool(x: Array[Float],c: Int,h: Int,w: Int): Array[Float]=
    val oh=h/2;val ow=w/2;val y=new Array[Float](c*oh*ow)
    DepthAnythingV2.parallel(c) { ch =>
      var oy=0
      while oy<oh do
        var ox=0
        while ox<ow do
          val b=ch*h*w+2*oy*w+2*ox
          y(ch*oh*ow+oy*ow+ox)=math.max(math.max(x(b),x(b+1)),math.max(x(b+w),x(b+w+1)));ox+=1
        oy+=1
    }
    y

  /** Runs the network on a normalised CHW BGR image whose sides are multiples of 8. */
  def infer(pixels: Array[Float],height: Int,width: Int,cancelled: () => Boolean=()=>false): OpenPoseOutputs=
    require(height>0 && width>0 && height%Stride==0 && width%Stride==0,s"OpenPose input must be a multiple of $Stride, got ${width}x$height")
    require(pixels.length==3*height*width,"pixel buffer size mismatch")
    require(height.toLong*width<=4000000L,"OpenPose input is too large")
    def check(): Unit=if cancelled() || Thread.currentThread().isInterrupted then throw new java.util.concurrent.CancellationException("openpose cancelled")
    var h=height;var w=width;var x=pixels;var c=3
    def step(name: String,co: Int): Unit={ x=conv(x,c,h,w,name,true);c=co }
    for (name,co) <- Vector("conv1_1"->64,"conv1_2"->64) do step(name,co)
    x=pool(x,c,h,w);h/=2;w/=2;check()
    for (name,co) <- Vector("conv2_1"->128,"conv2_2"->128) do step(name,co)
    x=pool(x,c,h,w);h/=2;w/=2;check()
    for (name,co) <- Vector("conv3_1"->256,"conv3_2"->256,"conv3_3"->256,"conv3_4"->256) do step(name,co)
    x=pool(x,c,h,w);h/=2;w/=2;check()
    for (name,co) <- Vector("conv4_1"->512,"conv4_2"->512,"conv4_3_CPM"->256,"conv4_4_CPM"->128) do step(name,co)
    val features=x
    def firstStage(branch: Int,out: Int): Array[Float]=
      var y=features
      for i <- 1 to 3 do y=conv(y,128,h,w,s"conv5_${i}_CPM_L$branch",true)
      y=conv(y,128,h,w,s"conv5_4_CPM_L$branch",true)
      conv(y,512,h,w,s"conv5_5_CPM_L$branch",false)
    var paf=firstStage(1,PafChannels);var heat=firstStage(2,HeatChannels);check()
    for stage <- 2 to 6 do
      val n=h*w
      val joined=new Array[Float]((PafChannels+HeatChannels+128)*n)
      System.arraycopy(paf,0,joined,0,paf.length);System.arraycopy(heat,0,joined,paf.length,heat.length)
      System.arraycopy(features,0,joined,paf.length+heat.length,features.length)
      def refine(branch: Int): Array[Float]=
        var y=joined;var ci=PafChannels+HeatChannels+128
        for i <- 1 to 5 do { y=conv(y,ci,h,w,s"Mconv${i}_stage${stage}_L$branch",true);ci=128 }
        y=conv(y,128,h,w,s"Mconv6_stage${stage}_L$branch",true)
        conv(y,128,h,w,s"Mconv7_stage${stage}_L$branch",false)
      val (p,t)=(refine(1),refine(2));paf=p;heat=t;check()
    OpenPoseOutputs(paf,heat,h,w)

object OpenPoseBody:
  val Stride=8;val BoxSize=368;val PadValue=128
  val PafChannels=38;val HeatChannels=19

  /** 0-based COCO-18 part order: nose, neck, r/l shoulder-elbow-wrist, r/l hip-knee-ankle, r/l eye, r/l ear. */
  val PartNames=Vector("nose","neck","rsho","relb","rwri","lsho","lelb","lwri","rhip","rkne","rank","lhip","lkne","lank","reye","leye","rear","lear")
  /** The 19 limbs, 1-based part indices (the last two, shoulder-ear, only help the association). Drawing uses the first 17. */
  val LimbSeq=Vector((2,3),(2,6),(3,4),(4,5),(6,7),(7,8),(2,9),(9,10),(10,11),(2,12),(12,13),(13,14),(2,1),(1,15),(15,17),(1,16),(16,18),(3,17),(6,18))
  /** The x/y PAF channel pair of each limb. */
  val PafIndex=Vector((12,13),(20,21),(14,15),(16,17),(22,23),(24,25),(0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(28,29),(30,31),(34,35),(32,33),(36,37),(18,19),(26,27))

  private val kernelSizes: Map[String,Int]=expectedShapes.collect { case (n,Seq(_,_,k,_)) if n.endsWith(".weight") => n.stripSuffix(".weight")->k }.toMap

  /** Strictly loads the PyTorch `body_pose_model.pth` state dict; unknown or mis-shaped weights are rejected. */
  def load(file: Path,cancelled: () => Boolean=()=>false): OpenPoseBody=
    require(Files.isRegularFile(file),s"OpenPose body weights not found: $file")
    // body_pose_model.pth is a pre-1.6 (non-ZIP) torch.save file; newer ZIP checkpoints are read as well.
    val entries: Map[String,PyTorchTensorArchive.Value]=
      if LegacyPyTorchStateDict.isLegacy(file) then
        LegacyPyTorchStateDict.read(file,PyTorchArchiveLimits.ImageModelWeights,cancelled).view.mapValues(PyTorchTensorArchive.TensorValue(_)).toMap
      else PyTorchTensorArchive.read(file,PyTorchArchiveLimits.ImageModelWeights,cancelled).value match
        case PyTorchTensorArchive.DictValue(e) => e
        case other => throw IllegalArgumentException(s"OpenPose weights must be a state dict, got ${other.getClass.getSimpleName}")
    val expected=expectedShapes
    val unknown=entries.keySet.diff(expected.keySet)
    require(unknown.isEmpty,s"unexpected OpenPose weights: ${unknown.take(5).mkString(",")}")
    val loaded=expected.map { (name,dims) =>
      entries.getOrElse(name,throw IllegalArgumentException(s"missing OpenPose weight: $name")) match
        case PyTorchTensorArchive.TensorValue(t) =>
          require(t.shape.map(_.toInt)==dims.toVector,s"OpenPose weight shape mismatch: $name, expected ${dims.mkString("x")}, got ${t.shape.mkString("x")}")
          name->t.values
        case _ => throw IllegalArgumentException(s"OpenPose weight is not a tensor: $name")
    }
    new OpenPoseBody(loaded)

  private def expectedShapes: Map[String,Seq[Int]]=
    val m=mutable.LinkedHashMap.empty[String,Seq[Int]]
    def cv(name: String,co: Int,ci: Int,k: Int): Unit={ m(s"$name.weight")=Seq(co,ci,k,k);m(s"$name.bias")=Seq(co) }
    val front=Vector("conv1_1"->(3,64),"conv1_2"->(64,64),"conv2_1"->(64,128),"conv2_2"->(128,128),"conv3_1"->(128,256),
      "conv3_2"->(256,256),"conv3_3"->(256,256),"conv3_4"->(256,256),"conv4_1"->(256,512),"conv4_2"->(512,512),
      "conv4_3_CPM"->(512,256),"conv4_4_CPM"->(256,128))
    for (n,(ci,co)) <- front do cv(n,co,ci,3)
    for (b,out) <- Vector(1->PafChannels,2->HeatChannels) do
      for i <- 1 to 3 do cv(s"conv5_${i}_CPM_L$b",128,128,3)
      cv(s"conv5_4_CPM_L$b",512,128,1);cv(s"conv5_5_CPM_L$b",out,512,1)
      for s <- 2 to 6 do
        cv(s"Mconv1_stage${s}_L$b",128,PafChannels+HeatChannels+128,7)
        for i <- 2 to 5 do cv(s"Mconv${i}_stage${s}_L$b",128,128,7)
        cv(s"Mconv6_stage${s}_L$b",128,128,1);cv(s"Mconv7_stage${s}_L$b",out,128,1)
    m.toMap

  // ---------------------------------------------------------------------------------------------------------------
  // Image resampling and filters

  /** Bicubic (a = -0.75, replicated borders) resize of a CHW map; `invX`/`invY` are source pixels per destination pixel. */
  def cubicResize(x: Array[Float],c: Int,h: Int,w: Int,oh: Int,ow: Int,invX: Double,invY: Double): Array[Float]=
    val a= -0.75
    def axis(inSize: Int,outSize: Int,inv: Double): (Array[Array[Int]],Array[Array[Float]])=
      val idx=new Array[Array[Int]](outSize);val cf=new Array[Array[Float]](outSize)
      for d <- 0 until outSize do
        val src=(d+0.5)*inv-0.5;val f=math.floor(src).toInt;val t=src-f
        idx(d)=Array.tabulate(4)(i=>math.max(0,math.min(inSize-1,f-1+i)))
        def far(u: Double)=((a*u-5*a)*u+8*a)*u-4*a
        def near(u: Double)=((a+2)*u-(a+3))*u*u+1
        val c0=far(t+1);val c1=near(t);val c2=near(1-t)
        cf(d)=Array(c0.toFloat,c1.toFloat,c2.toFloat,(1-c0-c1-c2).toFloat)
      (idx,cf)
    val (xi,xc)=axis(w,ow,invX);val (yi,yc)=axis(h,oh,invY);val out=new Array[Float](c*oh*ow)
    DepthAnythingV2.parallel(c) { ch =>
      val tmp=new Array[Float](h*ow);val b=ch*h*w
      var y=0
      while y<h do
        var ox=0
        while ox<ow do
          val i=xi(ox);val k=xc(ox);val r=b+y*w
          tmp(y*ow+ox)=x(r+i(0))*k(0)+x(r+i(1))*k(1)+x(r+i(2))*k(2)+x(r+i(3))*k(3);ox+=1
        y+=1
      var oy=0
      while oy<oh do
        val i=yi(oy);val k=yc(oy);var ox=0
        while ox<ow do
          out(ch*oh*ow+oy*ow+ox)=tmp(i(0)*ow+ox)*k(0)+tmp(i(1)*ow+ox)*k(1)+tmp(i(2)*ow+ox)*k(2)+tmp(i(3)*ow+ox)*k(3);ox+=1
        oy+=1
    }
    out

  /** Separable Gaussian smoothing of one h*w plane with reflected borders (truncated at four sigma). */
  def gaussian(plane: Array[Float],h: Int,w: Int,sigma: Double): Array[Float]=
    val radius=(4*sigma+0.5).toInt
    val k=Array.tabulate(2*radius+1)(i=>math.exp(-0.5*(i-radius)*(i-radius)/(sigma*sigma)))
    val sum=k.sum;for i <- k.indices do k(i)/=sum
    def reflect(i: Int,n: Int): Int=
      var v=i;val period=2*n
      v=((v%period)+period)%period
      if v>=n then period-v-1 else v
    val tmp=new Array[Float](plane.length);val out=new Array[Float](plane.length)
    for y <- 0 until h; xx <- 0 until w do
      var acc=0.0;var t= -radius
      while t<=radius do { acc+=k(t+radius)*plane(y*w+reflect(xx+t,w));t+=1 }
      tmp(y*w+xx)=acc.toFloat
    for y <- 0 until h; xx <- 0 until w do
      var acc=0.0;var t= -radius
      while t<=radius do { acc+=k(t+radius)*tmp(reflect(y+t,h)*w+xx);t+=1 }
      out(y*w+xx)=acc.toFloat
    out

  // ---------------------------------------------------------------------------------------------------------------
  // Detection

  /** Scales a BGR HWC 0..255 image to the 368-pixel box height, pads right/down to a multiple of 8 with 128 and normalises
    * (`v / 256 - 0.5`) to CHW. Returns the tensor and its (height, width). */
  def prepareInput(bgr: Array[Float],height: Int,width: Int): (Array[Float],Int,Int,Int,Int)=
    val chw=new Array[Float](3*height*width)
    for i <- 0 until height*width; c <- 0 until 3 do chw(c*height*width+i)=bgr(i*3+c)
    val scale=BoxSize.toDouble/height
    val nh=math.max(1,math.round(height*scale).toInt);val nw=math.max(1,math.round(width*scale).toInt)
    val scaled=cubicResize(chw,3,height,width,nh,nw,1.0/scale,1.0/scale)
    val ph=(Stride-nh%Stride)%Stride;val pw=(Stride-nw%Stride)%Stride
    val th=nh+ph;val tw=nw+pw;val out=new Array[Float](3*th*tw)
    for c <- 0 until 3; y <- 0 until th; x <- 0 until tw do
      val v=if y<nh && x<nw then math.rint(math.max(0.0,math.min(255.0,scaled(c*nh*nw+y*nw+x)))).toFloat else PadValue.toFloat
      out(c*th*tw+y*tw+x)=v/256f-0.5f
    (out,th,tw,ph,pw)

  /** Full single-scale detection on a BGR HWC 0..255 image. */
  def detect(model: OpenPoseBody,bgr: Array[Float],height: Int,width: Int,cancelled: () => Boolean=()=>false): OpenPoseDetection=
    val (input,th,tw,ph,pw)=prepareInput(bgr,height,width)
    val out=model.infer(input,th,tw,cancelled)
    postprocess(out,th-ph,tw-pw,height,width,cancelled)

  /** Upsamples the stride-8 maps to the source size, finds heat-map peaks and associates them into people using the PAFs.
    * `cropHeight`/`cropWidth` are the scaled image sizes without padding. */
  def postprocess(out: OpenPoseOutputs,cropHeight: Int,cropWidth: Int,height: Int,width: Int,cancelled: () => Boolean=()=>false): OpenPoseDetection=
    def check(): Unit=if cancelled() || Thread.currentThread().isInterrupted then throw new java.util.concurrent.CancellationException("openpose cancelled")
    val gh=out.gridHeight;val gw=out.gridWidth
    def upsample(m: Array[Float],c: Int): Array[Float]=
      val big=cubicResize(m,c,gh,gw,gh*Stride,gw*Stride,1.0/Stride,1.0/Stride)
      val cropped=new Array[Float](c*cropHeight*cropWidth)
      for ch <- 0 until c; y <- 0 until cropHeight do
        System.arraycopy(big,ch*gh*Stride*gw*Stride+y*gw*Stride,cropped,ch*cropHeight*cropWidth+y*cropWidth,cropWidth)
      cubicResize(cropped,c,cropHeight,cropWidth,height,width,cropWidth.toDouble/width,cropHeight.toDouble/height)
    val heat=upsample(out.heat,HeatChannels);check()
    val paf=upsample(out.paf,PafChannels);check()
    val plane=height*width
    var nextId=0
    val peaks=Vector.tabulate(18) { part =>
      val m=gaussian(java.util.Arrays.copyOfRange(heat,part*plane,(part+1)*plane),height,width,3.0)
      def at(y: Int,x: Int): Float=if y<0 || y>=height || x<0 || x>=width then 0f else m(y*width+x)
      val found=ArrayBuffer.empty[OpenPosePeak]
      for y <- 0 until height; x <- 0 until width do
        val v=m(y*width+x)
        if v>0.1f && v>=at(y,x-1) && v>=at(y,x+1) && v>=at(y-1,x) && v>=at(y+1,x) then
          found+=OpenPosePeak(x,y,v,nextId+found.size)
      nextId+=found.size
      found.toVector
    }
    check()
    val byId=peaks.flatten.sortBy(_.id)
    // limb connection candidates: (index in A, index in B, score)
    val connections=LimbSeq.indices.map { k =>
      val a=peaks(LimbSeq(k)._1-1);val b=peaks(LimbSeq(k)._2-1)
      if a.isEmpty || b.isEmpty then Vector.empty[(Int,Int,Double)]
      else
        val (cx,cy)=PafIndex(k);val candidates=ArrayBuffer.empty[(Int,Int,Double)]
        for (pa,i) <- a.zipWithIndex; (pb,j) <- b.zipWithIndex do
          val dx=(pb.x-pa.x).toDouble;val dy=(pb.y-pa.y).toDouble
          val norm=math.max(math.hypot(dx,dy),1e-3);val vx=dx/norm;val vy=dy/norm
          var total=0.0;var good=0;var s=0
          while s<MidSamples do
            val t=s.toDouble/(MidSamples-1)
            val sx=math.rint(pa.x+(pb.x-pa.x)*t).toInt;val sy=math.rint(pa.y+(pb.y-pa.y)*t).toInt
            val score=paf(cx*plane+sy*width+sx)*vx+paf(cy*plane+sy*width+sx)*vy
            total+=score;if score>0.05 then good+=1
            s+=1
          val prior=total/MidSamples+math.min(0.5*height/norm-1,0.0)
          if good>0.8*MidSamples && prior>0 then candidates+=((i,j,prior))
        val usedA=mutable.Set.empty[Int];val usedB=mutable.Set.empty[Int];val chosen=ArrayBuffer.empty[(Int,Int,Double)]
        for c <- candidates.sortBy(-_._3) if chosen.size<math.min(a.size,b.size) do
          if !usedA(c._1) && !usedB(c._2) then { usedA+=c._1;usedB+=c._2;chosen+=c }
        chosen.toVector
    }
    val subset=ArrayBuffer.empty[Array[Double]]
    for k <- LimbSeq.indices if connections(k).nonEmpty do
      val ia=LimbSeq(k)._1-1;val ib=LimbSeq(k)._2-1;val a=peaks(ia);val b=peaks(ib)
      for (i,j,score) <- connections(k) do
        val pa=a(i).id;val pb=b(j).id
        val hits=subset.indices.filter(n=>subset(n)(ia)==pa || subset(n)(ib)==pb).take(2)
        if hits.size==1 then
          val s=subset(hits(0))
          if s(ib)!=pb then { s(ib)=pb;s(19)+=1;s(18)+=byId(pb).score+score }
        else if hits.size==2 then
          val s1=subset(hits(0));val s2=subset(hits(1))
          val overlap=(0 until 18).exists(t=>s1(t)>=0 && s2(t)>=0)
          if !overlap then
            for t <- 0 until 18 do s1(t)+=s2(t)+1
            s1(18)+=s2(18)+score;s1(19)+=s2(19);subset.remove(hits(1))
          else { s1(ib)=pb;s1(19)+=1;s1(18)+=byId(pb).score+score }
        else if k<17 then
          val row=Array.fill(20)(-1.0);row(ia)=pa;row(ib)=pb;row(19)=2;row(18)=byId(pa).score+byId(pb).score+score
          subset+=row
    val people=subset.filter(s=>s(19)>=4 && s(18)/s(19)>=0.4).toVector
    OpenPoseDetection(peaks,people,width,height)

  private val MidSamples=10

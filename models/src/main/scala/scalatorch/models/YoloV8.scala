package scalatorch.models

import java.awt.image.BufferedImage
import java.nio.file.Path
import scalatorch.io.SafeTensorSet

/** A YOLOv8 object detector / instance segmenter on the CPU (plain Float32 arrays, no Python at runtime).
  *
  * Architecture (from the published YOLOv8 model description, `yolov8*.yaml` / `yolov8*-seg.yaml`): a backbone of
  * `Conv(3x3 s2)` and `C2f` stages ending in `SPPF`, a PAN neck (nearest 2x upsampling, concatenation, `C2f`, stride-2 `Conv`),
  * and a head over strides 8/16/32. `Conv` is conv (no bias) + BatchNorm (eps 1e-3) + SiLU. Per level the `Detect` head has a
  * box branch (two 3x3 Conv + a 1x1 conv to `4*regMax` distribution logits, decoded by DFL: softmax over `regMax` bins, expectation,
  * left/top/right/bottom distances from the cell centre, times stride) and a class branch (two 3x3 Conv + 1x1 conv to `nc`,
  * sigmoid). The `Segment` head adds a mask-coefficient branch (`nm` channels) and a prototype network on the stride-8 map
  * (3x3 Conv, 2x2 stride-2 ConvTranspose, 3x3 Conv, 1x1 Conv to `nm`).
  *
  * The network is derived from the checkpoint itself: the channel widths, the C2f repeat counts, `nc`, `regMax`, `nm`, `npr` are read from tensor
  * shapes (this covers the n/s/m/l/x scales, which differ only by the depth/width multipliers), then EVERY tensor name and shape of the
  * architecture is checked against the file: missing, unknown and mis-shaped tensors are rejected. Conv+BatchNorm are folded at load.
  *
  * Weights: plain float32 safetensors written once, offline, by scripts/convert_yolov8_to_safetensors.py from the Ultralytics `.pt`
  * pickles (which the repo's PyTorch readers cannot and should not load). The scala side never needs Python.
  *
  * Licence: Ultralytics YOLOv8 is AGPL-3.0 and the Bingsu/adetailer weights (face_yolov8n, hand_yolov8n, person_yolov8n-seg, ...) are
  * distributed under AGPL-3.0 on their Hugging Face model card; the converted files inherit those terms. They are not bundled with this
  * repository (kept in the NUI `models/annotators/adetailer` folder). This code was written from the architecture description and the
  * documented pre/post-processing behaviour (letterbox, confidence threshold, class-wise greedy NMS at IoU 0.7, mask = sigmoid(coefficients
  * x prototypes) cropped to the box); no Ultralytics source was copied. */
final class YoloV8 private (val spec: YoloV8.Spec,run: (Array[Float],Int,Int) => YoloV8.Raw):
  import YoloV8.*

  /** Raw head outputs for a CHW RGB image scaled to 0..1 (sides multiples of 32). */
  def forward(chw: Array[Float],height: Int,width: Int): Raw=
    require(height>=32 && width>=32 && height%32==0 && width%32==0 && chw.length==3*height*width,"YOLOv8 input must be CHW RGB with sides that are multiples of 32")
    run(chw,height,width)

  /** Letterbox + forward + threshold + class-wise NMS + (for segmentation models) masks. Boxes are in input-image pixels. */
  def detect(image: BufferedImage,conf: Double,iou: Double=0.7,imgsz: Int=640,maxDet: Int=300,withMasks: Boolean=true,cancelled: () => Boolean=()=>false): Result=
    val lb=letterbox(image,imgsz)
    if cancelled() then throw new java.util.concurrent.CancellationException("yolov8 cancelled")
    val raw=forward(lb.chw,lb.height,lb.width)
    val pred=decode(raw,lb.height,lb.width)
    val kept=nms(pred,conf.toFloat,iou.toFloat,maxDet)
    val w=image.getWidth;val h=image.getHeight
    val boxes=kept.map { i =>
      def cl(v: Double,hi: Int)=math.max(0.0,math.min(hi.toDouble,v))
      val x1=(pred.box(4*i)-lb.padLeft)/lb.gain;val y1=(pred.box(4*i+1)-lb.padTop)/lb.gain
      val x2=(pred.box(4*i+2)-lb.padLeft)/lb.gain;val y2=(pred.box(4*i+3)-lb.padTop)/lb.gain
      (cl(x1,w),cl(y1,h),cl(x2,w),cl(y2,h))
    }
    val masks=
      if withMasks && raw.proto.isDefined && kept.nonEmpty then
        Some(kept.map(i=>mask(raw.proto.get,pred.coef(i),(pred.box(4*i),pred.box(4*i+1),pred.box(4*i+2),pred.box(4*i+3)),lb,w,h)))
      else None
    Result(boxes,kept.map(i=>pred.score(i).toDouble),kept.map(pred.cls),masks)

object YoloV8:
  /** Network sizes recovered from the checkpoint tensor shapes. */
  final case class Spec(stem: Vector[Int],repeats: Vector[Int],headRepeats: Int,nc: Int,regMax: Int,nm: Int,npr: Int,c2: Int,c3: Int,c4: Int):
    def segment: Boolean=nm>0
  /** Raw per-level head outputs, each a CHW map: box distribution logits (`4*regMax`), class logits (`nc`), mask coefficients (`nm`, may be empty), plus the prototypes. */
  final case class Raw(box: Vector[Fm],cls: Vector[Fm],coef: Vector[Fm],proto: Option[Fm])
  /** A CHW feature map. */
  type Fm=(Array[Float],Int,Int,Int)
  /** Flattened anchors (levels 0..2, row-major): xyxy boxes in input pixels, best class score (sigmoid) and class, per-class scores, mask coefficients. */
  final case class Predictions(n: Int,box: Array[Float],scores: Array[Float],score: Array[Float],cls: Array[Int],coef: Array[Array[Float]],nc: Int)
  final case class Result(boxes: Vector[(Double,Double,Double,Double)],scores: Vector[Double],classes: Vector[Int],masks: Option[Vector[Array[Int]]])
  final case class Letterboxed(chw: Array[Float],height: Int,width: Int,gain: Double,padLeft: Int,padTop: Int,padRight: Int,padBottom: Int)

  private val Eps=1e-3 // BatchNorm eps of the YOLOv8 Conv block
  private final class P(val w: Array[Float],val b: Array[Float],val ci: Int,val co: Int,val k: Int,val s: Int)

  def load(file: Path): YoloV8=
    require(java.nio.file.Files.isRegularFile(file),s"YOLOv8 weights not found: $file")
    val set=SafeTensorSet.openFile(file)
    try
      val shapes=set.tensors.map((k,v)=>k->v.shape.dims.map(_.toInt).toArray)
      val spec=deriveSpec(shapes)
      val used=scala.collection.mutable.Set.empty[String]
      def tensor(name: String,dims: Seq[Int]): Array[Float]=
        val got=shapes.getOrElse(name,throw IllegalArgumentException(s"missing YOLOv8 weight: $name"))
        require(got.toSeq==dims,s"YOLOv8 weight shape mismatch: $name, expected ${dims.mkString("x")}, got ${got.mkString("x")}")
        used+=name
        set.readFloat(name).toFloatArray
      def convBn(name: String,ci: Int,co: Int,k: Int,s: Int): P=
        val w=tensor(name+".conv.weight",Seq(co,ci,k,k));val g=tensor(name+".bn.weight",Seq(co));val be=tensor(name+".bn.bias",Seq(co))
        val mean=tensor(name+".bn.running_mean",Seq(co));val v=tensor(name+".bn.running_var",Seq(co))
        val per=ci*k*k;val fw=new Array[Float](w.length);val fb=new Array[Float](co)
        for o <- 0 until co do
          val sc=(g(o)/math.sqrt(v(o).toDouble+Eps)).toFloat
          var i=0
          while i<per do { fw(o*per+i)=w(o*per+i)*sc;i+=1 }
          fb(o)=be(o)-mean(o)*sc
        new P(fw,fb,ci,co,k,s)
      def plain(name: String,ci: Int,co: Int): P=new P(tensor(name+".weight",Seq(co,ci,1,1)),tensor(name+".bias",Seq(co)),ci,co,1,1)

      // ---- build the network; every `convBn`/`plain`/`tensor` call both checks a tensor and records its use
      def cv(name: String,ci: Int,co: Int,k: Int,s: Int): Fm=>Fm=
        val p=convBn(name,ci,co,k,s);x => silu(conv(x,p))
      def c2f(name: String,c1: Int,c2: Int,n: Int,shortcut: Boolean): Fm=>Fm=
        val hid=c2/2;val a=cv(name+".cv1",c1,2*hid,1,1);val out=cv(name+".cv2",(2+n)*hid,c2,1,1)
        val bots=(0 until n).map { i =>
          val b1=cv(s"$name.m.$i.cv1",hid,hid,3,1);val b2=cv(s"$name.m.$i.cv2",hid,hid,3,1)
          (x: Fm) => val y=b2(b1(x));if shortcut then add(y,x) else y
        }
        x =>
          val y0=a(x);val parts=scala.collection.mutable.ArrayBuffer(slice(y0,0,hid),slice(y0,hid,2*hid))
          bots.foreach(b=>parts+=b(parts.last))
          out(concatAll(parts.toSeq))
      def sppf(name: String,c1: Int,c2: Int): Fm=>Fm=
        val hid=c1/2;val a=cv(name+".cv1",c1,hid,1,1);val b=cv(name+".cv2",4*hid,c2,1,1)
        x => val x0=a(x);val y1=maxPool5(x0);val y2=maxPool5(y1);val y3=maxPool5(y2);b(concatAll(Seq(x0,y1,y2,y3)))

      val Vector(s0,s1,s2,s3,s4)=spec.stem // widths of the five stages (layers 0,1,3,5,7)
      val r=spec.repeats;val nh=spec.headRepeats
      val l0=cv("model.0",3,s0,3,2);val l1=cv("model.1",s0,s1,3,2);val l2=c2f("model.2",s1,s1,r(0),true)
      val l3=cv("model.3",s1,s2,3,2);val l4=c2f("model.4",s2,s2,r(1),true)
      val l5=cv("model.5",s2,s3,3,2);val l6=c2f("model.6",s3,s3,r(2),true)
      val l7=cv("model.7",s3,s4,3,2);val l8=c2f("model.8",s4,s4,r(3),true);val l9=sppf("model.9",s4,s4)
      val l12=c2f("model.12",s4+s3,s3,nh,false);val l15=c2f("model.15",s3+s2,s2,nh,false)
      val l16=cv("model.16",s2,s2,3,2);val l18=c2f("model.18",s2+s3,s3,nh,false)
      val l19=cv("model.19",s3,s3,3,2);val l21=c2f("model.21",s3+s4,s4,nh,false)
      val chs=Vector(s2,s3,s4)
      val rm4=4*spec.regMax
      val boxB=chs.zipWithIndex.map((c,i)=>(cv(s"model.22.cv2.$i.0",c,spec.c2,3,1),cv(s"model.22.cv2.$i.1",spec.c2,spec.c2,3,1),plain(s"model.22.cv2.$i.2",spec.c2,rm4)))
      val clsB=chs.zipWithIndex.map((c,i)=>(cv(s"model.22.cv3.$i.0",c,spec.c3,3,1),cv(s"model.22.cv3.$i.1",spec.c3,spec.c3,3,1),plain(s"model.22.cv3.$i.2",spec.c3,spec.nc)))
      val coefB=if spec.segment then chs.zipWithIndex.map((c,i)=>(cv(s"model.22.cv4.$i.0",c,spec.c4,3,1),cv(s"model.22.cv4.$i.1",spec.c4,spec.c4,3,1),plain(s"model.22.cv4.$i.2",spec.c4,spec.nm))) else Vector.empty
      val dfl=tensor("model.22.dfl.conv.weight",Seq(1,spec.regMax,1,1))
      require((0 until spec.regMax).forall(i=>dfl(i)==i.toFloat),"YOLOv8 DFL weights must be the fixed 0..regMax-1 ramp")
      val protoNet=
        if spec.segment then
          val a=cv("model.22.proto.cv1",s2,spec.npr,3,1)
          val upW=tensor("model.22.proto.upsample.weight",Seq(spec.npr,spec.npr,2,2));val upB=tensor("model.22.proto.upsample.bias",Seq(spec.npr))
          val b=cv("model.22.proto.cv2",spec.npr,spec.npr,3,1);val c=cv("model.22.proto.cv3",spec.npr,spec.nm,1,1)
          Some((x: Fm) =>
            val y=a(x);val (d,oh,ow)=(AnnotatorOps.convTranspose(y._1,y._2,y._3,y._4,upW,upB,spec.npr,2,2,0,0),y._3*2,y._4*2)
            c(b((d,spec.npr,oh,ow))))
        else None
      val unknown=shapes.keySet.diff(used.toSet)
      require(unknown.isEmpty,s"unexpected YOLOv8 weights: ${unknown.toSeq.sorted.take(5).mkString(",")}")

      val runner=(chw: Array[Float],h: Int,w: Int) =>
        val x0: Fm=(chw,3,h,w)
        val y0=l0(x0);val y1=l1(y0);val y2=l2(y1);val y3=l3(y2);val y4=l4(y3);val y5=l5(y4);val y6=l6(y5);val y7=l7(y6);val y8=l8(y7);val y9=l9(y8)
        val y12=l12(concatAll(Seq(upsample2(y9),y6)))
        val y15=l15(concatAll(Seq(upsample2(y12),y4)))
        val y18=l18(concatAll(Seq(l16(y15),y12)))
        val y21=l21(concatAll(Seq(l19(y18),y9)))
        val feats=Vector(y15,y18,y21)
        def branch(b: (Fm=>Fm,Fm=>Fm,P),f: Fm): Fm=conv(b._2(b._1(f)),b._3)
        Raw(feats.zip(boxB).map((f,b)=>branch(b,f)),feats.zip(clsB).map((f,b)=>branch(b,f)),feats.zip(coefB).map((f,b)=>branch(b,f)),protoNet.map(_(y15)))
      new YoloV8(spec,runner)
    finally set.close()

  /** Recovers the architecture sizes from tensor shapes (the full name/shape check happens while the network is built). */
  private def deriveSpec(shapes: Map[String,Array[Int]]): Spec=
    def dim(name: String,axis: Int)=shapes.getOrElse(name,throw IllegalArgumentException(s"missing YOLOv8 weight: $name")).apply(axis)
    def repeats(layer: Int)=Iterator.from(0).takeWhile(i=>shapes.contains(s"model.$layer.m.$i.cv1.conv.weight")).size
    val stem=Vector(0,1,3,5,7).map(l=>dim(s"model.$l.conv.weight",0))
    val rep=Vector(2,4,6,8).map(repeats)
    require(rep.forall(_>0),"YOLOv8 backbone C2f stages must have at least one bottleneck")
    val nh=Vector(12,15,18,21).map(repeats).distinct
    require(nh.size==1 && nh.head>0,"YOLOv8 head C2f stages must share one repeat count")
    val seg=shapes.contains("model.22.proto.cv1.conv.weight")
    Spec(stem,rep,nh.head,nc=dim("model.22.cv3.0.2.weight",0),regMax=dim("model.22.cv2.0.2.weight",0)/4,
      nm=if seg then dim("model.22.cv4.0.2.weight",0) else 0,npr=if seg then dim("model.22.proto.cv1.conv.weight",0) else 0,
      c2=dim("model.22.cv2.0.0.conv.weight",0),c3=dim("model.22.cv3.0.0.conv.weight",0),c4=if seg then dim("model.22.cv4.0.0.conv.weight",0) else 0)

  // ---------------------------------------------------------------- layers
  private def silu(x: Fm): Fm=
    val d=x._1;DepthAnythingV2.parallel(x._2) { c =>
      val n=x._3*x._4;var i=c*n;val e=i+n
      while i<e do { val v=d(i);d(i)=(v/(1.0+math.exp(-v.toDouble))).toFloat;i+=1 }
    }
    x

  private def add(a: Fm,b: Fm): Fm=
    require(a._1.length==b._1.length,"YOLOv8 residual size mismatch")
    var i=0;while i<a._1.length do { a._1(i)+=b._1(i);i+=1 }
    a

  private def slice(x: Fm,c0: Int,c1: Int): Fm=
    val n=x._3*x._4;(java.util.Arrays.copyOfRange(x._1,c0*n,c1*n),c1-c0,x._3,x._4)

  private def concatAll(xs: Seq[Fm]): Fm=
    val h=xs.head._3;val w=xs.head._4
    require(xs.forall(f=>f._3==h && f._4==w),"YOLOv8 concat size mismatch")
    val out=new Array[Float](xs.map(_._1.length).sum);var off=0
    xs.foreach { f => System.arraycopy(f._1,0,out,off,f._1.length);off+=f._1.length }
    (out,xs.map(_._2).sum,h,w)

  private def upsample2(x: Fm): Fm=
    val (d,c,h,w)=x;val oh=h*2;val ow=w*2;val out=new Array[Float](c*oh*ow)
    DepthAnythingV2.parallel(c) { ch =>
      var y=0
      while y<oh do
        val sb=ch*h*w+(y/2)*w;val ob=ch*oh*ow+y*ow;var xx=0
        while xx<ow do { out(ob+xx)=d(sb+xx/2);xx+=1 }
        y+=1
    }
    (out,c,oh,ow)

  /** 5x5 max pool, stride 1, padding 2 (out-of-range cells are ignored). */
  private def maxPool5(x: Fm): Fm=
    val (d,c,h,w)=x;val out=new Array[Float](d.length);val tmp=new Array[Float](d.length)
    DepthAnythingV2.parallel(c) { ch =>
      val b=ch*h*w
      var y=0
      while y<h do
        var xx=0
        while xx<w do
          var m=Float.NegativeInfinity;var k=math.max(0,xx-2)
          while k<=math.min(w-1,xx+2) do { if d(b+y*w+k)>m then m=d(b+y*w+k);k+=1 }
          tmp(b+y*w+xx)=m;xx+=1
        y+=1
      y=0
      while y<h do
        var xx=0
        while xx<w do
          var m=Float.NegativeInfinity;var k=math.max(0,y-2)
          while k<=math.min(h-1,y+2) do { if tmp(b+k*w+xx)>m then m=tmp(b+k*w+xx);k+=1 }
          out(b+y*w+xx)=m;xx+=1
        y+=1
    }
    (out,c,h,w)

  /** `Conv2d` (groups 1, padding k/2) with the layer's folded weights. */
  private def conv(x: Fm,p: P): Fm=
    val (d,ci,h,w)=x
    require(ci==p.ci,s"YOLOv8 channel mismatch: ${p.ci} expected, got $ci")
    val k=p.k;val s=p.s;val pad=k/2;val oh=(h+2*pad-k)/s+1;val ow=(w+2*pad-k)/s+1
    val out=new Array[Float](p.co*oh*ow)
    DepthAnythingV2.parallel(p.co) { o =>
      val ob=o*oh*ow;java.util.Arrays.fill(out,ob,ob+oh*ow,p.b(o))
      var c=0
      while c<ci do
        val xb=c*h*w
        var ky=0
        while ky<k do
          var kx=0
          while kx<k do
            val wv=p.w(((o*ci+c)*k+ky)*k+kx)
            // valid output columns: 0 <= ox*s-pad+kx < w
            val ox0=if pad-kx>0 then (pad-kx+s-1)/s else 0
            val ox1=math.min(ow-1,(w-1+pad-kx)/s)
            var oy=0
            while oy<oh do
              val iy=oy*s-pad+ky
              if iy>=0 && iy<h then
                val xr=xb+iy*w-pad+kx;val orow=ob+oy*ow
                var ox=ox0
                if s==1 then while ox<=ox1 do { out(orow+ox)+=wv*d(xr+ox);ox+=1 }
                else while ox<=ox1 do { out(orow+ox)+=wv*d(xr+ox*s);ox+=1 }
              oy+=1
            kx+=1
          ky+=1
        c+=1
    }
    (out,p.co,oh,ow)

  // ---------------------------------------------------------------- pre / post processing
  /** Documented letterbox: scale to fit `imgsz` keeping aspect (bilinear), pad the shorter side with 114 grey up to a multiple of 32 (minimal rectangle, centred). Input scaled to 0..1. */
  def letterbox(image: BufferedImage,imgsz: Int=640): Letterboxed=
    val w=image.getWidth;val h=image.getHeight
    val r=math.min(imgsz.toDouble/h,imgsz.toDouble/w)
    val nw=math.rint(w*r).toInt;val nh=math.rint(h*r).toInt
    val dw=((imgsz-nw)%32)/2.0;val dh=((imgsz-nh)%32)/2.0
    val top=math.rint(dh-0.1).toInt;val bottom=math.rint(dh+0.1).toInt
    val left=math.rint(dw-0.1).toInt;val right=math.rint(dw+0.1).toInt
    val src=new Array[Float](3*w*h)
    for y <- 0 until h; x <- 0 until w do
      val c=image.getRGB(x,y);val i=y*w+x
      src(i)=((c>>>16)&255).toFloat;src(w*h+i)=((c>>>8)&255).toFloat;src(2*w*h+i)=(c&255).toFloat
    val resized=if nw==w && nh==h then src else DepthAnythingV2.bilinear(src,3,h,w,nh,nw,false)
    val oh=nh+top+bottom;val ow=nw+left+right
    val out=new Array[Float](3*oh*ow);java.util.Arrays.fill(out,114f/255f)
    for c <- 0 until 3; y <- 0 until nh; x <- 0 until nw do
      // cv2 resizes uint8 images, so the resampled values are rounded and clamped to 0..255
      val v=math.max(0.0,math.min(255.0,math.rint(resized(c*nh*nw+y*nw+x).toDouble)))
      out(c*oh*ow+(y+top)*ow+x+left)=(v/255.0).toFloat
    Letterboxed(out,oh,ow,r,left,top,right,bottom)

  /** Decodes the raw outputs: DFL boxes (xyxy, input pixels), sigmoid class scores, mask coefficients. */
  def decode(raw: Raw,inH: Int,inW: Int): Predictions=
    val levels=raw.box.indices
    val n=levels.map(l=>raw.box(l)._3*raw.box(l)._4).sum
    val nc=raw.cls.head._2;val regMax=raw.box.head._2/4;val nm=raw.coef.headOption.map(_._2).getOrElse(0)
    val box=new Array[Float](4*n);val scores=new Array[Float](n*nc);val best=new Array[Float](n);val cls=new Array[Int](n)
    val coef=new Array[Array[Float]](n)
    var base=0
    for l <- levels do
      val (bd,_,h,w)=raw.box(l);val (cd,_,_,_)=raw.cls(l);val hw=h*w
      val stride=inH.toFloat/h
      DepthAnythingV2.parallel(hw) { a =>
        val gx=a%w+0.5f;val gy=a/w+0.5f;val dist=new Array[Float](4);val e=new Array[Double](regMax)
        for side <- 0 until 4 do
          var mx=Double.NegativeInfinity;var j=0
          while j<regMax do { val v=bd((side*regMax+j)*hw+a).toDouble;e(j)=v;if v>mx then mx=v;j+=1 }
          var sum=0.0;var ex=0.0;j=0
          while j<regMax do { val p=math.exp(e(j)-mx);sum+=p;ex+=p*j;j+=1 }
          dist(side)=(ex/sum).toFloat
        val i=base+a
        box(4*i)=(gx-dist(0))*stride;box(4*i+1)=(gy-dist(1))*stride;box(4*i+2)=(gx+dist(2))*stride;box(4*i+3)=(gy+dist(3))*stride
        var bs=Float.NegativeInfinity;var bc=0;var c=0
        while c<nc do
          val s=AnnotatorOps.sigmoid(cd(c*hw+a));scores(i*nc+c)=s
          if s>bs then { bs=s;bc=c };c+=1
        best(i)=bs;cls(i)=bc
        if nm>0 then
          val md=raw.coef(l)._1;val cf=new Array[Float](nm);var m=0
          while m<nm do { cf(m)=md(m*hw+a);m+=1 }
          coef(i)=cf
      }
      base+=hw
    Predictions(n,box,scores,best,cls,coef,nc)

  /** Candidates whose best class score is `>= conf`, then greedy class-wise NMS (IoU `> iou` suppresses), best first, at most `maxDet`. Returns anchor indices. */
  def nms(p: Predictions,conf: Float,iou: Float,maxDet: Int): Vector[Int]=
    val cand=(0 until p.n).filter(i=>p.score(i)>conf).sortBy(i=>(-p.score(i),i))
    val kept=scala.collection.mutable.ArrayBuffer.empty[Int]
    val it=cand.iterator
    while it.hasNext && kept.size<maxDet do
      val i=it.next()
      val ax1=p.box(4*i);val ay1=p.box(4*i+1);val ax2=p.box(4*i+2);val ay2=p.box(4*i+3);val aa=(ax2-ax1)*(ay2-ay1)
      val clash=kept.exists { j =>
        p.cls(j)==p.cls(i) && {
          val iw=math.min(ax2,p.box(4*j+2))-math.max(ax1,p.box(4*j));val ih=math.min(ay2,p.box(4*j+3))-math.max(ay1,p.box(4*j+1))
          val inter=if iw>0f && ih>0f then iw*ih else 0f
          val ba=(p.box(4*j+2)-p.box(4*j))*(p.box(4*j+3)-p.box(4*j+1))
          inter/(aa+ba-inter)>iou
        }
      }
      if !clash then kept+=i
    kept.toVector

  /** Instance mask: prototypes x coefficients (logits), zeroed outside the box (box mapped to prototype resolution), bilinearly upsampled to the
    * letterboxed input, thresholded at 0 (= sigmoid 0.5), padding removed and resized to the source image. Values 0 or 255, `w*h`. */
  def mask(proto: Fm,coef: Array[Float],box: (Float,Float,Float,Float),lb: Letterboxed,w: Int,h: Int): Array[Int]=
    val (pd,nm,ph,pw)=proto;val n=ph*pw;val logit=new Array[Float](n)
    require(coef.length==nm,"mask coefficient count mismatch")
    val sx=pw.toFloat/lb.width;val sy=ph.toFloat/lb.height
    val bx1=box._1*sx;val by1=box._2*sy;val bx2=box._3*sx;val by2=box._4*sy
    var y=0
    while y<ph do
      if y>=by1 && y<by2 then
        var x=0
        while x<pw do
          if x>=bx1 && x<bx2 then
            var s=0f;var m=0
            while m<nm do { s+=coef(m)*pd(m*n+y*pw+x);m+=1 }
            logit(y*pw+x)=s
          x+=1
      y+=1
    val up=DepthAnythingV2.bilinear(logit,1,ph,pw,lb.height,lb.width,false)
    val ih=lb.height-lb.padTop-lb.padBottom;val iw=lb.width-lb.padLeft-lb.padRight
    val crop=new Array[Float](ih*iw)
    for yy <- 0 until ih; xx <- 0 until iw do crop(yy*iw+xx)=if up((yy+lb.padTop)*lb.width+xx+lb.padLeft)>0f then 1f else 0f
    val full=if iw==w && ih==h then crop else DepthAnythingV2.bilinear(crop,1,ih,iw,h,w,false)
    full.map(v=>if v>0.5f then 255 else 0)

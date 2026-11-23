package scalatorch.models

import java.nio.file.Path
import scalatorch.io.SafeTensorSet

/** U2-Net (full) salient-object segmentation network, CPU Float32, plain arrays (CHW).
  *
  * Architecture from the U2-Net paper (Qin et al., "U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection", Pattern
  * Recognition 2020; the reference code xuebinqin/U-2-Net is Apache-2.0, nothing is copied here): six encoder stages RSU-7/6/5/4/4F/4F joined
  * by 2x2 max pooling (ceil mode), five decoder stages, one 3x3 side output per stage (bilinearly upsampled to the stage-1 size), and a 1x1
  * fuse convolution over the six concatenated side maps. Every `REBNCONV` is conv3x3 (+ dilation) + BatchNorm + ReLU; the weights come from
  * rembg's `u2net.onnx` export where BatchNorm is already folded into the conv, converted offline by
  * scripts/convert_u2net_onnx_to_safetensors.py (names `stage1.rebnconvin.weight`, `side3.bias`, `outconv.weight`, ...), so each layer is
  * conv + bias + ReLU. Upsampling is bilinear with align_corners=False. Weights: the U2-Net authors' `u2net.pth` re-hosted by rembg (the
  * weight licence is not stated in the files; see docs). Loading is strict: unknown, missing or mis-shaped tensors are rejected.
  *
  * `infer` returns the seven sigmoid maps d0 (fused) and d1..d6 (side outputs) at the input size. */
final class U2Net private (run: U2Net.Fm => Vector[Array[Float]]):
  /** The seven sigmoid maps (d0 fused, d1..d6 sides), each h*w, for a normalised CHW input of `h` x `w`. */
  def infer(chw: Array[Float],h: Int,w: Int): Vector[Array[Float]]=
    require(chw.length==3*h*w,"U2-Net input must be 3 x h x w")
    run((chw,3,h,w)).map(m=>m.map(AnnotatorOps.sigmoid))

object U2Net:
  /** A CHW feature map. */
  type Fm=(Array[Float],Int,Int,Int)
  private final class Cv(val w: Array[Float],val b: Array[Float],val ci: Int,val co: Int,val k: Int,val d: Int,val relu: Boolean)

  /** Strict loader: every tensor of the converted checkpoint must be a layer of the network and vice versa. */
  def load(file: Path): U2Net=
    require(java.nio.file.Files.isRegularFile(file),s"U2-Net weights not found: $file")
    val set=SafeTensorSet.openFile(file)
    try
      val shapes=set.tensors.map((k,v)=>k->v.shape.dims.map(_.toInt).toSeq)
      val used=scala.collection.mutable.Set.empty[String]
      def cv(name: String,ci: Int,co: Int,dil: Int,k: Int=3,relu: Boolean=true): Cv=
        def tensor(n: String,dims: Seq[Int]): Array[Float]=
          val got=shapes.getOrElse(n,throw IllegalArgumentException(s"missing U2-Net weight: $n"))
          require(got==dims,s"U2-Net weight shape mismatch: $n, expected ${dims.mkString("x")}, got ${got.mkString("x")}")
          used+=n;set.readFloat(n).toFloatArray
        new Cv(tensor(name+".weight",Seq(co,ci,k,k)),tensor(name+".bias",Seq(co)),ci,co,k,dil,relu)

      /** RSU-n block (n = 7,6,5,4): n-1 pooled encoder levels, a dilated bottom conv, mirrored decoder convs, residual from the input conv. */
      def rsu(name: String,n: Int,cin: Int,mid: Int,cout: Int): Fm=>Fm=
        val in=cv(s"$name.rebnconvin",cin,cout,1)
        val enc=(1 until n).map(i=>cv(s"$name.rebnconv$i",if i==1 then cout else mid,mid,1))
        val bottom=cv(s"$name.rebnconv$n",mid,mid,2)
        val dec=(n-1 to 1 by -1).map(i=>cv(s"$name.rebnconv${i}d",2*mid,if i==1 then cout else mid,1)) // dec(0) is level n-1
        x =>
          val xin=conv(x,in)
          val hs=new Array[Fm](n)
          hs(1)=conv(xin,enc(0))
          var i=2
          while i<n do { hs(i)=conv(maxPool2(hs(i-1)),enc(i-1));i+=1 }
          var d=conv(concat(conv(hs(n-1),bottom),hs(n-1)),dec(0))
          i=n-2
          while i>=1 do { d=conv(concat(upsampleLike(d,hs(i)),hs(i)),dec(n-1-i));i-=1 }
          add(d,xin)

      /** RSU-4F block: no pooling, dilations 1,2,4,8 instead. */
      def rsu4f(name: String,cin: Int,mid: Int,cout: Int): Fm=>Fm=
        val in=cv(s"$name.rebnconvin",cin,cout,1)
        val c1=cv(s"$name.rebnconv1",cout,mid,1);val c2=cv(s"$name.rebnconv2",mid,mid,2)
        val c3=cv(s"$name.rebnconv3",mid,mid,4);val c4=cv(s"$name.rebnconv4",mid,mid,8)
        val d3=cv(s"$name.rebnconv3d",2*mid,mid,4);val d2=cv(s"$name.rebnconv2d",2*mid,mid,2);val d1=cv(s"$name.rebnconv1d",2*mid,cout,1)
        x =>
          val xin=conv(x,in)
          val h1=conv(xin,c1);val h2=conv(h1,c2);val h3=conv(h2,c3);val h4=conv(h3,c4)
          val y3=conv(concat(h4,h3),d3);val y2=conv(concat(y3,h2),d2);val y1=conv(concat(y2,h1),d1)
          add(y1,xin)

      val stage1=rsu("stage1",7,3,32,64);val stage2=rsu("stage2",6,64,32,128)
      val stage3=rsu("stage3",5,128,64,256);val stage4=rsu("stage4",4,256,128,512)
      val stage5=rsu4f("stage5",512,256,512);val stage6=rsu4f("stage6",512,256,512)
      val stage5d=rsu4f("stage5d",1024,256,512);val stage4d=rsu("stage4d",4,1024,128,256)
      val stage3d=rsu("stage3d",5,512,64,128);val stage2d=rsu("stage2d",6,256,32,64);val stage1d=rsu("stage1d",7,128,16,64)
      val sides=Vector(64,64,128,256,512,512).zipWithIndex.map((c,i)=>cv(s"side${i+1}",c,1,1,relu=false))
      val outconv=cv("outconv",6,1,1,k=1,relu=false)
      val unused=shapes.keySet.diff(used.toSet)
      require(unused.isEmpty,s"unexpected U2-Net weights: ${unused.take(5).mkString(",")}")

      val run: Fm=>Vector[Array[Float]]= x =>
        val h1=stage1(x);val h2=stage2(maxPool2(h1));val h3=stage3(maxPool2(h2))
        val h4=stage4(maxPool2(h3));val h5=stage5(maxPool2(h4));val h6=stage6(maxPool2(h5))
        val h5d=stage5d(concat(upsampleLike(h6,h5),h5))
        val h4d=stage4d(concat(upsampleLike(h5d,h4),h4))
        val h3d=stage3d(concat(upsampleLike(h4d,h3),h3))
        val h2d=stage2d(concat(upsampleLike(h3d,h2),h2))
        val h1d=stage1d(concat(upsampleLike(h2d,h1),h1))
        val maps=Vector(h1d,h2d,h3d,h4d,h5d,h6).zip(sides).map((f,s)=>upsampleLike(conv(f,s),h1))
        val d0=conv(maps.reduce(concat),outconv)
        (d0+:maps).map(_._1)
      new U2Net(run)
    finally set.close()

  private def concat(a: Fm,b: Fm): Fm=
    require(a._3==b._3 && a._4==b._4,"U2-Net concat size mismatch")
    (AnnotatorOps.concat(a._1,b._1),a._2+b._2,a._3,a._4)

  private def add(a: Fm,b: Fm): Fm=
    val o=new Array[Float](a._1.length);var i=0
    while i<o.length do { o(i)=a._1(i)+b._1(i);i+=1 }
    (o,a._2,a._3,a._4)

  /** `MaxPool2d(2, stride 2, ceil_mode=True)`: partial border windows are pooled over the pixels that exist. */
  private def maxPool2(x: Fm): Fm=
    val (d,c,h,w)=x;val oh=(h+1)/2;val ow=(w+1)/2;val out=new Array[Float](c*oh*ow)
    DepthAnythingV2.parallel(c) { ch =>
      var y=0
      while y<oh do
        var xx=0
        while xx<ow do
          var m=Float.NegativeInfinity;var dy=0
          while dy<2 do
            val sy=2*y+dy
            if sy<h then
              var dx=0
              while dx<2 do { val sx=2*xx+dx;if sx<w then m=math.max(m,d(ch*h*w+sy*w+sx));dx+=1 }
            dy+=1
          out(ch*oh*ow+y*ow+xx)=m;xx+=1
        y+=1
    }
    (out,c,oh,ow)

  /** Bilinear resize (align_corners=False, half-pixel centres, no antialiasing) of every channel to `oh` x `ow`. */
  def bilinear(d: Array[Float],c: Int,h: Int,w: Int,oh: Int,ow: Int): Array[Float]=
    if h==oh && w==ow then return d
    val out=new Array[Float](c*oh*ow)
    val sy=h.toDouble/oh;val sx=w.toDouble/ow
    val y0=new Array[Int](oh);val y1=new Array[Int](oh);val fy=new Array[Float](oh)
    for y <- 0 until oh do
      val s=math.max(0.0,(y+0.5)*sy-0.5);val i=math.min(s.toInt,h-1)
      y0(y)=i;y1(y)=math.min(i+1,h-1);fy(y)=(s-i).toFloat
    val x0=new Array[Int](ow);val x1=new Array[Int](ow);val fx=new Array[Float](ow)
    for x <- 0 until ow do
      val s=math.max(0.0,(x+0.5)*sx-0.5);val i=math.min(s.toInt,w-1)
      x0(x)=i;x1(x)=math.min(i+1,w-1);fx(x)=(s-i).toFloat
    DepthAnythingV2.parallel(c) { ch =>
      val b=ch*h*w;val ob=ch*oh*ow;var y=0
      while y<oh do
        var x=0
        while x<ow do
          val a=d(b+y0(y)*w+x0(x))*(1-fx(x))+d(b+y0(y)*w+x1(x))*fx(x)
          val bb=d(b+y1(y)*w+x0(x))*(1-fx(x))+d(b+y1(y)*w+x1(x))*fx(x)
          out(ob+y*ow+x)=a*(1-fy(y))+bb*fy(y);x+=1
        y+=1
    }
    out

  private def upsampleLike(x: Fm,target: Fm): Fm=(bilinear(x._1,x._2,x._3,x._4,target._3,target._4),x._2,target._3,target._4)

  /** Stride-1 `Conv2d` with zero padding `d*(k/2)` and dilation `d`, bias, and optional fused ReLU. */
  private def conv(x: Fm,p: Cv): Fm=
    val (d,ci,h,w)=x
    require(ci==p.ci,s"U2-Net channel mismatch: ${p.ci} expected, got $ci")
    val k=p.k;val dil=p.d;val pad=dil*(k/2);val out=new Array[Float](p.co*h*w)
    DepthAnythingV2.parallel(p.co) { o =>
      val ob=o*h*w;java.util.Arrays.fill(out,ob,ob+h*w,p.b(o))
      var c=0
      while c<ci do
        val xb=c*h*w
        var ky=0
        while ky<k do
          val dy=ky*dil-pad
          var kx=0
          while kx<k do
            val dx=kx*dil-pad
            val wv=p.w(((o*ci+c)*k+ky)*k+kx)
            val ox0=math.max(0,-dx);val ox1=math.min(w-1,w-1-dx)
            if ox0<=ox1 then
              var oy=math.max(0,-dy)
              val oyEnd=math.min(h-1,h-1-dy)
              while oy<=oyEnd do
                val xr=xb+(oy+dy)*w+dx;val orow=ob+oy*w
                var ox=ox0
                while ox<=ox1 do { out(orow+ox)+=wv*d(xr+ox);ox+=1 }
                oy+=1
            kx+=1
          ky+=1
        c+=1
      if p.relu then
        var i=ob
        while i<ob+h*w do { if out(i)<0f then out(i)=0f;i+=1 }
    }
    (out,p.co,h,w)

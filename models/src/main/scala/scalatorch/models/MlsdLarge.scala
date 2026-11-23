package scalatorch.models

import java.nio.file.Path

/** The M-LSD "large" line-segment detector (`mlsd_large_512_fp32.pth`, MobileV2_MLSD_Large) as used by the ControlNet `mlsd`
  * preprocessor. Network: a MobileNetV2 trunk (features 0..13, 4 input channels = RGB plus a constant plane, ReLU6) whose
  * outputs after features 1, 3, 6, 10 and 13 (strides 2, 4, 8, 16, 16) feed a decoder of three kinds of block: A (1x1 projections
  * of the skip and of the deeper map, bilinear 2x upsampling with align_corners of the deeper one, channel concatenation), B
  * (3x3 conv + residual, then a 3x3 conv) and C (a dilated 3x3 conv, a 3x3 conv, a 1x1 conv to 16 channels). Channels 7..15 of
  * the 16-channel stride-2 output are the `tp_map`: a centre logit plus four displacement channels (and four unused ones). Plain
  * CPU Float32; batch norm is folded into the convolutions at load time.
  *
  * Decoding follows the documented procedure: sigmoid of the centre channel, 3x3 non-maximum suppression, the best 200 peaks,
  * keep those scoring above `thrV` whose segment (displacement vector length, in tp_map pixels) is longer than `thrD`, and
  * scale endpoints by 2 to the 512 input.
  *
  * Licence note: M-LSD (Gu et al., "Towards Light-weight and Real-time Line Segment Detection", AAAI 2022) is released by NAVER
  * under Apache-2.0; the weights are distributed through lllyasviel/Annotators (ControlNet). The weights are not bundled with
  * this repository (kept in the NUI `models/annotators` folder). Written from the checkpoint keys and the documented algorithm,
  * not copied from any repository. */
final class MlsdLarge private (weights: Map[String,Array[Float]]):
  import MlsdLarge.*

  /** conv name -> (weight, bias) with the following eval-mode BatchNorm (eps 1e-5) folded in. */
  private val folded: Map[String,(Array[Float],Array[Float])]=
    layout._2.map { (convName,bn) =>
      val w=weights(convName+".weight");val g=weights(bn+".weight");val b=weights(bn+".bias")
      val mean=weights(bn+".running_mean");val v=weights(bn+".running_var")
      val cb=weights.get(convName+".bias");val co=g.length;val per=w.length/co
      val fw=new Array[Float](w.length);val fb=new Array[Float](co)
      for o <- 0 until co do
        val s=(g(o)/math.sqrt(v(o).toDouble+1e-5)).toFloat
        var i=0
        while i<per do { fw(o*per+i)=w(o*per+i)*s;i+=1 }
        fb(o)=b(o)+(cb.map(_(o)).getOrElse(0f)-mean(o))*s
      convName->(fw,fb)
    }.toMap

  /** act: 0 none, 1 ReLU, 2 ReLU6. */
  private def unit(x: (Array[Float],Int,Int,Int),conv: String,co: Int,k: Int,stride: Int,pad: Int,dil: Int,groups: Int,act: Int): (Array[Float],Int,Int,Int)=
    val (fw,fb)=folded(conv)
    val (y,oh,ow)=grouped(x._1,x._2,x._3,x._4,fw,fb,co,k,stride,pad,dil,groups)
    act match
      case 1 => AnnotatorOps.relu(y)
      case 2 => var i=0;while i<y.length do { y(i)=math.min(6f,math.max(0f,y(i)));i+=1 }
      case _ =>
    (y,co,oh,ow)

  /** The 9-channel `tp_map` (CHW, stride-2 resolution) for an RGB CHW image with values 0..255 (sides multiples of 32). */
  def tpMap(rgb: Array[Float],height: Int,width: Int,cancelled: () => Boolean=()=>false): (Array[Float],Int,Int)=
    require(height>=64 && width>=64 && height%32==0 && width%32==0 && rgb.length==3*height*width,"M-LSD input must be CHW RGB with sides that are multiples of 32")
    require(height.toLong*width<=4194304L,"M-LSD input is too large")
    def check(): Unit=if cancelled() || Thread.currentThread().isInterrupted then throw new java.util.concurrent.CancellationException("mlsd cancelled")
    val n=height*width;val in=new Array[Float](4*n)
    for i <- 0 until 3*n do in(i)=rgb(i)/127.5f-1f
    java.util.Arrays.fill(in,3*n,4*n,1f/127.5f-1f) // the constant plane is 1, normalised like the image
    type T=(Array[Float],Int,Int,Int)
    var x: T=unit((in,4,height,width),"backbone.features.0.0",32,3,2,1,1,1,2)
    val taps=scala.collection.mutable.HashMap.empty[Int,T]
    locally {
      val d=unit(x,"backbone.features.1.conv.0.0",32,3,1,1,1,32,2)
      x=unit(d,"backbone.features.1.conv.1",16,1,1,0,1,1,0);taps(1)=x
    }
    for ((cin,cout,stride),idx) <- Inverted.zipWithIndex do
      check()
      val f=idx+2;val p=s"backbone.features.$f.conv";val hidden=cin*6
      val e=unit(x,s"$p.0.0",hidden,1,1,0,1,1,2)
      val d=unit(e,s"$p.1.0",hidden,3,stride,1,1,hidden,2)
      val q=unit(d,s"$p.2",cout,1,1,0,1,1,0)
      if stride==1 && cin==cout then { var i=0;while i<q._1.length do { q._1(i)+=x._1(i);i+=1 } }
      x=q
      if Taps(f) then taps(f)=x
    taps(13)=x

    def blockA(skip: T,deep: T,name: String,upscale: Boolean): T=
      val b0=unit(deep,s"$name.conv1.0",64,1,1,0,1,1,1)
      val a=unit(skip,s"$name.conv2.0",64,1,1,0,1,1,1)
      val b=if upscale then (DepthAnythingV2.bilinear(b0._1,64,b0._3,b0._4,b0._3*2,b0._4*2,true),64,b0._3*2,b0._4*2) else b0
      require(b._3==a._3 && b._4==a._4,s"M-LSD $name size mismatch")
      (AnnotatorOps.concat(a._1,b._1),128,a._3,a._4)
    def blockB(in: T,name: String): T=
      val y=unit(in,s"$name.conv1.0",128,3,1,1,1,1,1)
      var i=0;while i<y._1.length do { y._1(i)+=in._1(i);i+=1 }
      unit(y,s"$name.conv2.0",64,3,1,1,1,1,1)
    check()
    var cur=blockB(blockA(taps(10),taps(13),"block15",false),"block16")
    check();cur=blockB(blockA(taps(6),cur,"block17",true),"block18")
    check();cur=blockB(blockA(taps(3),cur,"block19",true),"block20")
    check();cur=blockB(blockA(taps(1),cur,"block21",true),"block22")
    check()
    val c1=unit(cur,"block23.conv1.0",64,3,1,5,5,1,1) // dilation 5
    val c2=unit(c1,"block23.conv2.0",64,3,1,1,1,1,1)
    val (out,oh,ow)=grouped(c2._1,64,c2._3,c2._4,weights("block23.conv3.weight"),weights("block23.conv3.bias"),16,1,1,0,1,1)
    (java.util.Arrays.copyOfRange(out,7*oh*ow,16*oh*ow),oh,ow)

object MlsdLarge:
  /** A segment as (x0, y0, x1, y1) in input pixels. */
  type Segment=(Float,Float,Float,Float)
  /** (in, out, stride) of MobileNetV2 features 2..13. */
  val Inverted=Vector((16,24,2),(24,24,1),(24,32,2),(32,32,1),(32,32,1),(32,64,2),(64,64,1),(64,64,1),(64,64,1),(64,96,1),(96,96,1),(96,96,1))
  private val Taps=Set(3,6,10)

  def load(file: Path,cancelled: () => Boolean=()=>false): MlsdLarge=
    new MlsdLarge(AnnotatorOps.loadStrict(file,layout._1,"M-LSD",cancelled))

  /** Expected tensor shapes and the (conv name, batch-norm name) pairs. */
  private lazy val layout: (Map[String,Seq[Int]],Vector[(String,String)])=
    val m=scala.collection.mutable.LinkedHashMap.empty[String,Seq[Int]]
    val pairs=Vector.newBuilder[(String,String)]
    def bn(conv: String,name: String,c: Int): Unit=
      m(name+".weight")=Seq(c);m(name+".bias")=Seq(c);m(name+".running_mean")=Seq(c);m(name+".running_var")=Seq(c);m(name+".num_batches_tracked")=Seq()
      pairs+=((conv,name))
    m("backbone.features.0.0.weight")=Seq(32,4,3,3);bn("backbone.features.0.0","backbone.features.0.1",32)
    m("backbone.features.1.conv.0.0.weight")=Seq(32,1,3,3);bn("backbone.features.1.conv.0.0","backbone.features.1.conv.0.1",32)
    m("backbone.features.1.conv.1.weight")=Seq(16,32,1,1);bn("backbone.features.1.conv.1","backbone.features.1.conv.2",16)
    for ((cin,cout,_),i) <- Inverted.zipWithIndex do
      val p=s"backbone.features.${i+2}.conv";val hid=cin*6
      m(s"$p.0.0.weight")=Seq(hid,cin,1,1);bn(s"$p.0.0",s"$p.0.1",hid)
      m(s"$p.1.0.weight")=Seq(hid,1,3,3);bn(s"$p.1.0",s"$p.1.1",hid)
      m(s"$p.2.weight")=Seq(cout,hid,1,1);bn(s"$p.2",s"$p.3",cout)
    def cbr(name: String,co: Int,ci: Int,k: Int): Unit=
      m(s"$name.0.weight")=Seq(co,ci,k,k);m(s"$name.0.bias")=Seq(co);bn(s"$name.0",s"$name.1",co)
    for (b,skipC) <- Seq(("block15",64),("block17",32),("block19",24),("block21",16)) do
      cbr(s"$b.conv1",64,if b=="block15" then 96 else 64,1);cbr(s"$b.conv2",64,skipC,1)
    for b <- Seq("block16","block18","block20","block22") do { cbr(s"$b.conv1",128,128,3);cbr(s"$b.conv2",64,128,3) }
    cbr("block23.conv1",64,64,3);cbr("block23.conv2",64,64,3)
    m("block23.conv3.weight")=Seq(16,64,1,1);m("block23.conv3.bias")=Seq(16)
    (m.toMap,pairs.result())

  /** Grouped / dilated `Conv2d` with weights in torch's (out, in/groups, k, k) layout. Returns (data, outH, outW). */
  def grouped(x: Array[Float],ci: Int,h: Int,w: Int,wt: Array[Float],bias: Array[Float],co: Int,k: Int,stride: Int,pad: Int,dil: Int,groups: Int): (Array[Float],Int,Int)=
    val oh=(h+2*pad-dil*(k-1)-1)/stride+1;val ow=(w+2*pad-dil*(k-1)-1)/stride+1
    val out=new Array[Float](co*oh*ow);val cig=ci/groups;val cog=co/groups
    DepthAnythingV2.parallel(co) { o =>
      val base=o*oh*ow;val acc=new Array[Double](oh*ow);java.util.Arrays.fill(acc,bias(o).toDouble) // double accumulation keeps the 25-layer net within fp32 noise of torch
      val g=o/cog
      var c=0
      while c<cig do
        val xb=(g*cig+c)*h*w
        var ky=0
        while ky<k do
          var kx=0
          while kx<k do
            val wv=wt(((o*cig+c)*k+ky)*k+kx)
            var oy=0
            while oy<oh do
              val iy=oy*stride-pad+ky*dil
              if iy>=0 && iy<h then
                val ob=oy*ow;val xr=xb+iy*w
                var ox=0
                while ox<ow do
                  val ix=ox*stride-pad+kx*dil
                  if ix>=0 && ix<w then acc(ob+ox)+=wv*x(xr+ix)
                  ox+=1
              oy+=1
            kx+=1
          ky+=1
        c+=1
      var i=0
      while i<acc.length do { out(base+i)=acc(i).toFloat;i+=1 }
    }
    (out,oh,ow)

  /** Segments from a `tp_map` (9 x h x w): sigmoid centres, `pool` x `pool` non-maximum suppression, top `topK`, `score > thrV`
    * and displacement length `> thrD`; endpoints are scaled by 2 (tp_map pixels to input pixels). Best score first. */
  def decode(tp: Array[Float],h: Int,w: Int,thrV: Float,thrD: Float,topK: Int=200,pool: Int=3): Vector[(Float,Segment)]=
    val n=h*w;val heat=new Array[Float](n)
    var i=0
    while i<n do { heat(i)=AnnotatorOps.sigmoid(tp(i));i+=1 }
    val r=(pool-1)/2;val peaks=Vector.newBuilder[(Float,Int)]
    for y <- 0 until h; x <- 0 until w do
      val v=heat(y*w+x);var m=v;var yy=math.max(0,y-r)
      while yy<=math.min(h-1,y+r) do
        var xx=math.max(0,x-r)
        while xx<=math.min(w-1,x+r) do { if heat(yy*w+xx)>m then m=heat(yy*w+xx);xx+=1 }
        yy+=1
      if m==v then peaks+=((v,y*w+x))
    val best=peaks.result().sortBy(p=>(-p._1,p._2)).take(topK)
    best.flatMap { (score,idx) =>
      val y=idx/w;val x=idx%w
      val sx=tp(n+idx);val sy=tp(2*n+idx);val ex=tp(3*n+idx);val ey=tp(4*n+idx)
      val dist=math.sqrt(((sx-ex)*(sx-ex)+(sy-ey)*(sy-ey)).toDouble)
      if score>thrV && dist>thrD then Some((score,(2f*(x+sx),2f*(y+sy),2f*(x+ex),2f*(y+ey)))) else None
    }

  /** 1-pixel 8-connected Bresenham line (integer endpoints, as `cv2.line` with thickness 1) clipped to the image. */
  def drawLine(pix: Array[Boolean],w: Int,h: Int,x0: Int,y0: Int,x1: Int,y1: Int): Unit=
    var x=x0;var y=y0;val dx=math.abs(x1-x0);val dy=math.abs(y1-y0)
    val sx=if x0<x1 then 1 else -1;val sy=if y0<y1 then 1 else -1
    var err=dx-dy;var done=false
    while !done do
      if x>=0 && x<w && y>=0 && y<h then pix(y*w+x)=true
      if x==x1 && y==y1 then done=true
      else
        val e2=2*err
        if e2> -dy then { err-=dy;x+=sx }
        if e2<dx then { err+=dx;y+=sy }

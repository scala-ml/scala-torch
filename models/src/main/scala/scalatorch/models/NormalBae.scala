package scalatorch.models

import java.nio.file.{Files,Path}
import scalatorch.io.{LegacyPyTorchStateDict,PyTorchArchiveLimits,PyTorchTensorArchive}

/** The NormalBAE surface-normal estimator behind ControlNet's `normal_bae` preprocessor (Bae, Budvytis, Cipolla, "Estimating and
  * Exploiting the Aleatoric Uncertainty in Surface Normal Estimation", ICCV 2021; checkpoint `scannet.pt` of lllyasviel/Annotators),
  * on the CPU in plain Float32 arrays (CHW).
  *
  * Architecture, derived from the checkpoint keys/shapes and the published description:
  *  - Encoder: EfficientNet-B5 ("tf_efficientnet_b5_ap" layout). `conv_stem` (3x3, stride 2) + BN + SiLU, then seven block stages
  *    (3 depthwise-separable blocks 24ch; 5 x MBConv6 k3 s2 40ch; 5 x k5 s2 64ch; 7 x k3 s2 128ch; 7 x k5 176ch; 9 x k5 s2 304ch;
  *    3 x k3 512ch) with squeeze-excite (in_chs/4 channels, SiLU), a 1x1 `conv_head` to 2048 channels. All convolutions use TensorFlow
  *    'same' padding (dynamic, asymmetric for stride 2), BatchNorm eps 1e-3. The decoder takes the outputs of blocks 0, 1, 2, 4 and the raw
  *    `conv_head` output (before bn2, which the checkpoint holds but the network never uses).
  *  - Decoder: 1x1 conv, then four UpSampleBN stages (bilinear align_corners=True to the skip size, concat, [3x3 conv + BN +
  *    LeakyReLU(0.01)] x 2) giving 1/16, 1/8, 1/4, 1/2 resolution maps; a 3x3 head on the 1/8 map predicts (nx, ny, nz, kappa)
  *    ("norm_normalize": xyz L2-normalised, kappa = elu(k) + 1.01); three per-pixel MLP refinements (1x1 convs, ReLU, 128 wide) at
  *    1/4, 1/2 and 1/1 resolution take the bilinearly (align_corners=True) 2x-upsampled previous decoder feature and previous
  *    prediction. In inference every pixel is refined (the uncertainty-guided point sampling is a training device).
  *
  * Written from the checkpoint structure and the documented behaviour; no code is copied. Licences: the NormalBAE code
  * (baegwangbin/surface_normal_uncertainty, and its ControlNet-annotator port) is MIT, the EfficientNet port (rwightman
  * gen-efficientnet-pytorch) and TF-ported weights it started from are Apache-2.0; the licence of the `scannet.pt` checkpoint as
  * redistributed in lllyasviel/Annotators is not stated. Loading is strict (unknown, missing, mis-shaped tensors are rejected);
  * BatchNorm is folded into the preceding convolution at load time. */
final class NormalBae private (stem: NormalBae.Conv,blocks: Vector[NormalBae.Block],head: NormalBae.Conv,dec: NormalBae.Decoder):
  import NormalBae.*

  /** Raw network output (4 x h x w: normal xyz, unit length, and kappa) for a normalised CHW input; h and w must be multiples of 32. */
  def infer(chw: Array[Float],h: Int,w: Int,cancelled: () => Boolean=()=>false): Array[Float]=
    require(h>=32 && w>=32 && h%32==0 && w%32==0,"NormalBAE input sides must be multiples of 32")
    require(chw.length==3*h*w,"NormalBAE input must be 3 x h x w")
    require(h.toLong*w<=4194304L,"NormalBAE input is too large")
    def check(): Unit=if cancelled() || Thread.currentThread().isInterrupted then throw new java.util.concurrent.CancellationException("normal_bae cancelled")
    var x=stemConv(chw,h,w,stem);var fh=(h+1)/2;var fw=(w+1)/2
    val taps=new Array[(Array[Float],Int,Int,Int)](7) // (data, channels, h, w) after each stage
    var si=0;var c=stem.co
    for b <- blocks do
      check()
      val inp=x;val (y,nh,nw)=b.run(x,c,fh,fw);x=y;fh=nh;fw=nw
      if b.residual then { var i=0;while i<x.length do { x(i)+=inp(i);i+=1 } }
      c=b.cout
      if b.last then { taps(si)=(x,c,fh,fw);si+=1 }
    check()
    val f11=pointwise(x,c,fh*fw,head.w,head.b,head.co,ActNone)
    val d0=pointwise(f11,head.co,fh*fw,dec.conv2.w,dec.conv2.b,dec.conv2.co,ActNone)
    check()
    val skip=Vector(taps(4),taps(2),taps(1),taps(0)) // blocks.4 (176 @1/16), blocks.2 (64 @1/8), blocks.1 (40 @1/4), blocks.0 (24 @1/2)
    var cur=(d0,dec.conv2.co,fh,fw);val maps=Vector.newBuilder[(Array[Float],Int,Int,Int)]
    for (u,s) <- dec.ups.zip(skip) do
      check()
      val (sd,sc,sh,sw)=s
      val upx=DepthAnythingV2.bilinear(cur._1,cur._2,cur._3,cur._4,sh,sw,true)
      val a=conv3x3(Vector((upx,cur._2),(sd,sc)),sh,sw,u.c1,ActLeaky)
      val b=conv3x3(Vector((a,u.c1.co)),sh,sw,u.c2,ActLeaky)
      cur=(b,u.c2.co,sh,sw);maps+=cur
    val Vector(d1,d2,d3,d4)=maps.result()
    val r8=conv3x3(Vector((d2._1,d2._2)),d2._3,d2._4,dec.res8,ActNone)
    normNormalize(r8,d2._3*d2._4)
    check()
    val r4=refine(d2,r8,4,dec.res4);check()
    val r2=refine(d3,r4,4,dec.res2);check()
    refine(d4,r2,4,dec.res1)

  /** One refinement level: both inputs are upsampled 2x (bilinear, align_corners=True), concatenated and passed per pixel through the
    * ReLU MLP; the result is norm_normalized. Processed in row chunks so the full-resolution intermediates are never materialised. */
  private def refine(feat: (Array[Float],Int,Int,Int),prev: Array[Float],pc: Int,mlp: Vector[Conv]): Array[Float]=
    val (fd,fc,fh,fw)=feat;val oh=fh*2;val ow=fw*2;val cin=fc+pc
    val out=new Array[Float](4*oh*ow)
    def axis(in: Int,outN: Int)=
      val i0=new Array[Int](outN);val i1=new Array[Int](outN);val l=new Array[Float](outN)
      for d <- 0 until outN do
        val src=if outN>1 then d.toDouble*(in-1)/(outN-1) else 0.0
        val f=math.min(src.toInt,in-1);i0(d)=f;i1(d)=math.min(f+1,in-1);l(d)=(src-f).toFloat
      (i0,i1,l)
    val (y0,y1,yl)=axis(fh,oh);val (x0,x1,xl)=axis(fw,ow)
    val chunk=math.min(ow,256);val perRow=(ow+chunk-1)/chunk
    DepthAnythingV2.parallel(oh*perRow) { id =>
      val oy=id/perRow;val px0=(id%perRow)*chunk;val t=math.min(chunk,ow-px0)
      val in=new Array[Float](cin*t)
      var c=0
      while c<cin do
        val (src,sw0,base)=if c<fc then (fd,fw,c*fh*fw) else (prev,fw,(c-fc)*fh*fw)
        val r0=base+y0(oy)*sw0;val r1=base+y1(oy)*sw0;val ly=yl(oy);var p=0
        while p<t do
          val ox=px0+p;val lx=xl(ox);val a=x0(ox);val b=x1(ox)
          val top=src(r0+a)*(1-lx)+src(r0+b)*lx;val bot=src(r1+a)*(1-lx)+src(r1+b)*lx
          in(c*t+p)=top*(1-ly)+bot*ly;p+=1
        c+=1
      var cur=in;var ci=cin
      for (m,i) <- mlp.zipWithIndex do
        val nxt=new Array[Float](m.co*t);gemmTile(m.w,m.b,ci,cur,0,t,nxt,0,t,t,0,m.co,if i<mlp.size-1 then ActRelu else ActNone)
        cur=nxt;ci=m.co
      // norm_normalize
      var p=0
      while p<t do
        val nx=cur(p);val ny=cur(t+p);val nz=cur(2*t+p);val k=cur(3*t+p)
        val n=math.sqrt(nx.toDouble*nx+ny.toDouble*ny+nz.toDouble*nz).toFloat+1e-10f
        val o=oy*ow+px0+p
        out(o)=nx/n;out(oh*ow+o)=ny/n;out(2*oh*ow+o)=nz/n;out(3*oh*ow+o)=eluPlus(k);p+=1
    }
    out

object NormalBae:
  private[models] val ActNone=0;private[models] val ActSilu=1;private[models] val ActRelu=2;private[models] val ActLeaky=3

  /** A convolution with BatchNorm folded in; `w` is (co, ci/groups, k, k) (or (co, ci) for 1x1). */
  final class Conv(val w: Array[Float],val b: Array[Float],val ci: Int,val co: Int,val k: Int)
  final class Se(val rw: Array[Float],val rb: Array[Float],val ew: Array[Float],val eb: Array[Float],val c: Int,val r: Int)
  /** `expand` is the 1x1 expansion (None for the depthwise-separable first stage), `dw` the depthwise conv, `project` the 1x1 projection. */
  final class Block(val expand: Option[Conv],val dw: Conv,val se: Se,val project: Conv,val stride: Int,val residual: Boolean,val cout: Int,val last: Boolean):
    def run(x: Array[Float],cin: Int,h: Int,w: Int): (Array[Float],Int,Int) =
      val e=expand match
        case Some(p) => pointwise(x,cin,h*w,p.w,p.b,p.co,ActSilu)
        case None => x
      val (d,oh,ow)=depthwise(e,dw.co,h,w,dw,stride)
      squeezeExcite(d,se,oh*ow)
      (pointwise(d,dw.co,oh*ow,project.w,project.b,project.co,ActNone),oh,ow)
  final class Up(val c1: Conv,val c2: Conv)
  final class Decoder(val conv2: Conv,val ups: Vector[Up],val res8: Conv,val res4: Vector[Conv],val res2: Vector[Conv],val res1: Vector[Conv])

  /** (repeats, kernel, stride, out channels, depthwise-separable) of the seven stages. */
  private val Stages=Vector((3,3,1,24,true),(5,3,2,40,false),(5,5,2,64,false),(7,3,2,128,false),(7,5,1,176,false),(9,5,2,304,false),(3,3,1,512,false))
  private val P="encoder.original_model."
  private val EncoderEps=1e-3f;private val DecoderEps=1e-5f
  val FileName="scannet.pt"

  /** Reads `scannet.pt` (ZIP or legacy pickle; `{"model": state_dict}` or a bare state dict; optional `module.` prefixes) into name -> (shape, values). */
  private def readCheckpoint(file: Path,cancelled: () => Boolean): Map[String,(Vector[Int],Array[Float])]=
    val limits=PyTorchArchiveLimits.ImageModelWeights.copy(maxArchiveBytes=1024L*1024*1024,maxEntryBytes=256L*1024*1024,
      maxTotalUncompressedBytes=1024L*1024*1024,maxElements=32L*1024*1024)
    val top: Map[String,PyTorchTensorArchive.Value]=
      if LegacyPyTorchStateDict.isLegacy(file) then
        LegacyPyTorchStateDict.read(file,limits,cancelled).view.mapValues(PyTorchTensorArchive.TensorValue(_)).toMap
      else PyTorchTensorArchive.read(file,limits,cancelled).value match
        case PyTorchTensorArchive.DictValue(e) => e
        case other => throw IllegalArgumentException(s"NormalBAE weights must be a state dict, got ${other.getClass.getSimpleName}")
    val entries=top.get("model") match
      case Some(PyTorchTensorArchive.DictValue(inner)) if top.size==1 => inner
      case _ => top
    entries.map { (k,v) =>
      val name=if k.startsWith("module.") then k.drop(7) else k
      v match
        case PyTorchTensorArchive.TensorValue(t) => name->((t.shape.map(_.toInt),t.values))
        case PyTorchTensorArchive.IntegerTensorValue(t) if name.endsWith(".num_batches_tracked") => name->((t.shape.map(_.toInt),Array.empty[Float]))
        case _ => throw IllegalArgumentException(s"NormalBAE weight is not a tensor: $k")
    }

  private def expectedShapes: Map[String,Vector[Int]]=
    val m=scala.collection.mutable.LinkedHashMap.empty[String,Vector[Int]]
    def bn(n: String,c: Int): Unit=
      for s <- Seq("weight","bias","running_mean","running_var") do m(s"$n.$s")=Vector(c)
      m(s"$n.num_batches_tracked")=Vector()
    m(P+"conv_stem.weight")=Vector(48,3,3,3);bn(P+"bn1",48)
    var cin=48
    for ((n,k,_,cout,ds),si) <- Stages.zipWithIndex; bi <- 0 until n do
      val q=s"${P}blocks.$si.$bi";val mid=if ds then cin else cin*6
      if ds then
        m(s"$q.conv_dw.weight")=Vector(mid,1,k,k);bn(s"$q.bn1",mid)
      else
        m(s"$q.conv_pw.weight")=Vector(mid,cin,1,1);bn(s"$q.bn1",mid)
        m(s"$q.conv_dw.weight")=Vector(mid,1,k,k);bn(s"$q.bn2",mid)
      m(s"$q.se.conv_reduce.weight")=Vector(cin/4,mid,1,1);m(s"$q.se.conv_reduce.bias")=Vector(cin/4)
      m(s"$q.se.conv_expand.weight")=Vector(mid,cin/4,1,1);m(s"$q.se.conv_expand.bias")=Vector(mid)
      if ds then { m(s"$q.conv_pw.weight")=Vector(cout,mid,1,1);bn(s"$q.bn2",cout) }
      else { m(s"$q.conv_pwl.weight")=Vector(cout,mid,1,1);bn(s"$q.bn3",cout) }
      cin=cout
    m(P+"conv_head.weight")=Vector(2048,512,1,1);bn(P+"bn2",2048) // bn2 is stored but the decoder takes the raw conv_head output
    m("decoder.conv2.weight")=Vector(2048,2048,1,1);m("decoder.conv2.bias")=Vector(2048)
    for (u,ci,co) <- Seq(("up1",2224,1024),("up2",1088,512),("up3",552,256),("up4",280,128)) do
      m(s"decoder.$u._net.0.weight")=Vector(co,ci,3,3);m(s"decoder.$u._net.0.bias")=Vector(co);bn(s"decoder.$u._net.1",co)
      m(s"decoder.$u._net.3.weight")=Vector(co,co,3,3);m(s"decoder.$u._net.3.bias")=Vector(co);bn(s"decoder.$u._net.4",co)
    m("decoder.out_conv_res8.weight")=Vector(4,512,3,3);m("decoder.out_conv_res8.bias")=Vector(4)
    for (r,ci) <- Seq(("out_conv_res4",516),("out_conv_res2",260),("out_conv_res1",132)) do
      for (i,a,b) <- Seq((0,128,ci),(2,128,128),(4,128,128),(6,4,128)) do { m(s"decoder.$r.$i.weight")=Vector(a,b,1);m(s"decoder.$r.$i.bias")=Vector(a) }
    m.toMap

  def load(file: Path,cancelled: () => Boolean=()=>false): NormalBae=
    require(Files.isRegularFile(file),s"NormalBAE weights not found: $file")
    val raw=readCheckpoint(file,cancelled)
    val expected=expectedShapes
    val unknown=raw.keySet.diff(expected.keySet)
    require(unknown.isEmpty,s"unexpected NormalBAE weights: ${unknown.take(5).mkString(",")}")
    for (name,dims) <- expected do
      val t=raw.getOrElse(name,throw IllegalArgumentException(s"missing NormalBAE weight: $name"))
      require(t._1==dims,s"NormalBAE weight shape mismatch: $name, expected ${dims.mkString("x")}, got ${t._1.mkString("x")}")
    def t(n: String)=raw(n)._2
    def fold(conv: String,bn: String,eps: Float,hasBias: Boolean=false): Conv=
      val shape=raw(conv+".weight")._1;val co=shape(0)
      val w=t(conv+".weight").clone();val per=w.length/co
      val g=t(bn+".weight");val be=t(bn+".bias");val mu=t(bn+".running_mean");val va=t(bn+".running_var")
      val b=new Array[Float](co);val cb=if hasBias then t(conv+".bias") else null
      for o <- 0 until co do
        val s=(g(o)/math.sqrt(va(o).toDouble+eps)).toFloat
        var i=o*per;val end=i+per
        while i<end do { w(i)*=s;i+=1 }
        b(o)=be(o)-mu(o)*s+(if hasBias then cb(o)*s else 0f)
      new Conv(w,b,shape(1),co,shape(2))
    def plain(n: String): Conv=
      val s=raw(n+".weight")._1
      new Conv(t(n+".weight"),t(n+".bias"),s(1),s(0),if s.size==4 then s(2) else 1)
    val stem=fold(P+"conv_stem",P+"bn1",EncoderEps)
    val blocks=Vector.newBuilder[Block];var cin=48
    for ((n,_,s0,cout,ds),si) <- Stages.zipWithIndex; bi <- 0 until n do
      val q=s"${P}blocks.$si.$bi";val stride=if bi==0 then s0 else 1
      val mid=if ds then cin else cin*6
      val expand=if ds then None else Some(fold(s"$q.conv_pw",s"$q.bn1",EncoderEps))
      val dw=fold(s"$q.conv_dw",if ds then s"$q.bn1" else s"$q.bn2",EncoderEps)
      val se=new Se(t(s"$q.se.conv_reduce.weight"),t(s"$q.se.conv_reduce.bias"),t(s"$q.se.conv_expand.weight"),t(s"$q.se.conv_expand.bias"),mid,cin/4)
      val project=if ds then fold(s"$q.conv_pw",s"$q.bn2",EncoderEps) else fold(s"$q.conv_pwl",s"$q.bn3",EncoderEps)
      blocks+=new Block(expand,dw,se,project,stride,stride==1 && cin==cout,cout,bi==n-1)
      cin=cout
    val ups=Vector("up1","up2","up3","up4").map(u=>new Up(fold(s"decoder.$u._net.0",s"decoder.$u._net.1",DecoderEps,hasBias=true),
      fold(s"decoder.$u._net.3",s"decoder.$u._net.4",DecoderEps,hasBias=true)))
    def mlp(n: String)=Vector(0,2,4,6).map(i=>plain(s"decoder.$n.$i"))
    val dec=new Decoder(plain("decoder.conv2"),ups,plain("decoder.out_conv_res8"),mlp("out_conv_res4"),mlp("out_conv_res2"),mlp("out_conv_res1"))
    val head=new Conv(t(P+"conv_head.weight"),new Array[Float](2048),512,2048,1) // conv_head has no bias
    new NormalBae(stem,blocks.result(),head,dec)

  /** ControlNet's output: ((n + 1) * 0.5) clipped to 0..1, times 255, truncated to 8 bits; xyz map to RGB in that order (no flip). */
  def toRgb8(out: Array[Float],h: Int,w: Int): Array[Int]=
    val n=h*w;val px=new Array[Int](n)
    def q(v: Float)=math.max(0,math.min(255,(math.max(0f,math.min(1f,(v+1f)*0.5f))*255f).toInt))
    var i=0
    while i<n do { px(i)=0xff000000.toInt|(q(out(i))<<16)|(q(out(n+i))<<8)|q(out(2*n+i));i+=1 }
    px

  // ------------------------------------------------------------------ kernels

  private[models] def eluPlus(k: Float): Float=if k>0f then k+1.01f else math.exp(k.toDouble).toFloat+0.01f

  private def normNormalize(x: Array[Float],hw: Int): Unit=
    var i=0
    while i<hw do
      val nx=x(i);val ny=x(hw+i);val nz=x(2*hw+i)
      val n=math.sqrt(nx.toDouble*nx+ny.toDouble*ny+nz.toDouble*nz).toFloat+1e-10f
      x(i)=nx/n;x(hw+i)=ny/n;x(2*hw+i)=nz/n;x(3*hw+i)=eluPlus(x(3*hw+i));i+=1

  private def silu(v: Float): Float=(v/(1.0+math.exp(-v.toDouble))).toFloat

  private def activate(a: Array[Float],from: Int,to: Int,act: Int): Unit=
    var i=from
    act match
      case ActSilu => while i<to do { a(i)=silu(a(i));i+=1 }
      case ActRelu => while i<to do { if a(i)<0f then a(i)=0f;i+=1 }
      case ActLeaky => while i<to do { if a(i)<0f then a(i)*=0.01f;i+=1 }
      case _ =>

  /** `out[o][p] = act(b[o] + sum_c w[o][c] * in[c][p])` for o in [o0,o1) and p in [0,t); rows of `in`/`out` are `inStride`/`outStride`
    * apart. Four output channels share each input row load. */
  private[models] def gemmTile(w: Array[Float],b: Array[Float],ci: Int,in: Array[Float],inOff: Int,inStride: Int,out: Array[Float],outOff: Int,outStride: Int,
      t: Int,o0: Int,o1: Int,act: Int): Unit=
    val a0=new Array[Float](t);val a1=new Array[Float](t);val a2=new Array[Float](t);val a3=new Array[Float](t)
    var o=o0
    while o+3<o1 do
      java.util.Arrays.fill(a0,b(o));java.util.Arrays.fill(a1,b(o+1));java.util.Arrays.fill(a2,b(o+2));java.util.Arrays.fill(a3,b(o+3))
      var c=0
      while c<ci do
        val w0=w(o*ci+c);val w1=w((o+1)*ci+c);val w2=w((o+2)*ci+c);val w3=w((o+3)*ci+c);val xo=inOff+c*inStride;var p=0
        while p<t do
          val xv=in(xo+p);a0(p)+=w0*xv;a1(p)+=w1*xv;a2(p)+=w2*xv;a3(p)+=w3*xv;p+=1
        c+=1
      var j=0
      while j<4 do
        val a=if j==0 then a0 else if j==1 then a1 else if j==2 then a2 else a3
        activate(a,0,t,act);System.arraycopy(a,0,out,outOff+(o+j)*outStride,t);j+=1
      o+=4
    while o<o1 do
      java.util.Arrays.fill(a0,b(o));var c=0
      while c<ci do
        val w0=w(o*ci+c);val xo=inOff+c*inStride;var p=0
        while p<t do { a0(p)+=w0*in(xo+p);p+=1 }
        c+=1
      activate(a0,0,t,act);System.arraycopy(a0,0,out,outOff+o*outStride,t);o+=1

  /** 1x1 convolution over a CHW map flattened to (ci, hw), tiled over pixels and output-channel groups. */
  private[models] def pointwise(x: Array[Float],ci: Int,hw: Int,w: Array[Float],b: Array[Float],co: Int,act: Int): Array[Float]=
    val out=new Array[Float](co*hw)
    val tile=math.min(hw,256);val tiles=(hw+tile-1)/tile
    var g=32
    while g>4 && tiles*((co+g-1)/g)<64 do g/=2
    val groups=(co+g-1)/g
    DepthAnythingV2.parallel(tiles*groups) { id =>
      val p0=(id%tiles)*tile;val grp=id/tiles
      gemmTile(w,b,ci,x,p0,hw,out,p0,hw,math.min(tile,hw-p0),grp*g,math.min(co,grp*g+g),act)
    }
    out

  /** 3x3 stride-1 convolution with padding 1 over the channel-concatenated inputs, then `act`. Works on zero-padded planes of width
    * w+2 so each tap is one shifted contiguous slice; the two junk columns per row are dropped at the end. */
  private def conv3x3(inputs: Vector[(Array[Float],Int)],h: Int,w: Int,cv: Conv,act: Int): Array[Float]=
    val ci=inputs.map(_._2).sum;require(ci==cv.ci,s"conv input channels $ci != ${cv.ci}")
    val wp=w+2;val ps=(h+2)*wp+2;val co=cv.co;val q=h*wp
    val xp=new Array[Float](ci*ps)
    var base=0
    for (src,c) <- inputs do
      val b0=base
      DepthAnythingV2.parallel(c) { ch =>
        var y=0
        while y<h do { System.arraycopy(src,(ch*h+y)*w,xp,(b0+ch)*ps+(y+1)*wp+1,w);y+=1 }
      }
      base+=c
    val flat=new Array[Float](co*q)
    val tile=math.min(q,512);val tiles=(q+tile-1)/tile
    var g=32
    while g>4 && tiles*((co+g-1)/g)<64 do g/=2
    val groups=(co+g-1)/g
    DepthAnythingV2.parallel(tiles*groups) { id =>
      val q0=(id%tiles)*tile;val t=math.min(tile,q-q0);val grp=id/tiles
      val o0=grp*g;val o1=math.min(co,o0+g)
      val a0=new Array[Float](t);val a1=new Array[Float](t);val a2=new Array[Float](t);val a3=new Array[Float](t)
      var o=o0
      while o<o1 do
        val n=math.min(4,o1-o)
        val o1c=math.min(o+1,co-1);val o2c=math.min(o+2,co-1);val o3c=math.min(o+3,co-1)
        java.util.Arrays.fill(a0,cv.b(o));java.util.Arrays.fill(a1,cv.b(o1c));java.util.Arrays.fill(a2,cv.b(o2c));java.util.Arrays.fill(a3,cv.b(o3c))
        var c=0
        while c<ci do
          val xb=c*ps+q0;var k=0
          while k<9 do
            val xo=xb+(k/3)*wp+k%3
            val wa=cv.w((o*ci+c)*9+k);val wb=cv.w((o1c*ci+c)*9+k);val wc=cv.w((o2c*ci+c)*9+k);val wd=cv.w((o3c*ci+c)*9+k)
            var p=0
            while p<t do { val xv=xp(xo+p);a0(p)+=wa*xv;a1(p)+=wb*xv;a2(p)+=wc*xv;a3(p)+=wd*xv;p+=1 }
            k+=1
          c+=1
        var j=0
        while j<n do
          val a=if j==0 then a0 else if j==1 then a1 else if j==2 then a2 else a3
          activate(a,0,t,act);System.arraycopy(a,0,flat,(o+j)*q+q0,t);j+=1
        o+=n
    }
    val out=new Array[Float](co*h*w)
    DepthAnythingV2.parallel(co) { o =>
      var y=0
      while y<h do { System.arraycopy(flat,o*q+y*wp,out,(o*h+y)*w,w);y+=1 }
    }
    out

  /** The stem: 3x3 stride-2 convolution with TF 'same' padding, folded BN, SiLU. */
  private def stemConv(x: Array[Float],h: Int,w: Int,cv: Conv): Array[Float]=
    val (padT,padB)=samePad(h,3,2);val (padL,padR)=samePad(w,3,2);val hp=h+padT+padB;val wpad=w+padL+padR
    val xp=new Array[Float](3*hp*wpad)
    for c <- 0 until 3; y <- 0 until h do System.arraycopy(x,(c*h+y)*w,xp,(c*hp+y+padT)*wpad+padL,w)
    val out=DepthAnythingV2.conv2d(xp,3,hp,wpad,cv.w,Some(cv.b),cv.co,3,2,0)
    activate(out,0,out.length,ActSilu);out

  /** TF 'same' padding (before, after) for one axis. */
  private[models] def samePad(size: Int,k: Int,stride: Int): (Int,Int)=
    val out=(size+stride-1)/stride;val total=math.max((out-1)*stride+k-size,0);(total/2,total-total/2)

  /** Depthwise convolution with TF 'same' padding, folded BN and SiLU. */
  private[models] def depthwise(x: Array[Float],c: Int,h: Int,w: Int,cv: Conv,stride: Int): (Array[Float],Int,Int)=
    val k=cv.k;val oh=(h+stride-1)/stride;val ow=(w+stride-1)/stride
    val (padT,_)=samePad(h,k,stride);val (padL,_)=samePad(w,k,stride)
    val out=new Array[Float](c*oh*ow)
    DepthAnythingV2.parallel(c) { ch =>
      var oy=0
      while oy<oh do
        val orow=(ch*oh+oy)*ow;java.util.Arrays.fill(out,orow,orow+ow,cv.b(ch))
        var ky=0
        while ky<k do
          val iy=oy*stride-padT+ky
          if iy>=0 && iy<h then
            val xr=(ch*h+iy)*w;var kx=0
            while kx<k do
              val wv=cv.w((ch*k+ky)*k+kx)
              val lo=math.max(0,Math.floorDiv(padL-kx+stride-1,stride));val hi=math.min(ow-1,Math.floorDiv(w-1+padL-kx,stride))
              var ox=lo
              while ox<=hi do { out(orow+ox)+=wv*x(xr+ox*stride-padL+kx);ox+=1 }
              kx+=1
          ky+=1
        activate(out,orow,orow+ow,ActSilu)
        oy+=1
    }
    (out,oh,ow)

  /** Squeeze-excite in place: channel means -> 1x1 reduce (SiLU) -> 1x1 expand -> sigmoid gate. */
  private def squeezeExcite(x: Array[Float],se: Se,hw: Int): Unit=
    val mean=new Array[Float](se.c)
    DepthAnythingV2.parallel(se.c) { ch =>
      var s=0.0;var i=ch*hw;val end=i+hw
      while i<end do { s+=x(i);i+=1 }
      mean(ch)=(s/hw).toFloat
    }
    val r=new Array[Float](se.r)
    for j <- 0 until se.r do
      var s=se.rb(j).toDouble;var c=0
      while c<se.c do { s+=se.rw(j*se.c+c).toDouble*mean(c);c+=1 }
      r(j)=silu(s.toFloat)
    val gate=new Array[Float](se.c)
    for c <- 0 until se.c do
      var s=se.eb(c).toDouble;var j=0
      while j<se.r do { s+=se.ew(c*se.r+j).toDouble*r(j);j+=1 }
      gate(c)=(1.0/(1.0+math.exp(-s))).toFloat
    DepthAnythingV2.parallel(se.c) { ch =>
      val g=gate(ch);var i=ch*hw;val end=i+hw
      while i<end do { x(i)*=g;i+=1 }
    }

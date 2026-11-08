package scalatorch.models

import scalatorch.io.SafeTensorSet
import java.nio.file.{Files,Path}
import scala.collection.mutable

/** A CHW float feature map. */
final case class DepthFeature(channels: Int,height: Int,width: Int,data: Array[Float]):
  require(data.length.toLong==channels.toLong*height*width,"feature size mismatch")

/** Intermediate tensors for parity checks: embeddings (tokens*384), 13 hidden states (tokens*384), neck outputs, depth (h*w). */
final case class DepthAnythingOutputs(embeddings: Array[Float],hiddenStates: Vector[Array[Float]],
    neck: Vector[DepthFeature],depth: Array[Float])

/** Depth Anything V2 Small (DINOv2 ViT-S/14 + DPT neck/head), plain CPU Float32 inference from the HF safetensors. */
final class DepthAnythingV2 private (weights: Map[String,Array[Float]]):
  import DepthAnythingV2.*
  private def w(name: String): Array[Float]=weights(name)

  private def linear(x: Array[Float],n: Int,in: Int,prefix: String,out: Int): Array[Float]=
    val wt=w(s"$prefix.weight");val b=w(s"$prefix.bias");val y=new Array[Float](n*out)
    parallel(n) { i =>
      val xo=i*in
      var o=0
      while o<out do
        var acc=0f;val wo=o*in;var k=0
        while k<in do { acc+=x(xo+k)*wt(wo+k);k+=1 }
        y(i*out+o)=acc+b(o);o+=1
    }
    y

  private def layerNorm(x: Array[Float],n: Int,prefix: String): Array[Float]=
    val g=w(s"$prefix.weight");val b=w(s"$prefix.bias");val y=new Array[Float](x.length)
    parallel(n) { i =>
      val o=i*Hidden;var mean=0.0;var k=0
      while k<Hidden do { mean+=x(o+k);k+=1 }
      mean/=Hidden;var v=0.0;k=0
      while k<Hidden do { val d=x(o+k)-mean;v+=d*d;k+=1 }
      val inv=1.0/math.sqrt(v/Hidden+1e-6);k=0
      while k<Hidden do { y(o+k)=(((x(o+k)-mean)*inv)*g(k)+b(k)).toFloat;k+=1 }
    }
    y

  private def attention(x: Array[Float],n: Int,prefix: String): Array[Float]=
    val q=linear(x,n,Hidden,s"$prefix.attention.query",Hidden)
    val k=linear(x,n,Hidden,s"$prefix.attention.key",Hidden)
    val v=linear(x,n,Hidden,s"$prefix.attention.value",Hidden)
    val ctx=new Array[Float](n*Hidden);val scale=(1.0/math.sqrt(HeadDim)).toFloat
    parallel(Heads*n) { job =>
      val h=job/n;val i=job%n;val ho=h*HeadDim;val s=new Array[Float](n)
      var max=Float.NegativeInfinity;var j=0
      while j<n do
        var acc=0f;var d=0
        while d<HeadDim do { acc+=q(i*Hidden+ho+d)*k(j*Hidden+ho+d);d+=1 }
        s(j)=acc*scale;if s(j)>max then max=s(j)
        j+=1
      var sum=0.0;j=0
      while j<n do { val e=math.exp(s(j)-max);s(j)=e.toFloat;sum+=e;j+=1 }
      val inv=(1.0/sum).toFloat;var d=0
      while d<HeadDim do
        var acc=0f;j=0
        while j<n do { acc+=s(j)*v(j*Hidden+ho+d);j+=1 }
        ctx(i*Hidden+ho+d)=acc*inv;d+=1
    }
    linear(ctx,n,Hidden,s"$prefix.output.dense",Hidden)

  private def block(x: Array[Float],n: Int,i: Int): Array[Float]=
    val p=s"backbone.encoder.layer.$i"
    val a=attention(layerNorm(x,n,s"$p.norm1"),n,s"$p.attention")
    val l1=w(s"$p.layer_scale1.lambda1");val l2=w(s"$p.layer_scale2.lambda1")
    val r=new Array[Float](x.length);var t=0
    while t<r.length do { r(t)=x(t)+a(t)*l1(t%Hidden);t+=1 }
    val h=linear(layerNorm(r,n,s"$p.norm2"),n,Hidden,s"$p.mlp.fc1",Hidden*4)
    t=0;while t<h.length do { h(t)=gelu(h(t));t+=1 }
    val m=linear(h,n,Hidden*4,s"$p.mlp.fc2",Hidden)
    t=0;while t<r.length do { r(t)+=m(t)*l2(t%Hidden);t+=1 }
    r

  private def embed(pixels: Array[Float],height: Int,width: Int): Array[Float]=
    val gh=height/Patch;val gw=width/Patch;val n=1+gh*gw
    val pw=w("backbone.embeddings.patch_embeddings.projection.weight")
    val pb=w("backbone.embeddings.patch_embeddings.projection.bias")
    val patches=conv2d(pixels,3,height,width,pw,Some(pb),Hidden,Patch,Patch,0)
    val pos=w("backbone.embeddings.position_embeddings");val cls=w("backbone.embeddings.cls_token")
    val base=java.util.Arrays.copyOfRange(pos,Hidden,pos.length)
    val patchPos=if gh==PosGrid && gw==PosGrid then base else bicubicHwc(base,PosGrid,PosGrid,Hidden,gh,gw)
    val out=new Array[Float](n*Hidden);var c=0
    while c<Hidden do { out(c)=cls(c)+pos(c);c+=1 }
    var t=0
    while t<gh*gw do
      c=0
      while c<Hidden do { out((t+1)*Hidden+c)=patches(c*gh*gw+t)+patchPos(t*Hidden+c);c+=1 }
      t+=1
    out

  private def conv(x: DepthFeature,prefix: String,co: Int,k: Int,stride: Int,pad: Int,bias: Boolean=true): DepthFeature=
    val (oh,ow)=((x.height+2*pad-k)/stride+1,(x.width+2*pad-k)/stride+1)
    DepthFeature(co,oh,ow,conv2d(x.data,x.channels,x.height,x.width,w(s"$prefix.weight"),
      if bias then Some(w(s"$prefix.bias")) else None,co,k,stride,pad))

  private def relu(x: DepthFeature): DepthFeature=DepthFeature(x.channels,x.height,x.width,x.data.map(v=>if v>0f then v else 0f))
  private def add(a: DepthFeature,b: DepthFeature): DepthFeature=
    require(a.data.length==b.data.length,"feature add shape mismatch")
    DepthFeature(a.channels,a.height,a.width,Array.tabulate(a.data.length)(i=>a.data(i)+b.data(i)))
  private def resize(x: DepthFeature,h: Int,wd: Int,align: Boolean): DepthFeature=
    if x.height==h && x.width==wd then x else DepthFeature(x.channels,h,wd,bilinear(x.data,x.channels,x.height,x.width,h,wd,align))

  private def residual(x: DepthFeature,prefix: String): DepthFeature=
    add(conv(relu(conv(relu(x),s"$prefix.convolution1",Fusion,3,1,1)),s"$prefix.convolution2",Fusion,3,1,1),x)

  private def fusion(i: Int,hidden: DepthFeature,res: Option[DepthFeature],size: Option[(Int,Int)]): DepthFeature=
    val p=s"neck.fusion_stage.layers.$i"
    val merged=res match
      case Some(r) => add(hidden,residual(resize(r,hidden.height,hidden.width,false),s"$p.residual_layer1"))
      case None => hidden
    val m=residual(merged,s"$p.residual_layer2")
    val (th,tw)=size.getOrElse((m.height*2,m.width*2))
    conv(resize(m,th,tw,true),s"$p.projection",Fusion,1,1,0)

  private def deconv(x: DepthFeature,prefix: String,k: Int): DepthFeature=
    val c=x.channels;val wt=w(s"$prefix.weight");val b=w(s"$prefix.bias");val oh=x.height*k;val ow=x.width*k
    val y=new Array[Float](c*oh*ow)
    parallel(c) { co =>
      var h=0
      while h<x.height do
        var wd=0
        while wd<x.width do
          var i=0
          while i<k do
            var j=0
            while j<k do
              var acc=b(co);var ci=0
              while ci<c do { acc+=x.data(ci*x.height*x.width+h*x.width+wd)*wt(((ci*c+co)*k+i)*k+j);ci+=1 }
              y(co*oh*ow+(h*k+i)*ow+wd*k+j)=acc;j+=1
            i+=1
          wd+=1
        h+=1
    }
    DepthFeature(c,oh,ow,y)

  /** Runs the network on a normalised CHW image whose sides are multiples of 14; the result is the relative inverse depth (h*w). */
  def infer(pixels: Array[Float],height: Int,width: Int,cancelled: () => Boolean=()=>false): DepthAnythingOutputs=
    require(height>0 && width>0 && height%Patch==0 && width%Patch==0,s"Depth Anything input must be a multiple of $Patch, got ${width}x$height")
    require(pixels.length==3*height*width,"pixel buffer size mismatch")
    require(height.toLong*width<=4000000L,"Depth Anything input is too large")
    def check(): Unit=if cancelled() || Thread.currentThread().isInterrupted then throw new java.util.concurrent.CancellationException("depth cancelled")
    val gh=height/Patch;val gw=width/Patch;val n=1+gh*gw
    val emb=embed(pixels,height,width);var x=emb
    val hidden=Vector.newBuilder[Array[Float]];hidden+=emb
    for i <- 0 until Layers do { check();x=block(x,n,i);hidden+=x }
    val states=hidden.result()
    val taps=Vector(3,6,9,12).map { idx =>
      val ln=layerNorm(states(idx),n,"backbone.layernorm")
      val grid=new Array[Float](Hidden*gh*gw);var t=0
      while t<gh*gw do { var c=0;while c<Hidden do { grid(c*gh*gw+t)=ln((t+1)*Hidden+c);c+=1 };t+=1 }
      DepthFeature(Hidden,gh,gw,grid)
    }
    check()
    val reassembled=taps.zipWithIndex.map { (f,i) =>
      val p=s"neck.reassemble_stage.layers.$i"
      val proj=conv(f,s"$p.projection",NeckSizes(i),1,1,0)
      i match
        case 0 => deconv(proj,s"$p.resize",4)
        case 1 => deconv(proj,s"$p.resize",2)
        case 2 => proj
        case _ => conv(proj,s"$p.resize",NeckSizes(3),3,2,1)
    }
    val features=reassembled.zipWithIndex.map((f,i)=>conv(f,s"neck.convs.$i",Fusion,3,1,1,bias=false)).reverse
    var fused=fusion(0,features(0),None,Some((features(1).height,features(1).width)))
    val outs=Vector.newBuilder[DepthFeature];outs+=fused
    for i <- 1 until 4 do
      check()
      fused=fusion(i,fused,Some(features(i)),if i<3 then Some((features(i+1).height,features(i+1).width)) else None)
      outs+=fused
    val h1=conv(fused,"head.conv1",Fusion/2,3,1,1)
    val up=resize(h1,height,width,true)
    val h2=relu(conv(up,"head.conv2",HeadHidden,3,1,1))
    val d=conv(h2,"head.conv3",1,1,1,0)
    DepthAnythingOutputs(emb,states,outs.result(),d.data.map(v=>if v>0f then v else 0f))

object DepthAnythingV2:
  val Hidden=384;val Heads=6;val HeadDim=64;val Patch=14;val Layers=12;val PosGrid=37
  val Fusion=64;val HeadHidden=32;val NeckSizes=Vector(48,96,192,384)

  /** Strictly loads `model.safetensors` from a directory (or the file itself); unknown or mis-shaped weights are rejected. */
  def load(path: Path): DepthAnythingV2=
    val file=if Files.isDirectory(path) then path.resolve("model.safetensors") else path
    require(Files.isRegularFile(file),s"Depth Anything V2 weights not found: $file")
    val set=SafeTensorSet.openFile(file)
    try
      val expected=expectedShapes
      val unknown=set.tensors.keySet.diff(expected.keySet).diff(Set("backbone.embeddings.mask_token"))
      require(unknown.isEmpty,s"unexpected Depth Anything weights: ${unknown.take(5).mkString(",")}")
      val loaded=expected.map { (name,dims) =>
        val info=set.tensors.getOrElse(name,throw IllegalArgumentException(s"missing Depth Anything weight: $name"))
        require(info.shape.dims.map(_.toInt)==dims.toVector,s"Depth Anything weight shape mismatch: $name, expected ${dims.mkString("x")}, got ${info.shape}")
        name->set.readFloat(name).toFloatArray
      }
      new DepthAnythingV2(loaded)
    finally set.close()

  private def expectedShapes: Map[String,Seq[Int]]=
    val m=mutable.LinkedHashMap.empty[String,Seq[Int]]
    m("backbone.embeddings.cls_token")=Seq(1,1,Hidden)
    m("backbone.embeddings.patch_embeddings.projection.weight")=Seq(Hidden,3,Patch,Patch)
    m("backbone.embeddings.patch_embeddings.projection.bias")=Seq(Hidden)
    m("backbone.embeddings.position_embeddings")=Seq(1,1+PosGrid*PosGrid,Hidden)
    def lin(p: String,o: Int,i: Int): Unit={m(s"$p.weight")=Seq(o,i);m(s"$p.bias")=Seq(o)}
    def ln(p: String): Unit={m(s"$p.weight")=Seq(Hidden);m(s"$p.bias")=Seq(Hidden)}
    for i <- 0 until Layers do
      val p=s"backbone.encoder.layer.$i"
      for a <- Seq("query","key","value") do lin(s"$p.attention.attention.$a",Hidden,Hidden)
      lin(s"$p.attention.output.dense",Hidden,Hidden);lin(s"$p.mlp.fc1",Hidden*4,Hidden);lin(s"$p.mlp.fc2",Hidden,Hidden*4)
      ln(s"$p.norm1");ln(s"$p.norm2");m(s"$p.layer_scale1.lambda1")=Seq(Hidden);m(s"$p.layer_scale2.lambda1")=Seq(Hidden)
    ln("backbone.layernorm")
    def conv(p: String,o: Int,i: Int,k: Int,bias: Boolean=true): Unit=
      m(s"$p.weight")=Seq(o,i,k,k);if bias then m(s"$p.bias")=Seq(o)
    for i <- 0 until 4 do
      val c=NeckSizes(i);val p=s"neck.reassemble_stage.layers.$i"
      conv(s"$p.projection",c,Hidden,1)
      i match
        case 0 => conv(s"$p.resize",c,c,4)
        case 1 => conv(s"$p.resize",c,c,2)
        case 3 => conv(s"$p.resize",c,c,3)
        case _ => ()
      conv(s"neck.convs.$i",Fusion,c,3,bias=false)
      val f=s"neck.fusion_stage.layers.$i"
      conv(s"$f.projection",Fusion,Fusion,1)
      for r <- Seq("residual_layer1","residual_layer2"); c2 <- Seq("convolution1","convolution2") do conv(s"$f.$r.$c2",Fusion,Fusion,3)
    conv("head.conv1",Fusion/2,Fusion,3);conv("head.conv2",HeadHidden,Fusion/2,3);conv("head.conv3",1,HeadHidden,1)
    m.toMap

  private[models] def parallel(n: Int)(f: Int => Unit): Unit=
    java.util.stream.IntStream.range(0,n).parallel().forEach(i=>f(i))

  private def gelu(v: Float): Float=(0.5*v*(1.0+erf(v/math.sqrt(2.0)))).toFloat

  /** erf to ~1e-12 (Taylor series for small, continued fraction for large arguments). */
  private def erf(x: Double): Double=
    val a=math.abs(x)
    val r=if a<2.5 then
      var term=a;var sum=a;var n=0
      while math.abs(term)>1e-17*math.abs(sum) && n<200 do { n+=1;term= -term*a*a/n;sum+=term/(2*n+1) }
      2.0/math.sqrt(math.Pi)*sum
    else
      var f=0.0;var k=60
      while k>=1 do { f=k/2.0/(a+f);k-=1 }
      1.0-math.exp(-a*a)/math.sqrt(math.Pi)/(a+f)
    if x<0 then -r else r

  /** Direct convolution of a CHW image with an OIHW kernel. */
  def conv2d(x: Array[Float],ci: Int,h: Int,wd: Int,wt: Array[Float],bias: Option[Array[Float]],co: Int,k: Int,stride: Int,pad: Int): Array[Float]=
    val oh=(h+2*pad-k)/stride+1;val ow=(wd+2*pad-k)/stride+1;val out=new Array[Float](co*oh*ow)
    parallel(co) { o =>
      val base=o*oh*ow
      if bias.isDefined then java.util.Arrays.fill(out,base,base+oh*ow,bias.get(o))
      var c=0
      while c<ci do
        var ky=0
        while ky<k do
          var kx=0
          while kx<k do
            val wv=wt(((o*ci+c)*k+ky)*k+kx)
            val lo=math.max(0,Math.floorDiv(pad-kx+stride-1,stride))
            val hi=math.min(ow-1,Math.floorDiv(wd-1+pad-kx,stride))
            var oy=0
            while oy<oh do
              val iy=oy*stride-pad+ky
              if iy>=0 && iy<h then
                val xr=c*h*wd+iy*wd;val orow=base+oy*ow;var ox=lo
                while ox<=hi do { out(orow+ox)+=wv*x(xr+ox*stride-pad+kx);ox+=1 }
              oy+=1
            kx+=1
          ky+=1
        c+=1
    }
    out

  /** torch `interpolate(mode="bilinear")` on a CHW map. */
  def bilinear(x: Array[Float],c: Int,h: Int,w: Int,oh: Int,ow: Int,alignCorners: Boolean): Array[Float]=
    def axis(inSize: Int,outSize: Int): (Array[Int],Array[Int],Array[Float])=
      val i0=new Array[Int](outSize);val i1=new Array[Int](outSize);val l=new Array[Float](outSize)
      for d <- 0 until outSize do
        val src=if alignCorners then (if outSize>1 then d.toDouble*(inSize-1)/(outSize-1) else 0.0)
          else math.max((d+0.5)*inSize/outSize-0.5,0.0)
        val f=math.min(src.toInt,inSize-1)
        i0(d)=f;i1(d)=math.min(f+1,inSize-1);l(d)=(src-f).toFloat
      (i0,i1,l)
    val (y0,y1,yl)=axis(h,oh);val (x0,x1,xl)=axis(w,ow);val out=new Array[Float](c*oh*ow)
    parallel(c) { ch =>
      val b=ch*h*w
      for oy <- 0 until oh; ox <- 0 until ow do
        val top=x(b+y0(oy)*w+x0(ox))*(1-xl(ox))+x(b+y0(oy)*w+x1(ox))*xl(ox)
        val bot=x(b+y1(oy)*w+x0(ox))*(1-xl(ox))+x(b+y1(oy)*w+x1(ox))*xl(ox)
        out(ch*oh*ow+oy*ow+ox)=top*(1-yl(oy))+bot*yl(oy)
    }
    out

  /** torch bicubic (A=-0.75, align_corners=False) over an HWC grid (channels last). */
  private def bicubicHwc(x: Array[Float],h: Int,w: Int,c: Int,oh: Int,ow: Int): Array[Float]=
    def coeffs(t: Double): Array[Double]=
      val a= -0.75
      def near(d: Double)=((a+2)*d-(a+3))*d*d+1
      def far(d: Double)=((a*d-5*a)*d+8*a)*d-4*a
      Array(far(t+1),near(t),near(1-t),far(2-t))
    def axis(inSize: Int,outSize: Int): (Array[Array[Int]],Array[Array[Double]])=
      val idx=new Array[Array[Int]](outSize);val cf=new Array[Array[Double]](outSize)
      for d <- 0 until outSize do
        val src=(d+0.5)*inSize/outSize-0.5;val f=math.floor(src).toInt
        idx(d)=Array.tabulate(4)(t=>math.max(0,math.min(inSize-1,f-1+t)));cf(d)=coeffs(src-f)
      (idx,cf)
    val (yi,yc)=axis(h,oh);val (xi,xc)=axis(w,ow);val out=new Array[Float](oh*ow*c)
    parallel(oh) { oy =>
      for ox <- 0 until ow; ch <- 0 until c do
        var acc=0.0
        for ty <- 0 until 4; tx <- 0 until 4 do acc+=yc(oy)(ty)*xc(ox)(tx)*x((yi(oy)(ty)*w+xi(ox)(tx))*c+ch)
        out((oy*ow+ox)*c+ch)=acc.toFloat
    }
    out

package scalatorch.models

import java.nio.file.Path
import scalatorch.io.SafeTensorSet

/** waifu2x `upconv_7` 2x super-resolution / denoising network (nagadomi's waifu2x, models `art` and `photo`, noise 0..3), CPU Float32,
  * plain arrays (CHW, RGB in 0..1).
  *
  * Architecture, from the model files' own layer descriptions and the waifu2x documentation (nothing is copied from any repository): six
  * unpadded 3x3 convolutions 3-16-32-64-128-128-256, each followed by LeakyReLU(0.1), then a transposed convolution 256->3 (4x4, stride 2,
  * padding 3, no bias). Every layer shrinks or grows the map so that an input padded by 7 replicated border pixels per side yields exactly
  * twice the original size: (h+14-12) = h+2 feature rows, 2(h+2)-4 = 2h output rows. Large images are processed in square tiles with a 7 pixel
  * source overlap, which makes the tiled output identical (up to float summation order, in practice bit-identical) to the whole-image output.
  * The result is NOT clamped here; callers clamp to 0..1 before quantising.
  *
  * Weights: converted offline from nagadomi/waifu2x `models/upconv_7/{art,photo}/noiseN_scale2.0x_model.json` by
  * scripts/convert_waifu2x_json_to_safetensors.py (names `conv0..conv5.weight|bias`, `deconv.weight|bias`). Upstream code is MIT; the model
  * files carry no separate licence statement. Loading is strict: unknown, missing or mis-shaped tensors are rejected. */
final class Waifu2x private (convs: Array[Waifu2x.Conv],deconvW: Array[Float]):
  import Waifu2x.*

  /** Network forward pass for an image already padded by `Border` pixels on every side: `3 x ph x pw` in, `3 x 2(ph-2*Border) x 2(pw-2*Border)` out. */
  def forward(padded: Array[Float],ph: Int,pw: Int): Array[Float]=
    require(padded.length==3*ph*pw && ph>2*Border && pw>2*Border,"waifu2x input must be 3 x h x w with more than 14 rows and columns")
    var x=padded;var h=ph;var w=pw
    for cv <- convs do { x=conv3x3LeakyValid(x,cv.ci,h,w,cv.w,cv.b,cv.co);h-=2;w-=2 }
    deconv(x,256,h,w,deconvW)

  /** One 2x pass over a CHW image: replicate-pad by 7, then run `tile`x`tile` source tiles (0 = whole image at once). */
  def upscale2x(chw: Array[Float],h: Int,w: Int,tile: Int=256,cancelled: () => Boolean=()=>false,progress: (Int,Int)=>Unit=(_,_)=>()): Array[Float]=
    require(chw.length==3*h*w && h>0 && w>0,"waifu2x input must be 3 x h x w")
    val padded=replicatePad(chw,h,w,Border);val ph=h+2*Border;val pw=w+2*Border
    val oh=2*h;val ow=2*w
    if tile<=0 || (h<=tile && w<=tile) then
      progress(0,1);val o=forward(padded,ph,pw);progress(1,1);o
    else
      require(tile>=16,"waifu2x tile must be at least 16")
      val out=new Array[Float](3*oh*ow)
      val tilesY=(h+tile-1)/tile;val tilesX=(w+tile-1)/tile;var done=0
      var ty=0
      while ty<h do
        val ch=math.min(tile,h-ty);var tx=0
        while tx<w do
          if cancelled() || Thread.currentThread().isInterrupted then throw new java.util.concurrent.CancellationException("waifu2x cancelled")
          val cw=math.min(tile,w-tx);val th=ch+2*Border;val tw=cw+2*Border
          val sub=new Array[Float](3*th*tw)
          for c <- 0 until 3; y <- 0 until th do System.arraycopy(padded,(c*ph+ty+y)*pw+tx,sub,(c*th+y)*tw,tw)
          val r=forward(sub,th,tw)
          for c <- 0 until 3; y <- 0 until 2*ch do System.arraycopy(r,(c*2*ch+y)*2*cw,out,(c*oh+2*ty+y)*ow+2*tx,2*cw)
          done+=1;progress(done,tilesY*tilesX);tx+=tile
        ty+=tile
      out

object Waifu2x:
  /** Source pixels of context the network needs on every side. */
  val Border=7
  private val Channels=Vector(3,16,32,64,128,128,256)
  final class Conv(val w: Array[Float],val b: Array[Float],val ci: Int,val co: Int)

  def fileName(style: String,noise: Int): String=
    require(style=="art" || style=="photo","waifu2x style must be art or photo")
    require(noise>=0 && noise<=3,"waifu2x noise must be 0..3")
    s"waifu2x/$style-noise$noise.safetensors"

  def load(file: Path): Waifu2x=
    require(java.nio.file.Files.isRegularFile(file),s"waifu2x weights not found: $file")
    val set=SafeTensorSet.openFile(file)
    try
      val shapes=set.tensors.map((k,v)=>k->v.shape.dims.map(_.toInt).toSeq)
      val used=scala.collection.mutable.Set.empty[String]
      def tensor(n: String,dims: Seq[Int]): Array[Float]=
        val got=shapes.getOrElse(n,throw IllegalArgumentException(s"missing waifu2x weight: $n"))
        require(got==dims,s"waifu2x weight shape mismatch: $n, expected ${dims.mkString("x")}, got ${got.mkString("x")}")
        used+=n;val a=set.readFloat(n).toFloatArray
        require(a.forall(v=>java.lang.Float.isFinite(v)),s"waifu2x weight is not finite: $n");a
      val convs=(0 until 6).map(i=>new Conv(tensor(s"conv$i.weight",Seq(Channels(i+1),Channels(i),3,3)),tensor(s"conv$i.bias",Seq(Channels(i+1))),Channels(i),Channels(i+1))).toArray
      val dw=tensor("deconv.weight",Seq(256,3,4,4));val db=tensor("deconv.bias",Seq(3))
      require(db.forall(_==0f),"waifu2x deconv bias must be zero")
      val unused=shapes.keySet.diff(used.toSet)
      require(unused.isEmpty,s"unexpected waifu2x weights: ${unused.take(5).mkString(",")}")
      new Waifu2x(convs,dw)
    finally set.close()

  /** `ReplicationPad2d(p)`. */
  def replicatePad(x: Array[Float],h: Int,w: Int,p: Int): Array[Float]=
    val oh=h+2*p;val ow=w+2*p;val out=new Array[Float](3*oh*ow)
    for c <- 0 until 3; y <- 0 until oh do
      val sy=math.max(0,math.min(h-1,y-p));val src=(c*h+sy)*w;val dst=(c*oh+y)*ow
      System.arraycopy(x,src,out,dst+p,w)
      java.util.Arrays.fill(out,dst,dst+p,x(src));java.util.Arrays.fill(out,dst+p+w,dst+ow,x(src+w-1))
    out

  /** Unpadded 3x3 stride-1 convolution + bias + LeakyReLU(0.1) on a CHW map (output `co x (h-2) x (w-2)`). Parallel over (row, channel). */
  def conv3x3LeakyValid(x: Array[Float],ci: Int,h: Int,w: Int,wt: Array[Float],bias: Array[Float],co: Int): Array[Float]=
    val oh=h-2;val ow=w-2;val out=new Array[Float](co*oh*ow)
    DepthAnythingV2.parallel(oh*co) { idx =>
      val oy=idx/co;val o=idx%co;val orow=(o*oh+oy)*ow
      java.util.Arrays.fill(out,orow,orow+ow,bias(o))
      var c=0
      while c<ci do
        var ky=0
        while ky<3 do
          val wi=((o*ci+c)*3+ky)*3;val w0=wt(wi);val w1=wt(wi+1);val w2=wt(wi+2);val xr=(c*h+oy+ky)*w
          var ox=0
          while ox<ow do { out(orow+ox)+=w0*x(xr+ox)+w1*x(xr+ox+1)+w2*x(xr+ox+2);ox+=1 }
          ky+=1
        c+=1
      var ox=0
      while ox<ow do { val v=out(orow+ox);if v<0f then out(orow+ox)=v*0.1f;ox+=1 }
    }
    out

  /** `ConvTranspose2d(ci,3,k=4,stride=2,padding=3,bias=False)` with weights in (in,out,kH,kW) layout; output `3 x (2h-4) x (2w-4)`.
    * Written as a gather: output row `oy` takes input row `(oy+3-ky)/2` for the two kernel rows `ky` of matching parity (same for columns). */
  def deconv(x: Array[Float],ci: Int,h: Int,w: Int,wt: Array[Float]): Array[Float]=
    val co=3;val oh=2*h-4;val ow=2*w-4;val half=ow/2;val out=new Array[Float](co*oh*ow)
    DepthAnythingV2.parallel(oh*co) { idx =>
      val oy=idx/co;val o=idx%co
      val acc=Array.ofDim[Float](2,half)
      var ky=(oy+1)&1
      while ky<4 do
        val iy=(oy+3-ky)/2
        if iy>=0 && iy<h then
          var c=0
          while c<ci do
            val xr=(c*h+iy)*w;val wb=((c*co+o)*4+ky)*4
            var p=0
            while p<2 do
              val a=acc(p);val k0=(p+1)&1
              var kx=k0
              while kx<4 do
                val d=(p+3-kx)/2;val wv=wt(wb+kx)  // input column = m + d for output column 2m+p
                val lo=math.max(0,-d);val hi=math.min(half-1,w-1-d);var m=lo
                while m<=hi do { a(m)+=wv*x(xr+m+d);m+=1 }
                kx+=2
              p+=1
            c+=1
        ky+=2
      val base=(o*oh+oy)*ow;var m=0
      while m<half do { out(base+2*m)=acc(0)(m);out(base+2*m+1)=acc(1)(m);m+=1 }
    }
    out

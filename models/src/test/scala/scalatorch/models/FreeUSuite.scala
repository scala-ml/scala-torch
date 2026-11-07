package scalatorch.models.worker

import scalatorch.{Shape,Tensor}
import scalatorch.autograd.Variable
import scalatorch.models.*
import java.nio.file.Path

class FreeUSuite extends munit.FunSuite:
  private def tensor(dims: Seq[Int])(f: Int => Float): Tensor[Float] =
    Tensor.fromFloats(Array.tabulate(dims.product)(f),Shape.from(dims.map(_.toLong)))

  /** Literal port of diffusers fourier_filter: full DFT, fftshift, python-slice mask, ifftshift, inverse DFT, real part. */
  private def reference(x: Tensor[Float],threshold: Int,scale: Float): Array[Float] =
    val Seq(n,c,h,w)=x.shape.dims.map(_.toInt)
    val in=x.toFloatArray
    val out=new Array[Float](in.length)
    def pySlice(start: Int,end: Int,size: Int): Range =
      val s=if start<0 then math.max(start+size,0) else math.min(start,size)
      val e=if end<0 then math.max(end+size,0) else math.min(end,size)
      s until e
    val rows=pySlice(h/2-threshold,h/2+threshold,h).toSet
    val cols=pySlice(w/2-threshold,w/2+threshold,w).toSet
    for plane <- 0 until n*c do
      val base=plane*h*w
      val re=Array.ofDim[Double](h,w);val im=Array.ofDim[Double](h,w)
      for ky <- 0 until h; kx <- 0 until w do
        var sr=0.0;var si=0.0
        for y <- 0 until h; xx <- 0 until w do
          val a= -2*math.Pi*(ky.toDouble*y/h+kx.toDouble*xx/w)
          val v=in(base+y*w+xx)
          sr+=v*math.cos(a);si+=v*math.sin(a)
        // shifted index of frequency (ky,kx) is ((ky + h/2) mod h, (kx + w/2) mod w)
        val m=if rows((ky+h/2)%h) && cols((kx+w/2)%w) then scale.toDouble else 1.0
        re(ky)(kx)=sr*m;im(ky)(kx)=si*m
      for y <- 0 until h; xx <- 0 until w do
        var sr=0.0
        for ky <- 0 until h; kx <- 0 until w do
          val a=2*math.Pi*(ky.toDouble*y/h+kx.toDouble*xx/w)
          sr+=re(ky)(kx)*math.cos(a)-im(ky)(kx)*math.sin(a)
        out(base+y*w+xx)=(sr/(h*w)).toFloat
    out

  private def maxAbs(a: Array[Float],b: Array[Float]): Double =
    a.indices.map(i => math.abs(a(i).toDouble-b(i))).max

  test("fourier filter with scale 1 is the identity") {
    val x=tensor(Seq(1,2,6,5))(i => math.sin(i*0.37).toFloat)
    assert(maxAbs(FreeU.fourierFilter(x,1,1f).toFloatArray,x.toFloatArray)<=1e-6)
  }

  test("fourier filter scales a constant plane by the low-frequency factor") {
    val x=tensor(Seq(1,1,4,4))(_ => 2f)
    assert(maxAbs(FreeU.fourierFilter(x,1,0.25f).toFloatArray,Array.fill(16)(0.5f))<=1e-6)
  }

  test("fourier filter matches the diffusers FFT mask for even, odd and degenerate sizes") {
    for dims <- Seq(Seq(1,2,8,8),Seq(2,1,5,7),Seq(1,3,6,3),Seq(1,1,1,4),Seq(1,1,2,2)) do
      val x=tensor(dims)(i => (math.sin(i*0.91)+0.3*math.cos(i*2.3)).toFloat)
      val actual=FreeU.fourierFilter(x,1,0.2f).toFloatArray
      val expected=reference(x,1,0.2f)
      assert(maxAbs(actual,expected)<=1e-5,s"dims=$dims max=${maxAbs(actual,expected)}")
  }

  test("backbone scaling multiplies only the first half of the channels") {
    val x=tensor(Seq(1,5,1,2))(_ => 1f)
    val y=FreeU.scaleBackbone(x,1.5f).toFloatArray.toVector
    assertEquals(y,Vector(1.5f,1.5f,1.5f,1.5f,1f,1f,1f,1f,1f,1f))
  }

  test("only up-block stages 0 and 1 are modified") {
    val c=FreeUConfig(b1=1.5f,b2=1.6f,s1=0.9f,s2=0.2f)
    val h=tensor(Seq(1,2,2,2))(_ => 1f);val s=tensor(Seq(1,2,2,2))(i => i.toFloat)
    val (h0,_)=FreeU.apply(c,0,h,s)
    val (h1,_)=FreeU.apply(c,1,h,s)
    val (h2,s2)=FreeU.apply(c,2,h,s)
    assertEquals(h0.toFloatArray.take(4).toVector,Vector.fill(4)(1.5f))
    assertEquals(h1.toFloatArray.take(4).toVector,Vector.fill(4)(1.6f))
    assertEquals(h2.toFloatArray.toVector,h.toFloatArray.toVector)
    assertEquals(s2.toFloatArray.toVector,s.toFloatArray.toVector)
  }

  test("tiny U-Net: identity FreeU leaves output unchanged and active FreeU changes it; scope is request-local") {
    val dir=sys.props.get("scalatorch.unet.dir")
    assume(dir.nonEmpty,"tiny U-Net required")
    val unet=PretrainedUNet.load(Path.of(dir.get))
    val width=unet.midBlock.attention.blocks.head.attn2.toK.weight.data.shape.dims(1).toInt
    val sample=Variable(tensor(Seq(1,unet.convIn.weight.data.shape.dims(1).toInt,16,16))(i => math.sin(i*0.13).toFloat),false)
    val context=Variable(tensor(Seq(1,4,width))(i => math.cos(i*0.07).toFloat),false)
    val t=Tensor.fromFloats(Array(500f),Shape.from(Seq(1L)))
    val plain=unet(sample,t,context).data.toFloatArray
    val identity=FreeU.withConfig(Some(FreeUConfig(1f,1f,1f,1f)))(unet(sample,t,context).data.toFloatArray)
    assert(maxAbs(plain,identity)<=1e-5,s"identity FreeU changed output by ${maxAbs(plain,identity)}")
    val active=FreeU.withConfig(Some(FreeUConfig(1.5f,1.6f,0.9f,0.2f)))(unet(sample,t,context).data.toFloatArray)
    assert(maxAbs(plain,active)>1e-4,"active FreeU must change the U-Net output")
    assertEquals(FreeU.current,None)
    val after=unet(sample,t,context).data.toFloatArray
    assert(maxAbs(plain,after)==0.0,"FreeU must not leak outside its scope")
  }

  test("NativeDiffusionBackend advertises model_keyword and freeu and executes freeu hook") {
    import scalatorch.models.worker.*
    import NativeJson.*
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty,"tiny model required")
    val root=Path.of(configured.get).getParent
    val output=java.nio.file.Files.createTempDirectory("native-freeu-")
    try
      val backend=new NativeDiffusionBackend(root,root.resolve("euler-demo.json"))
      assert(backend.features.contains("model_keyword"))
      assert(backend.features.contains("freeu"))
      val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
      backend.execute(NativeRequest("load","load_model",Map("path"->Str(root.toString)),None),context)
      def generate(hooks: Map[String,NativeJson]): Vector[Int] =
        val params=Obj(Map("prompt"->Str("a cat"),"width"->Number(16),"height"->Number(16),
          "steps"->Number(2),"seed"->Number(42),"hooks"->Obj(hooks)))
        val res=backend.execute(NativeRequest("gen","txt2img",
          Map("output_dir"->Str(output.toString),"params"->params),None),context)
        val path=res("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
        val img=javax.imageio.ImageIO.read(Path.of(path).toFile)
        img.getRGB(0,0,16,16,null,0,16).toVector
      val baseline=generate(Map.empty)
      val identity=generate(Map("freeu"->Obj(Map("enabled"->Bool(true),"b1"->Number(1),"b2"->Number(1),"s1"->Number(1),"s2"->Number(1)))))
      assertEquals(identity,baseline,"identity FreeU hook must match baseline generation")
      val disabled=generate(Map("freeu"->Obj(Map("enabled"->Bool(false),"b1"->Number(2),"b2"->Number(2),"s1"->Number(0),"s2"->Number(0)))))
      assertEquals(disabled,baseline,"disabled FreeU hook must match baseline generation")
      val active=generate(Map("freeu"->Obj(Map("enabled"->Bool(true),"b1"->Number(BigDecimal("1.9")),"b2"->Number(BigDecimal("1.9")),"s1"->Number(BigDecimal("0.1")),"s2"->Number(BigDecimal("0.1"))))))
      assertNotEquals(active,baseline,"active FreeU hook must modify generated image")
      intercept[IllegalArgumentException] {
        generate(Map("freeu"->Obj(Map("enabled"->Bool(true),"unknown_opt"->Number(1)))))
      }
    finally
      val files=java.nio.file.Files.walk(output)
      try files.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(p => { java.nio.file.Files.delete(p);() })
      finally files.close()
  }

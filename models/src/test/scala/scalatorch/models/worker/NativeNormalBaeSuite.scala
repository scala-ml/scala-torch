package scalatorch.models.worker

import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import scalatorch.io.SafeTensorSet
import scalatorch.models.NormalBae

/** NormalBAE (`normal_bae` preprocessor) tests. Real-weights parity needs SDFX_TEST_NORMALBAE (path of scannet.pt) and SDFX_TEST_NORMALBAE_REF
  * (safetensors written by scripts/reference_normalbae.py, with `input` and `output`). The end-to-end test needs the weights reachable through
  * SDFX_SCALATORCH_ANNOTATOR_DIR (or models/annotators), SDFX_TEST_NORMALBAE_IMAGES (comma-separated images) and SDFX_TEST_OUT (normal maps are
  * written there). NormalBAE code is MIT, the EfficientNet encoder code/weights Apache-2.0; the scannet.pt licence is unconfirmed. */
class NativeNormalBaeSuite extends munit.FunSuite:
  test("missing NormalBAE weights fail loudly instead of producing an approximation") {
    val none=Files.createTempDirectory("nbae-none")
    val e=intercept[IllegalStateException](NativeNormalBae.locate(Map.empty,none))
    assert(e.getMessage.contains("weights not found"))
  }

  test("TF same padding puts the odd pixel at the end") {
    assertEquals(NormalBae.samePad(512,3,2),(0,1))
    assertEquals(NormalBae.samePad(512,5,2),(1,2))
    assertEquals(NormalBae.samePad(64,5,1),(2,2))
    assertEquals(NormalBae.samePad(64,3,1),(1,1))
  }

  test("normal map encoding is (n+1)/2*255 truncated, xyz to RGB without a flip") {
    // 3 pixels, channel-major: x=[0,1,-1] y=[0,0,0] z=[1,0,0] kappa ignored
    val out=Array(0f,1f,-1f, 0f,0f,0f, 1f,0f,0f, 1f,1f,1f)
    val px=NormalBae.toRgb8(out,1,3)
    def rgb(i: Int)=((px(i)>>16)&255,(px(i)>>8)&255,px(i)&255)
    assertEquals(rgb(0),(127,127,255))
    assertEquals(rgb(1),(255,127,127))
    assertEquals(rgb(2),(0,127,127))
  }

  test("working size and strict loading") {
    assertEquals(NativeAnnotatorFiles.workingSize(1000,500,512,64,roundUp=false),(1024,512))
    intercept[IllegalArgumentException](NormalBae.load(Path.of("no-such-scannet.pt")))
  }

  private def relErr(a: Array[Float],ref: Array[Float]): (Double,Double)=
    require(a.length==ref.length,s"length ${a.length} vs ${ref.length}")
    var maxAbs=0.0;var refMax=0.0;var i=0
    while i<a.length do { maxAbs=math.max(maxAbs,math.abs(a(i)-ref(i)));refMax=math.max(refMax,math.abs(ref(i)));i+=1 }
    (maxAbs,maxAbs/refMax)

  test("network matches the torch reference (raw 4-channel output, then the 8-bit normal map)") {
    val w=sys.env.get("SDFX_TEST_NORMALBAE");val r=sys.env.get("SDFX_TEST_NORMALBAE_REF")
    assume(w.isDefined && r.isDefined,"real-weights gate disabled")
    val t0=System.nanoTime
    val net=NormalBae.load(Path.of(w.get))
    println(f"[normalbae] load ${(System.nanoTime-t0)/1e9}%.1f s")
    for ref0 <- r.get.split(",").map(_.trim).filter(_.nonEmpty) do
      val set=SafeTensorSet.openFile(Path.of(ref0))
      try
        val x=set.readFloat("input").toFloatArray;val ref=set.readFloat("output").toFloatArray
        val dims=set.tensors("input").shape.dims.map(_.toInt);val h=dims(1);val w=dims(2)
        val t1=System.nanoTime
        val out=net.infer(x,h,w)
        println(f"[normalbae] ${w}x$h forward ${(System.nanoTime-t1)/1e9}%.1f s")
        val n=h*w
        val (abs,rel)=relErr(out,ref)
        val (xa,xr)=relErr(out.take(3*n),ref.take(3*n));val (ka,kr)=relErr(out.drop(3*n),ref.drop(3*n))
        println(f"[normalbae] all maxAbs=$abs%.3e rel=$rel%.3e | xyz maxAbs=$xa%.3e rel=$xr%.3e | kappa maxAbs=$ka%.3e rel=$kr%.3e")
        val a=NormalBae.toRgb8(out,h,w);val b=NormalBae.toRgb8(ref,h,w)
        var maxDiff=0;var differing=0;var i=0
        while i<n do
          for s <- Seq(16,8,0) do { val d=math.abs(((a(i)>>s)&255)-((b(i)>>s)&255));if d>0 then differing+=1;maxDiff=math.max(maxDiff,d) }
          i+=1
        println(s"[normalbae] 8-bit map: max channel difference $maxDiff, $differing of ${3*n} channel values differ")
        assert(rel<1e-4,s"relative error $rel")
        assert(maxDiff<=1)
      finally set.close()
  }

  test("preprocessor end to end on real images") {
    val imgs=sys.env.get("SDFX_TEST_NORMALBAE_IMAGES");val out=sys.env.get("SDFX_TEST_OUT")
    assume(imgs.isDefined && out.isDefined,"real-weights gate disabled")
    Files.createDirectories(Path.of(out.get))
    for file <- imgs.get.split(",").map(_.trim).filter(_.nonEmpty) do
      val source=ImageIO.read(Path.of(file).toFile)
      val t0=System.nanoTime
      val map=NativeNormalBae.preprocess(source,512,()=>false)
      val seconds=(System.nanoTime-t0)/1e9
      assertEquals((map.getWidth,map.getHeight),(source.getWidth,source.getHeight))
      val name=Path.of(file).getFileName.toString.replaceAll("\\.[^.]+$","")
      val dest=Path.of(out.get).resolve(s"$name-normal_bae.png");ImageIO.write(map,"png",dest.toFile)
      println(f"[normalbae] $file -> $dest in $seconds%.1f s")
  }

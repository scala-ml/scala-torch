package scalatorch.models.worker

import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import scalatorch.io.SafeTensorSet
import scalatorch.models.{AnnotatorOps,DepthAnythingV2,Waifu2x}
import NativeJson.*

/** waifu2x (upconv_7) tests. Weight-free checks run always. Real-weights parity needs SDFX_TEST_ANNOTATORS (folder holding
  * waifu2x/{art,photo}-noiseN.safetensors written by scripts/convert_waifu2x_json_to_safetensors.py) and SDFX_TEST_WAIFU2X_REF (folder written
  * by scripts/reference_waifu2x.py, the torch oracle built from the original JSON models). The end-to-end test needs the weights reachable
  * through SDFX_SCALATORCH_ANNOTATOR_DIR (or models/annotators), SDFX_TEST_WAIFU2X_IMAGE and SDFX_TEST_OUT. waifu2x code is MIT; the model
  * files have no separate licence statement. */
class NativeWaifu2xSuite extends munit.FunSuite:
  override val munitTimeout: scala.concurrent.duration.Duration=scala.concurrent.duration.Duration(10,"min")
  private def rnd(n: Int,seed: Long,scale: Float=1f): Array[Float]={ val r=new java.util.Random(seed);Array.fill(n)((r.nextFloat()-0.5f)*scale) }

  private def relErr(a: Array[Float],ref: Array[Float]): (Double,Double)=
    require(a.length==ref.length,s"length ${a.length} vs ${ref.length}")
    var maxAbs=0.0;var refMax=0.0;var i=0
    while i<a.length do { maxAbs=math.max(maxAbs,math.abs(a(i)-ref(i)));refMax=math.max(refMax,math.abs(ref(i)));i+=1 }
    (maxAbs,maxAbs/refMax)

  test("missing waifu2x weights fail loudly") {
    val none=Files.createTempDirectory("w2x-none")
    val e=intercept[IllegalStateException](NativeAnnotatorFiles.locate(Waifu2x.fileName("art",2),"waifu2x",Map.empty,none))
    assert(e.getMessage.contains("weights not found"))
    val e2=intercept[IllegalStateException](NativeWaifu2x.locate("photo",1,Map.empty,none))
    assert(e2.getMessage.contains("weights not found"))
  }

  test("styles map to model folders; CarnV2 is reported as unsupported") {
    assertEquals(NativeWaifu2x.styleKey("Anime"),"art")
    assertEquals(NativeWaifu2x.styleKey("Photo"),"photo")
    intercept[NativeBackendError](NativeWaifu2x.styleKey("CarnV2"))
    assertEquals(NativeWaifu2x.passes(100,100,200,200),1)
    assertEquals(NativeWaifu2x.passes(100,100,100,100),1)
    assertEquals(NativeWaifu2x.passes(100,100,201,100),2)
    assertEquals(NativeWaifu2x.passes(100,100,800,400),3)
  }

  test("gather deconvolution equals the scatter ConvTranspose2d reference") {
    val ci=5;val h=9;val w=11;val x=rnd(ci*h*w,1);val wt=rnd(ci*3*16,2)
    val a=Waifu2x.deconv(x,ci,h,w,wt)
    val b=AnnotatorOps.convTranspose(x,ci,h,w,wt,Array(0f,0f,0f),3,4,2,3,0)
    assertEquals(a.length,3*(2*h-4)*(2*w-4))
    assertEquals(b.length,a.length)
    val (abs,_)=relErr(a,b);assert(abs<1e-5,s"deconv maxAbs $abs")
  }

  test("valid 3x3 conv + LeakyReLU equals the generic conv2d") {
    val ci=4;val co=6;val h=10;val w=13;val x=rnd(ci*h*w,3);val wt=rnd(co*ci*9,4);val bias=rnd(co,5)
    val a=Waifu2x.conv3x3LeakyValid(x,ci,h,w,wt,bias,co)
    val b=AnnotatorOps.leakyRelu(DepthAnythingV2.conv2d(x,ci,h,w,wt,Some(bias),co,3,1,0),0.1f)
    val (abs,_)=relErr(a,b);assert(abs<1e-5,s"conv maxAbs $abs")
  }

  test("replicate padding repeats the edge pixels") {
    val p=Waifu2x.replicatePad(Array(1f,2f,3f,4f,5f,6f,7f,8f,9f,10f,11f,12f),2,2,1)
    assertEquals(p.take(4).toSeq,Seq(1f,1f,2f,2f))
    assertEquals(p.slice(8,12).toSeq,Seq(3f,3f,4f,4f))
  }

  private def annotators: Option[Path]=sys.env.get("SDFX_TEST_ANNOTATORS").map(Path.of(_))

  test("waifu2x matches the torch reference (all styles/noise levels, 128x128 crop, before 8-bit quantisation)") {
    val a=annotators;val r=sys.env.get("SDFX_TEST_WAIFU2X_REF")
    assume(a.isDefined && r.isDefined,"real-weights gate disabled")
    val set=SafeTensorSet.openFile(Path.of(r.get).resolve("waifu2x_ref.safetensors"))
    try
      val x=set.readFloat("input").toFloatArray
      for style <- Seq("art","photo"); noise <- 0 to 3 do
        val net=Waifu2x.load(a.get.resolve(Waifu2x.fileName(style,noise)))
        val t0=System.nanoTime
        val whole=net.upscale2x(x,128,128,tile=0)
        val secs=(System.nanoTime-t0)/1e9
        val ref=set.readFloat(s"${style}_n$noise").toFloatArray
        val (abs,rel)=relErr(whole,ref)
        // tiling with a 40 px tile (ragged edges: 128 = 3*40+8) must match the whole-image pass
        val tiled=net.upscale2x(x,128,128,tile=40)
        val (tabs,trel)=relErr(tiled,whole)
        println(f"[waifu2x] $style noise$noise vs torch maxAbs=$abs%.2e rel=$rel%.2e | tiled(40) vs whole maxAbs=$tabs%.2e rel=$trel%.2e | ${secs}%.2f s")
        assert(rel<1e-4,s"$style noise$noise relative error vs torch $rel")
        assert(trel<1e-4,s"$style noise$noise tiled vs whole $trel")
    finally set.close()
  }

  test("image_tool waifu2x runs for real on a real image (and Photo / noise variants)") {
    val image=sys.env.get("SDFX_TEST_WAIFU2X_IMAGE");val out=sys.env.get("SDFX_TEST_OUT")
    assume(image.isDefined && out.isDefined && scala.util.Try(NativeWaifu2x.locate("art",2)).isSuccess,"real-weights gate disabled")
    val root=Files.createTempDirectory("native-waifu2x-real")
    val scheduler=root.resolve("scheduler.json");Files.writeString(scheduler,"{}")
    val output=Path.of(out.get);Files.createDirectories(output)
    val backend=new NativeDiffusionBackend(root,scheduler)
    val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    val src=ImageIO.read(Path.of(image.get).toFile)
    val stem=Path.of(image.get).getFileName.toString.stripSuffix(".png")
    for (style,noise,scale) <- Seq(("Anime",2,2.0),("Photo",1,2.0),("Anime",3,2.0),("Anime",0,1.5),("Anime",1,4.0),("Photo",3,2.0)) do
      val t0=System.nanoTime
      val params=Map[String,NativeJson]("tool"->Str("waifu2x"),"image_path"->Str(image.get),"output_dir"->Str(output.toString),
        "style"->Str(style),"noise"->Number(BigDecimal(noise)),"scale"->Number(BigDecimal(scale)))
      val result=backend.execute(NativeRequest("w2x","image_tool",params,None),context)
      val secs=(System.nanoTime-t0)/1e9
      val dst=Path.of(result("path").asInstanceOf[Str].value)
      val up=ImageIO.read(dst.toFile)
      assertEquals((up.getWidth,up.getHeight),(math.round(src.getWidth*scale).toInt,math.round(src.getHeight*scale).toInt))
      val named=output.resolve(s"$stem-${style.toLowerCase}-n$noise-x$scale.png");Files.copy(dst,named,java.nio.file.StandardCopyOption.REPLACE_EXISTING)
      println(f"[waifu2x] $style noise$noise x$scale ${src.getWidth}x${src.getHeight} -> ${up.getWidth}x${up.getHeight} in $secs%.1f s: $named")
  }

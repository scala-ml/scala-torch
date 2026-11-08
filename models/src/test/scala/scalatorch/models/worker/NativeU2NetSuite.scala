package scalatorch.models.worker

import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import scalatorch.io.SafeTensorSet
import scalatorch.models.U2Net
import NativeJson.*

/** U2-Net (remove_background) tests. Real-weights parity needs SDFX_TEST_ANNOTATORS (folder holding u2net/u2net.safetensors written by
  * scripts/convert_u2net_onnx_to_safetensors.py) and SDFX_TEST_U2NET_REF (folder written by scripts/reference_u2net.py, the onnxruntime oracle).
  * The end-to-end test needs the weights reachable through SDFX_SCALATORCH_ANNOTATOR_DIR (or models/annotators), SDFX_TEST_U2NET_IMAGE and
  * SDFX_TEST_OUT (cut-outs are written there). U2-Net code is Apache-2.0; the rembg weights carry no stated licence. */
class NativeU2NetSuite extends munit.FunSuite:
  test("missing U2-Net weights fail loudly") {
    val none=Files.createTempDirectory("u2-none")
    val e=intercept[IllegalStateException](NativeAnnotatorFiles.locate(NativeU2Net.FileName,"U2-Net (u2net)",Map.empty,none))
    assert(e.getMessage.contains("weights not found"))
  }

  test("bilinear resize uses half-pixel centres (align_corners=False)") {
    val up=U2Net.bilinear(Array(0f,1f,2f,3f),1,2,2,4,4)
    // row 0 of the 2x2 -> 4x4 torch result: 0, 0.25, 0.75, 1
    assertEquals(up.take(4).toSeq,Seq(0f,0.25f,0.75f,1f))
    assertEquals(up(15),3f)
  }

  test("matte is min-max normalised and resized back") {
    val d0=Array.tabulate(NativeU2Net.InputSize*NativeU2Net.InputSize)(i=>0.2f+0.4f*(i%NativeU2Net.InputSize)/(NativeU2Net.InputSize-1))
    val m=NativeU2Net.matte(d0,50,30)
    assertEquals(m.length,1500)
    assert(m.min>=0f && m.max<=1f && m(0)<0.05f && m(49)>0.95f)
  }

  private def relErr(a: Array[Float],ref: Array[Float]): (Double,Double)=
    require(a.length==ref.length,s"length ${a.length} vs ${ref.length}")
    var maxAbs=0.0;var refMax=0.0;var i=0
    while i<a.length do { maxAbs=math.max(maxAbs,math.abs(a(i)-ref(i)));refMax=math.max(refMax,math.abs(ref(i)));i+=1 }
    (maxAbs,maxAbs/refMax)

  test("U2-Net matches onnxruntime: d0..d6 and the min-max matte on the same 1x3x320x320 tensor") {
    val a=sys.env.get("SDFX_TEST_ANNOTATORS");val r=sys.env.get("SDFX_TEST_U2NET_REF")
    assume(a.isDefined && r.isDefined,"real-weights gate disabled")
    val net=U2Net.load(Path.of(a.get).resolve(NativeU2Net.FileName))
    val set=SafeTensorSet.openFile(Path.of(r.get).resolve("u2net_ref.safetensors"))
    try
      val x=set.readFloat("input").toFloatArray
      val t0=System.nanoTime
      val maps=net.infer(x,320,320)
      println(f"[u2net] inference ${(System.nanoTime-t0)/1e9}%.1f s")
      assertEquals(maps.size,7)
      for (m,i) <- maps.zipWithIndex do
        val (abs,rel)=relErr(m,set.readFloat(s"d$i").toFloatArray)
        println(f"[u2net] d$i maxAbs=$abs%.2e rel=$rel%.2e")
        assert(rel<1e-4,s"d$i relative error vs onnxruntime $rel")
      val (abs,rel)=relErr(NativeU2Net.matte(maps.head,320,320),set.readFloat("matte").toFloatArray)
      println(f"[u2net] matte maxAbs=$abs%.2e rel=$rel%.2e")
      assert(rel<1e-4)
    finally set.close()
  }

  test("remove_background cuts out the subject with the real U2-Net weights (transparent and white)") {
    val image=sys.env.get("SDFX_TEST_U2NET_IMAGE");val out=sys.env.get("SDFX_TEST_OUT")
    val located=scala.util.Try(NativeU2Net.locate()).isSuccess
    assume(image.isDefined && out.isDefined && located,"real-weights gate disabled")
    val root=Files.createTempDirectory("native-u2net")
    val scheduler=root.resolve("scheduler.json");Files.writeString(scheduler,"{}")
    val output=Path.of(out.get);Files.createDirectories(output)
    val backend=new NativeDiffusionBackend(root,scheduler)
    val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    val source=ImageIO.read(Path.of(image.get).toFile)
    val stem=Path.of(image.get).getFileName.toString.stripSuffix(".png")
    for bg <- Seq("transparent","white") do
      val params=Map[String,NativeJson]("tool"->Str("remove_background"),"image_path"->Str(image.get),"output_dir"->Str(output.toString),"background"->Str(bg))
      val result=backend.execute(NativeRequest("rb","image_tool",params,None),context)
      val dst=Path.of(result("path").asInstanceOf[Str].value)
      val cut=ImageIO.read(dst.toFile)
      assertEquals((cut.getWidth,cut.getHeight),(source.getWidth,source.getHeight))
      Files.copy(dst,output.resolve(s"$stem-$bg.png"),java.nio.file.StandardCopyOption.REPLACE_EXISTING)
      if bg=="transparent" then
        val alpha=for y <- 0 until cut.getHeight;x <- 0 until cut.getWidth yield (cut.getRGB(x,y)>>>24)
        val frac=alpha.count(_>127).toDouble/alpha.size
        println(f"[u2net] $stem foreground fraction $frac%.3f")
        assert(frac>0.005 && frac<0.95,s"implausible matte coverage $frac")
        val mask=Path.of(result("outputs").asInstanceOf[Obj].fields("mask").asInstanceOf[Str].value)
        Files.copy(mask,output.resolve(s"$stem-matte.png"),java.nio.file.StandardCopyOption.REPLACE_EXISTING)
  }

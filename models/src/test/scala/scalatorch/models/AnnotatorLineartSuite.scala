package scalatorch.models

import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import scalatorch.io.SafeTensorSet
import scalatorch.models.worker.{NativeAnnotatorFiles,NativeHed,NativeLineart,NativeLineartAnime}

/** Real-weights parity against torch references (scripts/reference_annotator_lineart.py): set SDFX_TEST_ANNOTATORS (the directory
  * holding ControlNetHED.pth, sk_model.pth, netG.pth) and SDFX_TEST_ANNOTATOR_REF (the directory with hed_ref / lineart_ref /
  * anime_ref .safetensors). SDFX_TEST_IMAGE + SDFX_TEST_OUT additionally write the three preprocessor PNGs there. */
class AnnotatorLineartSuite extends munit.FunSuite:
  private def relErr(a: Array[Float],b: Array[Float]): (Double,Double)=
    assertEquals(a.length,b.length)
    var maxAbs=0.0;var ref=0.0;var i=0
    while i<a.length do { maxAbs=math.max(maxAbs,math.abs(a(i)-b(i)));ref=math.max(ref,math.abs(b(i)));i+=1 }
    (maxAbs,maxAbs/math.max(ref,1e-12))

  private def gate: Option[(Path,Path)]=
    for a <- sys.env.get("SDFX_TEST_ANNOTATORS");r <- sys.env.get("SDFX_TEST_ANNOTATOR_REF") yield (Path.of(a),Path.of(r))

  private def parity(label: String,ref: Path,height: Int=512)(run: (Array[Float],Int,Int) => Array[Float]): Unit=
    val set=SafeTensorSet.openFile(ref)
    try
      val input=set.readFloat("input").toFloatArray
      val t0=System.nanoTime
      val out=run(input,height,height)
      val seconds=(System.nanoTime-t0)/1e9
      val (abs,rel)=relErr(out,set.readFloat("output").toFloatArray)
      println(s"[$label] ${height}x$height in ${"%.1f".format(seconds)}s maxAbs=${"%.3e".format(abs)} rel=${"%.3e".format(rel)}")
      assert(rel<1e-4,s"$label relative error $rel")
    finally set.close()

  test("missing weights fail with a clear error instead of a fake edge map") {
    val empty=Files.createTempDirectory("annotator-none")
    for (f,label) <- Seq(NativeHed.FileName->"HED",NativeLineart.FileName->"lineart",NativeLineartAnime.FileName->"anime lineart") do
      val e=intercept[IllegalStateException](NativeAnnotatorFiles.locate(f,label,Map.empty,empty))
      assert(e.getMessage.contains(s"$label weights not found"))
  }

  test("instance norm, reflection pad and transposed convolution behave like torch on small cases") {
    val x=Array(1f,2f,3f,4f);AnnotatorOps.instanceNorm(x,1,2,2)
    assert(math.abs(x.sum)<1e-5 && math.abs(x(3)-1.3416f)<1e-3)
    val p=AnnotatorOps.reflectPad(Array(1f,2f,3f,4f,5f,6f,7f,8f,9f),1,3,3,1)
    assertEquals(p.slice(0,5).toSeq,Seq(5f,4f,5f,6f,5f))
    // 1x1 input, 3x3 kernel, stride 2, pad 1, output padding 1 -> 2x2 taking the kernel's centre/right/bottom taps
    val t=AnnotatorOps.convTranspose(Array(2f),1,1,1,Array.tabulate(9)(_.toFloat),Array(0f),1,3,2,1,1)
    assertEquals(t.toSeq,Seq(8f,10f,14f,16f))
    assertEquals(ControlNetHed.safeStep(0.5f),0.5f)
  }

  test("HED matches the torch reference") {
    assume(gate.isDefined,"real-weights gate disabled")
    val (a,r)=gate.get;val m=ControlNetHed.load(a.resolve(NativeHed.FileName))
    parity("hed",r.resolve("hed_ref.safetensors"))((in,h,w)=>m.edges(in,h,w))
  }

  test("lineart matches the torch reference") {
    assume(gate.isDefined,"real-weights gate disabled")
    val (a,r)=gate.get;val m=LineartGenerator.load(a.resolve(NativeLineart.FileName))
    parity("lineart",r.resolve("lineart_ref.safetensors"))((in,h,w)=>m.infer(in,h,w))
  }

  test("lineart_anime matches the torch reference") {
    assume(gate.isDefined,"real-weights gate disabled")
    val (a,r)=gate.get;val m=AnimeLineartUnet.load(a.resolve(NativeLineartAnime.FileName))
    parity("lineart_anime",r.resolve("anime_ref.safetensors"))((in,h,w)=>m.infer(in,h,w))
  }

  test("preprocessors write sample images") {
    assume(gate.isDefined && sys.env.contains("SDFX_TEST_IMAGE") && sys.env.contains("SDFX_TEST_OUT"),"sample gate disabled")
    val env=Map("SDFX_SCALATORCH_ANNOTATOR_DIR"->gate.get._1.toString)
    val src=ImageIO.read(Path.of(sys.env("SDFX_TEST_IMAGE")).toFile);val out=Path.of(sys.env("SDFX_TEST_OUT"))
    Files.createDirectories(out);val never=()=>false
    ImageIO.write(NativeHed.preprocess(src,0,false,never,env),"png",out.resolve("hed.png").toFile)
    ImageIO.write(NativeLineart.preprocess(src,0,never,env),"png",out.resolve("lineart.png").toFile)
    ImageIO.write(NativeLineartAnime.preprocess(src,never,env),"png",out.resolve("lineart_anime.png").toFile)
  }

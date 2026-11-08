package scalatorch.models

import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import scalatorch.io.SafeTensorSet
import scalatorch.models.worker.{NativeAnnotatorFiles,NativeMlsd}

/** Real-weights parity for the M-LSD large annotator against scripts/reference_annotator_mlsd.py: set SDFX_TEST_ANNOTATORS (the
  * directory holding mlsd_large_512_fp32.pth) and SDFX_TEST_MLSD_REF (the directory with mlsd_ref.safetensors and
  * mlsd_segments.json). SDFX_TEST_IMAGE + SDFX_TEST_OUT additionally write the rendered preprocessor PNG (mlsd.png) there. */
class AnnotatorMlsdSuite extends munit.FunSuite:
  private def gate: Option[(Path,Path)]=
    for a <- sys.env.get("SDFX_TEST_ANNOTATORS");r <- sys.env.get("SDFX_TEST_MLSD_REF") yield (Path.of(a),Path.of(r))

  test("missing M-LSD weights fail with a clear error") {
    val e=intercept[IllegalStateException](NativeAnnotatorFiles.locate(NativeMlsd.FileName,"M-LSD",Map.empty,Files.createTempDirectory("mlsd-none")))
    assert(e.getMessage.contains("M-LSD weights not found"))
  }

  test("Bresenham lines are 8-connected and include both endpoints") {
    val pix=new Array[Boolean](8*8)
    MlsdLarge.drawLine(pix,8,8,0,0,7,3)
    assertEquals(pix.count(identity),8)
    assert(pix(0) && pix(3*8+7))
    val clipped=new Array[Boolean](4*4);MlsdLarge.drawLine(clipped,4,4,-3,1,9,1)
    assertEquals(clipped.count(identity),4)
  }

  test("decoder keeps only local maxima above the score and length thresholds") {
    val h=8;val w=8;val n=h*w;val tp=new Array[Float](9*n)
    java.util.Arrays.fill(tp,0,n,-10f)
    tp(2*w+2)=5f;tp(2*w+3)=4f // neighbour is suppressed by the peak
    tp(n+2*w+2)= -1f;tp(3*n+2*w+2)=1f;tp(2*n+2*w+2)=0f;tp(4*n+2*w+2)=0f // 2 tp_map pixels long, horizontal
    tp(5*w+5)=3f // peak with zero displacement: too short
    val got=MlsdLarge.decode(tp,h,w,0.1f,0.1f)
    assertEquals(got.size,1)
    assertEquals(got.head._2,(2f,4f,6f,4f))
  }

  test("M-LSD tp_map and segments match the torch reference") {
    assume(gate.isDefined,"real-weights gate disabled")
    val (a,r)=gate.get;val m=MlsdLarge.load(a.resolve(NativeMlsd.FileName))
    val set=SafeTensorSet.openFile(r.resolve("mlsd_ref.safetensors"))
    try
      val input=set.readFloat("input").toFloatArray;val ref=set.readFloat("output").toFloatArray
      val t0=System.nanoTime
      val (tp,h,w)=m.tpMap(input,512,512)
      val seconds=(System.nanoTime-t0)/1e9
      assertEquals((tp.length,h,w),(9*256*256,256,256))
      def rel(ref: Array[Float]): (Double,Double)=
        var maxAbs=0.0;var refMax=0.0;var i=0
        while i<tp.length do { maxAbs=math.max(maxAbs,math.abs(tp(i)-ref(i)));refMax=math.max(refMax,math.abs(ref(i)));i+=1 }
        (maxAbs,maxAbs/refMax)
      val (abs32,rel32)=rel(ref);val (abs64,rel64)=rel(set.readFloat("output_f64").toFloatArray)
      println(s"[mlsd] tp_map 9x256x256 in ${"%.1f".format(seconds)}s vs torch fp32: maxAbs=${"%.3e".format(abs32)} rel=${"%.3e".format(rel32)}; vs torch fp64: maxAbs=${"%.3e".format(abs64)} rel=${"%.3e".format(rel64)}")
      // the torch float32 result itself deviates ~1e-4 from float64 on this net (logits reach +-150), so about 1e-4 is the float32 noise floor and the gates below sit at 2x that
      assert(rel64<2e-4,s"M-LSD tp_map relative error vs float64 $rel64")
      assert(rel32<3e-4,s"M-LSD tp_map relative error vs float32 $rel32")
      // segments against the reference decoder (run on the reference tp_map, so only the Scala decoder is compared here)
      val expected="""\[([-0-9.eE]+), ([-0-9.eE]+), ([-0-9.eE]+), ([-0-9.eE]+), ([-0-9.eE]+)\]""".r
        .findAllMatchIn(Files.readString(r.resolve("mlsd_segments.json"))).map(m=>m.subgroups.map(_.toFloat)).toVector
      val got=MlsdLarge.decode(tp,h,w,0.1f,0.1f)
      val worst=got.zip(expected).map((g,e)=>Seq(g._2._1-e(1),g._2._2-e(2),g._2._3-e(3),g._2._4-e(4)).map(v=>math.abs(v)).max).maxOption.getOrElse(0f)
      println(s"[mlsd] segments scala=${got.size} reference=${expected.size} max endpoint diff=$worst px")
      assertEquals(got.size,expected.size)
      assert(worst<0.05f,s"segment endpoints differ by $worst")
    finally set.close()
  }

  test("mlsd preprocessor writes a sample image") {
    assume(gate.isDefined && sys.env.contains("SDFX_TEST_IMAGE") && sys.env.contains("SDFX_TEST_OUT"),"sample gate disabled")
    val env=Map("SDFX_SCALATORCH_ANNOTATOR_DIR"->gate.get._1.toString)
    val src=ImageIO.read(Path.of(sys.env("SDFX_TEST_IMAGE")).toFile);val out=Path.of(sys.env("SDFX_TEST_OUT"))
    Files.createDirectories(out)
    val image=NativeMlsd.preprocess(src,512,0.1f,0.1f,()=>false,env)
    assertEquals((image.getWidth,image.getHeight),(src.getWidth,src.getHeight))
    ImageIO.write(image,"png",out.resolve("mlsd.png").toFile)
  }

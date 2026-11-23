package scalatorch.models

import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import scalatorch.io.SafeTensorSet
import scalatorch.models.worker.NativeOpenPose

/** Real-weights parity against a torch reference (python openpose_ref.py): set SDFX_TEST_OPENPOSE (body_pose_model.pth or its
  * directory), SDFX_TEST_OPENPOSE_REF (safetensors with input/paf/heat) and optionally SDFX_TEST_OPENPOSE_TXT (peaks and
  * people from the reference post-processing) and SDFX_TEST_IMAGE (the image the reference ran on). SDFX_TEST_OUT also
  * writes the rendered pose map. */
class OpenPoseBodySuite extends munit.FunSuite:
  private def relErr(a: Array[Float],b: Array[Float]): (Double,Double)=
    assertEquals(a.length,b.length)
    var maxAbs=0.0;var ref=0.0;var i=0
    while i<a.length do { maxAbs=math.max(maxAbs,math.abs(a(i)-b(i)));ref=math.max(ref,math.abs(b(i)));i+=1 }
    (maxAbs,maxAbs/math.max(ref,1e-12))

  private def weightsFile(path: String): Path=
    val p=Path.of(path);if Files.isDirectory(p) then p.resolve(NativeOpenPose.FileName) else p

  test("missing weights fail with a clear error instead of a fake pose") {
    val empty=Files.createTempDirectory("openpose-none")
    val e=intercept[IllegalStateException](NativeOpenPose.locate(Map.empty,empty))
    assert(e.getMessage.contains("OpenPose body weights not found"))
  }

  test("limb tables are consistent") {
    assertEquals(OpenPoseBody.LimbSeq.size,19)
    assertEquals(OpenPoseBody.PafIndex.flatMap((a,b)=>Vector(a,b)).sorted,(0 until 38).toVector)
  }

  test("gaussian smoothing preserves a constant plane and spreads an impulse symmetrically") {
    val flat=OpenPoseBody.gaussian(Array.fill(40*30)(2f),30,40,3.0)
    assert(flat.forall(v=>math.abs(v-2f)<1e-5))
    val impulse=new Array[Float](41*41);impulse(20*41+20)=1f
    val g=OpenPoseBody.gaussian(impulse,41,41,3.0)
    assert(math.abs(g(20*41+17)-g(20*41+23))<1e-7 && g(20*41+20)>g(20*41+21))
  }

  test("network and detection match the torch reference") {
    val weights=sys.env.get("SDFX_TEST_OPENPOSE");val ref=sys.env.get("SDFX_TEST_OPENPOSE_REF")
    assume(weights.isDefined && ref.isDefined,"real-weights gate disabled")
    val model=OpenPoseBody.load(weightsFile(weights.get))
    val set=SafeTensorSet.openFile(Path.of(ref.get))
    try
      val shape=set.tensors("input").shape.dims.map(_.toInt);val (h,w)=(shape(1),shape(2))
      val t0=System.nanoTime
      val out=model.infer(set.readFloat("input").toFloatArray,h,w)
      val seconds=(System.nanoTime-t0)/1e9
      val paf=relErr(out.paf,set.readFloat("paf").toFloatArray);val heat=relErr(out.heat,set.readFloat("heat").toFloatArray)
      println(s"[openpose] ${w}x$h in ${"%.1f".format(seconds)}s")
      println(s"[openpose]   paf maxAbs=${"%.3e".format(paf._1)} rel=${"%.3e".format(paf._2)}")
      println(s"[openpose]   heat maxAbs=${"%.3e".format(heat._1)} rel=${"%.3e".format(heat._2)}")
      assert(paf._2<1e-4,s"paf relative error ${paf._2}")
      assert(heat._2<1e-4,s"heat relative error ${heat._2}")
      for txt <- sys.env.get("SDFX_TEST_OPENPOSE_TXT"); image <- sys.env.get("SDFX_TEST_IMAGE") do
        val source=ImageIO.read(Path.of(image).toFile)
        val (input,th,tw,ph,pw)=OpenPoseBody.prepareInput(NativeOpenPose.bgr(source),source.getHeight,source.getWidth)
        assertEquals((th,tw),(h,w))
        val diff=relErr(input,set.readFloat("input").toFloatArray)
        println(s"[openpose]   prepared input maxAbs=${"%.3e".format(diff._1)} (cv2 8-bit cubic rounding)")
        assert(diff._1<0.02)
        val detection=OpenPoseBody.postprocess(out,th-ph,tw-pw,source.getHeight,source.getWidth)
        val lines=Files.readAllLines(Path.of(txt)).toArray.map(_.toString)
        val refPeaks=lines.filter(_.startsWith("peak ")).map(_.split(" ").drop(1)).map(a=>(a(0).toInt,a(1).toInt,a(2).toInt,a(3).toDouble))
        val got=detection.peaks.zipWithIndex.flatMap((ps,part)=>ps.map(p=>(part,p.x,p.y,p.score.toDouble)))
        println(s"[openpose]   peaks scala=${got.length} reference=${refPeaks.length}")
        assertEquals(got.map(p=>(p._1,p._2,p._3)).toSeq,refPeaks.map(p=>(p._1,p._2,p._3)).toSeq)
        val scoreDiff=got.zip(refPeaks).map((a,b)=>math.abs(a._4-b._4)).maxOption.getOrElse(0.0)
        println(s"[openpose]   peak score maxAbs=${"%.3e".format(scoreDiff)}")
        assert(scoreDiff<1e-3)
        val refPeople=lines.filter(_.startsWith("person ")).map(_.split(" ").drop(1).map(_.toDouble))
        assertEquals(detection.people.length,refPeople.length)
        for (a,b) <- detection.people.zip(refPeople) do
          assertEquals(a.take(18).toSeq,b.take(18).toSeq)
          assertEquals(a(19),b(19));assert(math.abs(a(18)-b(18))<1e-3)
    finally set.close()
  }

  test("openpose preprocessor end to end on a real image") {
    val weights=sys.env.get("SDFX_TEST_OPENPOSE");val image=sys.env.get("SDFX_TEST_IMAGE");val out=sys.env.get("SDFX_TEST_OUT")
    assume(weights.isDefined && image.isDefined && out.isDefined,"real-weights gate disabled")
    val source=ImageIO.read(Path.of(image.get).toFile)
    val pose=NativeOpenPose.preprocess(source,512,()=>false,Map("SDFX_SCALATORCH_ANNOTATOR_DIR"->weightsFile(weights.get).getParent.toString))
    assertEquals((pose.getWidth,pose.getHeight),(source.getWidth,source.getHeight))
    val lit=(for y <- 0 until pose.getHeight; x <- 0 until pose.getWidth yield pose.getRGB(x,y)&0xffffff).count(_!=0)
    assert(lit>200,s"pose map is blank ($lit lit pixels)")
    Files.createDirectories(Path.of(out.get))
    val file=Path.of(out.get).resolve("pose.png");ImageIO.write(pose,"png",file.toFile)
    println(s"[openpose] wrote $file")
  }

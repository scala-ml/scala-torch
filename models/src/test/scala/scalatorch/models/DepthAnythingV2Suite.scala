package scalatorch.models

import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import scalatorch.io.SafeTensorSet
import scalatorch.models.worker.NativeDepthAnything

/** Real-weights parity against a transformers reference: set SDFX_TEST_DAV2 (weights dir) and SDFX_TEST_DAV2_REF
  * (comma-separated safetensors from the reference script). SDFX_TEST_IMAGE + SDFX_TEST_OUT also run the preprocessor end to end. */
class DepthAnythingV2Suite extends munit.FunSuite:
  private def relErr(a: Array[Float],b: Array[Float]): (Double,Double)=
    assertEquals(a.length,b.length)
    var maxAbs=0.0;var ref=0.0;var i=0
    while i<a.length do { maxAbs=math.max(maxAbs,math.abs(a(i)-b(i)));ref=math.max(ref,math.abs(b(i)));i+=1 }
    (maxAbs,maxAbs/math.max(ref,1e-12))

  test("missing weights fail with a clear error instead of a fake depth map") {
    val empty=Files.createTempDirectory("dav2-none")
    val e=intercept[IllegalStateException](NativeDepthAnything.locate(Map.empty,empty))
    assert(e.getMessage.contains("Depth Anything V2 weights not found"))
  }

  test("input size follows the HF rules") {
    assertEquals(NativeDepthAnything.inputSize(518,518),(518,518))
    assertEquals(NativeDepthAnything.inputSize(266,336),(406,518))
    assertEquals(NativeDepthAnything.inputSize(1024,768),(686,518))
  }

  test("weights match transformers DepthAnythingForDepthEstimation") {
    val weights=sys.env.get("SDFX_TEST_DAV2");val refs=sys.env.get("SDFX_TEST_DAV2_REF")
    assume(weights.isDefined && refs.isDefined,"real-weights gate disabled")
    val model=DepthAnythingV2.load(Path.of(weights.get))
    for file <- refs.get.split(",").map(_.trim).filter(_.nonEmpty) do
      val set=SafeTensorSet.openFile(Path.of(file))
      try
        def ref(name: String)=set.readFloat(name).toFloatArray
        val shape=set.tensors("pixel_values").shape.dims.map(_.toInt)
        val (h,w)=(shape(2),shape(3))
        val t0=System.nanoTime
        val out=model.infer(ref("pixel_values"),h,w)
        val seconds=(System.nanoTime-t0)/1e9
        val rows=Vector.newBuilder[(String,(Double,Double))]
        rows+=("embeddings"->relErr(out.embeddings,ref("embeddings")))
        for i <- Vector(3,6,9,12) do rows+=(s"hs$i"->relErr(out.hiddenStates(i),ref(s"hs$i")))
        for i <- 0 until 4 do rows+=(s"neck$i"->relErr(out.neck(i).data,ref(s"neck$i")))
        val depth=relErr(out.depth,ref("predicted_depth"))
        rows+=("depth"->depth)
        println(s"[dav2] $file ${w}x$h in ${"%.1f".format(seconds)}s")
        rows.result().foreach((n,e)=>println(s"[dav2]   $n maxAbs=${"%.3e".format(e._1)} rel=${"%.3e".format(e._2)}"))
        assert(depth._2<1e-4,s"depth relative error ${depth._2}")
      finally set.close()
  }

  test("depth preprocessor end to end on a real image") {
    val weights=sys.env.get("SDFX_TEST_DAV2");val image=sys.env.get("SDFX_TEST_IMAGE");val out=sys.env.get("SDFX_TEST_OUT")
    assume(weights.isDefined && image.isDefined && out.isDefined,"real-weights gate disabled")
    val source=ImageIO.read(Path.of(image.get).toFile)
    val depth=NativeDepthAnything.preprocess(source,()=>false,Map("SDFX_SCALATORCH_ANNOTATOR_DIR"->weights.get))
    assertEquals((depth.getWidth,depth.getHeight),(source.getWidth,source.getHeight))
    val values=(for y <- 0 until depth.getHeight; x <- 0 until depth.getWidth yield depth.getRGB(x,y)&255)
    assertEquals(values.min,0);assertEquals(values.max,255)
    Files.createDirectories(Path.of(out.get))
    val file=Path.of(out.get).resolve("depth.png");ImageIO.write(depth,"png",file.toFile)
    println(s"[dav2] wrote $file")
  }

package scalatorch.models

import java.awt.image.BufferedImage
import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import scalatorch.io.SafeTensorSet
import scalatorch.models.worker.NativeAnnotatorFiles

/** Aesthetic scorers (improved aesthetic predictor, Simulacra, cafe BEiT classifiers). The structural tests always run. Real-weights parity needs
  * the converted weights reachable through SDFX_SCALATORCH_ANNOTATOR_DIR (a folder holding aesthetic folder of safetensors files from
  * scripts/convert_aesthetic_to_safetensors.py), SDFX_TEST_AESTHETIC_REF (folder written by scripts/reference_aesthetic.py, the torch/transformers oracle)
  * and SDFX_TEST_AESTHETIC_IMAGES (comma-separated images the oracle ran on; the reference file is aesthetic_ref_<stem>.safetensors).
  * Licences: LAION improved predictor (Apache-2.0 repo, MLP weights without a separate statement), Simulacra models (MIT code, weights without a separate
  * statement), OpenAI CLIP (MIT), cafeai classifiers (AGPL-3.0). */
class AestheticModelsSuite extends munit.FunSuite:
  override val munitTimeout: scala.concurrent.duration.Duration = scala.concurrent.duration.Duration(30, "min") // real-weights parity runs CPU transformers
  private def relErr(a: Array[Float],b: Array[Float]): (Double,Double)=
    assertEquals(a.length,b.length)
    var maxAbs=0.0;var ref=0.0;var i=0
    while i<a.length do { maxAbs=math.max(maxAbs,math.abs(a(i)-b(i)));ref=math.max(ref,math.abs(b(i)));i+=1 }
    (maxAbs,maxAbs/math.max(ref,1e-12))

  test("missing aesthetic weights fail with a clear error instead of invented scores") {
    val none=Files.createTempDirectory("aes-none")
    for f <- Vector(AestheticModels.ClipLargeFile,AestheticModels.ImprovedFile,AestheticModels.simulacraFile("b16"),AestheticModels.cafeFile("cafe_style")) do
      val e=intercept[IllegalStateException](NativeAnnotatorFiles.locate(f,"aesthetic",Map.empty,none))
      assert(e.getMessage.contains("weights not found"),e.getMessage)
  }

  test("BEiT relative position index follows the HF layout") {
    val g=BeitClassifier.Grid;val idx=BeitClassifier.relativePositionIndex();val n=BeitClassifier.Tokens;val d=(2*g-1)*(2*g-1)+3
    assertEquals(idx.length,n*n)
    assertEquals(idx(0),d-1);assertEquals(idx(5),d-3);assertEquals(idx(3*n),d-2)
    val centre=(g-1)*(2*g-1)+(g-1)
    assertEquals(idx(7*n+7),centre)
    assertEquals(idx(2*n+1),centre+1)        // token (0,1) minus token (0,0): dx=+1
    assertEquals(idx(1*n+2),centre-1)        // reversed
  }

  test("CLIP preprocessing resizes the shorter side, centre-crops and normalises") {
    val img=new BufferedImage(40,20,BufferedImage.TYPE_INT_RGB)
    for y <- 0 until 20; x <- 0 until 40 do img.setRGB(x,y,0x808080)
    val px=AestheticModels.clipPixels(img,8)
    assertEquals(px.length,3*8*8)
    assertEqualsDouble(px(0).toDouble,((128/255.0)-0.48145466)/0.26862954,2e-3)
    assertEquals(AestheticModels.cafePixels(img).length,3*384*384)
  }

  test("scores and probabilities are real: parity with torch / transformers") {
    val refDir=sys.env.get("SDFX_TEST_AESTHETIC_REF");val images=sys.env.get("SDFX_TEST_AESTHETIC_IMAGES")
    assume(refDir.isDefined && images.isDefined,"real-weights gate disabled")
    for path <- images.get.split(",").map(_.trim).filter(_.nonEmpty) do
      val stem=Path.of(path).getFileName.toString.replaceAll("\\.[^.]*$","")
      val set=SafeTensorSet.openFile(Path.of(refDir.get).resolve(s"aesthetic_ref_$stem.safetensors"))
      try
        def ref(n: String)=set.readFloat(n).toFloatArray
        val image=ImageIO.read(Path.of(path).toFile)
        val log=Vector.newBuilder[String]
        def row(name: String,a: Array[Float],r: Array[Float],tol: Double): Unit=
          val (abs,rel)=relErr(a,r);log+=f"  $name%-16s maxAbs=$abs%.3e rel=$rel%.3e"
          assert(rel<tol,s"$stem $name relative error $rel")
        def scalar(name: String,v: Double,r: Float,tol: Double): Unit=
          val rel=math.abs(v-r)/math.max(math.abs(r),1e-6);log+=f"  $name%-16s scala=$v%.6f torch=$r%.6f rel=$rel%.3e"
          assert(rel<tol,s"$stem $name relative error $rel")
        // pixel pipelines (8-bit resize parity: one 8-bit level is about 0.015 in normalised units)
        val (pAbs,_)=relErr(AestheticModels.clipPixels(image),ref("clip_pixels"))
        val (sAbs,_)=relErr(AestheticModels.clipPixels(image,224,lanczos=true),ref("sim_pixels_l14"))
        val (cAbs,_)=relErr(AestheticModels.cafePixels(image),ref("cafe_pixels"))
        log+=f"  pixels maxAbs clip=$pAbs%.3e simulacra=$sAbs%.3e cafe=$cAbs%.3e"
        assert(pAbs<0.02 && sAbs<0.02 && cAbs<0.02,s"$stem preprocessing differs from the reference: $pAbs $sAbs $cAbs")
        // CLIP L/14 embedding on the reference pixels (isolates the network from the resize) and the improved MLP
        val clipL=ClipVision.load(NativeAnnotatorFiles.locate(AestheticModels.ClipLargeFile,"CLIP ViT-L/14"),ClipVisionConfig.L14)
        val t0=System.nanoTime
        val emb=clipL.embed(ref("clip_pixels"))
        val seconds=(System.nanoTime-t0)/1e9
        row("clip_embed",emb,ref("clip_embed"),1e-4)
        val unit={val n=math.sqrt(emb.map(x=>x.toDouble*x).sum).toFloat;emb.map(_/n)}
        scalar("improved(ref px)",AestheticModels.improvedHead(unit),ref("improved")(0),1e-4)
        scalar("improved(e2e)",AestheticModels.improved(image),ref("improved")(0),5e-3)
        val clipB=ClipVision.load(NativeAnnotatorFiles.locate(AestheticModels.ClipBaseFile,"CLIP ViT-B/16"),ClipVisionConfig.B16)
        row("sim_embed_b16",clipB.embed(ref("sim_pixels_b16")),ref("sim_embed_b16"),1e-4)
        row("sim_embed_l14",clipL.embed(ref("sim_pixels_l14")),ref("sim_embed_l14"),1e-4)
        scalar("simulacra_l14",AestheticModels.simulacra(image,"l14"),ref("sim_l14")(0),5e-3)
        scalar("simulacra_b16",AestheticModels.simulacra(image,"b16"),ref("sim_b16")(0),5e-3)
        for name <- Vector("cafe_aesthetic","cafe_style","cafe_waifu") do
          val net=BeitClassifier.load(NativeAnnotatorFiles.locate(AestheticModels.cafeFile(name),name))
          row(s"$name logits",net.logits(ref("cafe_pixels")),ref(name),1e-4)
          val probs=AestheticModels.cafe(image,name,AestheticScore.CafeLabels(name))
          val refP=ref(s"${name}_probs")
          probs.zipWithIndex.foreach((lp,i)=>assert(math.abs(lp._2-refP(i))<5e-3,s"$stem $name ${lp._1}: ${lp._2} vs ${refP(i)}"))
        println(s"[aesthetic] $stem (CLIP-L forward ${"%.1f".format(seconds)}s)\n"+log.result().mkString("\n"))
      finally set.close()
  }

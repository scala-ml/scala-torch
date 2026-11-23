package scalatorch.models

import java.awt.{BasicStroke,Color}
import java.awt.image.BufferedImage
import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import scalatorch.io.SafeTensorSet
import scalatorch.models.worker.NativeAnnotatorFiles

/** NudeNet v3 (320n) tests: the non-explicit paths are always checked; real-weights parity needs
  *  SDFX_TEST_ANNOTATORS (a folder with nudenet/320n.safetensors from scripts/convert_nudenet_onnx_to_safetensors.py),
  *  SDFX_TEST_NUDENET_REF (folder written by scripts/reference_nudenet.py, onnxruntime oracle) and SDFX_TEST_NUDENET_IMAGE (the image it ran on).
  *  The end-to-end censoring test needs the same weights reachable through SDFX_SCALATORCH_ANNOTATOR_DIR (or models/annotators) and SDFX_TEST_NUDENET_IMAGE.
  *  SDFX_TEST_OUT writes the detection overlay and the censored image.
  *  Licence: NudeNet and its weights are AGPL-3.0; only safe-for-work images are used here. */
class NudeNetSuite extends munit.FunSuite:
  private def gate: Option[(Path,Path,Path)]=
    for a <- sys.env.get("SDFX_TEST_ANNOTATORS");r <- sys.env.get("SDFX_TEST_NUDENET_REF");i <- sys.env.get("SDFX_TEST_NUDENET_IMAGE") yield (Path.of(a),Path.of(r),Path.of(i))

  test("class order of the checker matches NudeNet v3's documented 18 classes") {
    val upstream=Vector("FEMALE_GENITALIA_COVERED","FACE_FEMALE","BUTTOCKS_EXPOSED","FEMALE_BREAST_EXPOSED","FEMALE_GENITALIA_EXPOSED","MALE_BREAST_EXPOSED",
      "ANUS_EXPOSED","FEET_EXPOSED","BELLY_COVERED","FEET_COVERED","ARMPITS_COVERED","ARMPITS_EXPOSED","FACE_MALE","BELLY_EXPOSED","MALE_GENITALIA_EXPOSED",
      "ANUS_COVERED","FEMALE_BREAST_COVERED","BUTTOCKS_COVERED")
    assertEquals(NsfwChecker.Labels.map(_.toUpperCase),upstream)
    assert(Adetailer.isImplemented("nudenet") && !Adetailer.isImplemented("u2net"))
  }

  test("square preprocessing pads bottom/right with zeros and resizes to the model size") {
    val img=new BufferedImage(40,20,BufferedImage.TYPE_INT_RGB)
    for y <- 0 until 20; x <- 0 until 40 do img.setRGB(x,y,0xffffff)
    val lb=YoloV8.letterboxSquare(img,32)
    assertEquals((lb.width,lb.height,lb.gain,lb.padLeft,lb.padTop),(32,32,0.8,0,0))
    assertEqualsDouble(lb.chw(5*32+5).toDouble,1.0,1e-6)      // content
    assertEqualsDouble(lb.chw(30*32+5).toDouble,0.0,1e-6)     // zero padding below
    assertEqualsDouble(lb.chw(2*32*32+31*32+31).toDouble,0.0,1e-6)
  }

  test("class-agnostic NMS suppresses overlapping boxes of different classes") {
    val box=Array[Float](0,0,10,10, 0.5f,0.5f,10.5f,10.5f)
    val p=YoloV8.Predictions(2,box,Array.empty,Array(0.9f,0.8f),Array(0,1),Array.empty,2)
    assertEquals(YoloV8.nms(p,0.25f,0.45f,300),Vector(0,1))
    assertEquals(YoloV8.nms(p,0.25f,0.45f,300,agnostic=true),Vector(0))
  }

  test("missing NudeNet weights fail loudly") {
    val none=Files.createTempDirectory("nn-none")
    val e=intercept[IllegalStateException](NativeAnnotatorFiles.locate(Adetailer.NudeNetWeights,"NudeNet v3 320n",Map.empty,none))
    assert(e.getMessage.contains("weights not found"))
  }

  private def relErr(a: Array[Float],ref: Array[Float]): (Double,Double)=
    require(a.length==ref.length,s"length ${a.length} vs ${ref.length}")
    var maxAbs=0.0;var refMax=0.0;var i=0
    while i<a.length do { maxAbs=math.max(maxAbs,math.abs(a(i)-ref(i)));refMax=math.max(refMax,math.abs(ref(i)));i+=1 }
    (maxAbs,maxAbs/refMax)

  private def overlay(image: BufferedImage,det: Adetailer.Detection): BufferedImage=
    val canvas=new BufferedImage(image.getWidth,image.getHeight,BufferedImage.TYPE_INT_RGB);val g=canvas.createGraphics()
    g.drawImage(image,0,0,null);g.setStroke(new BasicStroke(2f))
    for ((b,k) <- det.boxes.zipWithIndex) do
      g.setColor(Color.RED);g.drawRect(b._1.toInt,b._2.toInt,(b._3-b._1).toInt,(b._4-b._2).toInt)
      g.setColor(Color.YELLOW);g.drawString(f"${NsfwChecker.Labels(det.classes(k))} ${det.confidences(k)}%.2f",b._1.toInt+3,b._2.toInt+13)
    g.dispose();canvas

  test("320n matches the onnxruntime reference (raw 22x2100 output) and yields the same detections") {
    assume(gate.isDefined,"real-weights gate disabled")
    val (a,r,imgPath)=gate.get
    val net=YoloV8.load(a.resolve(Adetailer.NudeNetWeights))
    assertEquals((net.spec.nc,net.spec.regMax,net.spec.segment),(18,16,false))
    val image=ImageIO.read(imgPath.toFile)
    val set=SafeTensorSet.openFile(r.resolve("nudenet_ref.safetensors"))
    try
      val refInput=set.readFloat("input").toFloatArray
      val lb=YoloV8.letterboxSquare(image,320)
      val (inAbs,_)=relErr(lb.chw,refInput)
      println(f"[nudenet] preprocessing vs cv2 input: maxAbs=$inAbs%.3e (1 grey level = ${1/255.0}%.3e)")
      // parity on the exact tensor fed to onnxruntime
      val raw=net.forward(refInput,320,320)
      val pred=YoloV8.decode(raw,320,320);val n=pred.n;val nc=pred.nc
      assertEquals((n,nc),(2100,18))
      val ours=new Array[Float]((4+nc)*n)
      for i <- 0 until n do
        val x1=pred.box(4*i);val y1=pred.box(4*i+1);val x2=pred.box(4*i+2);val y2=pred.box(4*i+3)
        ours(i)=(x1+x2)/2;ours(n+i)=(y1+y2)/2;ours(2*n+i)=x2-x1;ours(3*n+i)=y2-y1
        for c <- 0 until nc do ours((4+c)*n+i)=pred.scores(i*nc+c)
      val ref=set.readFloat("output").toFloatArray
      val (abs,rel)=relErr(ours,ref)
      // boxes and scores separately: the score block is the sensitive one (most entries are near zero)
      val (_,relBox)=relErr(ours.take(4*n),ref.take(4*n));val (absSc,relSc)=relErr(ours.drop(4*n),ref.drop(4*n))
      println(f"[nudenet] raw output 22x$n maxAbs=$abs%.2e rel=$rel%.2e (box rel=$relBox%.2e, scores maxAbs=$absSc%.2e rel=$relSc%.2e)")
      assert(rel<1e-4,s"relative error vs onnxruntime $rel")

      // end-to-end detections vs the documented post-processing of the reference
      val json=Files.readString(r.resolve("nudenet_ref.json"))
      def nums(key: String): Array[Double]=
        val body=("\""+key+"""": (\[[^a-z]*?\])(?=, "|\})""").r.findFirstMatchIn(json).map(_.group(1)).getOrElse("")
        """-?[0-9.]+(?:[eE][-+]?[0-9]+)?""".r.findAllIn(body).map(_.toDouble).toArray
      val refBoxes=nums("boxes").grouped(4).toVector;val refScores=nums("scores");val refCls=nums("cls").map(_.toInt)
      val det=Adetailer.detect(image,"nudenet",0.2)
      println(s"[nudenet] detections scala=${det.boxes.size} reference=${refBoxes.size}: "+det.classes.zip(det.confidences).map((c,s)=>f"${NsfwChecker.Labels(c)}%s ${s}%.3f").mkString(", "))
      assertEquals(det.boxes.size,refBoxes.size)
      assertEquals(det.classes,refCls.toVector)
      for (((b,s),(rb,rs))) <- det.boxes.zip(det.confidences).zip(refBoxes.zip(refScores)) do
        val d=Seq(b._1-rb(0),b._2-rb(1),b._3-rb(2),b._4-rb(3)).map(math.abs).max
        println(f"[nudenet] box diff $d%.3f px, score diff ${math.abs(s-rs)}%.2e")
        assert(d<1.0 && math.abs(s-rs)<1e-2,"differences stem from cv2 fixed-point resize (max 1 grey level), raw outputs match to 1e-6")
      sys.env.get("SDFX_TEST_OUT").foreach { o =>
        Files.createDirectories(Path.of(o));ImageIO.write(overlay(image,det),"png",Path.of(o).resolve("nudenet_detections.png").toFile)
      }
    finally set.close()
  }

  test("nsfw_checker censors exactly the regions of classes whose threshold is met (real NudeNet weights)") {
    val img=sys.env.get("SDFX_TEST_NUDENET_IMAGE").map(p=>ImageIO.read(Path.of(p).toFile))
    val located=scala.util.Try(NativeAnnotatorFiles.locate(Adetailer.NudeNetWeights,"NudeNet v3 320n")).isSuccess
    assume(img.isDefined && located,"real-weights gate disabled")
    val image=img.get
    def same(a: BufferedImage,b: BufferedImage)=(0 until a.getHeight).forall(y=>(0 until a.getWidth).forall(x=>a.getRGB(x,y)==b.getRGB(x,y)))
    val det=Adetailer.detect(image,"nudenet",0.1)
    val labels=NsfwChecker.Labels
    // (a) NUI defaults: only anus / female breast / female genitalia / male genitalia exposed are active
    val defaults=NsfwChecker.NsfwConfig(genFilterType="Fill color",fillColor="#000000",maskShape="Rectangle",maskBlendRadius=0.0)
    val hitDefault=det.boxes.indices.exists(i=>det.confidences(i)>=defaults.thresholds(det.classes(i)))
    val (outDefault,auxDefault)=NsfwChecker.censorFinal(image,defaults)
    println(s"[nudenet] defaults: active detections=$hitDefault, image changed=${!same(image,outDefault)}")
    assertEquals(!same(image,outDefault),hitDefault)
    if !hitDefault then assert(auxDefault.isEmpty)
    // (b) only non-explicit body-part classes enabled (belly / armpits / feet / face exposed) so the effect is visible on a swimsuit render
    val enabled=Set("Belly_exposed","Armpits_exposed","Feet_exposed","Face_female")
    val th=labels.map(l=>if enabled(l) then 0.3f else 1.0f)
    val cfg=defaults.copy(thresholds=th,saveBeforeCensor=true)
    val active=det.boxes.indices.filter(i=>det.confidences(i)>=th(det.classes(i)))
    assert(active.nonEmpty,"expected at least one belly/armpit/feet/face detection on the swimsuit render")
    val (out,aux)=NsfwChecker.censorFinal(image,cfg)
    val boxes=active.map(det.boxes)
    def inBox(x: Int,y: Int,m: Double)=boxes.exists(b=>x>=b._1-m && x<=b._3+m && y>=b._2-m && y<=b._4+m)
    var inside=0;var outsideChanged=0;var insideBlack=0
    for y <- 0 until image.getHeight; x <- 0 until image.getWidth do
      val changed=out.getRGB(x,y)!=image.getRGB(x,y)
      if !inBox(x,y,2.0) && changed then outsideChanged+=1
      if inBox(x,y,-1.0) then { inside+=1;if (out.getRGB(x,y)&0xffffff)==0 then insideBlack+=1 }
    println(s"[nudenet] censoring ${active.size} regions (${active.map(i=>labels(det.classes(i))).mkString(",")}): inside pixels=$inside black=$insideBlack, changed outside=$outsideChanged")
    assertEquals(outsideChanged,0)
    assertEquals(insideBlack,inside)
    assertEquals(aux.map(_._1),Vector("-before-censor"))
    assert(same(image,aux.head._2))
    // (c) everything switched off -> no detector call needed, untouched
    val (untouched,none)=NsfwChecker.censorFinal(image,defaults.copy(thresholds=labels.map(_=>1.0f)))
    assert(same(image,untouched) && none.isEmpty)
    sys.env.get("SDFX_TEST_OUT").foreach { o =>
      Files.createDirectories(Path.of(o))
      ImageIO.write(out,"png",Path.of(o).resolve("nudenet_censored_fill.png").toFile)
      val blur=NsfwChecker.censorFinal(image,cfg.copy(genFilterType="Gaussian Blur",maskShape="Ellipse",maskBlendRadius=5.0))._1
      ImageIO.write(blur,"png",Path.of(o).resolve("nudenet_censored_blur.png").toFile)
      val pix=NsfwChecker.censorFinal(image,cfg.copy(genFilterType="Pixelate",maskShape="Rectangle"))._1
      ImageIO.write(pix,"png",Path.of(o).resolve("nudenet_censored_pixelate.png").toFile)
    }
  }

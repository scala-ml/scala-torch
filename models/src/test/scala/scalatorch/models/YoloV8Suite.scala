package scalatorch.models

import java.awt.{BasicStroke,Color}
import java.awt.image.BufferedImage
import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import scalatorch.{Shape,Tensor}
import scalatorch.io.{SafeTensorSet,SafeTensorWrite,SafeTensorsWriter}
import scalatorch.models.worker.{NativeAnnotatorFiles,NativeBackendError}

/** YOLOv8 detector tests. Synthetic ones always run. Real-weights parity against scripts/reference_yolov8.py is gated:
  * SDFX_TEST_ANNOTATORS (directory containing adetailer/<name>.safetensors, written by scripts/convert_yolov8_to_safetensors.py),
  * SDFX_TEST_YOLO_REF (directory with <name>_ref.safetensors / _ref.json / _masks.npy) and SDFX_TEST_YOLO_IMAGE (the image the reference ran on).
  * SDFX_TEST_OUT additionally writes the detections drawn on the image (<name>_detections.png). */
class YoloV8Suite extends munit.FunSuite:
  private def gate: Option[(Path,Path,Path)]=
    for a <- sys.env.get("SDFX_TEST_ANNOTATORS");r <- sys.env.get("SDFX_TEST_YOLO_REF");i <- sys.env.get("SDFX_TEST_YOLO_IMAGE") yield (Path.of(a),Path.of(r),Path.of(i))

  private def f32(shape: Seq[Int],value: Float=>Float=identity): SafeTensorWrite=
    val n=shape.product;val r=new java.util.Random(n+shape.hashCode)
    SafeTensorWrite.F32(Tensor.fromFloats(Array.tabulate(n)(_=>value(r.nextFloat()*2f-1f)),Shape.from(shape.map(_.toLong))))

  /** A complete tiny YOLOv8-style checkpoint (n=1 repeats everywhere), built name by name from the architecture. */
  private def tiny(seg: Boolean,mutate: scala.collection.mutable.LinkedHashMap[String,SafeTensorWrite]=>Unit=_=>()): Path=
    val m=scala.collection.mutable.LinkedHashMap.empty[String,SafeTensorWrite]
    def cbn(name: String,ci: Int,co: Int,k: Int): Unit=
      m(name+".conv.weight")=f32(Seq(co,ci,k,k),_*0.2f);m(name+".bn.weight")=f32(Seq(co),v=>0.5f+0.4f*v);m(name+".bn.bias")=f32(Seq(co),_*0.1f)
      m(name+".bn.running_mean")=f32(Seq(co),_*0.1f);m(name+".bn.running_var")=f32(Seq(co),v=>1f+0.5f*v)
    def c2f(name: String,c1: Int,c2: Int): Unit=
      val h=c2/2;cbn(name+".cv1",c1,2*h,1);cbn(name+".cv2",3*h,c2,1);cbn(name+".m.0.cv1",h,h,3);cbn(name+".m.0.cv2",h,h,3)
    def plain(name: String,ci: Int,co: Int): Unit={ m(name+".weight")=f32(Seq(co,ci,1,1),_*0.2f);m(name+".bias")=f32(Seq(co),_*0.1f) }
    val (s0,s1,s2,s3,s4)=(4,8,16,16,32);val nc=2;val rm=4;val nm=4;val npr=8;val c2=8
    cbn("model.0",3,s0,3);cbn("model.1",s0,s1,3);c2f("model.2",s1,s1);cbn("model.3",s1,s2,3);c2f("model.4",s2,s2)
    cbn("model.5",s2,s3,3);c2f("model.6",s3,s3);cbn("model.7",s3,s4,3);c2f("model.8",s4,s4)
    cbn("model.9.cv1",s4,s4/2,1);cbn("model.9.cv2",2*s4,s4,1)
    c2f("model.12",s4+s3,s3);c2f("model.15",s3+s2,s2);cbn("model.16",s2,s2,3);c2f("model.18",s2+s3,s3);cbn("model.19",s3,s3,3);c2f("model.21",s3+s4,s4)
    for (c,i) <- Seq(s2,s3,s4).zipWithIndex do
      cbn(s"model.22.cv2.$i.0",c,c2,3);cbn(s"model.22.cv2.$i.1",c2,c2,3);plain(s"model.22.cv2.$i.2",c2,4*rm)
      cbn(s"model.22.cv3.$i.0",c,c2,3);cbn(s"model.22.cv3.$i.1",c2,c2,3);plain(s"model.22.cv3.$i.2",c2,nc)
      if seg then { cbn(s"model.22.cv4.$i.0",c,c2,3);cbn(s"model.22.cv4.$i.1",c2,c2,3);plain(s"model.22.cv4.$i.2",c2,nm) }
    m("model.22.dfl.conv.weight")=SafeTensorWrite.F32(Tensor.fromFloats(Array.tabulate(rm)(_.toFloat),Shape.from(Seq(1L,rm.toLong,1L,1L))))
    if seg then
      cbn("model.22.proto.cv1",s2,npr,3);m("model.22.proto.upsample.weight")=f32(Seq(npr,npr,2,2),_*0.2f);m("model.22.proto.upsample.bias")=f32(Seq(npr),_*0.1f)
      cbn("model.22.proto.cv2",npr,npr,3);cbn("model.22.proto.cv3",npr,nm,1)
    mutate(m)
    val path=Files.createTempFile("tiny-yolo",".safetensors");SafeTensorsWriter.write(path,m.toMap,overwrite=true);path

  test("a synthetic checkpoint loads strictly and runs end to end (detect and segment)") {
    for seg <- Seq(false,true) do
      val f=tiny(seg)
      try
        val net=YoloV8.load(f)
        assertEquals((net.spec.nc,net.spec.regMax,net.spec.nm,net.spec.segment),(2,4,if seg then 4 else 0,seg))
        val raw=net.forward(Array.tabulate(3*64*96)(i=>(i%251)/251f),64,96)
        assertEquals(raw.box.map(b=>(b._2,b._3,b._4)),Vector((16,8,12),(16,4,6),(16,2,3)))
        assertEquals(raw.cls.map(_._2),Vector(2,2,2))
        assertEquals(raw.proto.map(p=>(p._2,p._3,p._4)),if seg then Some((4,16,24)) else None)
        val pred=YoloV8.decode(raw,64,96)
        assertEquals(pred.n,8*12+4*6+2*3)
        assert(pred.box.forall(!_.isNaN) && pred.score.forall(s=>s>=0f && s<=1f))
      finally Files.deleteIfExists(f)
  }

  test("loader rejects unknown, missing and mis-shaped tensors") {
    val extra=tiny(false,_("model.22.stray.weight")=SafeTensorWrite.F32(Tensor.fromFloats(Array(1f),Shape.from(Seq(1L)))))
    val gone=tiny(false,_.remove("model.4.cv2.bn.bias"))
    val bad=tiny(false,m=>m("model.3.conv.weight")=f32(Seq(16,8,1,1)))
    val ramp=tiny(false,_("model.22.dfl.conv.weight")=f32(Seq(1,4,1,1)))
    try
      assert(intercept[IllegalArgumentException](YoloV8.load(extra)).getMessage.contains("unexpected YOLOv8 weights: model.22.stray.weight"))
      assert(intercept[IllegalArgumentException](YoloV8.load(gone)).getMessage.contains("missing YOLOv8 weight: model.4.cv2.bn.bias"))
      assert(intercept[IllegalArgumentException](YoloV8.load(bad)).getMessage.contains("shape mismatch"))
      assert(intercept[IllegalArgumentException](YoloV8.load(ramp)).getMessage.contains("DFL"))
    finally List(extra,gone,bad,ramp).foreach(Files.deleteIfExists)
  }

  test("letterbox pads to a multiple of 32 with 114 grey and keeps the aspect") {
    val img=new BufferedImage(200,105,BufferedImage.TYPE_INT_RGB)
    for y <- 0 until 105; x <- 0 until 200 do img.setRGB(x,y,0xff0000)
    val lb=YoloV8.letterbox(img,640)
    assertEquals((lb.width,lb.height,lb.gain),(640,352,3.2)) // 640x336 content, 16 rows of padding split 8/8
    assertEquals((lb.padTop,lb.padBottom,lb.padLeft,lb.padRight),(8,8,0,0))
    assertEquals(lb.chw(0),114f/255f) // top-left pixel is padding
    assertEqualsDouble(lb.chw(8*640).toDouble,1.0,1e-6)  // first content row, red channel
    assertEqualsDouble(lb.chw(352*640+8*640).toDouble,0.0,1e-6) // green channel, content
  }

  test("NMS is class-wise, best first, and honours the threshold and maxDet") {
    // 4 anchors: two heavily overlapping of class 0, one overlapping of class 1, one far away below threshold
    val box=Array[Float](0,0,10,10, 0.5f,0.5f,10.5f,10.5f, 0,0,10,10, 50,50,60,60)
    val p=YoloV8.Predictions(4,box,Array.empty,Array(0.9f,0.8f,0.7f,0.1f),Array(0,0,1,0),Array.empty,2)
    assertEquals(YoloV8.nms(p,0.25f,0.7f,300),Vector(0,2))
    assertEquals(YoloV8.nms(p,0.25f,0.7f,1),Vector(0))
    assertEquals(YoloV8.nms(p,0.05f,0.7f,300),Vector(0,2,3))
  }

  test("mask = sigmoid(coefficients x prototypes) cropped to the box, mapped back to the image") {
    val ph=8;val pw=8;val proto=(Array.fill(ph*pw)(1f),1,ph,pw)
    val lb=YoloV8.Letterboxed(Array.empty,64,64,1.0,0,0,0,0)
    val m=YoloV8.mask(proto,Array(2f),(16f,16f,40f,48f),lb,64,64) // proto cells 2..4 x 2..5 are inside
    assertEquals(m(30*64+30),255)
    assertEquals(m(2*64+2),0)
    assertEquals(m(60*64+60),0)
    val negative=YoloV8.mask(proto,Array(-2f),(16f,16f,40f,48f),lb,64,64)
    assertEquals(negative.count(_>0),0)
  }

  test("missing weights fail with a clear error and unimplemented models are refused, never faked") {
    val none=Files.createTempDirectory("yolo-none")
    val e=intercept[IllegalStateException](NativeAnnotatorFiles.locate("adetailer/face_yolov8n.safetensors","YOLOv8 face_yolov8n",Map.empty,none))
    assert(e.getMessage.contains("weights not found"))
    val img=new BufferedImage(32,32,BufferedImage.TYPE_INT_RGB)
    for model <- Seq("nudenet","u2net","opencv_face","face_yolov9c.pt") do
      val u=intercept[NativeBackendError](Adetailer.detect(img,model,0.3))
      assertEquals(u.code,"unsupported");assert(u.getMessage.contains("not implemented"))
    assert(Adetailer.isImplemented("person_yolov8n-seg.pt") && !Adetailer.isImplemented("nudenet"))
  }

  private def relErr(a: Array[Float],ref: Array[Float]): (Double,Double)=
    require(a.length==ref.length,s"length ${a.length} vs ${ref.length}")
    var maxAbs=0.0;var refMax=0.0;var i=0
    while i<a.length do { maxAbs=math.max(maxAbs,math.abs(a(i)-ref(i)));refMax=math.max(refMax,math.abs(ref(i)));i+=1 }
    (maxAbs,maxAbs/refMax)

  private def realCase(name: String): Unit=
    test(s"$name matches the torch/ultralytics reference (raw head outputs, decoded predictions, boxes and masks)") {
      assume(gate.isDefined,"real-weights gate disabled")
      val (a,r,imgPath)=gate.get
      val net=YoloV8.load(a.resolve("adetailer").resolve(name+".safetensors"))
      val set=SafeTensorSet.openFile(r.resolve(name+"_ref.safetensors"))
      try
        val input=set.readFloat("input").toFloatArray
        val t0=System.nanoTime
        val raw=net.forward(input,640,640)
        val seconds=(System.nanoTime-t0)/1e9
        var worst=0.0;var worst64=0.0
        def check(label: String,got: Array[Float],key: String): Unit=
          val (abs32,rel32)=relErr(got,set.readFloat(key).toFloatArray);val (_,rel64)=relErr(got,set.readFloat(key+"_f64").toFloatArray)
          println(s"[$name] $label maxAbs=${"%.2e".format(abs32)} rel(f32)=${"%.2e".format(rel32)} rel(f64)=${"%.2e".format(rel64)}")
          worst=math.max(worst,rel32);worst64=math.max(worst64,rel64)
        for i <- 0 until 3 do { check(s"box$i",raw.box(i)._1,s"box$i");check(s"cls$i",raw.cls(i)._1,s"cls$i") }
        for i <- raw.coef.indices do check(s"coef$i",raw.coef(i)._1,s"coef$i")
        raw.proto.foreach(p=>check("proto",p._1,"proto"))
        // decoded (xywh, sigmoid scores, coefficients), compared in the layout of the reference `pred` (channels x anchors)
        val pred=YoloV8.decode(raw,640,640);val n=pred.n;val nc=pred.nc;val nm=raw.coef.headOption.map(_._2).getOrElse(0)
        val ours=new Array[Float]((4+nc+nm)*n)
        for i <- 0 until n do
          val x1=pred.box(4*i);val y1=pred.box(4*i+1);val x2=pred.box(4*i+2);val y2=pred.box(4*i+3)
          ours(i)=(x1+x2)/2;ours(n+i)=(y1+y2)/2;ours(2*n+i)=x2-x1;ours(3*n+i)=y2-y1
          for c <- 0 until nc do ours((4+c)*n+i)=pred.scores(i*nc+c)
          for c <- 0 until nm do ours((4+nc+c)*n+i)=pred.coef(i)(c)
        val (abs,rel)=relErr(ours,set.readFloat("pred").toFloatArray)
        println(s"[$name] decoded ${4+nc+nm}x$n maxAbs=${"%.2e".format(abs)} rel=${"%.2e".format(rel)}; forward ${"%.1f".format(seconds)}s")
        assert(worst<1e-4,s"$name raw head output relative error vs float32 $worst")
        assert(worst64<1e-4,s"$name raw head output relative error vs float64 $worst64")
        assert(rel<1e-4,s"$name decoded predictions relative error $rel")

        // final detections against predict()
        val json=Files.readString(r.resolve(name+"_ref.json"))
        def nums(key: String): Array[Double]=
          val body=("\""+key+"""": (\[[^a-z]*?\])(?=, "|\})""").r.findFirstMatchIn(json).map(_.group(1)).getOrElse("")
          """-?[0-9.]+(?:[eE][-+]?[0-9]+)?""".r.findAllIn(body).map(_.toDouble).toArray
        val refBoxes=nums("boxes").grouped(4).toVector;val refScores=nums("scores")
        val image=ImageIO.read(imgPath.toFile)
        val det=net.detect(image,0.3,withMasks=true)
        println(s"[$name] detections scala=${det.boxes.size} reference=${refBoxes.size} scores=${det.scores.map(s=>"%.3f".format(s)).mkString(",")}")
        assertEquals(det.boxes.size,refBoxes.size)
        for ((b,s),(rb,rs)) <- det.boxes.zip(det.scores).zip(refBoxes.zip(refScores)) do
          val d=Seq(b._1-rb(0),b._2-rb(1),b._3-rb(2),b._4-rb(3)).map(math.abs).max
          println(s"[$name] box diff ${"%.3f".format(d)} px, score diff ${"%.2e".format(math.abs(s-rs))}")
          assert(d<1.0,s"box differs by $d px");assert(math.abs(s-rs)<5e-3,"score differs (preprocessing resize rounding aside, raw outputs match to 1e-6)")
        val maskFile=r.resolve(name+"_masks.npy")
        if det.masks.isDefined && Files.exists(maskFile) then
          val bytes=Files.readAllBytes(maskFile);val hdr=10+((bytes(8)&255)|((bytes(9)&255)<<8));val wh=image.getWidth*image.getHeight
          for (m,k) <- det.masks.get.zipWithIndex do
            var inter=0;var union=0;var i=0
            while i<wh do { val ref=(bytes(hdr+k*wh+i)&255)>0;val ours=m(i)>0;if ref&&ours then inter+=1;if ref||ours then union+=1;i+=1 }
            val iou=inter.toDouble/math.max(1,union)
            println(s"[$name] mask $k IoU vs reference=${"%.4f".format(iou)} (${m.count(_>0)} px)")
            assert(iou>0.98,s"mask IoU $iou")
        sys.env.get("SDFX_TEST_OUT").foreach { o =>
          val out=Path.of(o);Files.createDirectories(out)
          val canvas=new BufferedImage(image.getWidth,image.getHeight,BufferedImage.TYPE_INT_RGB);val g=canvas.createGraphics()
          g.drawImage(image,0,0,null)
          for ((b,k) <- det.boxes.zipWithIndex) do
            det.masks.foreach(ms=>for p <- 0 until ms(k).length if ms(k)(p)>0 do
              val x=p%image.getWidth;val y=p/image.getWidth;val c=canvas.getRGB(x,y)
              canvas.setRGB(x,y,(((c>>16)&255)/2+128)<<16|((c>>8)&255)/2<<8|(c&255)/2))
            g.setColor(Color.RED);g.setStroke(new BasicStroke(2f))
            g.drawRect(b._1.toInt,b._2.toInt,(b._3-b._1).toInt,(b._4-b._2).toInt)
            g.drawString(f"${det.scores(k)}%.2f",b._1.toInt+2,b._2.toInt+12)
          g.dispose();ImageIO.write(canvas,"png",out.resolve(name+"_detections.png").toFile)
        }
      finally set.close()
    }

  realCase("face_yolov8n")
  realCase("hand_yolov8n")
  realCase("person_yolov8n-seg")

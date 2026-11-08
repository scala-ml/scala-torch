package scalatorch.models.worker

import java.nio.{ByteBuffer,ByteOrder}
import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import scalatorch.Tensor
import scalatorch.models.{ClipFastTokenizer,MarigoldNormals,PretrainedClip}

/** Marigold normals (`normal_marigold` preprocessor) tests. Real-weights parity needs the weights (SDFX_SCALATORCH_ANNOTATOR_DIR or
  * models/annotators/marigold-normals-v1-1) and SDFX_TEST_MARIGOLD_CASES = comma separated `image=reference dir` pairs, the directories written
  * by scripts/reference_marigold_normals.py (noise, text, image_latent, pred_latent, decoded, normals as raw float32, meta.json). The end-to-end
  * test needs SDFX_TEST_MARIGOLD_IMAGES (comma separated) and SDFX_TEST_OUT. Run with SDFX_SCALATORCH_UNET_DEVICE=cuda (and VAE) and
  * SCALATORCH_CUDA_FP16=0 for the FP32 numbers. Marigold code is Apache-2.0; the weights are OpenRAIL++-M. */
class NativeMarigoldNormalsSuite extends munit.FunSuite:
  override val munitTimeout=scala.concurrent.duration.Duration(30,"min")

  test("missing Marigold weights fail loudly instead of producing an approximation") {
    val none=Files.createTempDirectory("marigold-none")
    val e=intercept[IllegalStateException](NativeMarigoldNormals.locate(Map.empty,none))
    assert(e.getMessage.contains("weights not found"),e.getMessage)
  }

  test("DDIM schedule: trailing timesteps, zero terminal SNR, alpha_bar[0] after the last step") {
    val s=MarigoldNormals.Schedule(4)
    assertEquals(s.timesteps,Vector(999,749,499,249))
    assertEquals(MarigoldNormals.Schedule(1).timesteps,Vector(999))
    // alpha_bar[999] == 0 and the previous alpha_bar is alpha_bar[0] = 1 - 0.00085 for a single step:
    // x_prev = sqrt(p) * (-v) + sqrt(1 - p) * x
    val p=1-0.00085
    val out=MarigoldNormals.Schedule(1).step(0,Array(2f),Array(3f))
    assertEqualsDouble(out.head.toDouble,-2*math.sqrt(p)+3*math.sqrt(1-p),1e-6)
  }

  test("antialiased bilinear resize equals the box average for an exact 2x shrink and is exact for constants") {
    val x=Array.tabulate(16)(_.toFloat) // 1 x 4 x 4
    val y=MarigoldNormals.resizeBilinear(x,1,4,4,2,2,antialias=true)
    // triangle support 2 at scale 2: centre 1.0 covers pixels 0..2 with weights 0.25,0.5... (torch/PIL), normalised
    assertEqualsDouble(y.sum.toDouble,y.length*7.5,1e-4)
    val c=MarigoldNormals.resizeBilinear(Array.fill(3*5*7)(0.25f),3,5,7,13,11,antialias=true)
    assert(c.forall(v=>math.abs(v-0.25f)<1e-6))
    val up=MarigoldNormals.resizeBilinear(Array(0f,1f),1,1,2,1,4,antialias=false)
    assertEquals(up.toVector,Vector(0f,0.25f,0.75f,1f))
  }

  test("prepare: longer side to the processing resolution, padded to a multiple of 8 by replication, mapped to [-1, 1]") {
    val p=MarigoldNormals.prepare(Array.fill(3*30*50)(1f),30,50,100)
    assertEquals((p.innerHeight,p.innerWidth),(60,100))
    assertEquals((p.height,p.width,p.padHeight,p.padWidth),(64,104,4,4))
    assert(p.data.forall(_==1f))
    val q=MarigoldNormals.prepare(Array.fill(3*8*8)(0f),8,8,0)
    assertEquals((q.height,q.width),(8,8));assert(q.data.forall(_== -1f))
  }

  private def floats(path: Path): Array[Float]=
    val b=ByteBuffer.wrap(Files.readAllBytes(path)).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
    val a=new Array[Float](b.remaining());b.get(a);a
  private def rmsRel(a: Array[Float],ref: Array[Float]): Double=
    var sumSq=0.0;var refSq=0.0;var i=0
    while i<a.length do { val d=a(i)-ref(i);sumSq+=d.toDouble*d;refSq+=ref(i).toDouble*ref(i);i+=1 }
    math.sqrt(sumSq/refSq)
  private def stats(a: Array[Float],ref: Array[Float]): String=
    require(a.length==ref.length,s"length ${a.length} vs ${ref.length}")
    var maxAbs=0.0;var refMax=0.0;var sumSq=0.0;var refSq=0.0;var i=0
    while i<a.length do
      val d=a(i)-ref(i);maxAbs=math.max(maxAbs,math.abs(d));refMax=math.max(refMax,math.abs(ref(i)));sumSq+=d.toDouble*d;refSq+=ref(i).toDouble*ref(i);i+=1
    f"maxAbs=$maxAbs%.3e maxAbs/maxRef=${maxAbs/refMax}%.3e rmsRel=${math.sqrt(sumSq/refSq)}%.3e"

  test("matches the Diffusers MarigoldNormalsPipeline reference with identical noise (text, image latent, target latent, normals)") {
    val cases=sys.env.get("SDFX_TEST_MARIGOLD_CASES").toVector.flatMap(_.split(",")).map(_.trim).filter(_.nonEmpty)
    assume(cases.nonEmpty && scala.util.Try(NativeMarigoldNormals.locate()).isSuccess,"real-weights gate disabled")
    val root=NativeMarigoldNormals.locate()
    val (unet,vae,attention,vaeAttention)=NativeMarigoldNormals.executions(sys.env)
    println(s"[marigold] unet=${unet.device} vae=${vae.device} fp16=${!sys.env.get("SCALATORCH_CUDA_FP16").contains("0")}")
    val t0=System.nanoTime
    val tokenizer=ClipFastTokenizer.loadDirectory(root.resolve("tokenizer"))
    val text=MarigoldNormals.emptyTextEmbedding(tokenizer,PretrainedClip.load(root.resolve("text_encoder"),maxScoreElements=attention))
    val model=MarigoldNormals.load(root,unet,vae,attention,vaeAttention,Some(text))
    println(f"[marigold] load ${(System.nanoTime-t0)/1e9}%.1f s")
    for entry <- cases do
      val Array(image,dir0)=entry.split("=",2);val dir=Path.of(dir0)
      val source=ImageIO.read(Path.of(image).toFile)
      val rgb=NativeAnnotatorFiles.rgbChw(source,1f/255f,0f)
      val h=source.getHeight;val w=source.getWidth
      println(s"[marigold] $image")
      println("[marigold]   text        "+stats(text.toFloatArray,floats(dir.resolve("text.f32"))))
      val prepared=MarigoldNormals.prepare(rgb,h,w,768)
      val t1=System.nanoTime
      val latent=model.encode(prepared)
      println(f"[marigold]   encode ${(System.nanoTime-t1)/1e9}%.2f s; image latent "+stats(latent.toFloatArray,floats(dir.resolve("image_latent.f32"))))
      val noise=floats(dir.resolve("noise.f32"))
      val t2=System.nanoTime
      val target=model.denoise(latent,Tensor.fromFloats(noise,latent.shape),4)
      println(f"[marigold]   denoise(4) ${(System.nanoTime-t2)/1e9}%.2f s; target latent "+stats(target.toFloatArray,floats(dir.resolve("pred_latent.f32"))))
      // the target latent of the reference, decoded here, isolates the VAE decoder
      val t3=System.nanoTime
      val decoded=model.decode(target)
      println(f"[marigold]   decode ${(System.nanoTime-t3)/1e9}%.2f s")
      val ph=prepared.innerHeight;val pw=prepared.innerWidth
      val inner=new Array[Float](3*ph*pw)
      for c <- 0 until 3; y <- 0 until ph do System.arraycopy(decoded,c*prepared.height*prepared.width+y*prepared.width,inner,c*ph*pw+y*pw,pw)
      println("[marigold]   decoded     "+stats(inner,floats(dir.resolve("decoded.f32"))))
      val refDecodedFromRefLatent=model.decode(Tensor.fromFloats(floats(dir.resolve("pred_latent.f32")),latent.shape))
      val inner2=new Array[Float](3*ph*pw)
      for c <- 0 until 3; y <- 0 until ph do System.arraycopy(refDecodedFromRefLatent,c*prepared.height*prepared.width+y*prepared.width,inner2,c*ph*pw+y*pw,pw)
      println("[marigold]   decoded (reference latent, VAE only) "+stats(inner2,floats(dir.resolve("decoded.f32"))))
      val normals=MarigoldNormals.finish(decoded,prepared,h,w)
      val refNormals=floats(dir.resolve("normals.f32"))
      println("[marigold]   normals     "+stats(normals,refNormals))
      val a=normals.map(v=>(math.max(0f,math.min(1f,(v+1f)*0.5f))*255f).toInt);val b=refNormals.map(v=>(math.max(0f,math.min(1f,(v+1f)*0.5f))*255f).toInt)
      var maxDiff=0;var differing=0
      for i <- a.indices do { val d=math.abs(a(i)-b(i));if d>0 then differing+=1;maxDiff=math.max(maxDiff,d) }
      println(s"[marigold]   8-bit map: max channel difference $maxDiff, $differing of ${a.length} channel values differ")
      assert(normals.forall(_.isFinite))
      // FP32 (SCALATORCH_CUDA_FP16=0) matches to ~1e-5; the default FP16 CUDA mode to ~2e-3 (relative RMS)
      val limit=if sys.env.get("SCALATORCH_CUDA_FP16").contains("0") && unet.device!="cpu" || unet.device=="cpu" then 1e-3 else 1e-2
      assert(rmsRel(latent.toFloatArray,floats(dir.resolve("image_latent.f32")))<limit,"image latent")
      assert(rmsRel(target.toFloatArray,floats(dir.resolve("pred_latent.f32")))<limit,"target latent")
      assert(rmsRel(normals,refNormals)<limit,"normals")
  }

  test("end to end: normal maps of real images") {
    val images=sys.env.get("SDFX_TEST_MARIGOLD_IMAGES").toVector.flatMap(_.split(",")).map(_.trim).filter(_.nonEmpty)
    val out=sys.env.get("SDFX_TEST_OUT")
    assume(images.nonEmpty && out.isDefined && scala.util.Try(NativeMarigoldNormals.locate()).isSuccess,"real-weights gate disabled")
    Files.createDirectories(Path.of(out.get))
    for image <- images do
      val source=ImageIO.read(Path.of(image).toFile)
      val t=System.nanoTime
      val map=NativeMarigoldNormals.preprocess(source,()=>false)
      println(f"[marigold] ${Path.of(image).getFileName} ${source.getWidth}x${source.getHeight} -> ${(System.nanoTime-t)/1e9}%.2f s")
      assertEquals((map.getWidth,map.getHeight),(source.getWidth,source.getHeight))
      ImageIO.write(map,"png",Path.of(out.get).resolve(Path.of(image).getFileName.toString.replaceFirst("\\.[^.]*$","")+"-normal_marigold.png").toFile)
      val again=NativeMarigoldNormals.preprocess(source,()=>false)
      assertEquals(again.getRGB(source.getWidth/2,source.getHeight/2),map.getRGB(source.getWidth/2,source.getHeight/2),"deterministic")
  }

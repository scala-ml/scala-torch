package scalatorch.models

import java.awt.image.BufferedImage
import java.nio.file.Path
import java.util.concurrent.ConcurrentHashMap
import scala.collection.mutable
import scalatorch.io.SafeTensorSet

/** Shared plain-array CPU Float32 transformer ops for the aesthetic scorers (token-major `n x dim` buffers). */
private[models] object ViTOps:
  def parallel(n: Int)(f: Int => Unit): Unit=DepthAnythingV2.parallel(n)(f)

  /** `y = x * w^T + b` with `w` stored `out x in`; `bias` may be empty (no bias). Parallel over output features so each weight row is reused for all tokens. */
  def linear(x: Array[Float],n: Int,in: Int,w: Array[Float],bias: Array[Float],out: Int): Array[Float]=
    require(x.length==n*in && w.length==out*in && (bias.isEmpty || bias.length==out),"linear shape mismatch")
    val y=new Array[Float](n*out)
    parallel(out) { o =>
      val wo=o*in;val b=if bias.isEmpty then 0f else bias(o);var i=0
      while i<n do
        val xo=i*in;var a0=0f;var a1=0f;var a2=0f;var a3=0f;var k=0
        while k+3<in do
          a0+=x(xo+k)*w(wo+k);a1+=x(xo+k+1)*w(wo+k+1);a2+=x(xo+k+2)*w(wo+k+2);a3+=x(xo+k+3)*w(wo+k+3);k+=4
        while k<in do { a0+=x(xo+k)*w(wo+k);k+=1 }
        y(i*out+o)=(a0+a1)+(a2+a3)+b;i+=1
    }
    y

  def layerNorm(x: Array[Float],n: Int,dim: Int,g: Array[Float],b: Array[Float],eps: Double): Array[Float]=
    val y=new Array[Float](x.length)
    parallel(n) { i =>
      val o=i*dim;var mean=0.0;var k=0
      while k<dim do { mean+=x(o+k);k+=1 }
      mean/=dim;var v=0.0;k=0
      while k<dim do { val d=x(o+k)-mean;v+=d*d;k+=1 }
      val inv=1.0/math.sqrt(v/dim+eps);k=0
      while k<dim do { y(o+k)=(((x(o+k)-mean)*inv)*g(k)+b(k)).toFloat;k+=1 }
    }
    y

  /** Multi-head softmax attention over already projected `q`,`k`,`v` (`n x dim`); `relBias` (`heads x n x n`, may be empty) is added to the logits. */
  def attend(q: Array[Float],k: Array[Float],v: Array[Float],n: Int,dim: Int,heads: Int,relBias: Array[Float]): Array[Float]=
    val hd=dim/heads;val scale=(1.0/math.sqrt(hd)).toFloat;val ctx=new Array[Float](n*dim)
    parallel(heads*n) { job =>
      val h=job/n;val i=job%n;val ho=h*hd;val s=new Array[Float](n)
      var max=Float.NegativeInfinity;var j=0
      while j<n do
        var acc=0f;var d=0
        while d<hd do { acc+=q(i*dim+ho+d)*k(j*dim+ho+d);d+=1 }
        acc*=scale
        if relBias.nonEmpty then acc+=relBias((h*n+i)*n+j)
        s(j)=acc;if acc>max then max=acc
        j+=1
      var sum=0.0;j=0
      while j<n do { val e=math.exp(s(j)-max);s(j)=e.toFloat;sum+=e;j+=1 }
      val inv=(1.0/sum).toFloat;var d=0
      while d<hd do
        var acc=0f;j=0
        while j<n do { acc+=s(j)*v(j*dim+ho+d);j+=1 }
        ctx(i*dim+ho+d)=acc*inv;d+=1
    }
    ctx

  def quickGelu(v: Float): Float=(v/(1.0+math.exp(-1.702*v))).toFloat

  /** Strictly loads every expected tensor (shape-checked); anything unexpected is rejected. */
  def loadStrict(file: Path,expected: Map[String,Seq[Int]],label: String): Map[String,Array[Float]]=
    require(java.nio.file.Files.isRegularFile(file),s"$label weights not found: $file")
    val set=SafeTensorSet.openFile(file)
    try
      val unknown=set.tensors.keySet.diff(expected.keySet)
      require(unknown.isEmpty,s"unexpected $label weights: ${unknown.take(5).mkString(",")}")
      expected.map { (name,dims) =>
        val info=set.tensors.getOrElse(name,throw IllegalArgumentException(s"missing $label weight: $name"))
        require(info.shape.dims.map(_.toInt)==dims.toVector,s"$label weight shape mismatch: $name, expected ${dims.mkString("x")}, got ${info.shape}")
        name->set.readFloat(name).toFloatArray
      }
    finally set.close()

/** OpenAI CLIP vision tower hyper-parameters. */
final case class ClipVisionConfig(hidden: Int,layers: Int,heads: Int,intermediate: Int,patch: Int,image: Int,projection: Int):
  def grid: Int=image/patch
  def tokens: Int=1+grid*grid

object ClipVisionConfig:
  val L14=ClipVisionConfig(1024,24,16,4096,14,224,768)
  val B16=ClipVisionConfig(768,12,12,3072,16,224,512)

/** CLIP image tower (HF `CLIPVisionModelWithProjection` layout): patch conv + class embedding + positions, pre-LN, pre-LN transformer with
  * quick-GELU, post-LN on the class token, bias-free visual projection. Output of `embed` is the un-normalised image embedding. */
final class ClipVision private (val config: ClipVisionConfig,weights: Map[String,Array[Float]]):
  import ViTOps.*
  private def w(name: String): Array[Float]=weights(name)
  private val H=config.hidden

  private def block(x: Array[Float],n: Int,i: Int): Array[Float]=
    val p=s"vision_model.encoder.layers.$i"
    val a=layerNorm(x,n,H,w(s"$p.layer_norm1.weight"),w(s"$p.layer_norm1.bias"),1e-5)
    val q=linear(a,n,H,w(s"$p.self_attn.q_proj.weight"),w(s"$p.self_attn.q_proj.bias"),H)
    val k=linear(a,n,H,w(s"$p.self_attn.k_proj.weight"),w(s"$p.self_attn.k_proj.bias"),H)
    val v=linear(a,n,H,w(s"$p.self_attn.v_proj.weight"),w(s"$p.self_attn.v_proj.bias"),H)
    val o=linear(attend(q,k,v,n,H,config.heads,Array.empty),n,H,w(s"$p.self_attn.out_proj.weight"),w(s"$p.self_attn.out_proj.bias"),H)
    val r=new Array[Float](x.length);var t=0
    while t<r.length do { r(t)=x(t)+o(t);t+=1 }
    val h=linear(layerNorm(r,n,H,w(s"$p.layer_norm2.weight"),w(s"$p.layer_norm2.bias"),1e-5),n,H,w(s"$p.mlp.fc1.weight"),w(s"$p.mlp.fc1.bias"),config.intermediate)
    t=0;while t<h.length do { h(t)=quickGelu(h(t));t+=1 }
    val m=linear(h,n,config.intermediate,w(s"$p.mlp.fc2.weight"),w(s"$p.mlp.fc2.bias"),H)
    t=0;while t<r.length do { r(t)+=m(t);t+=1 }
    r

  /** Embedded tokens (class + patches, positions added, before `pre_layrnorm`). */
  def embeddings(pixels: Array[Float]): Array[Float]=
    val (g,n,s)=(config.grid,config.tokens,config.image)
    require(pixels.length==3*s*s,s"CLIP input must be 3x${s}x$s")
    val patches=DepthAnythingV2.conv2d(pixels,3,s,s,w("vision_model.embeddings.patch_embedding.weight"),None,H,config.patch,config.patch,0)
    val pos=w("vision_model.embeddings.position_embedding.weight");val cls=w("vision_model.embeddings.class_embedding")
    val out=new Array[Float](n*H);var c=0
    while c<H do { out(c)=cls(c)+pos(c);c+=1 }
    var t=0
    while t<g*g do
      c=0
      while c<H do { out((t+1)*H+c)=patches(c*g*g+t)+pos((t+1)*H+c);c+=1 }
      t+=1
    out

  /** Returns `(lastHidden n*hidden, pooled hidden (post-LN of class token), embedding)`. `pixels` are normalised CHW. */
  def forward(pixels: Array[Float],cancelled: () => Boolean=()=>false): (Array[Float],Array[Float],Array[Float])=
    val n=config.tokens
    var x=layerNorm(embeddings(pixels),n,H,w("vision_model.pre_layrnorm.weight"),w("vision_model.pre_layrnorm.bias"),1e-5)
    for i <- 0 until config.layers do
      if cancelled() || Thread.currentThread().isInterrupted then throw new java.util.concurrent.CancellationException("aesthetic scoring cancelled")
      x=block(x,n,i)
    val pooled=layerNorm(java.util.Arrays.copyOfRange(x,0,H),1,H,w("vision_model.post_layernorm.weight"),w("vision_model.post_layernorm.bias"),1e-5)
    (x,pooled,linear(pooled,1,H,w("visual_projection.weight"),Array.empty,config.projection))

  def embed(pixels: Array[Float],cancelled: () => Boolean=()=>false): Array[Float]=forward(pixels,cancelled)._3

object ClipVision:
  def expectedShapes(c: ClipVisionConfig): Map[String,Seq[Int]]=
    val m=mutable.LinkedHashMap.empty[String,Seq[Int]]
    val H=c.hidden
    m("vision_model.embeddings.class_embedding")=Seq(H)
    m("vision_model.embeddings.patch_embedding.weight")=Seq(H,3,c.patch,c.patch)
    m("vision_model.embeddings.position_embedding.weight")=Seq(c.tokens,H)
    def ln(p: String): Unit={m(s"$p.weight")=Seq(H);m(s"$p.bias")=Seq(H)}
    def lin(p: String,o: Int,i: Int): Unit={m(s"$p.weight")=Seq(o,i);m(s"$p.bias")=Seq(o)}
    ln("vision_model.pre_layrnorm");ln("vision_model.post_layernorm")
    for i <- 0 until c.layers do
      val p=s"vision_model.encoder.layers.$i"
      for a <- Seq("q_proj","k_proj","v_proj","out_proj") do lin(s"$p.self_attn.$a",H,H)
      lin(s"$p.mlp.fc1",c.intermediate,H);lin(s"$p.mlp.fc2",H,c.intermediate);ln(s"$p.layer_norm1");ln(s"$p.layer_norm2")
    m("visual_projection.weight")=Seq(c.projection,H)
    m.toMap

  /** Loads a vision-only safetensors file (written by scripts/convert_aesthetic_to_safetensors.py from the OpenAI checkpoints). */
  def load(file: Path,config: ClipVisionConfig): ClipVision=
    new ClipVision(config,ViTOps.loadStrict(file,expectedShapes(config),"CLIP vision"))

/** BEiT-base-patch16-384 image classifier (HF `BeitForImageClassification` with mean pooling and per-layer relative position bias), as used by
  * the cafeai aesthetic / style / waifu classifiers. */
final class BeitClassifier private (val labels: Int,weights: Map[String,Array[Float]]):
  import ViTOps.*
  import BeitClassifier.*
  private def w(name: String): Array[Float]=weights(name)
  private val relIndex=relativePositionIndex()
  private val N=Tokens

  private def relBias(layer: Int): Array[Float]=
    val table=w(s"beit.encoder.layer.$layer.attention.attention.relative_position_bias.relative_position_bias_table")
    val out=new Array[Float](Heads*N*N)
    parallel(Heads) { h =>
      var ij=0
      while ij<N*N do { out(h*N*N+ij)=table(relIndex(ij)*Heads+h);ij+=1 }
    }
    out

  private def block(x: Array[Float],i: Int): Array[Float]=
    val p=s"beit.encoder.layer.$i"
    val a=layerNorm(x,N,Hidden,w(s"$p.layernorm_before.weight"),w(s"$p.layernorm_before.bias"),Eps)
    val q=linear(a,N,Hidden,w(s"$p.attention.attention.query.weight"),w(s"$p.attention.attention.query.bias"),Hidden)
    val k=linear(a,N,Hidden,w(s"$p.attention.attention.key.weight"),Array.empty,Hidden)
    val v=linear(a,N,Hidden,w(s"$p.attention.attention.value.weight"),w(s"$p.attention.attention.value.bias"),Hidden)
    val o=linear(attend(q,k,v,N,Hidden,Heads,relBias(i)),N,Hidden,w(s"$p.attention.output.dense.weight"),w(s"$p.attention.output.dense.bias"),Hidden)
    val l1=w(s"$p.lambda_1");val l2=w(s"$p.lambda_2")
    val r=new Array[Float](x.length);var t=0
    while t<r.length do { r(t)=x(t)+l1(t%Hidden)*o(t);t+=1 }
    val h=linear(layerNorm(r,N,Hidden,w(s"$p.layernorm_after.weight"),w(s"$p.layernorm_after.bias"),Eps),N,Hidden,w(s"$p.intermediate.dense.weight"),w(s"$p.intermediate.dense.bias"),Intermediate)
    t=0;while t<h.length do { h(t)=DepthAnythingV2.gelu(h(t));t+=1 }
    val m=linear(h,N,Intermediate,w(s"$p.output.dense.weight"),w(s"$p.output.dense.bias"),Hidden)
    t=0;while t<r.length do { r(t)+=l2(t%Hidden)*m(t);t+=1 }
    r

  /** `(hidden states after the last layer n*hidden, pooled (LN of the mean patch token), logits)`; `pixels` are normalised CHW 3x384x384. */
  def forward(pixels: Array[Float],cancelled: () => Boolean=()=>false): (Array[Float],Array[Float],Array[Float])=
    require(pixels.length==3*Image*Image,s"BEiT input must be 3x${Image}x$Image")
    val patches=DepthAnythingV2.conv2d(pixels,3,Image,Image,w("beit.embeddings.patch_embeddings.projection.weight"),Some(w("beit.embeddings.patch_embeddings.projection.bias")),Hidden,Patch,Patch,0)
    val cls=w("beit.embeddings.cls_token");val G=Grid
    var x=new Array[Float](N*Hidden);var c=0
    while c<Hidden do { x(c)=cls(c);c+=1 }
    var t=0
    while t<G*G do
      c=0
      while c<Hidden do { x((t+1)*Hidden+c)=patches(c*G*G+t);c+=1 }
      t+=1
    for i <- 0 until Layers do
      if cancelled() || Thread.currentThread().isInterrupted then throw new java.util.concurrent.CancellationException("aesthetic scoring cancelled")
      x=block(x,i)
    val mean=new Array[Float](Hidden);c=0
    while c<Hidden do
      var s=0.0;t=1
      while t<N do { s+=x(t*Hidden+c);t+=1 }
      mean(c)=(s/(N-1)).toFloat;c+=1
    val pooled=layerNorm(mean,1,Hidden,w("beit.pooler.layernorm.weight"),w("beit.pooler.layernorm.bias"),Eps)
    (x,pooled,linear(pooled,1,Hidden,w("classifier.weight"),w("classifier.bias"),labels))

  def logits(pixels: Array[Float],cancelled: () => Boolean=()=>false): Array[Float]=forward(pixels,cancelled)._3

object BeitClassifier:
  val Hidden=768;val Heads=12;val Layers=12;val Intermediate=3072;val Patch=16;val Image=384
  val Grid=Image/Patch;val Tokens=1+Grid*Grid;val Eps=1e-12

  /** HF `BeitRelativePositionBias` index (window 24x24 plus the three cls entries), flattened `tokens x tokens`. */
  def relativePositionIndex(): Array[Int]=
    val g=Grid;val distances=(2*g-1)*(2*g-1)+3;val idx=new Array[Int](Tokens*Tokens)
    var j=0
    while j<Tokens do { idx(j)=distances-3;j+=1 }            // row 0: cls -> anything
    var i=1
    while i<Tokens do
      idx(i*Tokens)=distances-2                              // column 0: anything -> cls
      val (yi,xi)=((i-1)/g,(i-1)%g)
      j=1
      while j<Tokens do
        val (yj,xj)=((j-1)/g,(j-1)%g)
        idx(i*Tokens+j)=(yi-yj+g-1)*(2*g-1)+(xi-xj+g-1);j+=1
      i+=1
    idx(0)=distances-1
    idx

  def expectedShapes(labels: Int): Map[String,Seq[Int]]=
    val m=mutable.LinkedHashMap.empty[String,Seq[Int]]
    m("beit.embeddings.cls_token")=Seq(1,1,Hidden)
    m("beit.embeddings.patch_embeddings.projection.weight")=Seq(Hidden,3,Patch,Patch)
    m("beit.embeddings.patch_embeddings.projection.bias")=Seq(Hidden)
    def lin(p: String,o: Int,i: Int,bias: Boolean=true): Unit={m(s"$p.weight")=Seq(o,i);if bias then m(s"$p.bias")=Seq(o)}
    def ln(p: String): Unit={m(s"$p.weight")=Seq(Hidden);m(s"$p.bias")=Seq(Hidden)}
    for i <- 0 until Layers do
      val p=s"beit.encoder.layer.$i"
      m(s"$p.lambda_1")=Seq(Hidden);m(s"$p.lambda_2")=Seq(Hidden)
      lin(s"$p.attention.attention.query",Hidden,Hidden);lin(s"$p.attention.attention.key",Hidden,Hidden,bias=false);lin(s"$p.attention.attention.value",Hidden,Hidden)
      m(s"$p.attention.attention.relative_position_bias.relative_position_bias_table")=Seq((2*Grid-1)*(2*Grid-1)+3,Heads)
      lin(s"$p.attention.output.dense",Hidden,Hidden);lin(s"$p.intermediate.dense",Intermediate,Hidden);lin(s"$p.output.dense",Hidden,Intermediate)
      ln(s"$p.layernorm_before");ln(s"$p.layernorm_after")
    ln("beit.pooler.layernorm");lin("classifier",labels,Hidden)
    m.toMap

  /** Loads a cafe classifier written by scripts/convert_aesthetic_to_safetensors.py; the label count is taken from `classifier.weight`. */
  def load(file: Path): BeitClassifier=
    require(java.nio.file.Files.isRegularFile(file),s"BEiT classifier weights not found: $file")
    val set=SafeTensorSet.openFile(file)
    val labels=try set.tensors.getOrElse("classifier.weight",throw IllegalArgumentException("missing BEiT weight: classifier.weight")).shape.dims(0).toInt
      finally set.close()
    new BeitClassifier(labels,ViTOps.loadStrict(file,expectedShapes(labels),"BEiT classifier"))

/** The real aesthetic scorers on the CPU: improved aesthetic predictor (CLIP ViT-L/14 + MLP), Simulacra aesthetic models (CLIP ViT-L/14 or B/16 +
  * linear head) and the cafeai BEiT classifiers. Weights live in `<annotator dir>/aesthetic/` (SDFX_SCALATORCH_ANNOTATOR_DIR or models/annotators);
  * missing weights raise IllegalStateException ("weights not found"), there is no fallback. */
object AestheticModels:
  val Dir="aesthetic"
  val ClipLargeFile=s"$Dir/clip-vit-large-patch14-vision.safetensors"
  val ClipBaseFile=s"$Dir/clip-vit-base-patch16-vision.safetensors"
  val ImprovedFile=s"$Dir/improved-aesthetic-sac-logos-ava1-l14-linearMSE.safetensors"
  def simulacraFile(tag: String): String=s"$Dir/simulacra_$tag.safetensors"
  def cafeFile(name: String): String=s"$Dir/$name.safetensors"
  val ClipMean=Array(0.48145466f,0.4578275f,0.40821073f)
  val ClipStd=Array(0.26862954f,0.26130258f,0.27577711f)
  private val MaxPixels=1L<<27

  private val cache=new ConcurrentHashMap[Path,AnyRef]()
  private def cached[T <: AnyRef](file: Path)(load: Path => T): T=
    cache.computeIfAbsent(file.toAbsolutePath.normalize,f=>load(f)).asInstanceOf[T]
  private def locate(file: String,label: String): Path=scalatorch.models.worker.NativeAnnotatorFiles.locate(file,label)

  private def rgb(image: BufferedImage): BufferedImage=
    if image.getType==BufferedImage.TYPE_INT_RGB then image
    else
      val out=new BufferedImage(image.getWidth,image.getHeight,BufferedImage.TYPE_INT_RGB)
      out.setRGB(0,0,image.getWidth,image.getHeight,image.getRGB(0,0,image.getWidth,image.getHeight,null,0,image.getWidth),0,image.getWidth)
      out

  /** CLIP preprocessing: shorter side to `size` (Pillow bicubic, or Lanczos for the Simulacra scorers), centre crop (torchvision rounding), CLIP mean/std, CHW. */
  def clipPixels(image: BufferedImage,size: Int=224,lanczos: Boolean=false): Array[Float]=
    val src=rgb(image);val (w,h)=(src.getWidth,src.getHeight)
    val (nw,nh)=if w<=h then (size,(size.toLong*h/w).toInt) else ((size.toLong*w/h).toInt,size)
    val resized=if nw==w && nh==h then src else if lanczos then DiffusionImageResize.lanczos(src,nw,nh,MaxPixels) else DiffusionImageResize.bicubic(src,nw,nh,MaxPixels)
    val top=math.rint((nh-size)/2.0).toInt;val left=math.rint((nw-size)/2.0).toInt
    val out=new Array[Float](3*size*size)
    for y <- 0 until size; x <- 0 until size do
      val c=resized.getRGB(left+x,top+y);val i=y*size+x
      out(i)=(((c>>>16)&255)/255f-ClipMean(0))/ClipStd(0)
      out(size*size+i)=(((c>>>8)&255)/255f-ClipMean(1))/ClipStd(1)
      out(2*size*size+i)=((c&255)/255f-ClipMean(2))/ClipStd(2)
    out

  /** cafe (BEiT) preprocessing from preprocessor_config.json: stretch to 384x384 (Pillow bilinear), mean/std 0.5. */
  def cafePixels(image: BufferedImage): Array[Float]=
    val s=BeitClassifier.Image;val src=rgb(image)
    val resized=if src.getWidth==s && src.getHeight==s then src else DiffusionImageResize.bilinear(src,s,s,MaxPixels)
    val out=new Array[Float](3*s*s)
    for y <- 0 until s; x <- 0 until s do
      val c=resized.getRGB(x,y);val i=y*s+x
      out(i)=((c>>>16)&255)/127.5f-1f;out(s*s+i)=((c>>>8)&255)/127.5f-1f;out(2*s*s+i)=(c&255)/127.5f-1f
    out

  private def l2(v: Array[Float]): Array[Float]=
    var s=0.0;var i=0
    while i<v.length do { s+=v(i).toDouble*v(i);i+=1 }
    val n=math.sqrt(s).toFloat
    v.map(_/(if n==0f then 1f else n))

  private def clipModel(file: String,config: ClipVisionConfig,label: String): ClipVision=
    cached(locate(file,label))(f=>ClipVision.load(f,config))

  /** Unit-normalised CLIP image embedding of `image` (ViT-L/14 `"l14"` or ViT-B/16 `"b16"`), Simulacra/LAION preprocessing. */
  def clipEmbedding(image: BufferedImage,tag: String,lanczos: Boolean,cancelled: () => Boolean=()=>false): Array[Float]=
    val (file,cfg,label)=tag match
      case "l14" => (ClipLargeFile,ClipVisionConfig.L14,"CLIP ViT-L/14")
      case "b16" => (ClipBaseFile,ClipVisionConfig.B16,"CLIP ViT-B/16")
      case other => throw new IllegalArgumentException(s"unknown CLIP model $other")
    l2(clipModel(file,cfg,label).embed(clipPixels(image,cfg.image,lanczos),cancelled))

  /** The MLP 768-1024-128-64-16-1 of the improved aesthetic predictor (linear layers only; dropout is inactive at inference). */
  def improvedHead(embedding: Array[Float]): Double=
    val file=locate(ImprovedFile,"improved aesthetic predictor")
    val w=cached(file)(f=>ViTOps.loadStrict(f,Map(
      "layers.0.weight"->Seq(1024,768),"layers.0.bias"->Seq(1024),"layers.2.weight"->Seq(128,1024),"layers.2.bias"->Seq(128),
      "layers.4.weight"->Seq(64,128),"layers.4.bias"->Seq(64),"layers.6.weight"->Seq(16,64),"layers.6.bias"->Seq(16),
      "layers.7.weight"->Seq(1,16),"layers.7.bias"->Seq(1)),"improved aesthetic predictor"))
    var x=embedding
    for (name,out) <- Seq("layers.0"->1024,"layers.2"->128,"layers.4"->64,"layers.6"->16,"layers.7"->1) do
      x=ViTOps.linear(x,1,x.length,w(s"$name.weight"),w(s"$name.bias"),out)
    x(0).toDouble

  /** Improved aesthetic predictor score (raw, roughly 1..10). */
  def improved(image: BufferedImage,cancelled: () => Boolean=()=>false): Double=
    improvedHead(clipEmbedding(image,"l14",lanczos=false,cancelled))

  def simulacraHead(embedding: Array[Float],tag: String): Double=
    val w=cached(locate(simulacraFile(tag),s"Simulacra aesthetic model $tag"))(f=>
      ViTOps.loadStrict(f,Map("linear.weight"->Seq(1,embedding.length),"linear.bias"->Seq(1)),s"Simulacra aesthetic model $tag"))
    // AestheticMeanPredictionLinearModel: normalize(x) * sqrt(d), then a single linear unit
    val scaled=l2(embedding).map(_*math.sqrt(embedding.length).toFloat)
    ViTOps.linear(scaled,1,scaled.length,w("linear.weight"),w("linear.bias"),1)(0).toDouble

  /** Simulacra aesthetic model score (raw; `tag` is "l14" or "b16"; Lanczos resize like rank_images.py). */
  def simulacra(image: BufferedImage,tag: String,cancelled: () => Boolean=()=>false): Double=
    simulacraHead(clipEmbedding(image,tag,lanczos=true,cancelled),tag)

  def cafeLogits(image: BufferedImage,name: String,cancelled: () => Boolean=()=>false): Array[Float]=
    val net=cached(locate(cafeFile(name),s"cafe classifier $name"))(f=>BeitClassifier.load(f))
    net.logits(cafePixels(image),cancelled)

  def softmax(logits: Array[Float]): Array[Double]=
    val m=logits.max;val e=logits.map(v=>math.exp((v-m).toDouble));val s=e.sum
    e.map(_/s)

  /** cafe classifier probabilities in `id2label` order (`labels` must match the checkpoint's class count). */
  def cafe(image: BufferedImage,name: String,labels: Vector[String],cancelled: () => Boolean=()=>false): Vector[(String,Double)]=
    val p=softmax(cafeLogits(image,name,cancelled))
    require(p.length==labels.length,s"$name has ${p.length} classes, expected ${labels.length}")
    labels.zip(p)

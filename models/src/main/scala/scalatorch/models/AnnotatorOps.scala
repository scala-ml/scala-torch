package scalatorch.models

import scalatorch.io.{LegacyPyTorchStateDict,PyTorchArchiveLimits,PyTorchTensorArchive}
import java.nio.file.{Files,Path}

/** Plain-array CPU building blocks and the strict `.pth` loader shared by the HED, lineart and anime-lineart annotators.
  * All feature maps are CHW Float32. Written from the documented layer semantics (torch `Conv2d`, `ConvTranspose2d`,
  * `InstanceNorm2d(affine=False)`, `ReflectionPad2d`); no code is copied from any repository. */
object AnnotatorOps:
  /** Loads a `.pth` state dict (legacy or ZIP format) and rejects missing, unknown or mis-shaped tensors. */
  def loadStrict(file: Path,expected: Map[String,Seq[Int]],label: String,cancelled: () => Boolean=()=>false): Map[String,Array[Float]]=
    require(Files.isRegularFile(file),s"$label weights not found: $file")
    val limits=PyTorchArchiveLimits.ImageModelWeights.copy(maxElements=16L*1024*1024) // U-Net decoder kernels reach 8.4M elements
    val entries: Map[String,PyTorchTensorArchive.Value]=
      if LegacyPyTorchStateDict.isLegacy(file) then
        LegacyPyTorchStateDict.read(file,limits,cancelled).view.mapValues(PyTorchTensorArchive.TensorValue(_)).toMap
      else PyTorchTensorArchive.read(file,limits,cancelled).value match
        case PyTorchTensorArchive.DictValue(e) => e
        case other => throw IllegalArgumentException(s"$label weights must be a state dict, got ${other.getClass.getSimpleName}")
    val unknown=entries.keySet.diff(expected.keySet)
    require(unknown.isEmpty,s"unexpected $label weights: ${unknown.take(5).mkString(",")}")
    expected.map { (name,dims) =>
      entries.getOrElse(name,throw IllegalArgumentException(s"missing $label weight: $name")) match
        case PyTorchTensorArchive.TensorValue(t) =>
          require(t.shape.map(_.toInt)==dims.toVector,s"$label weight shape mismatch: $name, expected ${dims.mkString("x")}, got ${t.shape.mkString("x")}")
          name->t.values
        case PyTorchTensorArchive.IntegerTensorValue(t) if name.endsWith(".num_batches_tracked") => // BatchNorm bookkeeping, unused at inference
          require(t.shape.isEmpty,s"$label weight shape mismatch: $name")
          name->Array.empty[Float]
        case _ => throw IllegalArgumentException(s"$label weight is not a tensor: $name")
    }

  /** In-place ReLU. */
  def relu(x: Array[Float]): Array[Float]=
    var i=0
    while i<x.length do { if x(i)<0f then x(i)=0f;i+=1 }
    x

  def leakyRelu(x: Array[Float],slope: Float): Array[Float]=
    val y=new Array[Float](x.length);var i=0
    while i<x.length do { y(i)=if x(i)<0f then x(i)*slope else x(i);i+=1 }
    y

  def sigmoid(v: Float): Float=(1.0/(1.0+math.exp(-v.toDouble))).toFloat

  /** `InstanceNorm2d(affine=False, eps=1e-5)`: per-channel zero mean and unit (biased) variance, in place. */
  def instanceNorm(x: Array[Float],c: Int,h: Int,w: Int): Array[Float]=
    val n=h*w
    DepthAnythingV2.parallel(c) { ch =>
      val b=ch*n;var s=0.0;var i=0
      while i<n do { s+=x(b+i);i+=1 }
      val mean=s/n;var v=0.0;i=0
      while i<n do { val d=x(b+i)-mean;v+=d*d;i+=1 }
      val inv=1.0/math.sqrt(v/n+1e-5);i=0
      while i<n do { x(b+i)=((x(b+i)-mean)*inv).toFloat;i+=1 }
    }
    x

  /** `ReflectionPad2d(p)` (edge pixel not repeated). */
  def reflectPad(x: Array[Float],c: Int,h: Int,w: Int,p: Int): Array[Float]=
    require(p<h && p<w,"reflection padding must be smaller than the image")
    val oh=h+2*p;val ow=w+2*p;val out=new Array[Float](c*oh*ow)
    def refl(i: Int,n: Int)=if i<0 then -i else if i>=n then 2*n-2-i else i
    DepthAnythingV2.parallel(c) { ch =>
      var y=0
      while y<oh do
        val sy=refl(y-p,h);var xx=0
        while xx<ow do { out(ch*oh*ow+y*ow+xx)=x(ch*h*w+sy*w+refl(xx-p,w));xx+=1 }
        y+=1
    }
    out

  /** `ConvTranspose2d` with weights in torch's (in, out, k, k) layout. */
  def convTranspose(x: Array[Float],ci: Int,h: Int,w: Int,wt: Array[Float],bias: Array[Float],co: Int,k: Int,stride: Int,pad: Int,outPad: Int): Array[Float]=
    val oh=(h-1)*stride-2*pad+k+outPad;val ow=(w-1)*stride-2*pad+k+outPad
    val out=new Array[Float](co*oh*ow)
    DepthAnythingV2.parallel(co) { o =>
      val base=o*oh*ow;java.util.Arrays.fill(out,base,base+oh*ow,bias(o))
      var c=0
      while c<ci do
        var ky=0
        while ky<k do
          var kx=0
          while kx<k do
            val wv=wt(((c*co+o)*k+ky)*k+kx)
            var iy=0
            while iy<h do
              val oy=iy*stride-pad+ky
              if oy>=0 && oy<oh then
                var ix=0
                while ix<w do
                  val ox=ix*stride-pad+kx
                  if ox>=0 && ox<ow then out(base+oy*ow+ox)+=wv*x(c*h*w+iy*w+ix)
                  ix+=1
              iy+=1
            kx+=1
          ky+=1
        c+=1
    }
    out

  /** Channel concatenation of two CHW maps of equal spatial size. */
  def concat(a: Array[Float],b: Array[Float]): Array[Float]=
    val out=new Array[Float](a.length+b.length)
    System.arraycopy(a,0,out,0,a.length);System.arraycopy(b,0,out,a.length,b.length);out

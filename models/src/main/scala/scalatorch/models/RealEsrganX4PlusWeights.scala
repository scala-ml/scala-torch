package scalatorch.models

import scalatorch.io.{PyTorchArchiveLimits,PyTorchTensorArchive,ReadLimits,SafeDType,SafeTensors}
import java.nio.file.Path
import scala.util.Using

/** Strict state-dictionary reader for Real-ESRGAN's general-image RRDBNet x4 model. */
final case class RealEsrganX4PlusWeights private (tensors: Map[String,PyTorchTensorArchive.StoredTensor],blockCount: Int):
  def apply(name: String): PyTorchTensorArchive.StoredTensor =
    tensors.getOrElse(name,throw IllegalArgumentException(s"Real-ESRGAN weight not found: $name"))

object RealEsrganX4PlusWeights:
  private val Features=64
  private val Growth=32
  private val SupportedBlocks=Set(1,2,6,23)
  private val FloatKinds=Set(PyTorchTensorArchive.StorageDType.Float32,
    PyTorchTensorArchive.StorageDType.Float16,PyTorchTensorArchive.StorageDType.BFloat16)

  private def expectedShapes(blocks: Int): Map[String,Vector[Long]]=
    val result=scala.collection.mutable.LinkedHashMap.empty[String,Vector[Long]]
    def conv(prefix: String,in: Int,out: Int): Unit=
      result(s"$prefix.weight")=Vector(out.toLong,in.toLong,3L,3L)
      result(s"$prefix.bias")=Vector(out.toLong)
    conv("conv_first",3,Features)
    for block <- 0 until blocks do
      for dense <- 1 to 3 do
        val root=s"RRDB_trunk.$block.RDB$dense"
        conv(s"$root.conv1",Features,Growth)
        conv(s"$root.conv2",Features+Growth,Growth)
        conv(s"$root.conv3",Features+2*Growth,Growth)
        conv(s"$root.conv4",Features+3*Growth,Growth)
        conv(s"$root.conv5",Features+4*Growth,Features)
    conv("trunk_conv",Features,Features)
    conv("upconv1",Features,Features)
    conv("upconv2",Features,Features)
    conv("HRconv",Features,Features)
    conv("conv_last",Features,3)
    result.toMap

  private def canonicalName(name: String): String=
    name match
      case s"conv_body.$suffix" => s"trunk_conv.$suffix"
      case s"conv_up1.$suffix" => s"upconv1.$suffix"
      case s"conv_up2.$suffix" => s"upconv2.$suffix"
      case s"conv_hr.$suffix" => s"HRconv.$suffix"
      case BodyDense(block,dense,conv,kind) =>
        s"RRDB_trunk.${block}.RDB${dense}.conv${conv.drop(4)}.$kind"
      case other => other

  private val BodyDense="""^body\.([0-9]+)\.rdb([1-3])\.(conv[1-5])\.(weight|bias)$""".r

  private def canonicalEntries[A](entries: Map[String,A]): Map[String,A]=
    val out=scala.collection.mutable.LinkedHashMap.empty[String,A]
    entries.foreach { (source,value) =>
      val name=canonicalName(source)
      require(!out.contains(name),s"duplicate Real-ESRGAN weight after name normalization: $name")
      out(name)=value
    }
    out.toMap

  private def shapesFor(names: Set[String]): (Int,Map[String,Vector[Long]])=
    val blocks=names.iterator.flatMap {
      case RrdbBlock(index,_) => index.toIntOption
      case _ => None
    }.toSet
    require(blocks.nonEmpty && blocks==((0 until blocks.size).toSet),
      s"Real-ESRGAN RRDB blocks must be contiguous from zero; found ${blocks.toVector.sorted}")
    val count=blocks.size
    require(SupportedBlocks(count),s"unsupported Real-ESRGAN RRDB block count: $count (supported: 6 or 23)")
    (count,expectedShapes(count))

  private val RrdbBlock="""^RRDB_trunk\.([0-9]+)\.RDB[1-3]\.conv[1-5]\.(weight|bias)$""".r

  private def validateKeys(names: Set[String],shapes: Map[String,Vector[Long]]): Unit=
    val missing=shapes.keySet--names
    val unexpected=names--shapes.keySet
    require(missing.isEmpty,s"Real-ESRGAN checkpoint is missing weights: ${missing.toVector.sorted.take(8).mkString(", ")}")
    require(unexpected.isEmpty,s"Real-ESRGAN checkpoint has unexpected entries: ${unexpected.toVector.sorted.take(8).mkString(", ")}")

  private def stateDictionary(value: PyTorchTensorArchive.Value): Map[String,PyTorchTensorArchive.Value]=value match
    case PyTorchTensorArchive.DictValue(root) =>
      val selected=Seq("params_ema","params","state_dict").iterator.flatMap(root.get).take(1).toVector.headOption
        .getOrElse(value)
      selected match
        case PyTorchTensorArchive.DictValue(entries) => entries
        case _ => throw IllegalArgumentException("Real-ESRGAN checkpoint state dictionary must be a mapping")
    case _ => throw IllegalArgumentException("Real-ESRGAN checkpoint must contain a state dictionary")

  private def validate(entries: Map[String,PyTorchTensorArchive.StoredTensor],blockCount: Int,
      shapes: Map[String,Vector[Long]]): RealEsrganX4PlusWeights=
    validateKeys(entries.keySet,shapes)
    shapes.foreach { (name,shape) =>
      val weight=entries(name)
      require(FloatKinds(weight.dtype),s"Real-ESRGAN tensor $name must use a floating-point dtype")
      require(weight.shape==shape,s"Real-ESRGAN tensor $name has shape ${weight.shape}, expected $shape")
      require(weight.values.forall(_.isFinite),s"Real-ESRGAN tensor $name contains a non-finite value")
    }
    RealEsrganX4PlusWeights(entries,blockCount)

  private def fromSafeTensors(path: Path): RealEsrganX4PlusWeights=
    require(java.nio.file.Files.size(path)<=256L*1024*1024,"Real-ESRGAN safetensors file exceeds 256 MiB")
    Using.resource(SafeTensors.open(path,ReadLimits(maxHeaderBytes=16*1024*1024,maxTensors=4096,maxRank=8,
        maxTensorBytes=128L*1024*1024))) { archive =>
      val totalBytes=archive.tensors.valuesIterator.map(info=>info.end-info.begin)
        .foldLeft(0L)((total,size)=>Math.addExact(total,size))
      require(totalBytes<=512L*1024*1024,"Real-ESRGAN safetensors payload exceeds 512 MiB")
      val names=archive.tensors.keySet
      val prefix=Seq("params_ema.","params.","state_dict.").find(p=>names.nonEmpty && names.forall(_.startsWith(p)))
      val decoded=archive.tensors.keysIterator.map { original =>
        val name=prefix.fold(original)(p=>original.drop(p.length))
        val info=archive.tensors(original)
        val dtype=info.dtype match
          case SafeDType.F32 => PyTorchTensorArchive.StorageDType.Float32
          case SafeDType.F16 => PyTorchTensorArchive.StorageDType.Float16
          case SafeDType.BF16 => PyTorchTensorArchive.StorageDType.BFloat16
          case _ => throw IllegalArgumentException(s"Real-ESRGAN tensor $name must use a floating-point dtype")
        val values=archive.readFloat(original).toFloatArray
        name->PyTorchTensorArchive.StoredTensor(dtype,info.shape.dims,values)
      }.toMap
      val canonical=canonicalEntries(decoded)
      val (blocks,shapes)=shapesFor(canonical.keySet)
      validate(canonical,blocks,shapes)
    }

  private def tensor(name: String,value: PyTorchTensorArchive.Value,shape: Vector[Long]): PyTorchTensorArchive.StoredTensor=value match
    case PyTorchTensorArchive.TensorValue(t) =>
      require(FloatKinds(t.dtype),s"Real-ESRGAN tensor $name must use a floating-point dtype")
      require(t.shape==shape,s"Real-ESRGAN tensor $name has shape ${t.shape}, expected $shape")
      require(t.values.length==shape.foldLeft(1L)((size,dimension) => Math.multiplyExact(size,dimension)),
        s"Real-ESRGAN tensor $name has an invalid element count")
      require(t.values.forall(_.isFinite),s"Real-ESRGAN tensor $name contains a non-finite value")
      t
    case _ => throw IllegalArgumentException(s"Real-ESRGAN state entry $name is not a tensor")

  /** Reads the supported PyTorch ZIP state-dict subset without unpickling executable objects. */
  def read(path: Path): RealEsrganX4PlusWeights=
    require(java.nio.file.Files.isRegularFile(path),s"Real-ESRGAN checkpoint is missing: $path")
    path.getFileName.toString.toLowerCase(java.util.Locale.ROOT) match
      case name if name.endsWith(".safetensors") => fromSafeTensors(path)
      case name if name.endsWith(".pth") || name.endsWith(".pt") =>
        val archive=PyTorchTensorArchive.read(path,PyTorchCompatibility.limits(PyTorchArchiveLimits.ImageModelWeights))
        val entries=canonicalEntries(stateDictionary(archive.value))
        val (blocks,shapes)=shapesFor(entries.keySet)
        validateKeys(entries.keySet,shapes)
        val decoded=shapes.iterator.map { (name,shape) => name->tensor(name,entries(name),shape) }.toMap
        validate(decoded,blocks,shapes)
      case _ => throw IllegalArgumentException("Real-ESRGAN checkpoint must be a .pth, .pt or .safetensors file")

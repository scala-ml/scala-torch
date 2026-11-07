package scalatorch.models

import scalatorch.{Shape,Tensor}
import scalatorch.io.{PyTorchArchiveLimits,PyTorchTensorArchive,ReadLimits,SafeTensors,SafeTensorsEditCopy,SafeTensorsWriter,SafeTensorWrite}
import java.nio.file.{Files,Path}
import scala.util.Using

/** CPU Float32 vectors decoded from NUI-compatible safetensors textual-inversion files. */
final case class ClipTextualInversion(clipL: Vector[Array[Float]],clipG: Option[Vector[Array[Float]]])

object ClipTextualInversion:
  private val MaxFileBytes=64L*1024*1024
  private val MaxVectors=256
  /** Stage a restricted PyTorch embedding edit, preserving its wrapper, dtype and primitive metadata. */
  def stagePyTorchEditDraft(source: Path,stage: Path,edited: Map[String,Vector[Vector[Double]]],
      baseline: Map[String,Vector[Vector[Double]]],expectedSha256: String,cancelled: () => Boolean=() => false): Unit =
    import PyTorchTensorArchive.*
    require(source!=null && stage!=null && edited!=null && baseline!=null && cancelled!=null &&
      expectedSha256!=null && expectedSha256.matches("(?i)[0-9a-f]{64}"),"invalid PyTorch edit arguments")
    require((edited.keySet==Set("clip_l") || edited.keySet==Set("clip_l","clip_g")) && edited.keySet==baseline.keySet,
      "invalid embedding editor encoder set")
    def check(): Unit=if cancelled() || Thread.currentThread().isInterrupted then
      throw new java.util.concurrent.CancellationException("embedding edit cancelled")
    def digest(): String =
      check();require(Files.isRegularFile(source) && Files.size(source)<=MaxFileBytes,"invalid PyTorch embedding source")
      val hash=java.security.MessageDigest.getInstance("SHA-256")
      val input=Files.newInputStream(source)
      try
        val buffer=new Array[Byte](65536);var n=input.read(buffer)
        while n>=0 do
          check();if n>0 then hash.update(buffer,0,n)
          n=input.read(buffer)
      finally input.close()
      java.util.HexFormat.of().formatHex(hash.digest())
    def checkSource(): Unit=require(digest().equalsIgnoreCase(expectedSha256),"source file changed")
    def half(value: Double): Float =
      val sign=if (java.lang.Double.doubleToRawLongBits(value)&Long.MinValue)!=0 then 0x8000 else 0
      val magnitude=math.abs(value)
      require(magnitude<65520.0,"embedding value outside dtype range")
      val bits=if magnitude<math.pow(2.0,-14) then sign | math.rint(Math.scalb(magnitude,24)).toInt
      else
        var exponent=Math.getExponent(magnitude)
        var significand=math.rint(Math.scalb(magnitude,10-exponent)).toInt
        if significand==2048 then { exponent+=1;significand=1024 }
        require(exponent<=15,"embedding value outside dtype range")
        sign | ((exponent+15)<<10) | (significand-1024)
      java.lang.Float.float16ToFloat(bits.toShort)
    def replace(value: Value,key: String): Value =
      val tensor=value match
        case TensorValue(t) => t
        case _ => throw IllegalArgumentException("embedding value must be a tensor")
      val (count,width)=tensor.shape match
        case Vector(w) => (1L,w)
        case Vector(n,w) => (n,w)
        case _ => throw IllegalArgumentException("embedding tensor must have rank one or two")
      require(count>0 && count<=MaxVectors && width>0 && width<=Int.MaxValue,"invalid embedding tensor dimensions")
      val rows=edited(key);val old=baseline(key)
      def valid(data: Vector[Vector[Double]]): Boolean=data!=null && data.size==count &&
        data.forall(row=>row!=null && row.size==width && row.forall(_.isFinite))
      require(valid(rows) && valid(old),"invalid embedding vectors")
      val values=tensor.values.clone()
      for i <- rows.indices; j <- rows(i).indices do
        if rows(i)(j)!=old(i)(j) then
          val number=rows(i)(j)
          val converted=if tensor.dtype==StorageDType.Float16 then half(number) else number.toFloat
          require(converted.isFinite,"embedding value outside dtype range")
          values(i*width.toInt+j)=converted
      TensorValue(tensor.copy(values=values))
    checkSource()
    val sourceValue=PyTorchTensorArchive.read(source).value
    checkSource()
    val result=sourceValue match
      case tensor: TensorValue if edited.keySet==Set("clip_l") => replace(tensor,"clip_l")
      case DictValue(fields) if fields.contains("string_to_param") && edited.keySet==Set("clip_l") =>
        val nested=fields("string_to_param") match
          case DictValue(entries) if entries.keySet==Set("*") => DictValue(entries.updated("*",replace(entries("*"),"clip_l")))
          case _ => throw IllegalArgumentException("unsupported PyTorch string_to_param")
        DictValue(fields.updated("string_to_param",nested))
      case DictValue(fields) if fields.contains("emb_params") && edited.keySet==Set("clip_l") =>
        DictValue(fields.updated("emb_params",replace(fields("emb_params"),"clip_l")))
      case DictValue(fields) if edited.keySet==Set("clip_l","clip_g") && edited.keySet.subsetOf(fields.keySet) =>
        DictValue(fields.updated("clip_l",replace(fields("clip_l"),"clip_l")).updated("clip_g",replace(fields("clip_g"),"clip_g")))
      case DictValue(fields) if fields.size==1 && edited.keySet==Set("clip_l") =>
        val key=fields.keys.head;DictValue(fields.updated(key,replace(fields(key),"clip_l")))
      case _ => throw IllegalArgumentException("unsupported PyTorch embedding format")
    scalatorch.io.PyTorchTensorArchiveWriter.write(stage,result,cancelled=cancelled)
    val guard=stage.toAbsolutePath.normalize().resolveSibling(s".pt-edit-${java.util.UUID.randomUUID()}.identity")
    var keep=false
    var linked=false
    def ownsStage: Boolean =
      try linked && Files.isSameFile(stage,guard)
      catch case scala.util.control.NonFatal(_) => false
    try
      Files.createLink(guard,stage);linked=true
      checkSource();require(ownsStage,"embedding edit stage changed");keep=true
    finally
      if !keep && ownsStage then Files.deleteIfExists(stage)
      if linked then Files.deleteIfExists(guard)
  /** Stage an editor draft without changing the source or unrelated safetensors bytes. */
  def stageSafetensorsEdit(source: Path,stage: Path,edited: ClipTextualInversion,baseline: ClipTextualInversion,
      expectedSha256: String,cancelled: () => Boolean=() => false): Unit =
    require(source!=null && stage!=null && edited!=null && baseline!=null && edited.clipG!=null && baseline.clipG!=null,
      "invalid embedding editor draft")
    def values(rows: Vector[Array[Float]]): Vector[Vector[Double]] =
      require(rows!=null && rows.forall(_!=null),"invalid embedding vectors")
      rows.map(_.iterator.map(_.toDouble).toVector)
    def mapped(value: ClipTextualInversion): Map[String,Vector[Vector[Double]]] =
      Map("clip_l"->values(value.clipL))++value.clipG.map(rows=>"clip_g"->values(rows))
    stageSafetensorsEditDraft(source,stage,mapped(edited),mapped(baseline),expectedSha256,cancelled)

  /** Editor coordinates are doubles; do not narrow them before writing the source dtype. */
  def stageSafetensorsEditDraft(source: Path,stage: Path,edited: Map[String,Vector[Vector[Double]]],
      baseline: Map[String,Vector[Vector[Double]]],expectedSha256: String,
      cancelled: () => Boolean=() => false): Unit =
    require(source!=null && stage!=null && edited!=null && baseline!=null,"invalid embedding editor draft")
    require(edited.keySet==Set("clip_l") || edited.keySet==Set("clip_l","clip_g"),"invalid embedding editor encoder set")
    require(edited.keySet==baseline.keySet,"embedding editor encoder set differs")
    for values <- Vector(edited,baseline) do
      require(values.values.forall(rows=>rows!=null && rows.nonEmpty && rows.forall(row=>row!=null && row.nonEmpty)),
        "invalid embedding vectors")
      require(!values.contains("clip_g") || values("clip_l").size==values("clip_g").size,
        "SDXL textual-inversion encoders have different vector counts")
    require(Files.isRegularFile(source) && Files.size(source)<=512L*1024*1024,"invalid embedding source")
    val archive=SafeTensors.open(source,ReadLimits(maxHeaderBytes=16*1024*1024,maxTensors=100000,maxRank=64,
      maxTensorBytes=512L*1024*1024))
    val names=try archive.tensors.keySet finally archive.close()
    val left=if edited.contains("clip_g") then
      require(names.contains("clip_l") && names.contains("clip_g"),"SDXL embedding encoder names are missing")
      "clip_l"
    else if names.contains("clip_l") then "clip_l"
    else if names.contains("emb_params") then "emb_params"
    else if names.size==1 then names.head
    else throw IllegalArgumentException("embedding tensor names are ambiguous")
    def mapped(value: Map[String,Vector[Vector[Double]]]): Map[String,Vector[Vector[Double]]] =
      Map(left->value("clip_l"))++value.get("clip_g").map(rows=>"clip_g"->rows)
    SafeTensorsEditCopy.build(source,stage,mapped(edited),expectedSha256,Some(mapped(baseline)),cancelled)
  /** NUI-compatible SD1/SDXL Float32 embedding files, without a Python runtime. */
  def writeSafetensors(path: Path,embedding: ClipTextualInversion,step: Option[Long]=None,
      overwrite: Boolean=false,cancelled: () => Boolean=() => false): Unit =
    require(path!=null && path.getFileName.toString.toLowerCase(java.util.Locale.ROOT).endsWith(".safetensors"),
      "embedding output must be a .safetensors file")
    require(embedding!=null && embedding.clipL!=null && embedding.clipG!=null,"invalid textual-inversion embedding")
    val streams=Vector("clip_l"->embedding.clipL)++embedding.clipG.toVector.map("clip_g"->_)
    require(embedding.clipG.forall(rows=>rows!=null && rows.size==embedding.clipL.size),"SDXL textual-inversion encoders have different vector counts")
    var total=0L
    streams.foreach { (name,rows) =>
      require(rows!=null && rows.nonEmpty && rows.size<=MaxVectors && rows.head!=null && rows.head.nonEmpty,
        s"invalid textual-inversion vectors: $name")
      val width=rows.head.length
      require(rows.forall(row=>row!=null && row.length==width && row.forall(_.isFinite)),s"invalid textual-inversion vector contents: $name")
      total=Math.addExact(total,Math.multiplyExact(rows.size.toLong*width,4L))
      require(total<=MaxFileBytes,"textual-inversion tensors exceed 64 MiB")
    }
    val tensors=streams.map { (name,rows) =>
      name->SafeTensorWrite.F32(Tensor.fromFloats(rows.flatten.toArray,Shape.from(Seq(rows.size.toLong,rows.head.length.toLong))))
    }.toMap
    SafeTensorsWriter.write(path,tensors,step.map(value=>Map("step"->value.toString)).getOrElse(Map.empty),overwrite,cancelled,
      ReadLimits(maxHeaderBytes=1024*1024,maxTensors=8,maxRank=4,maxTensorBytes=MaxFileBytes))
  private def vectors(archive: SafeTensors,name: String,width: Int): Vector[Array[Float]] =
    require(width>0,s"invalid CLIP embedding width for $name")
    val info=archive.tensors.getOrElse(name,throw IllegalArgumentException(s"textual-inversion tensor missing: $name"))
    val dims=info.shape.dims
    val (rows,columns)=dims match
      case Vector(n) => (1L,n)
      case Vector(n,w) => (n,w)
      case _ => throw IllegalArgumentException(s"textual-inversion tensor must have rank one or two: $name")
    require(columns==width,s"textual-inversion embedding width mismatch for $name: expected $width, got $columns")
    require(rows>=1 && rows<=MaxVectors,s"textual-inversion vector count must be 1..$MaxVectors")
    val values=archive.readFloat(name).toFloatArray
    require(values.forall(_.isFinite),s"textual-inversion tensor contains nonfinite values: $name")
    Vector.tabulate(rows.toInt)(row=>java.util.Arrays.copyOfRange(values,row*width,(row+1)*width))

  /** Accepts NUI's one-tensor SD1 layout or its exact dual-tensor SDXL layout. */
  def readSafetensors(path: Path,isSdxl: Boolean,clipLWidth: Int,clipGWidth: Int): ClipTextualInversion =
    require(Files.isRegularFile(path) && Files.size(path)<=MaxFileBytes,
      "textual-inversion safetensors file is missing or exceeds 64 MiB")
    Using.resource(SafeTensors.open(path,ReadLimits(maxHeaderBytes=1024*1024,maxTensors=8,maxRank=4,
      maxTensorBytes=MaxFileBytes))) { archive =>
      if isSdxl then
        require(clipGWidth>0 && archive.tensors.keySet==Set("clip_l","clip_g"),
          "SDXL textual inversion requires exactly clip_l and clip_g tensors")
        val left=vectors(archive,"clip_l",clipLWidth)
        val right=vectors(archive,"clip_g",clipGWidth)
        require(left.size==right.size,"SDXL textual-inversion encoders have different vector counts")
        ClipTextualInversion(left,Some(right))
      else
        require(archive.tensors.size==1,"SD1 textual inversion requires exactly one CLIP tensor")
        ClipTextualInversion(vectors(archive,archive.tensors.keys.head,clipLWidth),None)
    }

  private def pytorchRows(value: PyTorchTensorArchive.Value,width: Int,name: String): Vector[Array[Float]]=
    val tensor=value match
      case PyTorchTensorArchive.TensorValue(result) => result
      case _ => throw IllegalArgumentException(s"textual-inversion value must be a tensor: $name")
    require(width>0,s"invalid CLIP embedding width for $name")
    val (rows,columns)=tensor.shape match
      case Vector(n) => (1L,n)
      case Vector(n,w) => (n,w)
      case _ => throw IllegalArgumentException(s"textual-inversion tensor must have rank one or two: $name")
    require(columns==width,s"textual-inversion embedding width mismatch for $name: expected $width, got $columns")
    require(rows>=1 && rows<=MaxVectors,s"textual-inversion vector count must be 1..$MaxVectors")
    require(tensor.values.length.toLong==Math.multiplyExact(rows,columns),s"textual-inversion tensor data shape mismatch: $name")
    require(tensor.values.forall(_.isFinite),s"textual-inversion tensor contains nonfinite values: $name")
    Vector.tabulate(rows.toInt)(row=>java.util.Arrays.copyOfRange(tensor.values,row*width,(row+1)*width))

  private def sd1Tensor(value: PyTorchTensorArchive.Value): PyTorchTensorArchive.Value=value match
    case tensor: PyTorchTensorArchive.TensorValue => tensor
    case PyTorchTensorArchive.DictValue(fields) if fields.contains("emb_params") =>
      validateMetadata(fields,Set("emb_params")); fields("emb_params")
    case PyTorchTensorArchive.DictValue(fields) if fields.contains("string_to_param") =>
      validateMetadata(fields,Set("string_to_param"))
      fields("string_to_param") match
        case PyTorchTensorArchive.DictValue(parameters) if parameters.keySet==Set("*") => parameters("*")
        case _ => throw IllegalArgumentException("PyTorch string_to_param must contain only the '*' embedding")
    case PyTorchTensorArchive.DictValue(fields) if fields.contains("clip_g") =>
      throw IllegalArgumentException("SDXL textual inversion cannot be loaded by an SD1 checkpoint")
    case PyTorchTensorArchive.DictValue(fields) if fields.contains("clip_l") =>
      throw IllegalArgumentException("CLIP-L/CLIP-G PyTorch textual inversion cannot be loaded by an SD1 checkpoint")
    case PyTorchTensorArchive.DictValue(fields) if fields.size==1 => fields.values.head
    case _ => throw IllegalArgumentException("unsupported SD1 PyTorch textual-inversion layout")

  private val AllowedMetadataKeys=Set("name","step","sd_checkpoint","sd_checkpoint_name")
  private def validateMetadata(fields: Map[String,PyTorchTensorArchive.Value],payloadKeys: Set[String]): Unit=
    val extra=fields.keySet--payloadKeys
    require(extra.subsetOf(AllowedMetadataKeys),s"unsupported PyTorch textual-inversion fields: ${extra.toVector.sorted.mkString(", ")}")
    extra.foreach { name => fields(name) match
      case _: PyTorchTensorArchive.StringValue | _: PyTorchTensorArchive.IntegerValue |
          _: PyTorchTensorArchive.BooleanValue | PyTorchTensorArchive.NoneValue => ()
      case _ => throw IllegalArgumentException(s"PyTorch textual-inversion metadata must be primitive: $name")
    }

  /** Reads only the NUI-compatible plain tensor layouts from a restricted current-format PyTorch archive. */
  def readPyTorch(path: Path,isSdxl: Boolean,clipLWidth: Int,clipGWidth: Int): ClipTextualInversion =
    val archive=PyTorchTensorArchive.read(path,PyTorchCompatibility.limits(PyTorchArchiveLimits(maxArchiveBytes=MaxFileBytes,
      maxEntries=64,maxEntryBytes=MaxFileBytes,maxTotalUncompressedBytes=MaxFileBytes,
      maxPickleBytes=1024*1024,maxPickleOps=100000,maxTensors=4,maxElements=1024L*1024)))
    if isSdxl then archive.value match
      case PyTorchTensorArchive.DictValue(fields) =>
        require(fields.keySet.contains("clip_l") && fields.keySet.contains("clip_g"),
          "SDXL PyTorch textual inversion requires clip_l and clip_g tensors")
        validateMetadata(fields,Set("clip_l","clip_g"))
        val left=pytorchRows(fields("clip_l"),clipLWidth,"clip_l")
        val right=pytorchRows(fields("clip_g"),clipGWidth,"clip_g")
        require(left.size==right.size,"SDXL textual-inversion encoders have different vector counts")
        ClipTextualInversion(left,Some(right))
      case _ => throw IllegalArgumentException("SDXL PyTorch textual inversion must be a CLIP-L/CLIP-G mapping")
    else ClipTextualInversion(pytorchRows(sd1Tensor(archive.value),clipLWidth,"embedding"),None)

package scalatorch.models

import scalatorch.*
import scalatorch.io.{ReadLimits, SafeTensorSet, SafeDType, TensorInfo, PyTorchArchiveLimits, PyTorchTensorArchive}
import scalatorch.nn.{LinearAdapterScope, LowRankAdapter, Parameter}
import java.nio.file.Path
import scala.collection.mutable
import scala.util.Using

/** Bounded loader for linear, convolution, and LyCORIS (LoCon/LoHa/LoKr/DoRA/IA3/Full) LoRA tensor archives. */
object DiffusionLoRA:
  private val maxTensors=20000
  private val maxRank=256L
  private val maxElements=1073741824L

  private trait LoraArchive extends AutoCloseable:
    def tensors: Map[String,TensorInfo]
    def readFloat(key: String): Tensor[Float]
    def readDouble(key: String): Tensor[Double]

  private def openArchive(path: Path,elementBudget: Long): LoraArchive =
    val name=path.getFileName.toString.toLowerCase(java.util.Locale.ROOT)
    if name.endsWith(".safetensors") then
      val source=SafeTensorSet.open(path.toAbsolutePath.normalize().getParent,
        ReadLimits(maxHeaderBytes=16*1024*1024,maxTensors=maxTensors,maxRank=4,maxTensorBytes=256L*1024*1024),
        filename=path.getFileName.toString)
      new LoraArchive:
        val tensors=source.tensors
        def readFloat(key: String)=source.readFloat(key)
        def readDouble(key: String)=source.readDouble(key)
        def close(): Unit=source.close()
    else
      require(Vector(".pt",".pth",".bin",".ckpt").exists(name.endsWith),"unsupported LoRA archive format")
      import PyTorchTensorArchive.*
      val limits=PyTorchArchiveLimits(maxArchiveBytes=512L*1024*1024,maxEntries=20010,
        maxEntryBytes=256L*1024*1024,maxTotalUncompressedBytes=512L*1024*1024,
        maxPickleBytes=16*1024*1024,maxPickleOps=4000000,maxTensors=maxTensors,maxElements=64000000L,
        maxTotalTensorElements=math.min(maxElements,Math.addExact(elementBudget,maxTensors.toLong)))
      val value=PyTorchTensorArchive.read(path,PyTorchCompatibility.limits(limits)).value
      val entries=value match
        case DictValue(fields) if fields.contains("state_dict") => fields("state_dict") match
          case DictValue(state) => state
          case _ => throw IllegalArgumentException("LoRA state_dict must be a tensor dictionary")
        case DictValue(fields) => fields
        case _ => throw IllegalArgumentException("LoRA archive must be a tensor dictionary")
      val stored=entries.map { (key,value) => value match
        case TensorValue(tensor) => key->tensor
        case _ => throw IllegalArgumentException(s"LoRA archive value is not a tensor: $key")
      }
      new LoraArchive:
        val tensors=stored.map { (key,tensor) =>
          val dtype=tensor.dtype match
            case StorageDType.Float32 => SafeDType.F32
            case StorageDType.Float16 => SafeDType.F16
            case StorageDType.BFloat16 => SafeDType.BF16
            case _ => throw IllegalArgumentException("floating-point LoRA tensor has an integer dtype")
          key->TensorInfo(dtype,Shape.from(tensor.shape),0L,tensor.values.length.toLong*dtype.bytes)
        }
        def readFloat(key: String)=Tensor.fromFloats(stored(key).values,tensors(key).shape)
        def readDouble(key: String): Tensor[Double]=throw IllegalArgumentException("PyTorch Float64 LoRA is unsupported")
        def close(): Unit=()

  /** Original LDM U-Net layer aliases for the standard two-resnet SD1/SDXL topology. */
  private def ldmAlias(name: String,targets: Map[String,Parameter]): Option[String] =
    val down="down_blocks\\.(\\d+)\\.resnets\\.(\\d+)\\.(.+)".r
    val downAttn="down_blocks\\.(\\d+)\\.attentions\\.(\\d+)\\.(.+)".r
    val downsample="down_blocks\\.(\\d+)\\.downsamplers\\.0\\.conv".r
    val up="up_blocks\\.(\\d+)\\.resnets\\.(\\d+)\\.(.+)".r
    val upAttn="up_blocks\\.(\\d+)\\.attentions\\.(\\d+)\\.(.+)".r
    val upsample="up_blocks\\.(\\d+)\\.upsamplers\\.0\\.conv".r
    val mid="mid_block\\.resnets\\.([01])\\.(.+)".r
    val midAttn="mid_block\\.attentions\\.0\\.(.+)".r
    def resnet(s: String)=Vector("norm1"->"in_layers.0","conv1"->"in_layers.2",
      "norm2"->"out_layers.0","conv2"->"out_layers.3","time_emb_proj"->"emb_layers.1",
      "conv_shortcut"->"skip_connection").find((key,_) => s==key || s.startsWith(key+"."))
      .map((key,value) => value+s.drop(key.length)).getOrElse(s)
    val mapped=name.stripPrefix("unet.") match
      case down(i,j,rest) => Some(s"input_blocks.${i.toInt*3+j.toInt+1}.0.${resnet(rest)}")
      case downAttn(i,j,rest) => Some(s"input_blocks.${i.toInt*3+j.toInt+1}.1.$rest")
      case downsample(i) => Some(s"input_blocks.${i.toInt*3+3}.0.op")
      case up(i,j,rest) => Some(s"output_blocks.${i.toInt*3+j.toInt}.0.${resnet(rest)}")
      case upAttn(i,j,rest) => Some(s"output_blocks.${i.toInt*3+j.toInt}.1.$rest")
      case upsample(i) =>
        val hasAttention=targets.keysIterator.exists(_.startsWith(s"unet.up_blocks.$i.attentions.2."))
        Some(s"output_blocks.${i.toInt*3+2}.${if hasAttention then 2 else 1}.conv")
      case mid(i,rest) => Some(s"middle_block.${i.toInt*2}.${resnet(rest)}")
      case midAttn(rest) => Some(s"middle_block.1.$rest")
      case "conv_in" => Some("input_blocks.0.0")
      case "conv_norm_out" => Some("out.0")
      case "conv_out" => Some("out.2")
      case "time_embedding.linear_1" => Some("time_embed.0")
      case "time_embedding.linear_2" => Some("time_embed.2")
      case "add_embedding.linear_1" => Some("label_emb.0.0")
      case "add_embedding.linear_2" => Some("label_emb.0.2")
      case _ => None
    mapped.map(value => "lora_unet_"+value.replace('.','_'))

  def load(path: Path,targets: Map[String,Parameter],textEncoderScale: Float=1f,
      unetScale: Float=1f,elementBudget: Long=maxElements,dynamicRank: Option[Int]=None): Map[Parameter,Vector[LinearAdapterScope.Applied]] =
    require(textEncoderScale.isFinite && unetScale.isFinite,"LoRA scales must be finite")
    require(elementBudget>=0 && elementBudget<=maxElements,"invalid LoRA allocation budget")
    val canonical=targets++targets.iterator.map((name,parameter)=>s"base_model.model.$name"->parameter)
    val unetWeights=targets.iterator.collect { case (name,parameter) if name.startsWith("unet.") => parameter }.toSet
    val kohya=targets.iterator.flatMap { (name,parameter) =>
      val flattened=if name.startsWith("unet.") then
          val tail=name.stripPrefix("unet.").replace('.','_')
          Vector("lora_unet_"+tail,"lycoris_unet_"+tail)
        else if name.startsWith("text_encoder_2.") then
          val tail=name.stripPrefix("text_encoder_2.").replace('.','_')
          Vector("lora_te2_"+tail,"lycoris_te2_"+tail)
        else if name.startsWith("text_encoder.") then
          val tail=name.stripPrefix("text_encoder.").replace('.','_')
          Vector("lora_te_"+tail,"lora_te1_"+tail,"lycoris_te_"+tail,"lycoris_te1_"+tail)
        else Vector.empty
      flattened.iterator.map(_->parameter)
    }.toMap
    val ldm=targets.iterator.filter(_._1.startsWith("unet.")).flatMap { (name,parameter) =>
      ldmAlias(name,targets).toVector.flatMap { loraKey =>
        val tail=loraKey.stripPrefix("lora_unet_")
        Vector(loraKey -> parameter, ("lycoris_unet_" + tail) -> parameter)
      }
    }.toMap
    val targetIndex=canonical++kohya++ldm
    Using.resource(openArchive(path,elementBudget)) { archive =>
      require(archive.tensors.size<=maxTensors,"LoRA tensor count exceeds limit")
      val layers=mutable.Map.empty[String,LycorisAdapter.LayerTensors]
      var elements=0L
      def readFloat32(key: String,info: scalatorch.io.TensorInfo): Tensor[Float] =
        if info.dtype != scalatorch.io.SafeDType.F64 then archive.readFloat(key)
        else
          val source=archive.readDouble(key).toDoubleArray
          val converted=source.map(_.toFloat)
          require(source.indices.forall(i => source(i).isFinite && converted(i).isFinite),
            s"LoRA F64 tensor cannot be represented as finite Float32: $key")
          Tensor.fromFloats(converted,info.shape)
      archive.tensors.toVector.sortBy(_._1).foreach { (key,info) =>
        def layer(base: String)=layers.getOrElseUpdate(base,new LycorisAdapter.LayerTensors)
        LycorisAdapter.TensorSuffixes.find((suffix,_)=>key.endsWith(suffix)) match
          case Some((suffix,canonicalName)) =>
            val base=key.dropRight(suffix.length)
            require(targetIndex.contains(base),s"LoRA target is not a loaded Linear: $base")
            require(info.dtype==scalatorch.io.SafeDType.F16 || info.dtype==scalatorch.io.SafeDType.BF16 ||
              info.dtype==scalatorch.io.SafeDType.F32 || info.dtype==scalatorch.io.SafeDType.F64,
              s"unsupported LoRA factor dtype: $key")
            require(info.shape.dims.forall(_>0),s"LoRA factor dimensions must be positive: $key")
            require(elements<=elementBudget-info.shape.numel,"LoRA factors exceed allocation budget")
            elements=Math.addExact(elements,info.shape.numel)
            val read=readFloat32(key,info)
            val entry=layer(base)
            require(!entry.fields.contains(canonicalName),s"duplicate LoRA/LyCORIS factor $canonicalName: $base")
            entry.fields(canonicalName)=read
          case None if key.endsWith(".alpha") =>
            val base=key.stripSuffix(".alpha")
            require(targetIndex.contains(base),s"LoRA alpha has no loaded Linear target: $base")
            require(info.shape.numel==1,s"LoRA alpha must be scalar: $key")
            val value=readFloat32(key,info).toFloatArray.head.toDouble
            require(value.isFinite && value>0,s"invalid LoRA alpha: $key")
            val entry=layer(base)
            require(entry.alpha.isEmpty,s"duplicate LoRA alpha: $base")
            entry.alpha=Some(value)
          case None => throw IllegalArgumentException(s"unsupported LoRA tensor: $key")
      }
      val result=mutable.Map.empty[Parameter,Vector[LinearAdapterScope.Applied]]
      layers.foreach { (base,entry) =>
        val target=targetIndex(base)
        val adapter=LycorisAdapter.build(base,entry,target,maxRank,dynamicRank)
        val scale=if unetWeights(target) then unetScale else textEncoderScale
        result(target)=result.getOrElse(target,Vector.empty):+(adapter->scale)
      }
      require(layers.nonEmpty,"LoRA archive has no supported linear adapters")
      result.toMap
    }

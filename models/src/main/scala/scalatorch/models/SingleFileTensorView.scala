package scalatorch.models

import scalatorch.{Shape,Tensor}
import scalatorch.io.{PyTorchTensorArchive,SafeDType,SafeTensorSet,TensorInfo}

/** A non-owning, lazy name view. The caller keeps the underlying archive open. */
private[models] final class SingleFileTensorView private (
    sourceInfo: Map[String,TensorInfo], readFloatSource: String => Tensor[Float],
    readLongSource: String => Tensor[Long], names: Map[String,String]):
  val tensors: Map[String,TensorInfo] = names.map { (target,source) =>
    target -> sourceInfo.getOrElse(source,throw IllegalArgumentException(s"missing checkpoint tensor: $source"))
  }
  def readFloat(name: String): Tensor[Float] = readFloatSource(source(name))
  def readLong(name: String): Tensor[Long] = readLongSource(source(name))
  private def source(name: String): String =
    names.getOrElse(name,throw IllegalArgumentException(s"unmapped checkpoint tensor: $name"))

private[models] object SingleFileTensorView:
  /** Accept pairs so duplicate destination names cannot be silently overwritten. */
  def apply(archive: SafeTensorSet, mappings: Iterable[(String,String)]): SingleFileTensorView =
    create(archive.tensors,archive.readFloat,archive.readLong,mappings)

  def pytorch(stored: Map[String,PyTorchTensorArchive.StoredTensor],mappings: Iterable[(String,String)]): SingleFileTensorView =
    pytorchValues(stored.map((name,tensor)=>name->PyTorchTensorArchive.TensorValue(tensor)),mappings.toVector)

  private[models] def pytorchInfos(stored: Map[String,PyTorchTensorArchive.Value]): Map[String,TensorInfo] =
    stored.map { (name,value) =>
      import PyTorchTensorArchive.*
      val (dtype,shape,size)=value match
        case TensorMetadataValue(storage,shape) =>
          val dtype=storage match
            case StorageDType.Float32 => SafeDType.F32
            case StorageDType.Float16 => SafeDType.F16
            case StorageDType.BFloat16 => SafeDType.BF16
            case StorageDType.Int32 => SafeDType.I32
            case StorageDType.Int64 => SafeDType.I64
          val elements=shape.foldLeft(1L)((count,dim)=>Math.multiplyExact(count,dim))
          (dtype,shape,elements)
        case TensorValue(tensor) =>
          val dtype=tensor.dtype match
            case StorageDType.Float32 => SafeDType.F32
            case StorageDType.Float16 => SafeDType.F16
            case StorageDType.BFloat16 => SafeDType.BF16
            case _ => throw IllegalArgumentException("floating-point PyTorch tensor has an integer dtype")
          (dtype,tensor.shape,tensor.values.length.toLong)
        case IntegerTensorValue(tensor) =>
          val dtype=tensor.dtype match
            case StorageDType.Int32 => SafeDType.I32
            case StorageDType.Int64 => SafeDType.I64
            case _ => throw IllegalArgumentException("integer PyTorch tensor has a floating-point dtype")
          (dtype,tensor.shape,tensor.values.length.toLong)
        case _ => throw IllegalArgumentException(s"PyTorch checkpoint value is not a tensor: $name")
      name -> TensorInfo(dtype,Shape.from(shape),0L,Math.multiplyExact(size.toLong,dtype.bytes.toLong))
    }

  private def pytorchValues(stored: Map[String,PyTorchTensorArchive.Value],mappings: Vector[(String,String)]): SingleFileTensorView =
    import PyTorchTensorArchive.*
    val selected=mappings.map(_._2).toSet
    val infos=pytorchInfos(stored.filter((name,_)=>selected(name)))
    val remaining=scala.collection.mutable.Map.from(stored.iterator.filter((name,_)=>selected(name)))
    def take(name: String): Value = remaining.remove(name).getOrElse(
      throw IllegalArgumentException(s"PyTorch tensor already consumed: $name"))
    create(infos,name => take(name) match
      case TensorValue(tensor) => Tensor.fromFloats(tensor.values,infos(name).shape)
      case _ => throw IllegalArgumentException(s"PyTorch tensor is not floating point: $name"),
      name => take(name) match
        case IntegerTensorValue(tensor) if tensor.dtype==StorageDType.Int64 => Tensor.fromLongs(tensor.values,infos(name).shape)
        case _ => throw IllegalArgumentException(s"PyTorch tensor is not Int64: $name"),mappings)

  private def create(infos: Map[String,TensorInfo],readFloat: String => Tensor[Float],readLong: String => Tensor[Long],
      mappings: Iterable[(String,String)]): SingleFileTensorView =
    val pairs=mappings.toVector
    require(pairs.forall((target,source) => target.nonEmpty && source.nonEmpty),"empty checkpoint tensor name")
    require(pairs.map(_._1).distinct.size==pairs.size,"duplicate mapped checkpoint destination")
    require(pairs.map(_._2).distinct.size==pairs.size,"checkpoint tensor mapped more than once")
    new SingleFileTensorView(infos,readFloat,readLong,pairs.toMap)

  def clip(archive: SafeTensorSet): SingleFileTensorView =
    apply(archive,clipMappings(archive.tensors.keysIterator))

  def pytorchClip(values: Map[String,PyTorchTensorArchive.Value]): SingleFileTensorView =
    pytorchValues(values,clipMappings(values.keysIterator))

  /** Maps the SDXL CLIP-L (`conditioner.embedders.0.transformer.*`) state dictionary to Transformers CLIP names. */
  def sdxlClip(archive: SafeTensorSet): SingleFileTensorView =
    apply(archive,clipMappings(archive.tensors.keysIterator,"conditioner.embedders.0.transformer."))

  private[models] def pytorchSdxlClip(values: Map[String,PyTorchTensorArchive.Value]): SingleFileTensorView =
    pytorchValues(values,clipMappings(values.keysIterator,"conditioner.embedders.0.transformer."))

  /** Maps the OpenCLIP state dictionary used by SD2 single-file checkpoints to Transformers CLIP names. */
  def sd2Clip(archive: SafeTensorSet): SingleFileTensorView =
    openClip(archive.tensors,archive.readFloat,archive.readLong,archive.tensors.keysIterator,SingleFileSd2Config.clipConfig.layers)

  private[models] def pytorchSd2Clip(values: Map[String,PyTorchTensorArchive.Value]): SingleFileTensorView =
    pytorchOpenClip(values,SingleFileSd2Config.clipConfig.layers,"cond_stage_model.model.",includeProjection=false)

  /** Maps the OpenCLIP ViT-bigG state dictionary (`conditioner.embedders.1.model.*`) used by SDXL single-file checkpoints. */
  def sdxlClip2(archive: SafeTensorSet): SingleFileTensorView =
    openClip(archive.tensors,archive.readFloat,archive.readLong,archive.tensors.keysIterator,
      SingleFileSdxlConfig.clip2Config.layers,"conditioner.embedders.1.model.",includeProjection=true)

  private[models] def pytorchSdxlClip2(values: Map[String,PyTorchTensorArchive.Value]): SingleFileTensorView =
    pytorchOpenClip(values,SingleFileSdxlConfig.clip2Config.layers,"conditioner.embedders.1.model.",includeProjection=true)

  private def pytorchOpenClip(values: Map[String,PyTorchTensorArchive.Value],layers: Int,prefix: String,
      includeProjection: Boolean): SingleFileTensorView =
    import PyTorchTensorArchive.*
    val infos=pytorchInfos(values)
    def readFloat(name: String): Tensor[Float] = values.getOrElse(name,
      throw IllegalArgumentException(s"missing OpenCLIP tensor: $name")) match
      case TensorValue(tensor) => Tensor.fromFloats(tensor.values,infos(name).shape)
      case _ => throw IllegalArgumentException(s"OpenCLIP tensor is not floating point: $name")
    def readLong(name: String): Tensor[Long] = values.getOrElse(name,
      throw IllegalArgumentException(s"missing OpenCLIP tensor: $name")) match
      case IntegerTensorValue(tensor) if tensor.dtype==StorageDType.Int64 => Tensor.fromLongs(tensor.values,infos(name).shape)
      case _ => throw IllegalArgumentException(s"OpenCLIP tensor is not Int64: $name")
    openClip(infos,readFloat,readLong,values.keysIterator,layers,prefix,includeProjection)

  private def openClip(sourceInfo: Map[String,TensorInfo],readFloatSource: String=>Tensor[Float],
      readLongSource: String=>Tensor[Long],names: Iterator[String],layers: Int,
      prefix: String="cond_stage_model.model.",includeProjection: Boolean=false): SingleFileTensorView =
    val mappings=Vector.newBuilder[(String,String)]
    val slices=scala.collection.mutable.Map.empty[String,(String,Int,Shape)]
    val transposed=scala.collection.mutable.Map.empty[String,(String,Shape)]
    def add(source: String,target: String): Unit=mappings+=target->source
    def mapLayer(source: String,tail: String): Unit=
      val layer="transformer\\.resblocks\\.(\\d+)\\.(.+)".r
      tail match
        case layer(index,rest) if index.toInt<layers =>
          val base=s"text_model.encoder.layers.$index."
          val direct=Vector(
            "ln_1.weight"->"layer_norm1.weight","ln_1.bias"->"layer_norm1.bias",
            "ln_2.weight"->"layer_norm2.weight","ln_2.bias"->"layer_norm2.bias",
            "attn.out_proj.weight"->"self_attn.out_proj.weight","attn.out_proj.bias"->"self_attn.out_proj.bias",
            "mlp.c_fc.weight"->"mlp.fc1.weight","mlp.c_fc.bias"->"mlp.fc1.bias",
            "mlp.c_proj.weight"->"mlp.fc2.weight","mlp.c_proj.bias"->"mlp.fc2.bias")
          direct.find((from,_)=>from==rest).foreach((_,to)=>add(source,base+to))
          val projection="attn\\.in_proj_(weight|bias)".r
          rest match
            case projection(kind) =>
              val info=sourceInfo.getOrElse(source,throw IllegalArgumentException(s"missing OpenCLIP tensor: $source"))
              val dims=info.shape.dims
              require(dims.nonEmpty && dims.head%3==0,s"invalid fused OpenCLIP QKV shape for $source: ${info.shape}")
              val rows=dims.head/3
              val shape=Shape.from(rows.toLong+:dims.tail)
              val suffix=if kind=="weight" then "weight" else "bias"
              Vector("q_proj","k_proj","v_proj").zipWithIndex.foreach { (name,part) =>
                val synthetic=s"$source#${name}"
                val target=s"${base}self_attn.$name.$suffix"
                slices(synthetic)=(source,part,shape)
                mappings+=target->synthetic
              }
            case _ => ()
        case _ => ()
    names.filter(_.startsWith(prefix)).foreach { source =>
      val tail=source.stripPrefix(prefix)
      tail match
        case "token_embedding.weight" => add(source,"text_model.embeddings.token_embedding.weight")
        case "positional_embedding" => add(source,"text_model.embeddings.position_embedding.weight")
        case "ln_final.weight" => add(source,"text_model.final_layer_norm.weight")
        case "ln_final.bias" => add(source,"text_model.final_layer_norm.bias")
        case "text_projection" if includeProjection =>
          val info=sourceInfo.getOrElse(source,throw IllegalArgumentException(s"missing OpenCLIP tensor: $source"))
          val dims=info.shape.dims
          require(dims.size==2,s"invalid OpenCLIP text_projection shape for $source: ${info.shape}")
          val shape=Shape.from(Seq(dims(1),dims(0)))
          val synthetic=s"$source#transpose"
          transposed(synthetic)=(source,shape)
          mappings+="text_projection.weight"->synthetic
        case "text_projection" | "logit_scale" => ()
        case value => mapLayer(source,value)
    }
    val pairs=mappings.result()
    require(pairs.nonEmpty,"missing OpenCLIP tensors")
    val syntheticInfo=slices.map { case (name,(source,_,shape)) =>
      val original=sourceInfo(source)
      val bytes=Math.multiplyExact(shape.numel,original.dtype.bytes.toLong)
      name->TensorInfo(original.dtype,shape,0L,bytes)
    }.toMap++transposed.map { case (name,(source,shape)) =>
      val original=sourceInfo(source)
      val bytes=Math.multiplyExact(shape.numel,original.dtype.bytes.toLong)
      name->TensorInfo(original.dtype,shape,0L,bytes)
    }
    val infos=sourceInfo++syntheticInfo
    val splitCache=scala.collection.mutable.Map.empty[String,Array[Float]]
    val readFloat=(name: String) => slices.get(name) match
      case Some((source,part,shape)) =>
        val values=splitCache.getOrElseUpdate(source,readFloatSource(source).toFloatArray)
        val size=shape.numel.toInt
        Tensor.fromFloats(java.util.Arrays.copyOfRange(values,part*size,(part+1)*size),shape)
      case None => transposed.get(name) match
        case Some((source,shape)) =>
          val original=readFloatSource(source)
          val rows=original.shape.dims(0).toInt
          val cols=original.shape.dims(1).toInt
          val in=original.toFloatArray
          val out=new Array[Float](in.length)
          var r=0
          while r<rows do
            var c=0
            while c<cols do
              out(c*rows+r)=in(r*cols+c)
              c+=1
            r+=1
          Tensor.fromFloats(out,shape)
        case None => readFloatSource(name)
    create(infos,readFloat,readLongSource,pairs)

  private def clipMappings(names: Iterator[String],prefix: String="cond_stage_model.transformer."): Vector[(String,String)] =
    val mappings=names.filter(_.startsWith(prefix)).map { source =>
      source.stripPrefix(prefix) -> source
    }.toVector
    require(mappings.nonEmpty,"missing SD1 CLIP tensors")
    mappings

  /** SD1 VAE names follow the LDM encoder/decoder block numbering. */
  def vae(archive: SafeTensorSet): SingleFileTensorView =
    apply(archive,vaeMappings(archive.tensors.keysIterator))

  def pytorchVae(values: Map[String,PyTorchTensorArchive.Value]): SingleFileTensorView =
    pytorchValues(values,vaeMappings(values.keysIterator))

  private def vaeMappings(names: Iterator[String]): Vector[(String,String)] =
    val prefix="first_stage_model."
    def rename(name: String): String =
      val down="encoder\\.down\\.(\\d+)\\.block\\.(\\d+)\\.(.+)".r
      val downsample="encoder\\.down\\.(\\d+)\\.downsample\\.conv\\.(.+)".r
      val up="decoder\\.up\\.(\\d+)\\.block\\.(\\d+)\\.(.+)".r
      val upsample="decoder\\.up\\.(\\d+)\\.upsample\\.conv\\.(.+)".r
      val midBlock="(encoder|decoder)\\.mid\\.block_([12])\\.(.+)".r
      val midAttention="(encoder|decoder)\\.mid\\.attn_1\\.(.+)".r
      def resnet(s: String)=if s.startsWith("nin_shortcut.") then "conv_shortcut."+s.stripPrefix("nin_shortcut.") else s
      name match
        case down(i,j,rest) => s"encoder.down_blocks.$i.resnets.$j.${resnet(rest)}"
        case downsample(i,rest) => s"encoder.down_blocks.$i.downsamplers.0.conv.$rest"
        case up(i,j,rest) => s"decoder.up_blocks.${3-i.toInt}.resnets.$j.${resnet(rest)}"
        case upsample(i,rest) => s"decoder.up_blocks.${3-i.toInt}.upsamplers.0.conv.$rest"
        case midBlock(side,i,rest) => s"$side.mid_block.resnets.${i.toInt-1}.${resnet(rest)}"
        case midAttention(side,rest) =>
          val mapped=Vector("norm."->"group_norm.","q."->"to_q.","k."->"to_k.",
            "v."->"to_v.","proj_out."->"to_out.0.").find((a,_) => rest.startsWith(a))
            .map((a,b) => b+rest.stripPrefix(a)).getOrElse(rest)
          s"$side.mid_block.attentions.0.$mapped"
        case other => other.replace("encoder.norm_out.","encoder.conv_norm_out.")
          .replace("decoder.norm_out.","decoder.conv_norm_out.")
    val mappings=names.filter(_.startsWith(prefix)).map(source =>
      rename(source.stripPrefix(prefix)) -> source).toVector
    require(mappings.nonEmpty,"missing SD1 VAE tensors")
    mappings

  def unet(archive: SafeTensorSet): SingleFileTensorView =
    apply(archive,unetMappings(archive.tensors.keysIterator))

  def pytorchUnet(values: Map[String,PyTorchTensorArchive.Value]): SingleFileTensorView =
    pytorchValues(values,unetMappings(values.keysIterator))

  private def unetMappings(names: Iterator[String]): Vector[(String,String)] =
    val prefix="model.diffusion_model."
    def resnet(s: String): String = s.replace("in_layers.0.","norm1.")
      .replace("in_layers.2.","conv1.").replace("out_layers.0.","norm2.")
      .replace("out_layers.3.","conv2.").replace("emb_layers.1.","time_emb_proj.")
      .replace("skip_connection.","conv_shortcut.")
    def rename(s: String): String =
      val input="input_blocks\\.(\\d+)\\.([01])\\.(.+)".r
      val middle="middle_block\\.([012])\\.(.+)".r
      val output="output_blocks\\.(\\d+)\\.([012])\\.(.+)".r
      s match
        case input("0","0",rest) => s"conv_in.$rest"
        case input(i,"0",rest) if rest.startsWith("op.") =>
          s"down_blocks.${(i.toInt-1)/3}.downsamplers.0.conv.${rest.stripPrefix("op.")}"
        case input(i,"0",rest) => s"down_blocks.${(i.toInt-1)/3}.resnets.${(i.toInt-1)%3}.${resnet(rest)}"
        case input(i,"1",rest) => s"down_blocks.${(i.toInt-1)/3}.attentions.${(i.toInt-1)%3}.$rest"
        case middle("1",rest) => s"mid_block.attentions.0.$rest"
        case middle(i,rest) => s"mid_block.resnets.${i.toInt/2}.${resnet(rest)}"
        case output(i,"0",rest) => s"up_blocks.${i.toInt/3}.resnets.${i.toInt%3}.${resnet(rest)}"
        case output(i,_,rest) if rest.startsWith("conv.") =>
          s"up_blocks.${i.toInt/3}.upsamplers.0.conv.${rest.stripPrefix("conv.")}"
        case output(i,"1",rest) => s"up_blocks.${i.toInt/3}.attentions.${i.toInt%3}.$rest"
        case other => Vector("time_embed.0."->"time_embedding.linear_1.","time_embed.2."->"time_embedding.linear_2.",
          "label_emb.0.0."->"add_embedding.linear_1.","label_emb.0.2."->"add_embedding.linear_2.",
          "out.0."->"conv_norm_out.","out.2."->"conv_out.").find((a,_) => other.startsWith(a))
          .map((a,b) => b+other.stripPrefix(a)).getOrElse(other)
    val mappings=names.filter(_.startsWith(prefix)).map(source =>
      rename(source.stripPrefix(prefix)) -> source).toVector
    require(mappings.nonEmpty,"missing SD1 U-Net tensors")
    mappings

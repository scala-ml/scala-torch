package scalatorch.models

import scalatorch.*
import scalatorch.io.{ReadLimits, SafeTensorSet}
import scalatorch.nn.{LinearAdapterScope, LowRankAdapter, Parameter}
import java.nio.file.Path
import scala.collection.mutable
import scala.util.Using

/** Bounded loader for standard linear LoRA safetensors. */
object DiffusionLoRA:
  private val maxTensors=20000
  private val maxRank=256L
  private val maxElements=50000000L

  def load(path: Path,targets: Map[String,Parameter],textEncoderScale: Float=1f,
      unetScale: Float=1f,elementBudget: Long=maxElements): Map[Parameter,Vector[LinearAdapterScope.Applied]] =
    require(textEncoderScale.isFinite && unetScale.isFinite,"LoRA scales must be finite")
    require(elementBudget>=0 && elementBudget<=maxElements,"invalid LoRA allocation budget")
    val canonical=targets++targets.iterator.map((name,parameter)=>s"base_model.model.$name"->parameter)
    val unetWeights=targets.iterator.collect { case (name,parameter) if name.startsWith("unet.") => parameter }.toSet
    val kohya=targets.iterator.flatMap { (name,parameter) =>
      val flattened=if name.startsWith("unet.") then Vector("lora_unet_"+name.stripPrefix("unet.").replace('.','_'))
        else if name.startsWith("text_encoder_2.") then Vector("lora_te2_"+name.stripPrefix("text_encoder_2.").replace('.','_'))
        else if name.startsWith("text_encoder.") then Vector("lora_te_"+name.stripPrefix("text_encoder.").replace('.','_'),
          "lora_te1_"+name.stripPrefix("text_encoder.").replace('.','_'))
        else Vector.empty
      flattened.iterator.map(_->parameter)
    }.toMap
    val targetIndex=canonical++kohya
    Using.resource(SafeTensorSet.open(path.toAbsolutePath.normalize().getParent,
      ReadLimits(maxHeaderBytes=16*1024*1024,maxTensors=maxTensors,maxRank=4,maxTensorBytes=256L*1024*1024),
      filename=path.getFileName.toString)) { archive =>
      require(archive.tensors.size<=maxTensors,"LoRA tensor count exceeds limit")
      final class Pair:
        var down: Option[Tensor[Float]]=None
        var up: Option[Tensor[Float]]=None
        var alpha: Option[Double]=None
      val pairs=mutable.Map.empty[String,Pair]
      var elements=0L
      archive.tensors.toVector.sortBy(_._1).foreach { (key,info) =>
        def pair(base: String)=pairs.getOrElseUpdate(base,new Pair)
        val suffixes=Vector(".lora_down.weight"->"down", ".lora_up.weight"->"up",
          ".lora_A.weight"->"down", ".lora_B.weight"->"up",
          ".lora_A.default.weight"->"down", ".lora_B.default.weight"->"up")
        suffixes.find((suffix,_)=>key.endsWith(suffix)) match
          case Some((suffix,kind)) =>
            val base=key.dropRight(suffix.length)
            val target=targetIndex.getOrElse(base,throw IllegalArgumentException(s"LoRA target is not a loaded Linear: $base"))
            require(info.dtype==scalatorch.io.SafeDType.F16 || info.dtype==scalatorch.io.SafeDType.BF16 ||
              info.dtype==scalatorch.io.SafeDType.F32 || info.dtype==scalatorch.io.SafeDType.F64,
              s"unsupported LoRA factor dtype: $key")
            require(info.shape.rank==2 && info.shape.dims.forall(_>0),s"LoRA factors must be nonempty matrices: $key")
            val baseShape=target.data.shape
            if kind=="down" then require(info.shape.dims(0)<=maxRank && info.shape.dims(1)==baseShape.dims(1),
              s"LoRA down factor shape does not match target: $base")
            else require(info.shape.dims(1)<=maxRank && info.shape.dims(0)==baseShape.dims(0),
              s"LoRA up factor shape does not match target: $base")
            require(elements<=elementBudget-info.shape.numel,"LoRA factors exceed allocation budget")
            elements=Math.addExact(elements,info.shape.numel)
            val tensor=archive.readFloat(key)
            if kind=="down" then
              require(pair(base).down.isEmpty,s"duplicate LoRA A/down factor: $base");pair(base).down=Some(tensor)
            else
              require(pair(base).up.isEmpty,s"duplicate LoRA B/up factor: $base");pair(base).up=Some(tensor)
          case None if key.endsWith(".alpha") =>
            val base=key.stripSuffix(".alpha")
            require(targetIndex.contains(base),s"LoRA alpha has no loaded Linear target: $base")
            require(info.shape.numel==1,s"LoRA alpha must be scalar: $key")
            val value=archive.readFloat(key).toFloatArray.head.toDouble
            require(value.isFinite && value>0,s"invalid LoRA alpha: $key")
            require(pair(base).alpha.isEmpty,s"duplicate LoRA alpha: $base");pair(base).alpha=Some(value)
          case None => throw IllegalArgumentException(s"unsupported LoRA tensor: $key")
      }
      val result=mutable.Map.empty[Parameter,Vector[LinearAdapterScope.Applied]]
      pairs.foreach { (base,pair) =>
        val down=pair.down.getOrElse(throw IllegalArgumentException(s"missing LoRA A/down factor: $base"))
        val up=pair.up.getOrElse(throw IllegalArgumentException(s"missing LoRA B/up factor: $base"))
        val target=targetIndex(base)
        val rank=down.shape.dims(0)
        require(rank<=maxRank && up.shape.dims(1)==rank && down.shape.dims(1)==target.data.shape.dims(1) &&
          up.shape.dims(0)==target.data.shape.dims(0),s"LoRA factor shape does not match target: $base")
        val adapter=LowRankAdapter(Parameter(down,false),Parameter(up,false),pair.alpha.getOrElse(rank.toDouble))
        val scale=if unetWeights(target) then unetScale else textEncoderScale
        result(target)=result.getOrElse(target,Vector.empty):+(adapter->scale)
      }
      require(pairs.nonEmpty,"LoRA archive has no supported linear adapters")
      result.toMap
    }

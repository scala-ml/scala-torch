package scalatorch.models

/** Strict bounded interpretation of the existing TrainRun v1 context, not executable configuration. */
private[models] object TrainingRunMetadata:
  final case class Adapter(rank: Int,alpha: Double,layers: Set[Int],seed: Long)
  final case class Settings(tokenizerHash: String,adapter: Option[Adapter])
  def parse(context: String): Settings =
    require(context.length<=4096,"oversized training metadata")
    val parts=context.split(";",-1).toVector
    require(parts.headOption.contains("scala-torch-train-v1"),
      "checkpoint lacks supported TrainRun metadata; use TrainedLlama.restore with a matching model")
    val entries=parts.tail.map { field =>
      val pair=field.split("=",2)
      require(pair.length==2,"invalid training metadata field")
      pair(0) -> pair(1)
    }
    require(entries.map(_._1).distinct.size==entries.size,"duplicate training metadata field")
    val fields=entries.toMap
    require(fields.keySet==Set("corpus","tokenizer","sequence","batch","mode"),"training metadata fields mismatch")
    require(Seq("corpus","tokenizer").forall(key => fields(key).matches("[0-9a-f]{64}")),"invalid training fingerprint")
    val sequence=fields("sequence").toInt;val batch=fields("batch").toInt
    require(sequence>0 && batch>0 && sequence.toLong*batch<=1048576,"invalid training dimensions")
    val adapter=if fields("mode")=="full" then None else
      val mode=fields("mode").split(":",-1)
      require(mode.length==5 && mode(0)=="lora","unsupported checkpoint training mode")
      val rank=mode(1).toInt;val alpha=mode(2).toDouble
      require(rank>0 && alpha.isFinite && alpha>0 && (alpha/rank).toFloat.isFinite && (alpha/rank).toFloat>0,"invalid checkpoint adapter scale")
      val layers=mode(3).split(",",-1).map(_.toInt)
      require(layers.nonEmpty && layers.forall(_>=0) && layers.distinct.length==layers.length,"invalid checkpoint adapter layers")
      Some(Adapter(rank,alpha,layers.toSet,mode(4).toLong))
    Settings(fields("tokenizer"),adapter)

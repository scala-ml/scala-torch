package scalatorch.models

import scalatorch.*
import scalatorch.cpu.CpuFloatBackend
import scalatorch.ffm.CudaSession
import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files,Path}
import java.security.MessageDigest
import scala.collection.mutable
import scala.util.Using

/** Offline deterministic text training. No download, Python runtime or implicit chat template. */
object TrainRun:
  val usage="""usage: TrainRun cpu|cuda MODEL_DIRECTORY DATA.txt CHECKPOINT TOTAL_STEPS
    |  [--resume] [--full] [--sequence-length N] [--batch-size N] [--learning-rate RATE]
    |  [--lora-rank N] [--lora-alpha A] [--lora-layers last|all|0,1,...] [--seed N]
    |  [--save-every N] [--device-bytes N]
    |Default: last-layer q/v LoRA, rank4/alpha8/seed42, sequence64/batch1, AdamW lr0.001,
    |save every10 updates and at completion, CUDA budget2GiB. TOTAL_STEPS is the final global step.
    |Data: bounded UTF-8 text, one document per nonempty line; EOS is added if absent.
    |Resume requires the same corpus, tokenizer, batch/adapter settings and optimizer configuration.
    |""".stripMargin
  private[models] final case class Options(backend: String,modelDirectory: Path,data: Path,checkpoint: Path,totalSteps: Long,
      resume: Boolean = false,full: Boolean = false,sequenceLength: Int = 64,batchSize: Int = 1,
      learningRate: Double = .001,rank: Int = 4,alpha: Double = 8,layers: String = "last",seed: Long = 42,
      saveEvery: Int = 10,deviceBytes: Long = 2147483648L):
    require(Set("cpu","cuda")(backend),"backend must be cpu or cuda")
    require(totalSteps>=0,"negative total steps")
    require(sequenceLength>0 && batchSize>0 && sequenceLength.toLong*batchSize<=1048576,"invalid training batch dimensions")
    require(saveEvery>0 && deviceBytes>0,"invalid checkpoint interval or device budget")
    require(rank>0 && alpha.isFinite && alpha>0 && (alpha/rank).toFloat.isFinite && (alpha/rank).toFloat>0,"invalid LoRA rank/alpha")
    val optimizer=AdamWConfig(learningRate)
    if layers!="last" && layers!="all" then
      val ids=layers.split(",",-1).map(_.toInt)
      require(ids.nonEmpty && ids.forall(_>=0) && ids.distinct.length==ids.length,"invalid LoRA layers")
  private[models] def parse(args: Array[String]): Options =
    require(args.length>=5,usage)
    val flags=mutable.Set.empty[String]
    val values=mutable.Map.empty[String,String]
    val boolean=Set("--resume","--full")
    val valued=Set("--sequence-length","--batch-size","--learning-rate","--lora-rank","--lora-alpha",
      "--lora-layers","--seed","--save-every","--device-bytes")
    var i=5
    while i<args.length do
      val key=args(i)
      require((boolean++valued)(key),s"unknown option: $key\n$usage")
      require(!flags(key),s"duplicate option: $key")
      flags+=key
      if valued(key) then
        require(i+1<args.length,s"missing value for $key")
        values(key)=args(i+1);i+=1
      i+=1
    require(!flags("--full") || !flags.exists(_.startsWith("--lora-")),"--full conflicts with LoRA options")
    Options(args(0),Path.of(args(1)),Path.of(args(2)),Path.of(args(3)),args(4).toLong,
      flags("--resume"),flags("--full"),values.getOrElse("--sequence-length","64").toInt,
      values.getOrElse("--batch-size","1").toInt,values.getOrElse("--learning-rate","0.001").toDouble,
      values.getOrElse("--lora-rank","4").toInt,values.getOrElse("--lora-alpha","8").toDouble,
      values.getOrElse("--lora-layers","last"),values.getOrElse("--seed","42").toLong,
      values.getOrElse("--save-every","10").toInt,values.getOrElse("--device-bytes","2147483648").toLong)
  private[models] def run(options: Options,log: String => Unit): Unit =
    require(!Files.exists(options.checkpoint) || options.resume,"checkpoint already exists; use --resume or choose another path")
    require(!options.resume || Files.isRegularFile(options.checkpoint),"resume checkpoint does not exist")
    val config=LlamaConfig.read(options.modelDirectory.resolve("config.json"))
    require(options.sequenceLength<=config.maxPositions,"sequence length exceeds model context")
    require(options.sequenceLength.toLong*options.batchSize*config.vocabSize<=LlamaLimits().maxLogitsElements,"training logits allocation limit exceeded")
    require(config.eosTokenIds.nonEmpty,"training data requires a configured EOS token")
    val bytes=Using.resource(Files.newInputStream(options.modelDirectory.resolve("tokenizer.json")))(_.readNBytes(TokenizerJson.maxBytes+1))
    require(bytes.length<=TokenizerJson.maxBytes,"tokenizer exceeds 32MiB")
    val tokenizer=ByteBpeTokenizer.fromJson(UTF_8.newDecoder().decode(ByteBuffer.wrap(bytes)).toString)
    require(tokenizer.vocabSize==config.vocabSize,"tokenizer/model vocabulary size mismatch")
    val tokenizerHash=MessageDigest.getInstance("SHA-256").digest(bytes).map(b => f"${b & 255}%02x").mkString
    val dataset=TextDataset.read(options.data,tokenizer,config.eosTokenIds.head,options.sequenceLength,options.batchSize)
    val selected=if options.full then Set.empty[Int] else options.layers match
      case "last" => Set(config.numLayers-1)
      case "all" => (0 until config.numLayers).toSet
      case ids => ids.split(",").map(_.toInt).toSet
    require(selected.forall(i => i>=0 && i<config.numLayers),"LoRA layer outside model")
    val mode=if options.full then "full" else
      s"lora:${options.rank}:${java.lang.Double.toHexString(options.alpha)}:${selected.toVector.sorted.mkString(",")}:${options.seed}"
    val context=s"scala-torch-train-v1;corpus=${dataset.fingerprint};tokenizer=$tokenizerHash;sequence=${options.sequenceLength};batch=${options.batchSize};mode=$mode"
    val base=LlamaModel.load(options.modelDirectory,trainable=options.full)
    val model=if options.full then base else LlamaLoRA.attach(base,options.rank,options.alpha,options.seed,targetLayers=Some(selected)).model
    def train[T](backend: AdamWBackend[T]): Unit = Using.resource(BackendLlamaTraining.prepareAdamW(model,backend,options.optimizer)) { prepared =>
      var step=if options.resume then prepared.loadCheckpoint(options.checkpoint,context=Some(context)) else 0L
      require(step<=options.totalSteps,"total steps precede saved checkpoint step")
      log(s"backend=${options.backend} mode=$mode tokens=${dataset.tokenCount} batches_per_epoch=${dataset.batchesPerEpoch} start_step=$step")
      var saved=if options.resume then step else -1L
      def save(): Unit =
        prepared.saveCheckpoint(options.checkpoint,step,context=context)
        saved=step;log(s"saved_step=$step checkpoint=${options.checkpoint}")
      while step<options.totalSteps do
        val batch=dataset.batch(step)
        val value=Using.resource(prepared.newStep()) { current =>
          val loss=current.forward(batch.input).crossEntropy(batch.targets)
          val value=backend.download(loss.data).toFloatArray.head
          require(value.isFinite,"nonfinite training loss")
          loss.backward();current.adamW();value
        }
        step=Math.addExact(step,1L)
        log(s"step=$step loss=$value target_tokens=${batch.targetCount}")
        if step%options.saveEvery==0 || step==options.totalSteps then save()
      if saved!=step then save()
      log(s"completed_step=$step")
    }
    if options.backend=="cuda" then CudaSession.withDevice(maxBytes=options.deviceBytes)(train(_))
    else train(CpuFloatBackend)
  def main(args: Array[String]): Unit =
    if args.sameElements(Array("--help")) then println(usage)
    else run(parse(args),println)

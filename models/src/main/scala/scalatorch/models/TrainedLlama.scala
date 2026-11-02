package scalatorch.models

import scalatorch.nn.Module
import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files,Path}
import java.security.MessageDigest
import scala.util.Using

/** Inference-ready host weights and tokenizer restored without retaining optimizer moments. */
final case class TrainedLlama(model: LlamaModel,tokenizer: ByteBpeTokenizer,trainingStep: Long)

object TrainedLlama:
  /** Restore into a matching full/LoRA template, including metadata-free and legacy checkpoints.
    * The template changes only after architecture, values, moments and checksum validate.
    */
  def restore(model: LlamaModel,checkpoint: Path,maxBytes: Long = 4294967296L,context: Option[String] = None): Long =
    val saved=BackendLlamaCheckpoint.loadWeights(model,checkpoint,maxBytes,context)
    Module.loadStateDict(model,saved.weights)
    saved.step

  /** Reconstruct a TrainRun model using its saved adapter metadata and exact tokenizer file. */
  def load(directory: Path,checkpoint: Path,maxBytes: Long = 4294967296L,limits: LlamaLimits = LlamaLimits()): TrainedLlama =
    val context=BackendLlamaCheckpoint.savedContext(checkpoint,maxBytes)
    val metadata=TrainingRunMetadata.parse(context)
    val bytes=Using.resource(Files.newInputStream(directory.resolve("tokenizer.json")))(_.readNBytes(TokenizerJson.maxBytes+1))
    require(bytes.length<=TokenizerJson.maxBytes,"tokenizer exceeds 32MiB")
    val hash=MessageDigest.getInstance("SHA-256").digest(bytes).map(b => f"${b & 255}%02x").mkString
    require(hash==metadata.tokenizerHash,"checkpoint tokenizer fingerprint mismatch")
    val tokenizer=ByteBpeTokenizer.fromJson(UTF_8.newDecoder().decode(ByteBuffer.wrap(bytes)).toString)
    val base=LlamaModel.load(directory,trainable=metadata.adapter.isEmpty,limits=limits)
    require(tokenizer.vocabSize==base.config.vocabSize,"tokenizer/model vocabulary size mismatch")
    val model=metadata.adapter.fold(base) { a =>
      LlamaLoRA.attach(base,a.rank,a.alpha,a.seed,targetLayers=Some(a.layers)).model
    }
    val step=restore(model,checkpoint,maxBytes,Some(context))
    TrainedLlama(model,tokenizer,step)

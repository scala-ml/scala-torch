package scalatorch.models

import scalatorch.FloatBackend
import scalatorch.cpu.CpuFloatBackend
import scalatorch.ffm.CudaSession
import java.nio.file.Path
import scala.util.Using

/** Offline text generation with resident CPU or CUDA weights and native tokenization. */
object BackendTextRun:
  private[models] final case class Options(backend: String,directory: Path,prompt: String,count: Int,
      chat: Boolean,checkpoint: Option[Path])
  private[models] def parse(args: Array[String]): Options =
    require(args.length>=3,
      "usage: BackendTextRun cpu|cuda MODEL_DIRECTORY PROMPT [MAX_NEW_TOKENS] [--smollm-chat] [--checkpoint PATH]")
    require(args(0)=="cpu" || args(0)=="cuda","backend must be cpu or cuda")
    var index=3
    val count=if index<args.length && !args(index).startsWith("--") then
      index+=1
      args(index-1).toInt
    else 32
    require(count>=0,"negative max new tokens")
    var chat=false
    var checkpoint=Option.empty[Path]
    while index<args.length do
      args(index) match
        case "--smollm-chat" =>
          require(!chat,"duplicate --smollm-chat")
          chat=true
        case "--checkpoint" =>
          require(checkpoint.isEmpty,"duplicate --checkpoint")
          index+=1
          require(index<args.length && !args(index).startsWith("--"),"missing checkpoint path")
          checkpoint=Some(Path.of(args(index)))
        case other => throw new IllegalArgumentException(s"unsupported option: $other")
      index+=1
    Options(args(0),Path.of(args(1)),args(2),count,chat,checkpoint)

  def main(args: Array[String]): Unit = println(run(parse(args)))

  private[models] def run(options: Options): String =
    val prompt=if options.chat then SmolChat.format(Vector(ChatMessage("user",options.prompt))) else options.prompt
    val (model,tokenizer)=options.checkpoint match
      case Some(path) =>
        val loaded=TrainedLlama.load(options.directory,path)
        (loaded.model,loaded.tokenizer)
      case None => (LlamaModel.load(options.directory),ByteBpeTokenizer.read(options.directory.resolve("tokenizer.json")))
    require(tokenizer.vocabSize==model.config.vocabSize,"tokenizer/model vocabulary size mismatch")
    val ids=tokenizer.encode(prompt).toVector
    def generate[T](backend: FloatBackend[T]): String =
      Using.resource(BackendLlama.prepare(model,backend)) { prepared =>
        tokenizer.decode(prepared.generateIds(ids,options.count).drop(ids.size).toArray)
      }
    if options.backend=="cuda" then CudaSession.withDevice()(generate(_)) else generate(CpuFloatBackend)

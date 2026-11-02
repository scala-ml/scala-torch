package scalatorch.models

import scalatorch.FloatBackend
import scalatorch.cpu.CpuFloatBackend
import scalatorch.ffm.CudaSession
import java.nio.file.Path
import scala.util.Using

/** Offline text generation with resident CPU or CUDA weights and native tokenization. */
object BackendTextRun:
  def main(args: Array[String]): Unit =
    require(args.length>=3 && args.length<=5,
      "usage: BackendTextRun cpu|cuda MODEL_DIRECTORY PROMPT [MAX_NEW_TOKENS [--smollm-chat]]")
    require(args(0)=="cpu" || args(0)=="cuda","backend must be cpu or cuda")
    require(args.length<5 || args(4)=="--smollm-chat","unsupported chat option")
    val count=if args.length>=4 then args(3).toInt else 32
    require(count>=0,"negative max new tokens")
    val directory=Path.of(args(1))
    val prompt=if args.length==5 then SmolChat.format(Vector(ChatMessage("user",args(2)))) else args(2)
    val tokenizer=ByteBpeTokenizer.read(directory.resolve("tokenizer.json"))
    val model=LlamaModel.load(directory)
    require(tokenizer.vocabSize==model.config.vocabSize,"tokenizer/model vocabulary size mismatch")
    val ids=tokenizer.encode(prompt).toVector
    def generate[T](backend: FloatBackend[T]): String =
      Using.resource(BackendLlama.prepare(model,backend)) { prepared =>
        tokenizer.decode(prepared.generateIds(ids,count).drop(ids.size).toArray)
      }
    val text=if args(0)=="cuda" then CudaSession.withDevice()(generate(_)) else generate(CpuFloatBackend)
    println(text)

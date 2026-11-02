package scalatorch.models

import java.nio.file.Path

/** Minimal token-ID inference entrypoint; use TextRun for text prompts. */
object LlamaRun:
  def main(args: Array[String]): Unit =
    require(args.length >= 2 && args.length <= 3,
      "usage: LlamaRun MODEL_DIRECTORY COMMA_SEPARATED_TOKEN_IDS [MAX_NEW_TOKENS]")
    val prompt = args(1).split(",").toVector.map(_.toLong)
    val count = if args.length == 3 then args(2).toInt else 1
    val model = LlamaModel.load(Path.of(args(0)))
    println(model.generateIds(prompt, count).mkString(","))
